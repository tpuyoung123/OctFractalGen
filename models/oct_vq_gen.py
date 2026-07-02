"""Terminal-level VQ generator for OctFractalGen.

Predicts VQ codes (BSQ32: 32 independent binary classifications) at the
terminal octree depth, then decodes geometry via the frozen pretrained VQVAE.
Analogous to FractalGen's PixelLoss as the final fractal level.

Uses OctFormer (sparse windowed attention with RoPE) as the backbone.
Input is VQ-code embeddings (teacher forcing during training, iterative
refinement during sampling), so revealed positions carry VQ semantics.
"""

import math
import copy
import numpy as np
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.octformer import OctFormer, OctreeT
from models.positional_embedding import SinPosEmb
from utils.utils import sample, depth2batch, batch2depth


class OctVQGenerator(nn.Module):
    """Terminal fractal level: predicts VQ codes at depth 6, decodes via VQVAE.

    Input is VQ-code embeddings (revealed positions) + parent condition.
    Masked positions use mask_token. This enables iterative refinement during
    MAR sampling: already-predicted VQ codes feed back into the encoder.
    """

    def __init__(self, depth, embed_dim, cond_embed_dim, num_blocks, num_heads,
                 vq_groups=32, num_iters=256, generator_type="mar",
                 patch_size=1024, dilation=2, use_swin=True, use_checkpoint=True,
                 attn_dropout=0.0, proj_dropout=0.1,
                 mask_ratio_min=0.5, random_flip=0.1,
                 remask_stage=0.7, remask_prob=0.1,
                 loss_weight=1.0, denoise_weight=0.3,
                 full_depth=3, max_depth=8):
        super().__init__()
        self.depth = depth                 # terminal depth, e.g. 6
        self.embed_dim = embed_dim
        self.vq_groups = vq_groups         # BSQ32 -> 32
        self.num_iters = num_iters
        self.generator_type = generator_type
        self.patch_size = patch_size
        self.dilation = dilation
        self.use_swin = use_swin
        self.random_flip = random_flip
        self.remask_stage = remask_stage
        self.remask_prob = remask_prob
        self.loss_weight = loss_weight
        self.denoise_weight = denoise_weight

        # VQ code projection: BSQ32 quantized code zq -> embedding.
        # For BSQ, zq is (-1/+1) / sqrt(32), matching OctGPT.
        self.vq_proj = nn.Linear(vq_groups, embed_dim)

        # condition projection: parent features -> current embed_dim (bias)
        self.cond_proj = nn.Linear(cond_embed_dim, embed_dim)

        # mask token for MAR (unrevealed positions)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        # OctFormer backbone (windowed attention with RoPE)
        self.transformer = OctFormer(
            channels=embed_dim, num_blocks=num_blocks, num_heads=num_heads,
            patch_size=patch_size, dilation=dilation,
            dropout=proj_dropout, attn_drop=attn_dropout, proj_drop=proj_dropout,
            nempty=False, use_checkpoint=use_checkpoint, use_swin=use_swin,
            pos_emb=SinPosEmb, norm_layer=nn.LayerNorm)
        self.norm = nn.LayerNorm(embed_dim)

        # VQ head: 32 independent binary classifications (BSQ32)
        # output shape: (N, 2 * vq_groups) -> reshape to (N, vq_groups, 2)
        self.vq_head = nn.Linear(embed_dim, 2 * vq_groups)

        # MAR masking ratio distribution, matching OctGPT's high-mask training.
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # BSQ / remask helpers
    # ------------------------------------------------------------------
    def _bsq_indices_to_code(self, indices):
        return (indices.float() * 2.0 - 1.0) * (1.0 / self.vq_groups ** 0.5)

    def _get_correct_topk(self, logits, targets, topk=1):
        topk = min(topk, logits.shape[-1] - 1)
        topk = torch.topk(logits, topk, dim=-1).indices
        return topk.eq(targets.unsqueeze(-1).expand_as(topk))

    def _get_remask(self, logits, tokens, mask, remask_prob=0.1, topk=5):
        correct_topk = self._get_correct_topk(logits, tokens, topk=topk)
        correct_by_group = correct_topk.any(dim=-1)
        num_incorrect = (~correct_by_group).sum(dim=-1)
        num_incorrect[mask] = 0

        num_remask = int(num_incorrect.bool().sum().item() * remask_prob)
        remask = torch.zeros_like(mask, dtype=torch.bool)
        if num_remask <= 0:
            return remask

        remask_scores = num_incorrect.float()
        remask_indices = torch.topk(remask_scores, num_remask).indices
        remask[remask_indices] = True
        return remask & ~mask

    # ------------------------------------------------------------------
    # shared encoder
    # ------------------------------------------------------------------
    def _encode(self, octree, vq_codes, cond_list, mask=None):
        """Encode nodes at self.depth.

        Args:
            octree: octree (GT during training, growing during sampling)
            vq_codes: (N_d, vq_groups) BSQ quantized code zq.
                Unrevealed positions can be any value (will be overwritten).
            cond_list: [parent_features] with shape (N_d, cond_embed_dim)
            mask: bool tensor (N_d,) True=unrevealed (use mask_token).
                None for no masking.
        Returns:
            x: (N_d, embed_dim) encoded features
        """
        parent_features = cond_list[0]
        cond = self.cond_proj(parent_features)               # (N_d, E) bias
        vq_codes = vq_codes.to(dtype=parent_features.dtype)
        x = self.vq_proj(vq_codes) + cond                    # (N_d, E) token + cond

        if mask is not None:
            masked_x = self.mask_token.to(x.dtype) + cond
            x = torch.where(mask.unsqueeze(1), masked_x, x)

        # Build OctreeT for windowed attention at this depth
        nnum_d = x.shape[0]
        octreeT = OctreeT(
            octree, nnum_d, self.patch_size, self.dilation,
            nempty=False, depth_list=[self.depth], buffer_size=0,
            use_swin=self.use_swin)

        # depth layout -> batch layout -> OctFormer -> batch -> depth layout
        x = depth2batch(x, octreeT.indices)
        x = self.transformer(x, octreeT, context=None)
        x = batch2depth(x, octreeT.indices)

        x = self.norm(x)
        return x

    # ------------------------------------------------------------------
    # training forward (teacher forcing) — called as next_fractal(octree, cond_list, targets)
    # ------------------------------------------------------------------
    def forward(self, octree, cond_list, targets):
        """Training: compute VQ loss and VQ prediction accuracy.

        Args:
            octree: GT octree
            cond_list: [parent_features] (N_6, cond_embed_dim)
            targets: dict with 'vq' key -> (N_6, vq_groups) in {0, 1}
        Returns:
            (weighted_vq_loss, metrics_dict): metrics_dict contains
            bit-level and full-code VQ accuracy.
            Note: BSQ32 has vq_size=2 (per-bit binary), so top5 degenerates
            to top1 (topk=min(5, 2-1)=1), consistent with OctGPT.
        """
        nnum_d = octree.nnum[self.depth]
        target_vq = targets['vq']  # (N, 32) in {0, 1}
        target_vq_code = targets.get('vq_zq')
        if target_vq_code is None:
            target_vq_code = self._bsq_indices_to_code(target_vq)
        target_vq_code = target_vq_code.to(octree.device)

        input_vq = target_vq
        input_vq_code = target_vq_code
        use_random_flip = self.training and self.random_flip > 0.0
        if use_random_flip:
            flip = torch.rand_like(target_vq.float()) < self.random_flip
            input_vq = torch.where(flip, 1 - target_vq, target_vq)
            input_vq_code = self._bsq_indices_to_code(input_vq).to(octree.device)

        if self.training and self.generator_type == "mar":
            mask_rate = self.mask_ratio_generator.rvs(1)[0]
            num_masked = max(int(np.ceil(nnum_d * mask_rate)), 1)
            orders = torch.randperm(nnum_d, device=octree.device)
            mask = torch.zeros(nnum_d, dtype=torch.bool, device=octree.device)
            mask[orders[:num_masked]] = True
        else:
            mask = None

        # teacher forcing: revealed positions use BSQ zq embedding, as in OctGPT.
        x = self._encode(octree, input_vq_code, cond_list, mask)
        vq_logits = self.vq_head(x).reshape(-1, self.vq_groups, 2)  # (N, 32, 2)

        # Loss computation: focus prediction gradient on masked positions.
        # When random_flip is active, OctGPT computes loss on ALL tokens
        # (denoising + prediction). But the small L3 model (120 dim / 2 blocks)
        # cannot handle both tasks effectively, causing vq_top5_acc to plateau
        # at ~0.75. Fix: main prediction loss on masked positions only; add a
        # low-weight denoising loss on flipped non-masked positions so the
        # model still learns noise robustness without diluting prediction
        # gradient.
        if mask is None:
            # eval / non-MAR: loss on all tokens
            logits_flat = vq_logits.reshape(-1, 2)
            target_flat = target_vq.reshape(-1)
            raw_loss = F.cross_entropy(logits_flat, target_flat)
            logits_for_metric = vq_logits
            target_for_metric = target_vq
        else:
            # MAR training: prediction loss on masked positions only
            pred_logits = vq_logits[mask].reshape(-1, 2)
            pred_target = target_vq[mask].reshape(-1)
            raw_loss = F.cross_entropy(pred_logits, pred_target)
            logits_for_metric = vq_logits[mask]
            target_for_metric = target_vq[mask]

            # low-weight denoising loss on flipped non-masked positions
            if use_random_flip:
                revealed = ~mask
                flipped = revealed & (input_vq != target_vq).any(dim=-1)
                if flipped.any():
                    denoise_logits = vq_logits[flipped].reshape(-1, 2)
                    denoise_target = target_vq[flipped].reshape(-1)
                    denoise_loss = F.cross_entropy(denoise_logits, denoise_target)
                    raw_loss = raw_loss + self.denoise_weight * denoise_loss

        loss = raw_loss * self.loss_weight

        with torch.no_grad():
            vq_pred = logits_for_metric.argmax(dim=-1)
            vq_bit_acc = (vq_pred == target_for_metric).float().mean()
            vq_code_acc = (vq_pred == target_for_metric).all(dim=-1).float().mean()
            vq_mask_ratio = (
                mask.float().mean()
                if mask is not None
                else torch.zeros((), device=octree.device)
            )

        return loss, {
            'vq_loss': raw_loss.detach(),
            'vq_loss_weighted': loss.detach(),
            'vq_bit_acc': vq_bit_acc,
            'vq_top5_acc': vq_bit_acc,  # backward-compatible alias
            'vq_code_acc': vq_code_acc,
            'vq_mask_ratio': vq_mask_ratio,
        }

    # ------------------------------------------------------------------
    # generation (MAR iterative sampling + VQVAE decode)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, cond_list, octree, vqvae, num_iter=None,
               temperature=0.5, visualize=False, return_raw_octree=False):
        """Generation: MAR sample VQ codes -> VQVAE decode -> neural_mpu.

        MaskGIT-style iterative refinement: revealed positions use
        vq_proj(pred), masked positions use mask_token.

        Args:
            return_raw_octree: if True, skip VQVAE decode and return
                (octree, vq_pred) at depth 6 instead of neural_mpu.
        """
        if num_iter is None:
            num_iter = self.num_iters

        nnum_d = octree.nnum[self.depth]
        device = octree.device

        # start: all masked, vq_pred/code initialized as placeholders
        mask = torch.ones(nnum_d, dtype=torch.bool, device=device)
        orders = torch.randperm(nnum_d, device=device)
        vq_pred = torch.zeros(nnum_d, self.vq_groups, dtype=torch.long,
                              device=device)
        vq_code = torch.zeros(nnum_d, self.vq_groups, dtype=torch.float,
                              device=device)

        num_iter = min(num_iter, nnum_d)

        for step in range(num_iter):
            # forward with current mask (revealed=BSQ zq, masked=mask_token+cond)
            x = self._encode(octree, vq_code, cond_list, mask)
            vq_logits = self.vq_head(x).reshape(-1, self.vq_groups, 2)

            # cosine schedule: number of masked positions for NEXT step
            mask_ratio = math.cos(math.pi / 2.0 * (step + 1) / num_iter)
            mask_len = int(np.floor(nnum_d * mask_ratio))
            mask_len = max(1, min(int(mask.sum().item()) - 1, mask_len))

            # mask_next: keep orders[:mask_len] masked (shrinking subset)
            mask_next = torch.zeros(nnum_d, dtype=torch.bool, device=device)
            mask_next[orders[:mask_len]] = True

            # mask_to_pred: positions revealed THIS step = mask - mask_next
            if step >= num_iter - 1:
                mask_to_pred = mask.clone()
            else:
                mask_to_pred = mask ^ mask_next

            mask = mask_next

            # OctGPT-style remask: revisit low-confidence revealed VQ codes.
            if step > num_iter * self.remask_stage and self.remask_prob > 0.0:
                remask = self._get_remask(
                    vq_logits, vq_pred, mask,
                    topk=5, remask_prob=self.remask_prob)
                mask_to_pred = mask_to_pred | remask

            # sample VQ codes at positions revealed this step
            cur_temperature = temperature * ((num_iter - step) / num_iter)
            if not mask_to_pred.any():
                continue
            sampled = sample(
                vq_logits[mask_to_pred].reshape(-1, 2),
                temperature=cur_temperature)
            sampled = sampled.reshape(-1, self.vq_groups)
            vq_pred[mask_to_pred] = sampled.long()
            zq = vqvae.quantizer.extract_code(sampled)
            vq_code[mask_to_pred] = zq.float()

        # ------ Early return: raw octree + VQ codes (skip VQVAE decode) ------
        if return_raw_octree:
            # Build neighs only up to depth 6 (terminal). octree.depth may be
            # 8 (from init_octree), but children[6/7] are None since we only
            # grew to depth 6 via seq2octree. construct_neigh(d) needs
            # children[d-1], so we can safely build up to d=self.depth.
            for d in range(octree.full_depth, self.depth + 1):
                if octree.neighs[d] is None:
                    octree.construct_neigh(d)
            return octree, vq_pred

        # ------ VQVAE decode: depth 6 -> depth 8 -> mesh ------
        zq = vqvae.quantizer.extract_code(vq_pred)  # (N_6, 32) continuous

        # Grow the octree from self.depth (6) to depth 8 with all-split
        # (split=1) so that the VQVAE decoder has a complete octree structure.
        # update_octree=False is used to avoid an OctreeD deepcopy batch_id bug.
        for d in range(self.depth, 8):
            split_one = torch.ones(
                octree.nnum[d], device=device).long()
            octree.octree_split(split_one, d)
            octree.octree_grow(d + 1)

        # build neighs for VQVAE convolutions (encoder + decoder)
        for d in range(octree.full_depth, octree.depth + 1):
            octree.construct_neigh(d)

        from ognn.octreed import OctreeD
        doctree = OctreeD(octree)
        # update_octree=False: use the all-split structure as-is;
        # the decoder only predicts signals (neural_mpu) without refining
        # the octree structure. This avoids OctreeD deepcopy batch_id bug.
        output = vqvae.decode_code(
            zq, self.depth, doctree,
            copy.deepcopy(doctree), update_octree=False)

        return output['neural_mpu']
