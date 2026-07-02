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

        # VQ code projection: BSQ32 (32-dim binary) -> embedding
        # Revealed positions use this; equivalent to OctGPT's vq_proj on zq.
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

        # MAR masking ratio distribution
        self.mask_ratio_generator = stats.truncnorm(-4, 0, loc=1.0, scale=0.25)

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
    # shared encoder
    # ------------------------------------------------------------------
    def _encode(self, octree, vq_codes, cond_list, mask=None):
        """Encode nodes at self.depth.

        Args:
            octree: octree (GT during training, growing during sampling)
            vq_codes: (N_d, vq_groups) long in {0, 1}; revealed VQ codes.
                Unrevealed positions can be any value (will be overwritten).
            cond_list: [parent_features] with shape (N_d, cond_embed_dim)
            mask: bool tensor (N_d,) True=unrevealed (use mask_token).
                None for no masking.
        Returns:
            x: (N_d, embed_dim) encoded features
        """
        parent_features = cond_list[0]
        cond = self.cond_proj(parent_features)               # (N_d, E) bias
        x = self.vq_proj(vq_codes.float()) + cond            # (N_d, E) token + cond

        if mask is not None:
            x = torch.where(mask.unsqueeze(1), self.mask_token.to(x.dtype), x)

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
        """Training: compute VQ loss and top5 accuracy.

        Args:
            octree: GT octree
            cond_list: [parent_features] (N_6, cond_embed_dim)
            targets: dict with 'vq' key -> (N_6, vq_groups) in {0, 1}
        Returns:
            (vq_loss, metrics_dict): metrics_dict contains 'vq_top5_acc'.
            Note: BSQ32 has vq_size=2 (per-bit binary), so top5 degenerates
            to top1 (topk=min(5, 2-1)=1), consistent with OctGPT.
        """
        nnum_d = octree.nnum[self.depth]
        target_vq = targets['vq']  # (N, 32) in {0, 1}

        if self.training and self.generator_type == "mar":
            mask_rate = self.mask_ratio_generator.rvs(1)[0]
            num_masked = max(int(np.ceil(nnum_d * mask_rate)), 1)
            orders = torch.randperm(nnum_d, device=octree.device)
            mask = torch.zeros(nnum_d, dtype=torch.bool, device=octree.device)
            mask[orders[:num_masked]] = True
        else:
            mask = None

        # teacher forcing: revealed positions use GT VQ code embedding
        x = self._encode(octree, target_vq, cond_list, mask)
        vq_logits = self.vq_head(x).reshape(-1, self.vq_groups, 2)  # (N, 32, 2)

        if mask is not None:
            logits_flat = vq_logits[mask].reshape(-1, 2)
            target_flat = target_vq[mask].reshape(-1)
            loss = F.cross_entropy(logits_flat, target_flat)
            # top5 accuracy (bit-level top1, since vq_size=2)
            with torch.no_grad():
                vq_pred = logits_flat.argmax(dim=-1)
                vq_top5_acc = (vq_pred == target_flat).float().mean()
        else:
            logits_flat = vq_logits.reshape(-1, 2)
            target_flat = target_vq.reshape(-1)
            loss = F.cross_entropy(logits_flat, target_flat)
            with torch.no_grad():
                vq_pred = logits_flat.argmax(dim=-1)
                vq_top5_acc = (vq_pred == target_flat).float().mean()

        return loss, {'vq_top5_acc': vq_top5_acc}

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

        # start: all masked, vq_pred initialized to 0 (placeholder)
        mask = torch.ones(nnum_d, dtype=torch.bool, device=device)
        orders = torch.randperm(nnum_d, device=device)
        vq_pred = torch.zeros(nnum_d, self.vq_groups, dtype=torch.long,
                              device=device)

        num_iter = min(num_iter, nnum_d)

        for step in range(num_iter):
            # forward with current mask (revealed=vq_proj, masked=mask_token)
            x = self._encode(octree, vq_pred, cond_list, mask)
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

            # sample VQ codes at positions revealed this step
            sampled = sample(
                vq_logits[mask_to_pred].reshape(-1, 2),
                temperature=temperature)
            sampled = sampled.reshape(-1, self.vq_groups)
            vq_pred[mask_to_pred] = sampled.long()

            mask = mask_next

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
