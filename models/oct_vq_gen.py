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
from utils.metrics import get_correct_topk, compute_vq_accuracy
from utils.utils import sample, depth2batch, batch2depth


def _patch_octreed_split_batch_id():
    """Patch OctreeD.octree_split for ocnn/ognn batch_id mismatch.

    OctreeD.batch_id(depth) returns dual-graph batch ids, whose length can
    include retained ancestor leaves. ocnn.Octree.octree_split expects
    current-depth octree batch ids with length == split.shape[0]. Temporarily
    switching graph.batch_id to keys[depth] >> 48 keeps OctreeD updates usable.
    """
    import ognn.octreed as _octreed

    orig_split = _octreed.OctreeD.octree_split
    if getattr(orig_split, "_octfractalgen_batch_id_patch", False):
        return

    def patched_split(self, split, depth):
        graph = self.graphs[depth] if depth < len(self.graphs) else None
        saved_batch_id = None
        if graph is not None and graph.batch_id is not None:
            keys = self.keys[depth] if depth < len(self.keys) else None
            if keys is not None and keys.shape[0] == split.shape[0]:
                saved_batch_id = graph.batch_id
                graph.batch_id = keys >> 48
        try:
            return orig_split(self, split, depth)
        finally:
            if saved_batch_id is not None:
                self.graphs[depth].batch_id = saved_batch_id

    patched_split._octfractalgen_batch_id_patch = True
    patched_split._octfractalgen_orig_split = orig_split
    _octreed.OctreeD.octree_split = patched_split


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
                 mask_ratio_min=0.5, mask_ratio_max=1.0,
                 mask_ratio_loc=1.0, mask_ratio_scale=0.25,
                 random_flip=0.1,
                 remask_stage=0.7, remask_prob=0.1,
                 loss_weight=1.0, denoise_weight=0.3,
                 loss_mode="masked", label_smoothing=0.0,
                 mask_loss_weight=2.0, reveal_loss_weight=0.5,
                 bit_weight_mode="uniform", bit_weight_ema_decay=0.99,
                 full_depth=3, max_depth=8,
                 cond_embed_dims=None,
                 use_bit_pos_emb=True, cond_injection="add",
                 cond_cross_attn_heads=4):
        super().__init__()
        if not (0.0 <= mask_ratio_min < mask_ratio_max <= 1.0):
            raise ValueError(
                "mask_ratio_min/max must satisfy 0 <= min < max <= 1")
        if mask_ratio_scale <= 0.0:
            raise ValueError("mask_ratio_scale must be > 0")
        if not (0.0 <= mask_ratio_loc <= 1.0):
            raise ValueError("mask_ratio_loc must be in [0, 1]")
        if loss_mode not in ("masked", "all_weighted"):
            raise ValueError(f"Unknown VQ loss mode: {loss_mode}")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1)")
        if mask_loss_weight < 0.0 or reveal_loss_weight < 0.0:
            raise ValueError("mask/reveal loss weights must be non-negative")
        if mask_loss_weight == 0.0 and reveal_loss_weight == 0.0:
            raise ValueError("At least one VQ loss position weight must be > 0")
        if bit_weight_mode not in ("uniform", "batch_var", "batch_var_ema"):
            raise ValueError(f"Unknown VQ bit weight mode: {bit_weight_mode}")
        if not (0.0 <= bit_weight_ema_decay < 1.0):
            raise ValueError("bit_weight_ema_decay must be in [0, 1)")
        if cond_embed_dims is None:
            cond_embed_dims = (cond_embed_dim,)
        else:
            cond_embed_dims = tuple(cond_embed_dims)
            if len(cond_embed_dims) == 0:
                raise ValueError("cond_embed_dims must not be empty")
            if cond_embed_dims[-1] != cond_embed_dim:
                raise ValueError(
                    "cond_embed_dims[-1] must match cond_embed_dim")
        if cond_injection not in ("add", "film", "cross_attn"):
            raise ValueError(
                f"Unknown cond_injection mode: {cond_injection!r}; "
                "expected one of 'add', 'film', 'cross_attn'")
        if cond_injection == "cross_attn" and cond_cross_attn_heads <= 0:
            raise ValueError("cond_cross_attn_heads must be > 0 for cross_attn")
        if cond_injection == "cross_attn" and embed_dim % cond_cross_attn_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"cond_cross_attn_heads ({cond_cross_attn_heads})")

        self.depth = depth                 # terminal depth, e.g. 6
        self.embed_dim = embed_dim
        self.vq_size = 2                   # BSQ bit classification: 0/1
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
        # P0.1 loss redesign
        self.loss_mode = loss_mode            # "masked" (legacy) | "all_weighted"
        self.label_smoothing = label_smoothing
        self.mask_loss_weight = mask_loss_weight
        self.reveal_loss_weight = reveal_loss_weight
        self.bit_weight_mode = bit_weight_mode
        self.bit_weight_ema_decay = bit_weight_ema_decay
        self.cond_embed_dims = cond_embed_dims
        self.cond_context_scale = len(cond_embed_dims) ** -0.5
        # P1.2/1.3: condition injection mode and bit position embedding
        self.use_bit_pos_emb = bool(use_bit_pos_emb)
        self.cond_injection = cond_injection
        self.cond_cross_attn_heads = cond_cross_attn_heads
        self.register_buffer(
            "bit_var_ema", torch.ones(vq_groups), persistent=False)

        # VQ code projection: BSQ32 quantized code zq -> embedding.
        # For BSQ, zq is (-1/+1) / sqrt(32), matching OctGPT.
        self.vq_proj = nn.Linear(vq_groups, embed_dim)

        # P2.8: bit position embedding so the model can distinguish BSQ bit
        # indices (the 32 bits carry different geometric information: high
        # vs low-order bits on the L2 sphere).
        if self.use_bit_pos_emb:
            self.bit_pos_emb = nn.Parameter(torch.zeros(vq_groups, embed_dim))
        else:
            self.bit_pos_emb = None

        # condition projections: multi-level split features -> current embed_dim
        self.extra_cond_projs = nn.ModuleList([
            nn.Linear(dim, embed_dim) for dim in cond_embed_dims[:-1]
        ])
        # Keep the direct-parent projection name stable for old diagnostics.
        self.cond_proj = nn.Linear(cond_embed_dim, embed_dim)

        # P1.2/1.3: condition injection modules.
        #   "add":      x = vq_proj(codes) + cond           (legacy)
        #   "film":     x = vq_proj(codes) * (1+gamma) + beta
        #   "cross_attn": x = CrossAttn(Q=vq_tokens, K=V=cond)
        if self.cond_injection == "film":
            self.film = nn.Sequential(
                nn.Linear(embed_dim, 2 * embed_dim),
                nn.GELU(),
                nn.Linear(2 * embed_dim, 2 * embed_dim),
            )
        else:
            self.film = None
        if self.cond_injection == "cross_attn":
            self.cond_cross_attn = nn.MultiheadAttention(
                embed_dim, cond_cross_attn_heads, batch_first=True)
        else:
            self.cond_cross_attn = None

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
        self.vq_head = nn.Linear(embed_dim, self.vq_size * vq_groups)

        # MAR masking ratio distribution, matching OctGPT's high-mask training.
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - mask_ratio_loc) / mask_ratio_scale,
            (mask_ratio_max - mask_ratio_loc) / mask_ratio_scale,
            loc=mask_ratio_loc, scale=mask_ratio_scale)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        if self.bit_pos_emb is not None:
            nn.init.normal_(self.bit_pos_emb, std=0.02)
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

    def _get_remask(self, logits, tokens, mask, remask_prob=0.1, topk=5):
        correct_topk = get_correct_topk(logits, tokens, topk=topk)
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

    def _get_bit_weight(self, target_vq, dtype):
        if self.bit_weight_mode == "uniform" or not self.training:
            return None

        bit_var = target_vq.float().var(dim=0, unbiased=False).clamp_min(1e-4)
        if self.bit_weight_mode == "batch_var_ema":
            with torch.no_grad():
                decay = self.bit_weight_ema_decay
                self.bit_var_ema.mul_(decay).add_(bit_var.detach(), alpha=1.0 - decay)
            bit_var = self.bit_var_ema

        bit_w = bit_var / bit_var.mean().clamp_min(1e-6)
        return bit_w.clamp(0.25, 4.0).to(dtype)

    # ------------------------------------------------------------------
    # shared encoder
    # ------------------------------------------------------------------
    def _encode(self, octree, vq_codes, cond_list, mask=None):
        """Encode nodes at self.depth.

        Args:
            octree: octree (GT during training, growing during sampling)
            vq_codes: (N_d, vq_groups) BSQ quantized code zq.
                Unrevealed positions can be any value (will be overwritten).
            cond_list: context feature tensors aligned to terminal depth.
                When multi-context is enabled this is [d3, d4, d5] split
                features after repeated unpooling to depth self.depth.
            mask: bool tensor (N_d,) True=unrevealed (use mask_token).
                None for no masking.
        Returns:
            x: (N_d, embed_dim) encoded features
        """
        if len(cond_list) != len(self.cond_embed_dims):
            if len(cond_list) == 1:
                context_features = cond_list
                context_projs = [self.cond_proj]
                context_scale = 1.0
            else:
                raise ValueError(
                    f"Expected {len(self.cond_embed_dims)} VQ context tensors, "
                    f"got {len(cond_list)}")
        else:
            context_features = cond_list
            context_projs = list(self.extra_cond_projs) + [self.cond_proj]
            context_scale = self.cond_context_scale

        # If the whole VQ generator is frozen, do not let its loss reshape the
        # parent split features through a fixed, untrainable VQ head.
        detach_context = (
            self.training and not any(p.requires_grad for p in self.parameters())
        )
        cond = None
        for features, proj in zip(context_features, context_projs):
            if features.shape[0] != vq_codes.shape[0]:
                raise ValueError(
                    "VQ context node count must match vq_codes node count: "
                    f"{features.shape[0]} vs {vq_codes.shape[0]}")
            if detach_context:
                features = features.detach()
            projected = proj(features)
            cond = projected if cond is None else cond + projected
        cond = cond * context_scale

        vq_codes = vq_codes.to(dtype=cond.dtype)
        vq_tokens = self.vq_proj(vq_codes)                 # (N_d, E)
        if self.bit_pos_emb is not None:
            # bit_pos_emb: (vq_groups, E). Aggregate per-bit position
            # priors weighted by the code value, producing a (N_d, E)
            # additive embedding that distinguishes bit indices.
            vq_tokens = vq_tokens + torch.matmul(
                vq_codes, self.bit_pos_emb)

        # Condition injection: add (legacy) / FiLM / cross-attention.
        if self.cond_injection == "add":
            x = vq_tokens + cond
            if mask is not None:
                masked_x = self.mask_token.to(x.dtype) + cond
                x = torch.where(mask.unsqueeze(1), masked_x, x)
        elif self.cond_injection == "film":
            gamma, beta = self.film(cond).chunk(2, dim=-1)
            x = vq_tokens * (1.0 + gamma) + beta
            if mask is not None:
                masked_x = self.mask_token.to(x.dtype) * (1.0 + gamma) + beta
                x = torch.where(mask.unsqueeze(1), masked_x, x)
        else:  # cross_attn
            # Q = vq_tokens (revealed) or mask_token (masked); K = V = cond.
            x_query = vq_tokens
            if mask is not None:
                masked_query = self.mask_token.to(x_query.dtype)
                x_query = torch.where(
                    mask.unsqueeze(1), masked_query, x_query)
            # nn.MultiheadAttention expects (N, E) -> (1, N, E)
            x_attn, _ = self.cond_cross_attn(
                x_query.unsqueeze(0), cond.unsqueeze(0), cond.unsqueeze(0),
                need_weights=False)
            x = x_attn.squeeze(0)

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
            cond_list: terminal-depth context feature tensors. With the
                default OctFractalGen wiring this is [d3, d4, d5] split
                features aligned to N_6 nodes.
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
        vq_logits = self.vq_head(x).reshape(
            -1, self.vq_groups, self.vq_size)  # (N, 32, 2)

        bit_w = self._get_bit_weight(target_vq, vq_logits.dtype)

        # ---- Loss computation ----
        # Two modes:
        #   "masked" (legacy): CE only on masked positions + optional denoise
        #       on flipped non-masked. Wastes ~50% revealed-position supervision.
        #   "all_weighted" (P0.1): CE on all positions, masked weighted by
        #       mask_loss_weight, revealed by reveal_loss_weight. Denoise is
        #       subsumed (flipped revealed positions are already penalized).
        if mask is None:
            # eval / non-MAR: loss on all tokens
            logits_flat = vq_logits.reshape(-1, self.vq_size)
            target_flat = target_vq.reshape(-1)
            raw_loss = F.cross_entropy(
                logits_flat, target_flat,
                label_smoothing=self.label_smoothing)
        elif self.loss_mode == "all_weighted":
            # P0.1: per-bit CE on all positions, then weight by mask/reveal
            ce_per_bit = F.cross_entropy(
                vq_logits.reshape(-1, self.vq_size),
                target_vq.reshape(-1),
                reduction='none',
                label_smoothing=self.label_smoothing,
            ).reshape(-1, self.vq_groups)  # (N, 32)
            if bit_w is not None:
                ce_per_bit = ce_per_bit * bit_w  # broadcast (32,)
            ce_per_pos = ce_per_bit.mean(dim=-1)  # (N,)
            weight = torch.where(
                mask,
                torch.full_like(ce_per_pos, self.mask_loss_weight),
                torch.full_like(ce_per_pos, self.reveal_loss_weight),
            )
            raw_loss = (ce_per_pos * weight).sum() / weight.sum().clamp_min(1e-6)
        else:
            # legacy "masked" mode: prediction loss on masked positions only
            pred_logits = vq_logits[mask].reshape(-1, self.vq_size)
            pred_target = target_vq[mask].reshape(-1)
            raw_loss = F.cross_entropy(
                pred_logits, pred_target,
                label_smoothing=self.label_smoothing)

            # low-weight denoising loss on flipped non-masked positions
            if use_random_flip and self.denoise_weight > 0.0:
                revealed = ~mask
                flipped = revealed & (input_vq != target_vq).any(dim=-1)
                if flipped.any():
                    denoise_logits = vq_logits[flipped].reshape(-1, self.vq_size)
                    denoise_target = target_vq[flipped].reshape(-1)
                    denoise_loss = F.cross_entropy(denoise_logits, denoise_target)
                    raw_loss = raw_loss + self.denoise_weight * denoise_loss

        loss = raw_loss * self.loss_weight

        with torch.no_grad():
            acc = compute_vq_accuracy(
                vq_logits, target_vq, mask=mask,
                vq_groups=self.vq_groups, vq_size=self.vq_size,
                topk=5, device=octree.device)

        return loss, {
            'vq_loss': raw_loss.detach(),
            'vq_loss_weighted': loss.detach(),
            **acc,
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
            vq_logits = self.vq_head(x).reshape(
                -1, self.vq_groups, self.vq_size)

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
                vq_logits[mask_to_pred].reshape(-1, self.vq_size),
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

        # Build neighborhoods only through the code depth. The VQVAE decoder
        # updates/grows d7-d8 itself when update_octree=True.
        for d in range(octree.full_depth, self.depth + 1):
            octree.construct_neigh(d)

        from ognn.octreed import OctreeD
        _patch_octreed_split_batch_id()

        # max_depth=self.depth avoids requiring pre-existing d7/d8 graphs.
        # The output dual octree is a separate copy so decoder split updates do
        # not mutate the input code octree used for encoder/skip alignment.
        doctree_in = OctreeD(octree, max_depth=self.depth)
        doctree_out = copy.deepcopy(doctree_in)
        output = vqvae.decode_code(
            zq, self.depth, doctree_in, doctree_out, update_octree=True)

        return output['neural_mpu']
