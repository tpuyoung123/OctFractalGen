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

    def __init__(
        self,
        depth,
        embed_dim,
        cond_embed_dim,
        num_blocks,
        num_heads,
        vq_groups=32,
        num_iters=256,
        patch_size=2048,
        dilation=2,
        use_swin=True,
        use_checkpoint=True,
        attn_dropout=0.0,
        proj_dropout=0.1,
        mask_ratio_min=0.5,
        random_flip=0.0,
        remask_stage=0.7,
        remask_prob=0.1,
        cond_embed_dims=None,
        use_bit_pos_emb=True,
        cond_injection="add",
        cond_cross_attn_heads=4,
    ):
        super().__init__()
        if not (0.0 <= mask_ratio_min < 1.0):
            raise ValueError("mask_ratio_min must be in [0, 1)")
        if cond_embed_dims is None:
            cond_embed_dims = (cond_embed_dim,)
        else:
            cond_embed_dims = tuple(cond_embed_dims)
            if len(cond_embed_dims) == 0:
                raise ValueError("cond_embed_dims must not be empty")
            if cond_embed_dims[-1] != cond_embed_dim:
                raise ValueError("cond_embed_dims[-1] must match cond_embed_dim")
        if cond_injection not in ("add", "film", "cross_attn"):
            raise ValueError(
                f"Unknown cond_injection mode: {cond_injection!r}; "
                "expected one of 'add', 'film', 'cross_attn'"
            )
        if cond_injection == "cross_attn" and cond_cross_attn_heads <= 0:
            raise ValueError("cond_cross_attn_heads must be > 0 for cross_attn")
        if cond_injection == "cross_attn" and embed_dim % cond_cross_attn_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"cond_cross_attn_heads ({cond_cross_attn_heads})"
            )

        self.depth = depth  # terminal depth, e.g. 6
        self.vq_size = 2  # BSQ bit classification: 0/1
        self.vq_groups = vq_groups  # BSQ32 -> 32
        self.num_iters = num_iters
        self.patch_size = patch_size
        self.dilation = dilation
        self.use_swin = use_swin
        self.random_flip = random_flip
        self.remask_stage = remask_stage
        self.remask_prob = remask_prob
        self.cond_embed_dims = cond_embed_dims
        self.cond_context_scale = len(cond_embed_dims) ** -0.5
        # P1.2/1.3: condition injection mode and bit position embedding
        self.use_bit_pos_emb = bool(use_bit_pos_emb)
        self.cond_injection = cond_injection

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
        self.extra_cond_projs = nn.ModuleList(
            [nn.Linear(dim, embed_dim) for dim in cond_embed_dims[:-1]]
        )
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
                embed_dim, cond_cross_attn_heads, batch_first=True
            )
        else:
            self.cond_cross_attn = None

        # mask token for MAR (unrevealed positions)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        # OctGPT-style encoder-decoder backbone. The encoder only processes
        # visible VQ tokens; the decoder then reasons over the full sequence
        # containing both encoded visible tokens and masked tokens.
        enc_blocks = max(num_blocks // 2, 1)
        dec_blocks = max(num_blocks - enc_blocks, 1)
        self.encoder = OctFormer(
            channels=embed_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            patch_size=patch_size,
            dilation=dilation,
            dropout=proj_dropout,
            attn_drop=attn_dropout,
            proj_drop=proj_dropout,
            nempty=False,
            use_checkpoint=use_checkpoint,
            use_swin=use_swin,
            pos_emb=SinPosEmb,
            norm_layer=nn.LayerNorm,
        )
        self.encoder_ln = nn.LayerNorm(embed_dim)

        self.decoder = OctFormer(
            channels=embed_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            patch_size=patch_size,
            dilation=dilation,
            dropout=proj_dropout,
            attn_drop=attn_dropout,
            proj_drop=proj_dropout,
            nempty=False,
            use_checkpoint=use_checkpoint,
            use_swin=use_swin,
            pos_emb=SinPosEmb,
            norm_layer=nn.LayerNorm,
        )
        self.decoder_ln = nn.LayerNorm(embed_dim)

        # VQ head: 32 independent binary classifications (BSQ32)
        # output shape: (N, 2 * vq_groups) -> reshape to (N, vq_groups, 2)
        self.vq_head = nn.Linear(embed_dim, self.vq_size * vq_groups)

        # MAR masking ratio distribution, matching OctGPT's high-mask training.
        # OctGPT hardcodes loc=1.0, scale=0.25, upper bound=1.0.
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0.0, loc=1.0, scale=0.25
        )

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
        return (indices.float() * 2.0 - 1.0) * (1.0 / self.vq_groups**0.5)

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

    def _forward_blocks(self, x, octreeT, blocks):
        x = depth2batch(x, octreeT.indices)
        x = blocks(x, octreeT, context=None)
        x = batch2depth(x, octreeT.indices)
        return x

    def _forward_model(self, x, octree, mask=None):
        nnum_d = x.shape[0]
        if mask is not None:
            if mask.shape[0] != nnum_d:
                raise ValueError(
                    f"VQ mask length must match token count: {mask.shape[0]} vs {nnum_d}"
                )
            mask = mask.to(device=x.device, dtype=torch.bool)
            visible = ~mask
        else:
            visible = None

        # Encoder: OctGPT keeps only visible tokens through data_mask. This
        # generator has no class/buffer tokens, so the all-masked first sample
        # step skips the encoder and lets the decoder process mask+condition.
        if visible is None:
            x_enc = x
            octreeT_encoder = OctreeT(
                octree,
                x_enc.shape[0],
                self.patch_size,
                self.dilation,
                nempty=False,
                depth_list=[self.depth],
                buffer_size=0,
                use_swin=self.use_swin,
            )
            x_enc = self._forward_blocks(x_enc, octreeT_encoder, self.encoder)
            x = self.encoder_ln(x_enc)
        elif visible.any():
            x_enc = x[visible]
            octreeT_encoder = OctreeT(
                octree,
                x_enc.shape[0],
                self.patch_size,
                self.dilation,
                nempty=False,
                depth_list=[self.depth],
                data_mask=mask,
                buffer_size=0,
                use_swin=self.use_swin,
            )
            x_enc = self._forward_blocks(x_enc, octreeT_encoder, self.encoder)
            x_enc = self.encoder_ln(x_enc)
            x = x.clone()
            x[visible] = x_enc

        # Decoder: full sequence, with visible tokens already encoded and
        # masked tokens still carrying mask+condition embeddings.
        octreeT_decoder = OctreeT(
            octree,
            nnum_d,
            self.patch_size,
            self.dilation,
            nempty=False,
            depth_list=[self.depth],
            buffer_size=0,
            use_swin=self.use_swin,
        )
        x = self._forward_blocks(x, octreeT_decoder, self.decoder)
        x = self.decoder_ln(x)
        return x

    # ------------------------------------------------------------------
    # shared encoder-decoder predictor
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
                    f"got {len(cond_list)}"
                )
        else:
            context_features = cond_list
            context_projs = list(self.extra_cond_projs) + [self.cond_proj]
            context_scale = self.cond_context_scale

        # If the whole VQ generator is frozen, do not let its loss reshape the
        # parent split features through a fixed, untrainable VQ head.
        detach_context = self.training and not any(
            p.requires_grad for p in self.parameters()
        )
        cond = None
        for features, proj in zip(context_features, context_projs):
            if features.shape[0] != vq_codes.shape[0]:
                raise ValueError(
                    "VQ context node count must match vq_codes node count: "
                    f"{features.shape[0]} vs {vq_codes.shape[0]}"
                )
            if detach_context:
                features = features.detach()
            projected = proj(features)
            cond = projected if cond is None else cond + projected
        cond = cond * context_scale

        vq_codes = vq_codes.to(dtype=cond.dtype)
        vq_tokens = self.vq_proj(vq_codes)  # (N_d, E)
        if self.bit_pos_emb is not None:
            # bit_pos_emb: (vq_groups, E). Aggregate per-bit position
            # priors weighted by the code value, producing a (N_d, E)
            # additive embedding that distinguishes bit indices.
            vq_tokens = vq_tokens + torch.matmul(vq_codes, self.bit_pos_emb)

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
                x_query = torch.where(mask.unsqueeze(1), masked_query, x_query)
            # nn.MultiheadAttention expects (N, E) -> (1, N, E)
            x_attn, _ = self.cond_cross_attn(
                x_query.unsqueeze(0),
                cond.unsqueeze(0),
                cond.unsqueeze(0),
                need_weights=False,
            )
            x = x_attn.squeeze(0)

        return self._forward_model(x, octree, mask)

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
            (vq_loss, metrics_dict): metrics_dict contains
            bit-level and full-code VQ accuracy.
            Note: BSQ32 has vq_size=2 (per-bit binary), so top5 degenerates
            to top1 (topk=min(5, 2-1)=1), consistent with OctGPT.
        """
        nnum_d = octree.nnum[self.depth]
        target_vq = targets["vq"]  # (N, 32) in {0, 1}
        target_vq_code = targets.get("vq_zq")
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

        if self.training:
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
            -1, self.vq_groups, self.vq_size
        )  # (N, 32, 2)

        # ---- Loss computation (OctGPT-aligned) ----
        # Plain masked cross_entropy: CE on masked positions only.
        # When random_flip > 0, OctGPT computes CE on all positions instead
        # (so flipped revealed positions are also supervised).
        if mask is None or use_random_flip:
            logits_flat = vq_logits.reshape(-1, self.vq_size)
            target_flat = target_vq.reshape(-1)
            loss = F.cross_entropy(logits_flat, target_flat)
        else:
            pred_logits = vq_logits[mask].reshape(-1, self.vq_size)
            pred_target = target_vq[mask].reshape(-1)
            loss = F.cross_entropy(pred_logits, pred_target)

        with torch.no_grad():
            acc = compute_vq_accuracy(
                vq_logits,
                target_vq,
                mask=mask,
                vq_groups=self.vq_groups,
                vq_size=self.vq_size,
                topk=5,
                device=octree.device,
            )

        return loss, {
            "vq_loss": loss.detach(),
            **acc,
        }

    # ------------------------------------------------------------------
    # generation (MAR iterative sampling + VQVAE decode)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        cond_list,
        octree,
        vqvae,
        num_iter=None,
        temperature=0.5,
        visualize=False,
        return_raw_octree=False,
    ):
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
        vq_pred = torch.zeros(nnum_d, self.vq_groups, dtype=torch.long, device=device)
        vq_code = torch.zeros(nnum_d, self.vq_groups, dtype=torch.float, device=device)

        num_iter = min(num_iter, nnum_d)

        for step in range(num_iter):
            # forward with current mask (revealed=BSQ zq, masked=mask_token+cond)
            x = self._encode(octree, vq_code, cond_list, mask)
            vq_logits = self.vq_head(x).reshape(-1, self.vq_groups, self.vq_size)

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
                    vq_logits, vq_pred, mask, topk=5, remask_prob=self.remask_prob
                )
                mask_to_pred = mask_to_pred | remask

            # sample VQ codes at positions revealed this step
            cur_temperature = temperature * ((num_iter - step) / num_iter)
            if not mask_to_pred.any():
                continue
            sampled = sample(
                vq_logits[mask_to_pred].reshape(-1, self.vq_size),
                temperature=cur_temperature,
            )
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
            zq, self.depth, doctree_in, doctree_out, update_octree=True
        )

        return output["neural_mpu"]
