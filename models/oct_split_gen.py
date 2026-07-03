import math
import numpy as np
import scipy.stats as stats

import torch
import torch.nn as nn

from models.octformer import OctFormer, OctreeT
from models.positional_embedding import SinPosEmb
from utils.utils import octree_copy_unpool, seq2octree, sample, depth2batch, batch2depth


class OctSplitGenerator(nn.Module):
    def __init__(
        self,
        depth,
        embed_dim,
        cond_embed_dim,
        num_blocks,
        num_heads,
        generator_type="mar",
        patch_size=1024,
        dilation=2,
        use_swin=True,
        use_checkpoint=True,
        attn_dropout=0.0,
        proj_dropout=0.1,
        propagate_cond_context=True,
    ):
        super().__init__()
        self.depth = depth
        self.generator_type = generator_type
        self.patch_size = patch_size
        self.dilation = dilation
        self.use_swin = use_swin
        self.propagate_cond_context = propagate_cond_context

        # split token embedding: 0=leaf, 1=split (revealed positions use this)
        self.split_emb = nn.Embedding(2, embed_dim)

        # condition projection: parent features -> current embed_dim (bias)
        self.cond_proj = nn.Linear(cond_embed_dim, embed_dim)

        # mask token for MAR (unrevealed positions)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        # OctFormer backbone (windowed attention with RoPE)
        self.transformer = OctFormer(
            channels=embed_dim,
            num_blocks=num_blocks,
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
        self.norm = nn.LayerNorm(embed_dim)

        # dual heads: split prediction + feature for next level
        self.split_head = nn.Linear(embed_dim, 2)  # 0=leaf, 1=split
        self.feature_head = nn.Linear(embed_dim, embed_dim)

        # MAR masking ratio distribution (truncated normal, like FractalGen MAR)
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
    def _encode(self, octree, split_tokens, cond_list, mask=None):
        """Encode nodes at self.depth.

        Args:
            octree: octree (GT during training, growing during sampling)
            split_tokens: (N_d,) long in {0, 1}; revealed split values.
                Unrevealed positions can be any value (will be overwritten).
            cond_list: context feature tensors aligned to depth self.depth.
                The last tensor is the direct parent feature for this level.
            mask: bool tensor (N_d,) True=unrevealed (use mask_token).
                None for no masking.
        Returns:
            x: (N_d, embed_dim) encoded features
        """
        parent_features = cond_list[-1]
        cond = self.cond_proj(parent_features)  # (N_d, E) condition bias
        x = self.split_emb(split_tokens) + cond  # (N_d, E) token + cond

        if mask is not None:
            x = torch.where(mask.unsqueeze(1), self.mask_token.to(x.dtype), x)

        # Build OctreeT for windowed attention at this depth
        nnum_d = x.shape[0]
        octreeT = OctreeT(
            octree,
            nnum_d,
            self.patch_size,
            self.dilation,
            nempty=False,
            depth_list=[self.depth],
            buffer_size=0,
            use_swin=self.use_swin,
        )

        # depth layout -> batch layout -> OctFormer -> batch -> depth layout
        x = depth2batch(x, octreeT.indices)
        x = self.transformer(x, octreeT, context=None)
        x = batch2depth(x, octreeT.indices)

        x = self.norm(x)
        return x

    # ------------------------------------------------------------------
    # training forward (teacher forcing)
    # ------------------------------------------------------------------
    def forward(self, octree, cond_list, target_split):
        """Training: predict split + unpool features for next level.

        Args:
            octree: GT octree
            cond_list: context feature tensors aligned to depth self.depth.
                The last tensor is the direct parent feature for this level.
            target_split: (N_d,) long in {0, 1} GT split tokens (teacher forcing)
        Returns:
            split_logits: (N_d, 2)
            cond_list_next: split feature tensors aligned to depth self.depth + 1
            aux_loss: 0.0 (no auxiliary loss)
        """
        nnum_d = octree.nnum[self.depth]

        if self.training and self.generator_type == "mar":
            # random masking for MAR training: revealed = ~mask
            mask_rate = self.mask_ratio_generator.rvs(1)[0]
            num_masked = max(int(np.ceil(nnum_d * mask_rate)), 1)
            orders = torch.randperm(nnum_d, device=octree.device)
            mask = torch.zeros(nnum_d, dtype=torch.bool, device=octree.device)
            mask[orders[:num_masked]] = True
        else:
            mask = None

        # teacher forcing: revealed positions use GT split embedding
        x = self._encode(octree, target_split, cond_list, mask)

        split_logits = self.split_head(x)  # (N_d, 2)
        features = self.feature_head(x)  # (N_d, embed_dim)

        # Unpool features to next depth (parent -> 8 children). Previous
        # split-level features are kept and re-aligned so terminal VQ can
        # condition on the whole d3-d5 hierarchy.
        cond_list_prev = []
        if self.propagate_cond_context:
            cond_list_prev = [
                octree_copy_unpool(cond, octree, self.depth, nempty=False)
                for cond in cond_list
            ]
        cond_next = octree_copy_unpool(features, octree, self.depth, nempty=False)
        cond_list_next = cond_list_prev + [cond_next]

        return split_logits, cond_list_next, 0.0

    # ------------------------------------------------------------------
    # generation (MAR iterative sampling)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        cond_list,
        octree,
        num_iter,
        temperature,
        next_level_sample_function,
        visualize=False,
    ):
        """Generation: MAR sample split -> grow octree -> recurse.

        MaskGIT-style iterative refinement: revealed positions use
        split_emb(pred), masked positions use mask_token. Each step reveals
        a subset of masked positions based on cosine schedule, so the model
        conditions on already-sampled split values for spatial coherence.
        """
        nnum_d = octree.nnum[self.depth]
        device = octree.device

        # start: all masked, split_pred initialized to 0 (placeholder)
        mask = torch.ones(nnum_d, dtype=torch.bool, device=device)
        orders = torch.randperm(nnum_d, device=device)
        split_pred = torch.zeros(nnum_d, dtype=torch.long, device=device)

        num_iter = min(num_iter, nnum_d)

        for step in range(num_iter):
            # forward with current mask (revealed=split_emb, masked=mask_token)
            x = self._encode(octree, split_pred, cond_list, mask)
            split_logits = self.split_head(x)  # (N_d, 2)

            # cosine schedule: number of masked positions for NEXT step
            mask_ratio = math.cos(math.pi / 2.0 * (step + 1) / num_iter)
            mask_len = int(np.floor(nnum_d * mask_ratio))
            mask_len = max(1, min(int(mask.sum().item()) - 1, mask_len))

            # mask_next: keep orders[:mask_len] masked (shrinking subset)
            mask_next = torch.zeros(nnum_d, dtype=torch.bool, device=device)
            mask_next[orders[:mask_len]] = True

            # mask_to_pred: positions revealed THIS step = mask - mask_next
            if step >= num_iter - 1:
                mask_to_pred = mask.clone()  # last step reveals all remaining
            else:
                mask_to_pred = mask ^ mask_next

            # OctGPT-style linear temperature decay:
            # temperature = start_temperature * ((num_iter - step) / num_iter)
            # so sampling is diverse early and sharpens to argmax at the end.
            cur_temperature = temperature * ((num_iter - step) / num_iter)

            # sample and update only the positions revealed this step
            sampled = sample(split_logits[mask_to_pred], temperature=cur_temperature)
            split_pred[mask_to_pred] = sampled.long()

            mask = mask_next

        # final forward with all revealed to get features for next level
        x = self._encode(octree, split_pred, cond_list, mask=None)
        features = self.feature_head(x)

        # grow octree with predicted split
        octree = seq2octree(octree, split_pred, self.depth, self.depth + 1)

        # Unpool current and previous split-level features to next depth.
        cond_list_prev = []
        if self.propagate_cond_context:
            cond_list_prev = [
                octree_copy_unpool(cond, octree, self.depth, nempty=False)
                for cond in cond_list
            ]
        cond_next = octree_copy_unpool(features, octree, self.depth, nempty=False)
        cond_list_next = cond_list_prev + [cond_next]

        # recurse to next level (pass grown octree forward)
        return next_level_sample_function(cond_list=cond_list_next, octree=octree)
