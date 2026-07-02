"""OctFractalGen: Fractal Generative Model for Octree-based 3D Generation.

Recursive coarse-to-fine generation on octrees, following the FractalGen
architecture style. Each fractal level is an independent generator operating
at a fixed octree depth:

  - Intermediate levels (depth 3, 4, 5): OctSplitGenerator
      predicts split tokens (0=leaf, 1=split) + features for next level
  - Terminal level (depth 6): OctVQGenerator
      predicts VQ codes (BSQ32), decoded by frozen pretrained VQVAE

The parent level's feature output is unpooled to 8 children and used as the
condition for the next level, forming a fractal recursion from coarse to fine.

Unconditional generation: a learnable root_token replaces the class embedding.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.oct_split_gen import OctSplitGenerator
from models.oct_vq_gen import OctVQGenerator


class OctFractalGen(nn.Module):
    """Fractal Generative Model for Octree-based 3D Generation."""

    def __init__(self,
                 depth_list,
                 embed_dim_list,
                 num_blocks_list,
                 num_heads_list,
                 generator_type_list,
                 num_iters_list,
                 vq_groups=32,
                 full_depth=3,
                 max_depth=8,
                 attn_dropout=0.0,
                 proj_dropout=0.1,
                 patch_size=1024,
                 dilation=2,
                 use_swin=True,
                 use_checkpoint=True,
                 fractal_level=0):
        super().__init__()

        # ----------------------------------------------------------------------
        # fractal specifics
        self.depth_list = list(depth_list)
        self.full_depth = full_depth
        self.max_depth = max_depth
        self.fractal_level = fractal_level
        self.num_fractal_levels = len(depth_list)

        # ----------------------------------------------------------------------
        # Root token for the first fractal level (unconditional generation)
        # Replaces class_emb + fake_latent from FractalGen
        if self.fractal_level == 0:
            self.root_token = nn.Parameter(torch.zeros(1, embed_dim_list[0]))
            nn.init.normal_(self.root_token, std=0.02)

        # ----------------------------------------------------------------------
        # Generator for the current level
        # Intermediate levels use OctSplitGenerator (like AR/MAR in FractalGen)
        self.generator = OctSplitGenerator(
            depth=depth_list[fractal_level],
            embed_dim=embed_dim_list[fractal_level],
            cond_embed_dim=embed_dim_list[fractal_level - 1] if fractal_level > 0 else embed_dim_list[0],
            num_blocks=num_blocks_list[fractal_level],
            num_heads=num_heads_list[fractal_level],
            generator_type=generator_type_list[fractal_level],
            num_iters=num_iters_list[fractal_level],
            patch_size=patch_size,
            dilation=dilation,
            use_swin=use_swin,
            use_checkpoint=use_checkpoint,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            full_depth=full_depth,
            max_depth=max_depth,
        )

        # ----------------------------------------------------------------------
        # Build the next fractal level recursively
        if self.fractal_level < self.num_fractal_levels - 2:
            self.next_fractal = OctFractalGen(
                depth_list=depth_list,
                embed_dim_list=embed_dim_list,
                num_blocks_list=num_blocks_list,
                num_heads_list=num_heads_list,
                generator_type_list=generator_type_list,
                num_iters_list=num_iters_list,
                vq_groups=vq_groups,
                full_depth=full_depth,
                max_depth=max_depth,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                patch_size=patch_size,
                dilation=dilation,
                use_swin=use_swin,
                use_checkpoint=use_checkpoint,
                fractal_level=fractal_level + 1,
            )
        else:
            # The final fractal level uses OctVQGenerator (predicts VQ codes).
            # Analogous to FractalGen using PixelLoss at the final level.
            self.next_fractal = OctVQGenerator(
                depth=depth_list[fractal_level + 1],
                embed_dim=embed_dim_list[fractal_level + 1],
                cond_embed_dim=embed_dim_list[fractal_level],
                num_blocks=num_blocks_list[fractal_level + 1],
                num_heads=num_heads_list[fractal_level + 1],
                vq_groups=vq_groups,
                num_iters=num_iters_list[fractal_level + 1],
                generator_type=generator_type_list[fractal_level + 1],
                patch_size=patch_size,
                dilation=dilation,
                use_swin=use_swin,
                use_checkpoint=use_checkpoint,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                full_depth=full_depth,
                max_depth=max_depth,
            )

    def forward(self, octree, cond_list=None, targets=None):
        """
        Forward pass to get loss recursively (teacher forcing).

        Args:
            octree: GT octree (depth=8, full_depth=3)
            cond_list: list of parent feature tensors; None at level 0
            targets: dict with 'split' (list per intermediate level) and
                     'vq' (terminal level target)
        Returns:
            (total_loss, metrics_dict): metrics_dict contains per-level
            'split_acc_l{i}' and terminal 'vq_top5_acc'.
        """
        if self.fractal_level == 0:
            # Unconditional: broadcast root_token to all nodes at depth_list[0]
            nnum_root = octree.nnum[self.depth_list[0]]
            root_features = self.root_token.expand(nnum_root, -1)
            cond_list = [root_features]

        # Generator forward: predict split + features for next level
        # Pass target_split for teacher forcing (revealed positions use GT
        # split embedding, masked positions use mask_token).
        target_split = targets['split'][self.fractal_level]
        split_logits, cond_list_next, aux_loss = self.generator(
            octree, cond_list, target_split)

        # Split loss + top1 accuracy at current level
        loss = F.cross_entropy(split_logits, target_split)
        with torch.no_grad():
            split_pred = split_logits.argmax(dim=-1)
            split_acc = (split_pred == target_split).float().mean()

        # Recursive: next level returns (loss, metrics)
        sub_loss, sub_metrics = self.next_fractal(octree, cond_list_next, targets)

        metrics = {f'split_acc_l{self.fractal_level}': split_acc}
        metrics.update(sub_metrics)
        return loss + aux_loss + sub_loss, metrics

    def sample(self, cond_list=None, octree=None, vqvae=None,
               num_iter_list=None, temperature=1.0, fractal_level=0,
               visualize=False, return_raw_octree=False):
        """
        Generate samples recursively (coarse-to-fine).

        Args:
            cond_list: parent feature tensors; None at level 0
            octree: growing octree (mutated during sampling)
            vqvae: frozen pretrained VQVAE for terminal decoding
            num_iter_list: MAR iterations per level
            temperature: sampling temperature
            fractal_level: current fractal level
            return_raw_octree: if True, skip VQVAE decode at terminal level
                and return (octree, vq_pred) at depth 6 instead of neural_mpu.
        Returns:
            neural_mpu callable from VQVAE decode (at terminal level), OR
            (octree, vq_pred) if return_raw_octree=True.
        """
        if fractal_level == 0:
            # Unconditional: broadcast root_token
            nnum_root = octree.nnum[self.depth_list[0]]
            root_features = self.root_token.expand(nnum_root, -1)
            cond_list = [root_features]

        # Prepare next level's sample function
        if fractal_level < self.num_fractal_levels - 2:
            # Next is another OctFractalGen level
            next_level_sample_function = partial(
                self.next_fractal.sample,
                vqvae=vqvae,
                num_iter_list=num_iter_list,
                temperature=temperature,
                fractal_level=fractal_level + 1,
                return_raw_octree=return_raw_octree,
            )
        else:
            # Terminal level: OctVQGenerator.sample -> VQVAE decode -> mesh
            # (or skip decode if return_raw_octree=True)
            next_level_sample_function = partial(
                self.next_fractal.sample,
                vqvae=vqvae,
                return_raw_octree=return_raw_octree,
            )

        # Recursively sample using the current generator
        return self.generator.sample(
            cond_list, octree, num_iter_list[fractal_level],
            temperature, next_level_sample_function, visualize)


# ---------------------------------------------------------------------------
# Model factory functions (following FractalGen convention)
# ---------------------------------------------------------------------------

def octfractalgen_shapenet(**kwargs):
    """OctFractalGen for ShapeNet airplane (unconditional, depth 3->6).

    4 fractal levels. Embed dims chosen so that (dim // heads) % 6 == 0,
    which is required by OctFormer's RotaryPosEmb (init_3d_freqs produces
    3 * (dim//heads//6) freqs, must equal (dim//heads)//2).
      Level 0 (d=3): OctSplitGenerator, 768 dim, 16 blocks (768/8=96, 96%6=0)
      Level 1 (d=4): OctSplitGenerator, 384 dim,  8 blocks (384/8=48, 48%6=0)
      Level 2 (d=5): OctSplitGenerator, 240 dim,  4 blocks (240/4=60, 60%6=0)
      Level 3 (d=6): OctVQGenerator,   120 dim,  2 blocks (120/4=30, 30%6=0) (BSQ32)
    L0 aligns with OctGPT (768 dim). Total ~132M params.
    """
    model = OctFractalGen(
        depth_list=(3, 4, 5, 6),
        embed_dim_list=(768, 384, 240, 120),
        num_blocks_list=(16, 8, 4, 2),
        num_heads_list=(8, 8, 4, 4),
        generator_type_list=("mar", "mar", "mar", "mar"),
        num_iters_list=(64, 128, 128, 256),
        fractal_level=0,
        **kwargs,
    )
    return model


def octfractalgen_small(**kwargs):
    """Smaller variant for fast experimentation (RoPE-compatible dims)."""
    model = OctFractalGen(
        depth_list=(3, 4, 5, 6),
        embed_dim_list=(288, 192, 120, 72),
        num_blocks_list=(8, 4, 2, 2),
        num_heads_list=(8, 8, 4, 4),
        generator_type_list=("mar", "mar", "mar", "mar"),
        num_iters_list=(32, 64, 64, 128),
        fractal_level=0,
        **kwargs,
    )
    return model
