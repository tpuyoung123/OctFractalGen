"""Generate 3D samples with trained OctFractalGen.

Usage:
  python generate_samples.py --ckpt logs/octfractalgen/uncond/latest.pth --num_samples 8

Pipeline:
  1. Root token -> Level 0 (d=3): MAR sample split -> grow octree
  2. Level 1 (d=4): MAR sample split -> grow octree
  3. Level 2 (d=5): MAR sample split -> grow octree
  4. Level 3 (d=6): MAR sample VQ codes (BSQ32)
  5. Frozen VQVAE decode (d=6 -> d=8) -> neural_mpu -> marching cubes -> .obj
"""

import os
import sys
import copy
import argparse
import importlib.util

# ---- ocnn environment setup (must happen before import ocnn) ----
_CACHE_DIR = os.path.join(os.path.dirname(__file__), '.ocnn_cache')
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault('OCNN_AUTOTUNE_CACHE_PATH',
                      os.path.join(_CACHE_DIR, 'autotune_cache.json'))
os.environ.setdefault('OCNN_AUTOSAVE_AUTOTUNE', '0')
os.environ.setdefault('OCNN_DISABLE_TRITON', '1')

import numpy as np
import torch
import ocnn
from ocnn.octree import Octree

from models.octfractalgen import octfractalgen_shapenet, octfractalgen_small
from models.vae_loader import build_vqvae
from utils import utils as frgen_utils  # octfractalgen/utils/utils.py


# ---- import octgpt utils for create_mesh/export_octree (isolated) ----
def _load_octgpt_utils():
    spec = importlib.util.spec_from_file_location(
        'octgpt_utils', r'd:\Python\3D_fractal_auto_regression\octgpt\utils\utils.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

octgpt_utils = _load_octgpt_utils()


def generate_one_sample(model, vqvae, device, full_depth=3, depth=8,
                        depth_stop=6, num_iters_list=None, temperature=1.0,
                        return_raw_octree=False):
    """Generate one octree -> mesh via OctFractalGen + VQVAE.

    Returns:
        octree_out: the grown octree (depth=8)
        vq_indices: (N_6, 32) predicted VQ codes
        neural_mpu: VQVAE decoder output callable
    """
    # 1. init a full octree at full_depth
    octree_out = Octree.init_octree(
        depth=depth, full_depth=full_depth, batch_size=1, device=device)

    # 2. recursive fractal sampling
    # The model.sample() handles the full recursion: split levels -> VQ level
    # -> VQVAE decode. It returns neural_mpu directly, OR (octree, vq_pred)
    # if return_raw_octree=True (skips VQVAE decode).
    result = model.sample(
        cond_list=None,
        octree=octree_out,
        vqvae=vqvae,
        num_iter_list=num_iters_list,
        temperature=temperature,
        fractal_level=0,
        return_raw_octree=return_raw_octree)

    return result


def export_mesh(neural_mpu, output_path, resolution=256, sdf_scale=0.9,
                points_scale=1.0):
    """Export neural_mpu to .obj via marching cubes (octgpt convention)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    octgpt_utils.create_mesh(
        neural_mpu,
        output_path,
        size=resolution,
        level=0.002,
        clean=True,
        bbmin=-sdf_scale,
        bbmax=sdf_scale,
        mesh_scale=points_scale,
        save_sdf=False)


def main():
    parser = argparse.ArgumentParser(description='Generate samples with OctFractalGen')
    parser.add_argument('--ckpt', type=str, default='logs/octfractalgen/uncond/latest.pth')
    parser.add_argument('--vqvae_ckpt', type=str,
                        default=r'd:\Python\3D_fractal_auto_regression\octgpt\saved_ckpt\vqvae_large_im5_uncond_bsq32.pth')
    parser.add_argument('--output_dir', type=str, default='logs/octfractalgen/uncond/samples')
    parser.add_argument('--model', type=str, default='shapenet', choices=['shapenet', 'small'])
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--sdf_scale', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--num_iters', type=int, nargs='+', default=[64, 128, 128, 256],
                        help='MAR iterations per fractal level (4 levels)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--raw_octree', action='store_true',
                        help='Export raw octree coarse occupancy at depth 6 '
                             '(skip VQVAE decode). Saves .obj voxel mesh.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ---- VQVAE ----
    print('Loading pretrained VQVAE ...')
    vqvae = build_vqvae(args.vqvae_ckpt, device=device, freeze=True)

    # ---- Model ----
    print('Building OctFractalGen ...')
    if args.model == 'shapenet':
        model = octfractalgen_shapenet()
    else:
        model = octfractalgen_small()
    model.to(device)

    # ---- Load checkpoint ----
    print(f'Loading checkpoint: {args.ckpt}')
    ck = torch.load(args.ckpt, weights_only=True, map_location='cpu')
    model.load_state_dict(ck['model'])
    model.eval()
    epoch = ck.get('epoch', '?')
    print(f'  Loaded (epoch {epoch})')

    num_iters_list = args.num_iters  # per fractal level

    print(f'\n=== Generating {args.num_samples} samples ===')
    if args.raw_octree:
        print('  (raw_octree mode: skip VQVAE decode, export depth=6 coarse occupancy)')
    for i in range(args.num_samples):
        seed = args.seed + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        t0 = __import__('time').time()
        try:
            result = generate_one_sample(
                model, vqvae, device,
                full_depth=3, depth=8, depth_stop=6,
                num_iters_list=num_iters_list,
                temperature=args.temperature,
                return_raw_octree=args.raw_octree)
            elapsed = __import__('time').time() - t0

            output_path = os.path.join(args.output_dir, f'{i}.obj')

            if args.raw_octree:
                # result = (octree, vq_pred) at depth 6
                octree_out, vq_pred = result
                # Export depth=6 coarse occupancy as voxel mesh
                os.makedirs(args.output_dir, exist_ok=True)
                octgpt_utils.export_octree(
                    octree_out, depth=6, save_dir=args.output_dir, index=i)
                # Report stats
                nnum_6 = octree_out.nnum[6].item()
                nnum_nempty_6 = octree_out.nnum_nempty[6].item()
                print(f'  [{i+1}/{args.num_samples}] seed={seed} | '
                      f'd6 nodes={nnum_6} nempty={nnum_nempty_6} '
                      f'vq_pred_sum={int(vq_pred.sum())} | '
                      f'{elapsed:.1f}s -> {output_path}')
            else:
                neural_mpu = result
                export_mesh(neural_mpu, output_path,
                            resolution=args.resolution,
                            sdf_scale=args.sdf_scale)
                # report vertex/face count
                import trimesh
                mesh = trimesh.load(output_path)
                print(f'  [{i+1}/{args.num_samples}] seed={seed} | '
                      f'verts={len(mesh.vertices)} faces={len(mesh.faces)} | '
                      f'{elapsed:.1f}s -> {output_path}')
        except Exception as e:
            print(f'  [{i+1}/{args.num_samples}] seed={seed} FAILED: {e}')
            import traceback
            traceback.print_exc()

    print(f'\n=== Done. Samples saved to {args.output_dir} ===')


if __name__ == '__main__':
    main()
