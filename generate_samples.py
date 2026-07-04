import os
import sys
import copy
import argparse
import numpy as np
import torch
import ocnn
from ocnn.octree import Octree

from models.octfractalgen import (
    octfractalgen_shapenet_vq120_b2,
    octfractalgen_shapenet_vq240_b4,
    octfractalgen_shapenet_vq384_b8,
    octfractalgen_shapenet_vq384_b16,
    octfractalgen_shapenet_vq576_b8,
    octfractalgen_shapenet_vq576_b12,
    octfractalgen_shapenet_vq576_b16,
    octfractalgen_shapenet_vq768_b24,
)
from models.vae_loader import build_vqvae
from utils import utils as octgpt_utils


def generate_one_sample(
    model,
    vqvae,
    device,
    full_depth=3,
    depth=8,
    depth_stop=6,
    num_iters_list=None,
    temperature=1.0,
    return_raw_octree=False,
):
    """Generate one octree -> mesh via OctFractalGen + VQVAE.

    Returns:
        octree_out: the grown octree (depth=8)
        vq_indices: (N_6, 32) predicted VQ codes
        neural_mpu: VQVAE decoder output callable
    """
    # 1. init a full octree at full_depth
    octree_out = Octree.init_octree(
        depth=depth, full_depth=full_depth, batch_size=1, device=device
    )

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
        return_raw_octree=return_raw_octree,
    )

    return result


def export_mesh(
    octgpt_utils,
    neural_mpu,
    output_path,
    resolution=256,
    sdf_scale=0.9,
    points_scale=1.0,
):
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
        save_sdf=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate samples with OctFractalGen")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the trained OctFractalGen checkpoint.",
    )
    parser.add_argument(
        "--vqvae_ckpt",
        type=str,
        required=True,
        help="Path to the pretrained OctGPT VQVAE checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for generated .obj samples.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="shapenet_vq576_b12",
        choices=[
            "shapenet_vq120_b2",
            "shapenet_vq240_b4",
            "shapenet_vq384_b8",
            "shapenet_vq384_b16",
            "shapenet_vq576_b8",
            "shapenet_vq576_b12",
            "shapenet_vq576_b16",
            "shapenet_vq768_b24",
        ],
    )
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--sdf_scale", type=float, default=0.9)
    parser.add_argument(
        "--temperature",
        type=float,
        nargs="+",
        default=[1.0, 1.2, 0.5, 0.5],
        help="Sampling start temperature per fractal level "
        "(OctGPT convention: [L0, L1, L2, L3]). Each level "
        "linearly decays from this value to 0 across its "
        "MAR iterations. A single value is broadcast to all levels.",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        nargs="+",
        default=[64, 128, 128, 256],
        help="MAR iterations per fractal level (4 levels)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--raw_octree",
        action="store_true",
        help="Export raw octree coarse occupancy at depth 6 "
        "(skip VQVAE decode). Saves .obj voxel mesh.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- VQVAE ----
    print("Loading pretrained VQVAE ...")
    vqvae = build_vqvae(
        args.vqvae_ckpt,
        device=device,
        freeze=True,
    )

    # ---- Load checkpoint first to read training args (model config) ----
    print(f"Loading checkpoint: {args.ckpt}")
    ck = torch.load(args.ckpt, weights_only=True, map_location="cpu")
    ck_args = ck.get("args", {})
    # Use ck_args model if --model not explicitly set or matches
    ck_model = ck_args.get("model", args.model)
    if ck_model != args.model:
        print(
            f"  Note: checkpoint model={ck_model}, CLI --model={args.model}; using checkpoint model."
        )
        args.model = ck_model

    # Build model kwargs from checkpoint args for VQ enhancements.
    # Detect actual state_dict keys to stay robust against old checkpoints
    # that predate the bit_pos_emb / cond_injection features.
    sd_keys = set(ck["model"].keys())
    has_bit_pos_emb = any(k.endswith(".bit_pos_emb") for k in sd_keys)
    has_film = any(".film." in k for k in sd_keys)
    has_cross_attn = any(".cond_cross_attn." in k for k in sd_keys)
    model_kwargs = {}
    model_kwargs["patch_size"] = ck_args.get("patch_size", 1024)
    model_kwargs["vq_buffer_size"] = ck_args.get("vq_buffer_size", 0)
    model_kwargs["vq_use_bit_pos_emb"] = has_bit_pos_emb
    if has_film:
        model_kwargs["vq_cond_injection"] = "film"
    elif has_cross_attn:
        model_kwargs["vq_cond_injection"] = "cross_attn"
    else:
        model_kwargs["vq_cond_injection"] = "add"
    print(
        f"  Detected from state_dict: bit_pos_emb={has_bit_pos_emb}, "
        f"cond_injection={model_kwargs['vq_cond_injection']}"
    )
    print(f"  Using patch_size={model_kwargs['patch_size']}")
    print(f"  Using vq_buffer_size={model_kwargs['vq_buffer_size']}")

    # ---- Model ----
    print("Building OctFractalGen ...")
    if args.model == "shapenet_vq120_b2":
        model = octfractalgen_shapenet_vq120_b2(**model_kwargs)
    elif args.model == "shapenet_vq240_b4":
        model = octfractalgen_shapenet_vq240_b4(**model_kwargs)
    elif args.model == "shapenet_vq384_b8":
        model = octfractalgen_shapenet_vq384_b8(**model_kwargs)
    elif args.model == "shapenet_vq384_b16":
        model = octfractalgen_shapenet_vq384_b16(**model_kwargs)
    elif args.model == "shapenet_vq576_b8":
        model = octfractalgen_shapenet_vq576_b8(**model_kwargs)
    elif args.model == "shapenet_vq576_b12":
        model = octfractalgen_shapenet_vq576_b12(**model_kwargs)
    elif args.model == "shapenet_vq576_b16":
        model = octfractalgen_shapenet_vq576_b16(**model_kwargs)
    else:
        model = octfractalgen_shapenet_vq768_b24(**model_kwargs)
    model.to(device)

    model.load_state_dict(ck["model"])
    model.eval()
    epoch = ck.get("epoch", "?")
    print(f"  Loaded (epoch {epoch})")

    num_iters_list = args.num_iters  # per fractal level

    # Normalize temperature: single value -> broadcast to 4 levels;
    # list stays as-is (OctGPT convention [L0, L1, L2, L3]).
    if len(args.temperature) == 1:
        temperature = args.temperature[0]
    else:
        temperature = args.temperature
    print(f"  Temperature: {temperature}")

    print(f"\n=== Generating {args.num_samples} samples ===")
    if args.raw_octree:
        print("  (raw_octree mode: skip VQVAE decode, export depth=6 coarse occupancy)")
    for i in range(args.num_samples):
        seed = args.seed + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        t0 = __import__("time").time()
        try:
            result = generate_one_sample(
                model,
                vqvae,
                device,
                full_depth=3,
                depth=8,
                depth_stop=6,
                num_iters_list=num_iters_list,
                temperature=temperature,
                return_raw_octree=args.raw_octree,
            )
            elapsed = __import__("time").time() - t0

            output_path = os.path.join(args.output_dir, f"{i}.obj")

            if args.raw_octree:
                # result = (octree, vq_pred) at depth 6
                octree_out, vq_pred = result
                # Export depth=6 coarse occupancy as voxel mesh
                os.makedirs(args.output_dir, exist_ok=True)
                octgpt_utils.export_octree(
                    octree_out, depth=6, save_dir=args.output_dir, index=i
                )
                # Report stats
                nnum_6 = octree_out.nnum[6].item()
                nnum_nempty_6 = octree_out.nnum_nempty[6].item()
                print(
                    f"  [{i + 1}/{args.num_samples}] seed={seed} | "
                    f"d6 nodes={nnum_6} nempty={nnum_nempty_6} "
                    f"vq_pred_sum={int(vq_pred.sum())} | "
                    f"{elapsed:.1f}s -> {output_path}"
                )
            else:
                neural_mpu = result
                export_mesh(
                    octgpt_utils,
                    neural_mpu,
                    output_path,
                    resolution=args.resolution,
                    sdf_scale=args.sdf_scale,
                )
                # report vertex/face count
                import trimesh

                mesh = trimesh.load(output_path)
                print(
                    f"  [{i + 1}/{args.num_samples}] seed={seed} | "
                    f"verts={len(mesh.vertices)} faces={len(mesh.faces)} | "
                    f"{elapsed:.1f}s -> {output_path}"
                )
        except Exception as e:
            print(f"  [{i + 1}/{args.num_samples}] seed={seed} FAILED: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n=== Done. Samples saved to {args.output_dir} ===")


if __name__ == "__main__":
    main()
