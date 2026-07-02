import os
import sys
import time
import math
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import ocnn
from ocnn.octree import Octree, Points
from models.octfractalgen import (
    octfractalgen_shapenet,
    octfractalgen_shapenet_vq120_b2,
    octfractalgen_shapenet_vq240_b4,
    octfractalgen_shapenet_vq384_b8,
    octfractalgen_shapenet_vq384_b16,
    octfractalgen_shapenet_vq576_b8,
    octfractalgen_shapenet_vq576_b12,
    octfractalgen_shapenet_vq576_b16,
    octfractalgen_shapenet_vqstrong,
    octfractalgen_small,
)
from models.vae_loader import build_vqvae
from utils.utils import octree2seq


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ShapeNetOctreeDataset(Dataset):
    """Loads .npz point clouds and builds octrees for OctFractalGen training."""

    def __init__(
        self, data_dir, depth=8, full_depth=3, points_scale=1.0, cache_dir=None
    ):
        self.data_dir = data_dir
        self.depth = depth
        self.full_depth = full_depth
        self.points_scale = points_scale
        self.cache_dir = cache_dir

        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in {data_dir}")
        print(f"[Dataset] Found {len(self.files)} samples in {data_dir}")

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.files)

    def _cache_path(self, idx):
        if not self.cache_dir:
            return None
        stem = os.path.splitext(os.path.basename(self.files[idx]))[0]
        return os.path.join(self.cache_dir, f"{stem}.octree.pth")

    def _build_octree(self, npz_path):
        raw = np.load(npz_path)
        points = torch.from_numpy(raw["points"]).float()
        normals = torch.from_numpy(raw["normals"]).float()
        points = points / self.points_scale  # normalize to [-1, 1]
        pts = Points(points=points, normals=normals)
        pts.clip(-1.0, 1.0)
        octree = Octree(depth=self.depth, full_depth=self.full_depth)
        octree.build_octree(pts)
        # build_octree uses update_neigh=False; construct neighs for conv ops
        for d in range(self.full_depth, self.depth + 1):
            octree.construct_neigh(d)
        return octree

    def __getitem__(self, idx):
        cache_path = self._cache_path(idx)
        # try cache
        if cache_path and os.path.exists(cache_path):
            try:
                octree = torch.load(cache_path, weights_only=False)
                return octree
            except Exception:
                pass  # cache corrupt, rebuild

        octree = self._build_octree(self.files[idx])

        # save cache
        if cache_path:
            try:
                torch.save(octree, cache_path)
            except Exception:
                pass
        return octree


def octree_collate_fn(batch):
    """Return list of octrees; merging is done in main process to avoid
    ocnn crashes under Windows multiprocessing (construct_neigh access violation)."""
    return batch


def merge_octree_batch(octrees, full_depth=3, depth=8):
    """Merge a list of octrees into one batched octree (main process only).

    construct_neigh is called here because it crashes ocnn C++ extension
    when run inside DataLoader worker processes on Windows.
    """
    if len(octrees) == 1:
        return octrees[0]
    batched = Octree.init_like(octrees[0])
    batched.merge_octrees(octrees)
    # merge_octrees does not carry over neighs; rebuild for conv ops
    for d in range(full_depth, depth + 1):
        batched.construct_neigh(d)
    return batched


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------
def extract_targets(octree, vqvae, full_depth=3, depth_stop=6, device="cuda"):
    """Extract split targets (depth 3,4,5) and VQ targets (depth 6) from GT octree."""
    with torch.no_grad():
        # ---- split targets ----
        split_seq = octree2seq(octree, full_depth, depth_stop)
        splits = []
        offset = 0
        for d in range(full_depth, depth_stop):  # 3, 4, 5
            nnum_d = octree.nnum[d]
            splits.append(split_seq[offset : offset + nnum_d].to(device))
            offset += nnum_d

        # ---- VQ targets ----
        vq_code = vqvae.extract_code(octree)  # (N_6, 32) continuous
        vq_zq, vq_indices, _ = vqvae.quantizer(vq_code)  # BSQ zq + {0, 1} indices

    return {"split": splits, "vq": vq_indices.to(device), "vq_zq": vq_zq.to(device)}


# ---------------------------------------------------------------------------
# LR scheduler (FractalGen-style per-iteration cosine with linear warmup)
# ---------------------------------------------------------------------------
def adjust_learning_rate(
    optimizer, current_epoch, base_lr, min_lr, warmup_epochs, total_epochs
):
    """Per-iteration cosine schedule with linear warmup.

    Args:
        current_epoch: float, fractional epoch (epoch + step/total_steps)
        base_lr: peak learning rate after warmup
        min_lr: minimum learning rate at the end
        warmup_epochs: number of linear warmup epochs
        total_epochs: total number of training epochs
    """
    if current_epoch < warmup_epochs:
        # linear warmup from 0 to base_lr
        lr = base_lr * current_epoch / warmup_epochs
    else:
        # cosine decay from base_lr to min_lr
        progress = (current_epoch - warmup_epochs) / max(
            total_epochs - warmup_epochs, 1
        )
        progress = min(max(progress, 0.0), 1.0)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def count_module_params(model, class_name):
    total = 0
    trainable = 0
    for module in model.modules():
        if module.__class__.__name__ != class_name:
            continue
        total += sum(p.numel() for p in module.parameters())
        trainable += sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(
    model,
    vqvae,
    dataloader,
    optimizer,
    scaler,
    device,
    epoch,
    log_interval=20,
    use_amp=True,
    args_full_depth=3,
    args_depth=8,
    # LR scheduler params
    base_lr=1e-4,
    min_lr=1e-5,
    warmup_epochs=20,
    total_epochs=400,
    grad_clip=3.0,
):
    model.train()
    total_loss = 0.0
    total_split_loss = 0.0
    total_vq_loss = 0.0
    # metrics accumulators
    metric_sums = {}  # key -> float sum
    metric_counts = {}  # key -> int count
    n_batches = 0
    t0 = time.time()
    n_steps = len(dataloader)

    for it, octrees in enumerate(dataloader):
        # per-iteration LR adjustment (FractalGen-style cosine with warmup)
        frac_epoch = epoch + it / max(n_steps, 1)
        cur_lr = adjust_learning_rate(
            optimizer, frac_epoch, base_lr, min_lr, warmup_epochs, total_epochs
        )

        # merge octrees in main process (ocnn construct_neigh crashes in workers)
        octree = merge_octree_batch(octrees, args_full_depth, args_depth)
        octree = octree.to(device)

        # extract targets with frozen VQVAE
        targets = extract_targets(octree, vqvae, device=device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast("cuda"):
                loss, metrics = model(octree, cond_list=None, targets=targets)
            scaler.scale(loss).backward()
            grad_norm = None
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, metrics = model(octree, cond_list=None, targets=targets)
            loss.backward()
            grad_norm = None
            if grad_clip is not None and grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip
                )
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if grad_norm is not None:
            metrics = dict(metrics)
            metrics["grad_norm"] = grad_norm.detach()

        # accumulate metrics (detached scalars)
        for k, v in metrics.items():
            val = v.item() if torch.is_tensor(v) else float(v)
            metric_sums[k] = metric_sums.get(k, 0.0) + val
            metric_counts[k] = metric_counts.get(k, 0) + 1

        if (it + 1) % log_interval == 0 or it == 0:
            avg = total_loss / n_batches
            elapsed = time.time() - t0
            batch_size = dataloader.batch_size
            sps = (it + 1) * batch_size / max(elapsed, 1e-6)  # samples/sec
            mem = (
                torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available()
                else 0
            )
            # current-step metrics (instantaneous)
            inst_metrics = " ".join(
                f"{k} {v.item() if torch.is_tensor(v) else v:.3f}"
                for k, v in metrics.items()
            )
            print(
                f"  [ep {epoch}] iter {it + 1}/{len(dataloader)} | "
                f"loss {loss.item():.4f} avg {avg:.4f} | "
                f"{inst_metrics} | "
                f"{sps:.2f} samples/s | mem {mem:.2f} GB | lr {cur_lr:.2e}"
            )

    elapsed = time.time() - t0
    # epoch-averaged metrics
    avg_metrics = {k: metric_sums[k] / max(metric_counts[k], 1) for k in metric_sums}
    return {
        "loss": total_loss / max(n_batches, 1),
        "time": elapsed,
        **{f"avg_{k}": v for k, v in avg_metrics.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Train OctFractalGen")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing ShapeNet .npz point-cloud samples.",
    )
    parser.add_argument(
        "--vqvae_ckpt",
        type=str,
        required=True,
        help="Path to the pretrained OctGPT VQVAE checkpoint.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Directory for checkpoints, history, and octree cache.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="shapenet_vq576_b12",
        choices=[
            "shapenet",
            "shapenet_vqstrong",
            "shapenet_vq120_b2",
            "shapenet_vq240_b4",
            "shapenet_vq384_b8",
            "shapenet_vq384_b16",
            "shapenet_vq576_b8",
            "shapenet_vq576_b12",
            "shapenet_vq576_b16",
            "small",
        ],
        help="Model variant. vq{dim}_b{blocks} = L3 (terminal VQ generator) "
        "embed_dim and num_blocks, all sharing L0=768/L1=384/L2=240.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--min_lr", type=float, default=1e-5, help="Minimum LR for cosine schedule"
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=20,
        help="Linear warmup epochs before cosine decay",
    )
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=3.0,
        help="Gradient clipping max norm; <=0 disables clipping",
    )
    parser.add_argument(
        "--vq_loss_weight",
        type=float,
        default=2.0,
        help="Weight applied to terminal VQ loss in the total loss",
    )
    parser.add_argument(
        "--vq_mask_ratio_min",
        type=float,
        default=0.5,
        help="Minimum MAR mask ratio for VQ prediction",
    )
    parser.add_argument(
        "--vq_random_flip",
        type=float,
        default=0.1,
        help="Random bit flip probability for VQ teacher forcing",
    )
    parser.add_argument(
        "--vq_remask_stage",
        type=float,
        default=0.7,
        help="Sampling remask starts after this fraction of VQ iterations",
    )
    parser.add_argument(
        "--vq_remask_prob",
        type=float,
        default=0.1,
        help="Fraction of low-confidence VQ positions to remask during sampling",
    )
    parser.add_argument(
        "--vq_denoise_weight",
        type=float,
        default=0.3,
        help="Weight for denoising loss on flipped non-masked VQ positions (0=disabled)",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--full_depth", type=int, default=3)
    parser.add_argument(
        "--cache", action="store_true", default=True, help="Cache built octrees to disk"
    )
    parser.add_argument("--no_amp", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Args: {vars(args)}")

    # ---- VQVAE (frozen) ----
    print("Loading pretrained VQVAE ...")
    vqvae = build_vqvae(
        args.vqvae_ckpt,
        device=device,
        freeze=True,
    )
    print(f"VQVAE loaded and frozen.")

    # ---- Model ----
    print("Building OctFractalGen ...")
    model_kwargs = dict(
        vq_mask_ratio_min=args.vq_mask_ratio_min,
        vq_random_flip=args.vq_random_flip,
        vq_remask_stage=args.vq_remask_stage,
        vq_remask_prob=args.vq_remask_prob,
        vq_loss_weight=args.vq_loss_weight,
        vq_denoise_weight=args.vq_denoise_weight,
    )
    if args.model == "shapenet":
        model = octfractalgen_shapenet(**model_kwargs)
    elif args.model == "shapenet_vqstrong":
        model = octfractalgen_shapenet_vqstrong(**model_kwargs)
    elif args.model == "shapenet_vq120_b2":
        model = octfractalgen_shapenet_vq120_b2(
            **model_kwargs,
        )
    elif args.model == "shapenet_vq240_b4":
        model = octfractalgen_shapenet_vq240_b4(
            **model_kwargs,
        )
    elif args.model == "shapenet_vq384_b8":
        model = octfractalgen_shapenet_vq384_b8(
            **model_kwargs,
        )
    elif args.model == "shapenet_vq384_b16":
        model = octfractalgen_shapenet_vq384_b16(
            **model_kwargs,
        )
    elif args.model == "shapenet_vq576_b8":
        model = octfractalgen_shapenet_vq576_b8(
            **model_kwargs,
        )
    elif args.model == "shapenet_vq576_b12":
        model = octfractalgen_shapenet_vq576_b12(
            **model_kwargs,
        )
    elif args.model == "shapenet_vq576_b16":
        model = octfractalgen_shapenet_vq576_b16(
            **model_kwargs,
        )
    else:
        model = octfractalgen_small(
            **model_kwargs,
        )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_vq, n_vq_trainable = count_module_params(model, "OctVQGenerator")
    print(f"  params: {n_params / 1e6:.2f}M (trainable: {n_trainable / 1e6:.2f}M)")
    print(
        f"  VQ predictor params: {n_vq / 1e6:.2f}M "
        f"(trainable: {n_vq_trainable / 1e6:.2f}M)"
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = GradScaler("cuda", enabled=not args.no_amp)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, weights_only=True, map_location="cpu")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # ---- Dataset ----
    cache_dir = os.path.join(args.logdir, "octree_cache") if args.cache else None
    dataset = ShapeNetOctreeDataset(
        args.data_dir, depth=args.depth, full_depth=args.full_depth, cache_dir=cache_dir
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=octree_collate_fn,
        pin_memory=True,  # P1: pinned memory for faster H2D
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4
        if args.num_workers > 0
        else None,  # P1: more prefetch buffers
    )
    print(
        f"Dataloader: {len(dataset)} samples, batch_size={args.batch_size}, "
        f"num_workers={args.num_workers}"
    )

    # ---- Training loop ----
    history = []
    print(f"\n=== Training started: {args.epochs} epochs ===\n")
    for epoch in range(start_epoch, args.epochs):
        stats = train_one_epoch(
            model,
            vqvae,
            dataloader,
            optimizer,
            scaler,
            device,
            epoch,
            log_interval=args.log_interval,
            use_amp=not args.no_amp,
            args_full_depth=args.full_depth,
            args_depth=args.depth,
            base_lr=args.lr,
            min_lr=args.min_lr,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            grad_clip=args.grad_clip,
        )

        lr = optimizer.param_groups[0]["lr"]
        # format avg metrics for epoch summary
        avg_metric_str = " ".join(
            f"{k} {v:.3f}" for k, v in stats.items() if k.startswith("avg_")
        )
        print(
            f"Epoch {epoch} done | loss {stats['loss']:.4f} | "
            f"{avg_metric_str} | "
            f"time {stats['time']:.1f}s | lr {lr:.2e}"
        )

        history.append({"epoch": epoch, **stats, "lr": lr})

        # checkpoint
        ckpt_path = os.path.join(args.logdir, "latest.pth")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )

        # periodic checkpoint
        if (epoch + 1) % 50 == 0:
            ep_path = os.path.join(args.logdir, f"epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": vars(args),
                },
                ep_path,
            )

        # save history
        with open(os.path.join(args.logdir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print("\n=== Training complete ===")
    print(f"Checkpoints in {args.logdir}")


if __name__ == "__main__":
    main()
