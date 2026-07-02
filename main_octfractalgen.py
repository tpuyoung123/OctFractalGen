"""Training script for OctFractalGen (unconditional, ShapeNet airplane 02691156).

Usage:
  python main_octfractalgen.py

Environment variables for ocnn (set automatically in main):
  OCNN_DISABLE_TRITON=1     - use block_gemm instead of triton (compatibility)
  OCNN_AUTOTUNE_CACHE_PATH  - redirect autotune cache to project dir
  OCNN_AUTOSAVE_AUTOTUNE=0  - disable cache writes (sandbox-safe)
"""

import os
import sys
import time
import glob
import json
import argparse

# ---- ocnn environment setup (must happen before import ocnn) ----
_CACHE_DIR = os.path.join(os.path.dirname(__file__), '.ocnn_cache')
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault('OCNN_AUTOTUNE_CACHE_PATH',
                      os.path.join(_CACHE_DIR, 'autotune_cache.json'))
os.environ.setdefault('OCNN_AUTOSAVE_AUTOTUNE', '0')
os.environ.setdefault('OCNN_DISABLE_TRITON', '1')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import ocnn
from ocnn.octree import Octree, Points

from models.octfractalgen import octfractalgen_shapenet, octfractalgen_small
from models.vae_loader import build_vqvae
from utils.utils import octree2seq


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ShapeNetOctreeDataset(Dataset):
    """Loads .npz point clouds and builds octrees for OctFractalGen training."""

    def __init__(self, data_dir, depth=8, full_depth=3, points_scale=1.0,
                 cache_dir=None):
        self.data_dir = data_dir
        self.depth = depth
        self.full_depth = full_depth
        self.points_scale = points_scale
        self.cache_dir = cache_dir

        self.files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        if len(self.files) == 0:
            raise RuntimeError(f'No .npz files found in {data_dir}')
        print(f'[Dataset] Found {len(self.files)} samples in {data_dir}')

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.files)

    def _cache_path(self, idx):
        if not self.cache_dir:
            return None
        stem = os.path.splitext(os.path.basename(self.files[idx]))[0]
        return os.path.join(self.cache_dir, f'{stem}.octree.pth')

    def _build_octree(self, npz_path):
        raw = np.load(npz_path)
        points = torch.from_numpy(raw['points']).float()
        normals = torch.from_numpy(raw['normals']).float()
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
def extract_targets(octree, vqvae, full_depth=3, depth_stop=6, device='cuda'):
    """Extract split targets (depth 3,4,5) and VQ targets (depth 6) from GT octree."""
    with torch.no_grad():
        # ---- split targets ----
        split_seq = octree2seq(octree, full_depth, depth_stop)
        splits = []
        offset = 0
        for d in range(full_depth, depth_stop):  # 3, 4, 5
            nnum_d = octree.nnum[d]
            splits.append(split_seq[offset:offset + nnum_d].to(device))
            offset += nnum_d

        # ---- VQ targets ----
        vq_code = vqvae.extract_code(octree)          # (N_6, 32) continuous
        _, vq_indices, _ = vqvae.quantizer(vq_code)   # (N_6, 32) in {0, 1}

    return {'split': splits, 'vq': vq_indices}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(model, vqvae, dataloader, optimizer, scaler, device,
                    epoch, log_interval=20, use_amp=True,
                    args_full_depth=3, args_depth=8):
    model.train()
    total_loss = 0.0
    total_split_loss = 0.0
    total_vq_loss = 0.0
    # metrics accumulators
    metric_sums = {}  # key -> float sum
    metric_counts = {}  # key -> int count
    n_batches = 0
    t0 = time.time()

    for it, octrees in enumerate(dataloader):
        # merge octrees in main process (ocnn construct_neigh crashes in workers)
        octree = merge_octree_batch(octrees, args_full_depth, args_depth)
        octree = octree.to(device)

        # extract targets with frozen VQVAE
        targets = extract_targets(octree, vqvae, device=device)

        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                loss, metrics = model(octree, cond_list=None, targets=targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, metrics = model(octree, cond_list=None, targets=targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # accumulate metrics (detached scalars)
        for k, v in metrics.items():
            val = v.item() if torch.is_tensor(v) else float(v)
            metric_sums[k] = metric_sums.get(k, 0.0) + val
            metric_counts[k] = metric_counts.get(k, 0) + 1

        if (it + 1) % log_interval == 0 or it == 0:
            avg = total_loss / n_batches
            elapsed = time.time() - t0
            its = (it + 1) / max(elapsed, 1e-6)
            mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            # current-step metrics (instantaneous)
            inst_metrics = ' '.join(
                f'{k} {v.item() if torch.is_tensor(v) else v:.3f}'
                for k, v in metrics.items())
            print(f'  [ep {epoch}] iter {it+1}/{len(dataloader)} | '
                  f'loss {loss.item():.4f} avg {avg:.4f} | '
                  f'{inst_metrics} | '
                  f'{its:.2f} it/s | mem {mem:.2f} GB')

    elapsed = time.time() - t0
    # epoch-averaged metrics
    avg_metrics = {k: metric_sums[k] / max(metric_counts[k], 1)
                   for k in metric_sums}
    return {
        'loss': total_loss / max(n_batches, 1),
        'time': elapsed,
        **{f'avg_{k}': v for k, v in avg_metrics.items()},
    }


def main():
    parser = argparse.ArgumentParser(description='Train OctFractalGen')
    parser.add_argument('--data_dir', type=str,
                        default=r'D:\data\ShapeNetV1\points.npz\points.npz\02691156')
    parser.add_argument('--vqvae_ckpt', type=str,
                        default=r'd:\Python\3D_fractal_auto_regression\octgpt\saved_ckpt\vqvae_large_im5_uncond_bsq32.pth')
    parser.add_argument('--logdir', type=str, default='logs/octfractalgen/uncond')
    parser.add_argument('--model', type=str, default='shapenet',
                        choices=['shapenet', 'small'])
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--full_depth', type=int, default=3)
    parser.add_argument('--cache', action='store_true', default=True,
                        help='Cache built octrees to disk')
    parser.add_argument('--no_amp', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Args: {vars(args)}')

    # ---- VQVAE (frozen) ----
    print('Loading pretrained VQVAE ...')
    vqvae = build_vqvae(args.vqvae_ckpt, device=device, freeze=True)
    print(f'VQVAE loaded and frozen.')

    # ---- Model ----
    print('Building OctFractalGen ...')
    if args.model == 'shapenet':
        model = octfractalgen_shapenet()
    else:
        model = octfractalgen_small()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  params: {n_params/1e6:.2f}M (trainable: {n_trainable/1e6:.2f}M)')

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95))
    scaler = GradScaler('cuda', enabled=not args.no_amp)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, weights_only=True, map_location='cpu')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        start_epoch = ck['epoch'] + 1
        print(f'Resumed from {args.resume} at epoch {start_epoch}')

    # ---- Dataset ----
    cache_dir = os.path.join(args.logdir, 'octree_cache') if args.cache else None
    dataset = ShapeNetOctreeDataset(
        args.data_dir, depth=args.depth, full_depth=args.full_depth,
        cache_dir=cache_dir)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=octree_collate_fn,
        pin_memory=False, persistent_workers=(args.num_workers > 0))
    print(f'Dataloader: {len(dataset)} samples, batch_size={args.batch_size}, '
          f'num_workers={args.num_workers}')

    # ---- Training loop ----
    history = []
    print(f'\n=== Training started: {args.epochs} epochs ===\n')
    for epoch in range(start_epoch, args.epochs):
        stats = train_one_epoch(
            model, vqvae, dataloader, optimizer, scaler, device,
            epoch, log_interval=args.log_interval, use_amp=not args.no_amp,
            args_full_depth=args.full_depth, args_depth=args.depth)

        lr = optimizer.param_groups[0]['lr']
        # format avg metrics for epoch summary
        avg_metric_str = ' '.join(
            f'{k} {v:.3f}' for k, v in stats.items()
            if k.startswith('avg_'))
        print(f'Epoch {epoch} done | loss {stats["loss"]:.4f} | '
              f'{avg_metric_str} | '
              f'time {stats["time"]:.1f}s | lr {lr:.2e}')

        history.append({'epoch': epoch, **stats, 'lr': lr})

        # checkpoint
        ckpt_path = os.path.join(args.logdir, 'latest.pth')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'args': vars(args),
        }, ckpt_path)

        # periodic checkpoint
        if (epoch + 1) % 50 == 0:
            ep_path = os.path.join(args.logdir, f'epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'args': vars(args),
            }, ep_path)

        # save history
        with open(os.path.join(args.logdir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print('\n=== Training complete ===')
    print(f'Checkpoints in {args.logdir}')


if __name__ == '__main__':
    main()
