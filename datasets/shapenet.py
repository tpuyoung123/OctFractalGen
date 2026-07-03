"""ShapeNet octree dataset and batching utilities for OctFractalGen.

Contains:
  - ShapeNetOctreeDataset: loads .npz point clouds and builds octrees
  - octree_collate_fn: Windows-safe collate (returns list of octrees)
  - merge_octree_batch: merge octrees in main process (construct_neigh
    crashes ocnn C++ extension inside DataLoader workers on Windows)
  - octree cache helpers: strip_octree_runtime_fields,
    octree_has_runtime_fields, ensure_octree_neighs
"""
import os
import glob

import torch
from torch.utils.data import Dataset
from ocnn.octree import Octree, Points

import numpy as np


# ---------------------------------------------------------------------------
# Octree runtime/cache helpers
# ---------------------------------------------------------------------------
def strip_octree_runtime_fields(octree):
    """Remove runtime-only tensors before writing an octree to disk cache."""
    if hasattr(octree, "neighs") and octree.neighs is not None:
        octree.neighs = [None for _ in octree.neighs]
    return octree


def octree_has_runtime_fields(octree):
    return any(x is not None for x in getattr(octree, "neighs", []) or [])


def ensure_octree_neighs(octree, full_depth=3, depth=8):
    for d in range(full_depth, depth + 1):
        if octree.neighs[d] is None:
            octree.construct_neigh(d)
    return octree


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ShapeNetOctreeDataset(Dataset):
    """Loads .npz point clouds and builds octrees for OctFractalGen training."""

    def __init__(
        self,
        data_dir,
        depth=8,
        full_depth=3,
        points_scale=1.0,
        cache_dir=None,
        compact_cache_on_load=True,
    ):
        self.data_dir = data_dir
        self.depth = depth
        self.full_depth = full_depth
        self.points_scale = points_scale
        self.cache_dir = cache_dir
        self.compact_cache_on_load = compact_cache_on_load

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
        return octree

    def __getitem__(self, idx):
        cache_path = self._cache_path(idx)
        # try cache
        if cache_path and os.path.exists(cache_path):
            try:
                octree = torch.load(cache_path, weights_only=False)
                if self.compact_cache_on_load and octree_has_runtime_fields(octree):
                    strip_octree_runtime_fields(octree)
                    torch.save(octree, cache_path)
                return octree
            except Exception:
                pass  # cache corrupt, rebuild

        octree = self._build_octree(self.files[idx])

        # save cache
        if cache_path:
            try:
                strip_octree_runtime_fields(octree)
                torch.save(octree, cache_path)
            except Exception:
                pass
        return octree


# ---------------------------------------------------------------------------
# Collate / batching
# ---------------------------------------------------------------------------
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
        return ensure_octree_neighs(octrees[0], full_depth, depth)
    batched = Octree.init_like(octrees[0])
    batched.merge_octrees(octrees)
    # merge_octrees does not carry over neighs; rebuild for conv ops
    return ensure_octree_neighs(batched, full_depth, depth)
