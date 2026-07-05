"""Voxel grid: group points into voxels and aggregate attributes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class VoxelSet:
    """Per-voxel aggregates plus the point→voxel mapping."""

    centroid: np.ndarray        # (V, 3) float64
    point_voxel: np.ndarray     # (N,) int64 — voxel index of every point
    counts: np.ndarray          # (V,) int32
    hag: np.ndarray             # (V,) float32 mean HAG
    intensity_mean: np.ndarray  # (V,) float32 (percentile-rank units)
    intensity_std: np.ndarray   # (V,) float32
    return_mean: np.ndarray     # (V,) float32
    multi_return_frac: np.ndarray  # (V,) float32
    label: np.ndarray | None = None  # (V,) int32 majority class (training only)


def _majority_label(voxel_idx: np.ndarray, labels: np.ndarray, n_voxels: int) -> np.ndarray:
    """Majority point class per voxel via bincount over (voxel, class) pairs."""
    n_cls = int(labels.max()) + 1
    pair = voxel_idx * n_cls + labels
    counts = np.bincount(pair, minlength=n_voxels * n_cls).reshape(n_voxels, n_cls)
    return counts.argmax(axis=1).astype(np.int32)


def build_voxels(
    xyz: np.ndarray,
    hag: np.ndarray,
    intensity_rank: np.ndarray,
    return_number: np.ndarray,
    number_of_returns: np.ndarray,
    cfg: Config,
    labels: np.ndarray | None = None,
) -> VoxelSet:
    vs = cfg.voxel_size
    key = np.floor((xyz - xyz.min(axis=0)) / vs).astype(np.int64)
    flat = (key[:, 0] << 42) | (key[:, 1] << 21) | key[:, 2]
    uniq, voxel_idx = np.unique(flat, return_inverse=True)
    n_vox = len(uniq)

    counts = np.bincount(voxel_idx, minlength=n_vox).astype(np.int32)
    inv_counts = 1.0 / counts

    def vmean(values: np.ndarray) -> np.ndarray:
        return (np.bincount(voxel_idx, weights=values, minlength=n_vox) * inv_counts).astype(
            np.float32
        )

    centroid = np.column_stack(
        [np.bincount(voxel_idx, weights=xyz[:, d], minlength=n_vox) * inv_counts for d in range(3)]
    )

    i_mean = vmean(intensity_rank)
    i_sq = np.bincount(voxel_idx, weights=intensity_rank.astype(np.float64) ** 2, minlength=n_vox)
    i_std = np.sqrt(np.maximum(i_sq * inv_counts - i_mean.astype(np.float64) ** 2, 0.0)).astype(
        np.float32
    )

    label = None
    if labels is not None:
        label = _majority_label(voxel_idx, labels.astype(np.int64), n_vox)

    return VoxelSet(
        centroid=centroid,
        point_voxel=voxel_idx.astype(np.int64),
        counts=counts,
        hag=vmean(hag),
        intensity_mean=i_mean,
        intensity_std=i_std,
        return_mean=vmean(return_number),
        multi_return_frac=vmean((number_of_returns > 1).astype(np.float32)),
        label=label,
    )
