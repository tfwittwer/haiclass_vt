"""Per-voxel feature computation: eigen-geometry at two scales + column context."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .config import Config
from .voxelize import VoxelSet


def _eigen_features(centroid: np.ndarray, neigh_idx: np.ndarray, chunk: int = 262144) -> np.ndarray:
    """Covariance eigen-features for each voxel from its neighbour set.

    Returns (V, 7): linearity, planarity, scattering, verticality, |normal_z|,
    neighbourhood radius, neighbourhood z-range. Chunked to bound memory.
    """
    if len(neigh_idx) > chunk:
        return np.concatenate(
            [
                _eigen_features(centroid, neigh_idx[i : i + chunk])
                for i in range(0, len(neigh_idx), chunk)
            ]
        )
    pts = centroid[neigh_idx]                      # (V, k, 3)
    mean = pts.mean(axis=1, keepdims=True)
    d = pts - mean
    cov = np.einsum("vki,vkj->vij", d, d) / d.shape[1]
    evals, evecs = np.linalg.eigh(cov)             # ascending
    evals = np.maximum(evals[:, ::-1], 1e-12)      # descending l1 >= l2 >= l3
    l1, l2, l3 = evals[:, 0], evals[:, 1], evals[:, 2]

    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    scattering = l3 / l1
    principal = evecs[:, :, 2]                     # eigenvector of largest eigenvalue
    normal = evecs[:, :, 0]                        # eigenvector of smallest eigenvalue
    verticality = np.abs(principal[:, 2])
    normal_z = np.abs(normal[:, 2])
    radius = np.sqrt((d**2).sum(axis=2)).max(axis=1)
    zrange = pts[:, :, 2].max(axis=1) - pts[:, :, 2].min(axis=1)

    return np.column_stack(
        [linearity, planarity, scattering, verticality, normal_z, radius, zrange]
    ).astype(np.float32)


def _column_context(centroid: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Relative z position within the voxel's XY column and the column height."""
    cell = cfg.column_cell
    ij = np.floor((centroid[:, :2] - centroid[:, :2].min(axis=0)) / cell).astype(np.int64)
    ny = ij[:, 1].max() + 1
    flat = ij[:, 0] * ny + ij[:, 1]
    uniq, inv = np.unique(flat, return_inverse=True)
    z = centroid[:, 2]
    zmin = np.full(len(uniq), np.inf)
    zmax = np.full(len(uniq), -np.inf)
    np.minimum.at(zmin, inv, z)
    np.maximum.at(zmax, inv, z)
    height = (zmax - zmin)[inv]
    z_rel = np.where(height > 1e-6, (z - zmin[inv]) / np.maximum(height, 1e-6), 0.5)
    return z_rel.astype(np.float32), height.astype(np.float32)


def compute_features(vox: VoxelSet, cfg: Config) -> np.ndarray:
    """Assemble the (V, F) float32 feature matrix. Order matches config.feature_names."""
    c = vox.centroid
    tree = cKDTree(c)
    _, idx_fine = tree.query(c, k=cfg.k_fine, workers=-1)
    _, idx_coarse = tree.query(c, k=cfg.k_coarse, workers=-1)

    z_rel, col_h = _column_context(c, cfg)

    base = np.column_stack(
        [
            vox.hag,
            np.log1p(np.maximum(vox.hag, 0.0)),
            z_rel,
            col_h,
            vox.intensity_mean,
            vox.intensity_std,
            vox.return_mean,
            vox.multi_return_frac,
            np.log1p(vox.counts.astype(np.float32)),
        ]
    ).astype(np.float32)

    fine = _eigen_features(c, idx_fine)
    coarse = _eigen_features(c, idx_coarse)
    return np.column_stack([base, fine, coarse])
