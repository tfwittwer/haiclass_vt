"""Robust ground grid and height-above-ground."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .config import Config


def height_above_ground(xyz: np.ndarray, cfg: Config) -> np.ndarray:
    """Estimate a DTM on a regular grid and return per-point HAG.

    Per-cell low percentile of z, median-filtered to reject sub-ground noise,
    nearest-filled holes, bilinear interpolation at point locations.
    """
    cell = cfg.ground_cell
    xy_min = xyz[:, :2].min(axis=0)
    ij = np.floor((xyz[:, :2] - xy_min) / cell).astype(np.int64)
    nx, ny = ij.max(axis=0) + 1
    flat = ij[:, 0] * ny + ij[:, 1]

    # per-cell low percentile via sort + segment offsets
    order = np.argsort(flat, kind="stable")
    fs = flat[order]
    zs = xyz[order, 2]
    starts = np.flatnonzero(np.r_[True, fs[1:] != fs[:-1]])
    counts = np.diff(np.r_[starts, len(fs)])
    # index of the qth percentile inside each sorted-z segment
    z_sorted = np.empty_like(zs)
    for s, c in zip(starts, counts):
        seg = np.sort(zs[s : s + c])
        z_sorted[s : s + c] = seg  # reuse buffer; only the picked index matters below
    pick = starts + np.minimum(
        (counts - 1), np.floor(counts * cfg.ground_percentile / 100.0).astype(np.int64)
    )
    cell_ids = fs[starts]
    cell_z = z_sorted[pick]

    grid = np.full((nx, ny), np.nan, dtype=np.float64)
    grid[cell_ids // ny, cell_ids % ny] = cell_z

    # median filter (ignoring NaNs) rejects isolated low-noise cells
    filled = np.where(np.isnan(grid), np.inf, grid)
    med = ndimage.median_filter(filled, size=5, mode="nearest")
    valid = ~np.isnan(grid) & np.isfinite(med)
    # clamp cells far below the local median (sub-ground noise)
    grid = np.where(valid & (grid < med - 1.0), med, grid)

    # fill holes by nearest valid cell
    mask = np.isnan(grid)
    if mask.any():
        idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
        grid = grid[tuple(idx)]

    # light smoothing for interpolation stability
    grid = ndimage.uniform_filter(grid, size=3, mode="nearest")

    # bilinear interpolation at point xy (cell-center registered)
    gx = (xyz[:, 0] - xy_min[0]) / cell - 0.5
    gy = (xyz[:, 1] - xy_min[1]) / cell - 0.5
    ground = ndimage.map_coordinates(grid, np.vstack([gx, gy]), order=1, mode="nearest")

    hag = xyz[:, 2] - ground
    return np.clip(hag, cfg.hag_clip[0], cfg.hag_clip[1]).astype(np.float32)
