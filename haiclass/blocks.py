"""Partition voxels into spatial attention blocks of ~block_target voxels."""

from __future__ import annotations

import numpy as np


def make_blocks(
    centroid: np.ndarray,
    target: int,
    shift: tuple[float, float] = (0.0, 0.0),
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Recursive median split in XY until every block has <= target voxels.

    `shift` (in metres) offsets the split planes indirectly by translating the
    coordinates first — used for shifted second passes and train-time jitter.
    """
    xy = centroid[:, :2] + np.asarray(shift)

    blocks: list[np.ndarray] = []
    stack = [np.arange(len(centroid), dtype=np.int64)]
    while stack:
        idx = stack.pop()
        if len(idx) <= target:
            blocks.append(idx)
            continue
        ext = xy[idx].max(axis=0) - xy[idx].min(axis=0)
        d = int(ext[1] > ext[0])
        med = np.median(xy[idx, d])
        left = xy[idx, d] <= med
        # degenerate split (many identical coords): fall back to even halves
        if left.all() or not left.any():
            order = np.argsort(xy[idx, d], kind="stable")
            half = len(idx) // 2
            stack += [idx[order[:half]], idx[order[half:]]]
        else:
            stack += [idx[left], idx[~left]]
    if rng is not None:
        rng.shuffle(blocks)
    return blocks
