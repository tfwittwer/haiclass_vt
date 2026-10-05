"""Partition voxels into spatial attention blocks of ~block_target voxels."""

from __future__ import annotations

import numpy as np


def make_blocks(
    centroid: np.ndarray,
    target: int,
    shift: tuple[float, float] = (0.0, 0.0),
    rng: np.random.Generator | None = None,
    jitter: float = 0.0,
) -> list[np.ndarray]:
    """Recursive median split in XY until every block has <= target voxels.

    `shift` (in metres) offsets the split planes indirectly by translating the
    coordinates first — used for shifted second passes and train-time jitter.

    A pure median is translation invariant, so `shift` alone never changes the
    partition. With `jitter` > 0 each split uses quantile 0.5 +- jitter drawn
    from a generator seeded by `shift`: same shift, same blocks; different
    shift, different blocks.
    """
    xy = centroid[:, :2] + np.asarray(shift)
    qrng = np.random.default_rng(abs(hash(tuple(float(s) for s in shift)))) if jitter > 0 else None

    blocks: list[np.ndarray] = []
    stack = [np.arange(len(centroid), dtype=np.int64)]
    while stack:
        idx = stack.pop()
        if len(idx) <= target:
            blocks.append(idx)
            continue
        ext = xy[idx].max(axis=0) - xy[idx].min(axis=0)
        d = int(ext[1] > ext[0])
        q = 0.5 if qrng is None else 0.5 + qrng.uniform(-jitter, jitter)
        med = np.quantile(xy[idx, d], q)
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
