"""Precompute per-file voxel features and (for training files) labels → npz cache.

Usage:
    uv run python -m haiclass.precompute            # training files
    uv run python -m haiclass.precompute --test     # test files
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .config import CACHE_DIR, TEST_DIR, TRAIN_DIR, Config
from .features import compute_features
from .ground import height_above_ground
from .lasio import intensity_rank, read_file
from .voxelize import build_voxels


def process_file(path: Path, cfg: Config, with_labels: bool) -> dict[str, np.ndarray]:
    _, arr = read_file(path)
    xyz = arr["xyz"]
    hag = height_above_ground(xyz, cfg)
    vox = build_voxels(
        xyz,
        hag,
        intensity_rank(arr["intensity"]),
        arr["return_number"],
        arr["number_of_returns"],
        cfg,
        labels=arr["classification"] if with_labels else None,
    )
    feats = compute_features(vox, cfg)
    out = {
        "centroid": vox.centroid.astype(np.float32) - vox.centroid.min(axis=0),
        "origin": vox.centroid.min(axis=0),
        "features": feats,
        "counts": vox.counts,
    }
    if with_labels:
        out["label"] = vox.label
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="process test files instead of training")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    src = TEST_DIR if args.test else TRAIN_DIR
    sub = "test" if args.test else "train"
    out_dir = CACHE_DIR / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.laz"))[: args.limit]
    for i, f in enumerate(files, 1):
        dst = out_dir / (f.stem + ".npz")
        if dst.exists():
            print(f"[{i}/{len(files)}] {f.name} cached, skip")
            continue
        t0 = time.time()
        data = process_file(f, cfg, with_labels=not args.test)
        np.savez_compressed(dst, **data)
        n = len(data["features"])
        print(f"[{i}/{len(files)}] {f.name}: {n:,} voxels in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
