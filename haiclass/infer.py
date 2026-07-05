"""Classify LAZ files with a trained checkpoint and write results.

Usage:
    uv run python -m haiclass.infer [--run spt01] [--limit N] [--files NAME ...]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from .blocks import make_blocks
from .config import OUT_DIR, RUNS_DIR, TEST_DIR, Config
from .features import compute_features
from .ground import height_above_ground
from .lasio import intensity_rank, read_file, write_classified
from .model import VoxelTransformer, pick_device
from .voxelize import build_voxels


def predict_logits(
    model: VoxelTransformer,
    feats: np.ndarray,
    centroid: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> np.ndarray:
    """Two shifted block passes, averaged logits. Returns (V, C) float32."""
    V = len(feats)
    logits = np.zeros((V, cfg.num_classes), dtype=np.float32)
    hits = np.zeros(V, dtype=np.float32)
    with torch.no_grad():
        for shift in ((0.0, 0.0), (23.7, 23.7)):
            blocks = make_blocks(centroid, cfg.block_target, shift=shift)
            for i in range(0, len(blocks), cfg.batch_blocks):
                chunk = blocks[i : i + cfg.batch_blocks]
                B = len(chunk)
                N = max(len(b) for b in chunk)
                f = np.zeros((B, N, feats.shape[1]), dtype=np.float32)
                c = np.zeros((B, N, 3), dtype=np.float32)
                pad = np.ones((B, N), dtype=bool)
                for j, b in enumerate(chunk):
                    f[j, : len(b)] = feats[b]
                    cc = centroid[b].astype(np.float32)
                    c[j, : len(b)] = cc - cc.mean(axis=0)
                    pad[j, : len(b)] = False
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out = model(
                        torch.from_numpy(f).to(device),
                        torch.from_numpy(c).to(device),
                        torch.from_numpy(pad).to(device),
                    )
                out = out.float().cpu().numpy()
                for j, b in enumerate(chunk):
                    logits[b] += out[j, : len(b)]
                    hits[b] += 1
    return logits / np.maximum(hits[:, None], 1)


def smooth_logits(logits: np.ndarray, centroid: np.ndarray, k: int = 8) -> np.ndarray:
    """Average logits over voxel k-NN — 'similar neighbours, same class' prior."""
    tree = cKDTree(centroid)
    _, idx = tree.query(centroid, k=k, workers=-1)
    out = np.zeros_like(logits)
    chunk = 500_000
    for i in range(0, len(logits), chunk):
        out[i : i + chunk] = logits[idx[i : i + chunk]].mean(axis=1)
    return 0.5 * logits + 0.5 * out


def classify_file(
    path: Path, model: VoxelTransformer, cfg: Config,
    mean: np.ndarray, std: np.ndarray, device: torch.device, out_dir: Path,
) -> dict:
    t0 = time.time()
    las, arr = read_file(path)
    xyz = arr["xyz"]
    hag = height_above_ground(xyz, cfg)
    vox = build_voxels(
        xyz, hag, intensity_rank(arr["intensity"]),
        arr["return_number"], arr["number_of_returns"], cfg,
    )
    feats = (compute_features(vox, cfg) - mean) / std
    centroid = (vox.centroid - vox.centroid.min(axis=0)).astype(np.float32)
    t_prep = time.time() - t0

    logits = predict_logits(model, feats, centroid, cfg, device)
    logits = smooth_logits(logits, centroid)
    voxel_cls = np.asarray(cfg.class_values, dtype=np.int32)[logits.argmax(axis=1)]
    point_cls = voxel_cls[vox.point_voxel]
    t_pred = time.time() - t0 - t_prep

    write_classified(las, point_cls, out_dir / path.name)
    vals, cnts = np.unique(point_cls, return_counts=True)
    return {
        "points": len(point_cls),
        "voxels": len(feats),
        "classes": {int(v): int(c) for v, c in zip(vals, cnts)},
        "t_prep": t_prep,
        "t_pred": t_pred,
        "t_total": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="spt01")
    ap.add_argument("--checkpoint", type=str, default="best.pt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--files", nargs="*", default=None, help="specific file stems")
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    ckpt = torch.load(RUNS_DIR / args.run / args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt["config"].items() if k in Config.__dataclass_fields__})
    mean, std = ckpt["mean"], ckpt["std"]

    device = pick_device()
    model = VoxelTransformer(cfg, len(mean)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.run}/{args.checkpoint} (epoch {ckpt.get('epoch')}, "
          f"mIoU {ckpt.get('miou', float('nan')):.4f}), device={device}")

    files = sorted(TEST_DIR.glob("*.laz"))
    if args.files:
        files = [f for f in files if f.stem in set(args.files)]
    files = files[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(files, 1):
        if (out_dir / f.name).exists():
            print(f"[{i}/{len(files)}] {f.name} exists, skip")
            continue
        info = classify_file(f, model, cfg, mean, std, device, out_dir)
        print(f"[{i}/{len(files)}] {f.name}: {info['points']:,} pts, {info['voxels']:,} vox, "
              f"prep {info['t_prep']:.0f}s + pred {info['t_pred']:.0f}s = {info['t_total']:.0f}s")

    print("done.")


if __name__ == "__main__":
    main()
