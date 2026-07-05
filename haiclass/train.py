"""Train the voxel transformer on cached training files.

Usage:
    uv run python -m haiclass.train [--epochs N]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .blocks import make_blocks
from .config import CACHE_DIR, RUNS_DIR, Config
from .model import VoxelTransformer, pick_device


def load_files(cfg: Config) -> list[dict]:
    caches = sorted((CACHE_DIR / "train").glob("*.npz"))
    if not caches:
        raise SystemExit("no training caches — run `python -m haiclass.precompute` first")
    files = []
    for f in caches:
        d = np.load(f)
        files.append(
            {
                "name": f.stem,
                "centroid": d["centroid"],
                "features": d["features"],
                "label": d["label"],
            }
        )
        print(f"  {f.stem}: {len(d['label']):,} voxels")
    return files


def build_class_map(files: list[dict], cfg: Config) -> np.ndarray:
    """Discover classes with enough support; returns array of raw class values."""
    counts: dict[int, int] = {}
    for t in files:
        vals, cnts = np.unique(t["label"], return_counts=True)
        for v, c in zip(vals, cnts):
            counts[int(v)] = counts.get(int(v), 0) + int(c)
    kept = sorted(v for v, c in counts.items() if c >= cfg.min_class_voxels and v > 1)
    dropped = {v: c for v, c in counts.items() if v not in kept}
    print(f"classes kept: {kept}")
    if dropped:
        print(f"classes dropped (low support / unclassified): {dropped}")
    return np.array(kept, dtype=np.int32)


class BlockDataset:
    """Voxel sets re-partitioned into blocks with a random shift each epoch.

    Train/val is a fixed voxel-level split: a deterministic hash of each voxel's
    25 m grid cell marks ~val_fraction of the area as validation. Val voxels are
    excluded from the training loss (label -1) and are the only ones scored at
    evaluation, so the split is stable no matter how blocks are drawn.
    """

    def __init__(self, files: list[dict], cfg: Config, rng: np.random.Generator):
        self.files = files
        self.cfg = cfg
        self.rng = rng
        # raw class value -> contiguous id (or -1 = ignore)
        self.remap = np.full(256, -1, dtype=np.int64)
        for fi, t in enumerate(files):
            g = np.floor(t["centroid"][:, :2] / 25.0).astype(np.int64)
            h = (g[:, 0] * 73856093) ^ (g[:, 1] * 19349663) ^ (fi * 83492791)
            t["val_mask"] = (h % 100) < int(cfg.val_fraction * 100)
        for i, v in enumerate(cfg.class_values):
            self.remap[v] = i

    def epoch_blocks(self, val: bool = False) -> list[tuple[int, np.ndarray]]:
        """(file_idx, voxel_idx) pairs covering all voxels."""
        out = []
        for fi, t in enumerate(self.files):
            shift = (0.0, 0.0) if val else tuple(self.rng.uniform(0, 30, 2))
            for b in make_blocks(t["centroid"], self.cfg.block_target, shift=shift):
                out.append((fi, b))
        if not val:
            self.rng.shuffle(out)
        return out

    def collate(self, items: list[tuple[int, np.ndarray]], augment: bool, val: bool = False):
        B = len(items)
        N = max(len(b) for _, b in items)
        F_dim = self.files[0]["features"].shape[1]
        feats = np.zeros((B, N, F_dim), dtype=np.float32)
        coords = np.zeros((B, N, 3), dtype=np.float32)
        labels = np.full((B, N), -1, dtype=np.int64)
        pad = np.ones((B, N), dtype=bool)
        for i, (fi, b) in enumerate(items):
            t = self.files[fi]
            n = len(b)
            f = t["features"][b].copy()
            c = t["centroid"][b].astype(np.float32)
            c = c - c.mean(axis=0)
            if augment:
                ang = self.rng.uniform(0, 2 * np.pi)
                ca, sa = np.cos(ang), np.sin(ang)
                rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
                c[:, :2] = c[:, :2] @ rot.T
                if self.rng.random() < 0.5:
                    c[:, 0] = -c[:, 0]
                f += self.rng.normal(0, 0.01, f.shape).astype(np.float32)
            feats[i, :n] = f
            coords[i, :n] = c
            lab = self.remap[t["label"][b]]
            # score only the split we're in; the other side is ignore_index
            vm = t["val_mask"][b]
            lab[vm != val] = -1
            labels[i, :n] = lab
            pad[i, :n] = False
        return (
            torch.from_numpy(feats),
            torch.from_numpy(coords),
            torch.from_numpy(labels),
            torch.from_numpy(pad),
        )


def normalize_stats(files: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    sample = np.concatenate([t["features"][:: max(1, len(t["features"]) // 100000)] for t in files])
    mean = sample.mean(axis=0)
    std = sample.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def evaluate(model, ds: BlockDataset, device, cfg: Config) -> dict:
    model.eval()
    C = cfg.num_classes
    conf = np.zeros((C, C), dtype=np.int64)
    val_blocks = ds.epoch_blocks(val=True)
    with torch.no_grad():
        for i in range(0, len(val_blocks), cfg.batch_blocks):
            feats, coords, labels, pad = ds.collate(
                val_blocks[i : i + cfg.batch_blocks], augment=False, val=True
            )
            logits = model(feats.to(device), coords.to(device), pad.to(device))
            pred = logits.argmax(-1).cpu().numpy().ravel()
            lab = labels.numpy().ravel()
            m = lab >= 0
            np.add.at(conf, (lab[m], pred[m]), 1)
    tp = np.diag(conf).astype(np.float64)
    iou = tp / np.maximum(conf.sum(0) + conf.sum(1) - tp, 1)
    acc = tp.sum() / max(conf.sum(), 1)
    model.train()
    return {"oa": acc, "miou": iou.mean(), "iou": iou, "conf": conf}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--run", type=str, default="spt01")
    args = ap.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.epochs = args.epochs
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    print("loading caches...")
    files = load_files(cfg)
    cfg.class_values = build_class_map(files, cfg).tolist()

    mean, std = normalize_stats(files)
    for t in files:
        t["features"] = (t["features"] - mean) / std

    ds = BlockDataset(files, cfg, rng)
    device = pick_device()
    print(f"device: {device}")
    model = VoxelTransformer(cfg, files[0]["features"].shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M params, {cfg.num_classes} classes")

    # class weights from voxel frequency
    total = np.zeros(cfg.num_classes, dtype=np.int64)
    for t in files:
        lab = ds.remap[t["label"]]
        m = lab >= 0
        total += np.bincount(lab[m], minlength=cfg.num_classes)
    weights = 1.0 / np.log(1.2 + total / total.sum())
    weights = weights / weights.mean()
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    print("class weights:", np.round(weights, 2))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(ds.epoch_blocks(val=False)) // cfg.batch_blocks)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.05
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    run_dir = RUNS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    np.savez(run_dir / "norm.npz", mean=mean, std=std)

    best_miou = 0.0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        blocks = ds.epoch_blocks(val=False)
        losses = []
        for i in range(0, len(blocks) - cfg.batch_blocks + 1, cfg.batch_blocks):
            feats, coords, labels, pad = ds.collate(blocks[i : i + cfg.batch_blocks], augment=True)
            feats, coords, labels, pad = (
                feats.to(device), coords.to(device), labels.to(device), pad.to(device),
            )
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(feats, coords, pad)
                loss = F.cross_entropy(
                    logits.reshape(-1, cfg.num_classes), labels.reshape(-1),
                    weight=w, ignore_index=-1,
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        msg = f"epoch {epoch:3d}/{cfg.epochs}  loss={np.mean(losses):.4f}  [{time.time()-t0:.0f}s]"
        if epoch % 5 == 0 or epoch == cfg.epochs:
            ev = evaluate(model, ds, device, cfg)
            msg += f"  val OA={ev['oa']:.4f} mIoU={ev['miou']:.4f}"
            per = {v: round(float(i), 3) for v, i in zip(cfg.class_values, ev["iou"])}
            print(msg)
            print(f"   per-class IoU: {per}")
            if ev["miou"] > best_miou:
                best_miou = ev["miou"]
                torch.save(
                    {"model": model.state_dict(), "config": asdict(cfg),
                     "mean": mean, "std": std, "epoch": epoch, "miou": best_miou},
                    run_dir / "best.pt",
                )
                print(f"   saved best (mIoU={best_miou:.4f})")
        else:
            print(msg)
        torch.save(
            {"model": model.state_dict(), "config": asdict(cfg),
             "mean": mean, "std": std, "epoch": epoch},
            run_dir / "last.pt",
        )

    print(f"done. best val mIoU={best_miou:.4f}  -> {run_dir}")


if __name__ == "__main__":
    main()
