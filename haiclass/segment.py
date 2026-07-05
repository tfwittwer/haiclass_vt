"""Object segmentation: group classified points into object instances.

Per class, voxel centroids are clustered by connected components with a
class-specific linking radius; every point inherits the instance of its voxel.
Instance ids are written to an `instance_id` extra dimension (uint32, 0 = not
segmented), classification and all other attributes stay untouched.

Usage:
    uv run python -m haiclass.segment                  # segment C:\\work\\haiclasst\\out in place
    uv run python -m haiclass.segment --src DIR --dst DIR --limit N
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import laspy
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .config import OUT_DIR, Config

# classes that form discrete objects worth instancing; everything else keeps id 0.
# ground (2), low vegetation (3) and diffuse/noise-like classes are excluded.
SEGMENT_CLASSES: set[int] = {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}

# linking radius in metres: max centroid gap that still connects two voxels
DEFAULT_RADIUS = 0.4
RADIUS_OVERRIDES: dict[int, float] = {
    13: 1.0,  # wire-like: sag + sparse sampling need a wider link
    14: 1.0,
    9: 0.8,
}

MIN_POINTS = 25  # instances smaller than this stay id 0


def segment_voxels(centroid: np.ndarray, radius: float) -> np.ndarray:
    """Connected components over voxel centroids within `radius`. Returns labels."""
    n = len(centroid)
    if n == 1:
        return np.zeros(1, dtype=np.int64)
    tree = cKDTree(centroid)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n, dtype=np.int64)
    graph = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64)


def segment_file(path: Path, dst: Path, cfg: Config) -> dict:
    t0 = time.time()
    las = laspy.read(path)
    xyz = np.column_stack([las.x, las.y, las.z])
    cls = np.asarray(las.classification)

    # point -> voxel (same grid resolution as classification)
    key = np.floor((xyz - xyz.min(axis=0)) / cfg.voxel_size).astype(np.int64)
    flat = (key[:, 0] << 42) | (key[:, 1] << 21) | key[:, 2]
    _, voxel_idx, v_counts = np.unique(flat, return_inverse=True, return_counts=True)
    n_vox = len(v_counts)
    v_centroid = np.column_stack(
        [np.bincount(voxel_idx, weights=xyz[:, d], minlength=n_vox) / v_counts for d in range(3)]
    )
    # voxel class = class of its points (uniform by construction of classification)
    v_cls = np.zeros(n_vox, dtype=np.int32)
    v_cls[voxel_idx] = cls

    v_inst = np.zeros(n_vox, dtype=np.uint32)
    next_id = 1
    n_objects = 0
    for c in sorted(SEGMENT_CLASSES):
        m = np.flatnonzero(v_cls == c)
        if len(m) == 0:
            continue
        labels = segment_voxels(v_centroid[m], RADIUS_OVERRIDES.get(c, DEFAULT_RADIUS))
        # count points per component, drop tiny ones
        comp_pts = np.bincount(labels, weights=v_counts[m])
        keep = comp_pts >= MIN_POINTS
        remap = np.zeros(len(comp_pts), dtype=np.uint32)
        ids = np.arange(keep.sum(), dtype=np.uint32) + next_id
        remap[keep] = ids
        v_inst[m] = remap[labels]
        next_id += int(keep.sum())
        n_objects += int(keep.sum())

    point_inst = v_inst[voxel_idx]

    if "instance_id" not in las.point_format.dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name="instance_id", type=np.uint32))
    las.instance_id = point_inst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.laz")
    las.write(tmp)
    tmp.replace(dst)

    return {
        "objects": n_objects,
        "segmented_pts": int((point_inst > 0).sum()),
        "points": len(cls),
        "t": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=str(OUT_DIR))
    ap.add_argument("--dst", type=str, default=None, help="default: in place (src)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--files", nargs="*", default=None)
    args = ap.parse_args()

    cfg = Config()
    src = Path(args.src)
    dst_dir = Path(args.dst) if args.dst else src

    files = sorted(src.glob("*.laz"))
    if args.files:
        files = [f for f in files if f.stem in set(args.files)]
    files = files[: args.limit]

    for i, f in enumerate(files, 1):
        info = segment_file(f, dst_dir / f.name, cfg)
        pct = info["segmented_pts"] / info["points"] * 100
        print(f"[{i}/{len(files)}] {f.name}: {info['objects']:,} objects, "
              f"{pct:.1f}% of points segmented [{info['t']:.0f}s]")

    print("done.")


if __name__ == "__main__":
    main()
