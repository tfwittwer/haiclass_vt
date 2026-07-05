"""LAZ read/write via laspy + lazrs."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np


def read_file(path: Path) -> tuple[laspy.LasData, dict[str, np.ndarray]]:
    """Read a LAZ file and extract the arrays the pipeline needs.

    Returns the full LasData (kept for lossless write-back) and a dict of
    float/int arrays: xyz, intensity, return_number, number_of_returns,
    classification.
    """
    las = laspy.read(path)
    dims = set(las.point_format.dimension_names)
    n = len(las.points)

    def get(name: str, default: float) -> np.ndarray:
        if name in dims:
            return np.asarray(las[name])
        return np.full(n, default, dtype=np.float32)

    arrays = {
        "xyz": np.column_stack([las.x, las.y, las.z]).astype(np.float64),
        "intensity": get("intensity", 0).astype(np.float32),
        "return_number": get("return_number", 1).astype(np.float32),
        "number_of_returns": get("number_of_returns", 1).astype(np.float32),
        "classification": np.asarray(las.classification).astype(np.int32),
    }
    return las, arrays


def write_classified(las: laspy.LasData, labels: np.ndarray, path: Path) -> None:
    """Write the file back with a new classification, preserving everything else."""
    labels = np.asarray(labels)
    max_allowed = 31 if las.header.point_format.id < 6 else 255
    if labels.max(initial=0) > max_allowed:
        raise ValueError(f"class value {labels.max()} exceeds format limit {max_allowed}")
    las.classification = labels.astype(las.classification.dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(path)


def intensity_rank(intensity: np.ndarray) -> np.ndarray:
    """Per-file percentile rank in [0, 1] — sensor-invariant intensity."""
    order = np.argsort(intensity, kind="stable")
    ranks = np.empty(len(intensity), dtype=np.float32)
    ranks[order] = np.arange(len(intensity), dtype=np.float32)
    if len(intensity) > 1:
        ranks /= len(intensity) - 1
    return ranks
