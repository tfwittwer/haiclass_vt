"""Central configuration for the haiclass pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECTPATH = Path(r"haiclass_vt")
TRAIN_DIR = PROJECTPATH / Path(r"training")
TEST_DIR = PROJECTPATH / Path(r"in")
OUT_DIR = PROJECTPATH / Path(r"out")
CACHE_DIR = PROJECTPATH / Path(r"cache")
RUNS_DIR = PROJECTPATH / Path(r"runs")


@dataclass
class Config:
    # --- voxels ---
    voxel_size: float = 0.10

    # --- ground grid ---
    ground_cell: float = 1.0
    ground_percentile: float = 5.0
    hag_clip: tuple[float, float] = (-2.0, 60.0)

    # --- features ---
    column_cell: float = 0.5
    k_fine: int = 10
    k_coarse: int = 32

    # --- blocks ---
    block_target: int = 4096  # voxels per attention block

    # --- model ---
    dim: int = 256
    depth: int = 6
    heads: int = 8
    dropout: float = 0.05

    # --- training ---
    lr: float = 3e-4
    weight_decay: float = 0.05
    epochs: int = 60
    batch_blocks: int = 8
    val_fraction: float = 0.10
    min_class_voxels: int = 2000  # drop classes with less total support
    seed: int = 42

    # classes are discovered from the training caches; stored on the checkpoint
    class_values: list[int] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.class_values)


def feature_names(cfg: Config) -> list[str]:
    """Order of the per-voxel feature vector (keep in sync with features.py)."""
    names = [
        "hag", "log_hag", "z_in_column", "column_height",
        "intensity_mean", "intensity_std",
        "return_mean", "multi_return_frac",
        "log_pts",
    ]
    for s in ("fine", "coarse"):
        names += [
            f"{s}_linearity", f"{s}_planarity", f"{s}_scattering",
            f"{s}_verticality", f"{s}_normal_z", f"{s}_radius", f"{s}_zrange",
        ]
    return names
