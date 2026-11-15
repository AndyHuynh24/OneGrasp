"""Read the canonical dataset (data/processed) the same way for every source.

from touch.data.catalog import load_catalog
seqs = load_catalog(split="train")                  # all sources
seqs = load_catalog(source="egopressure", split="val", family="index_press")
s = seqs[0]
s.labels["contact"], s.fingers["contact"][:, 1]     # hand-level, index finger
s.image_path("ego", i)                  # i = row in the arrays, not the raw frame id
s.landmarks("overhead")["lm_px"]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml

PROCESSED_ROOT = Path("data/processed")
SPLITS_FILE = Path("configs/splits.yaml")


def load_splits(path: Path = SPLITS_FILE) -> dict[str, str]:
    """participant id -> 'train' | 'val' | 'test'."""
    cfg = yaml.safe_load(Path(path).read_text())
    out: dict[str, str] = {}
    for source in cfg.values():
        for split, people in source.items():
            for p in people:
                assert p not in out, f"{p} listed twice in {path}"
                out[p] = split
    return out


@dataclass
class Sequence:
    root: Path
    meta: dict
    split: str | None = field(default=None)

    @property
    def source(self) -> str:
        return self.meta["source"]

    @property
    def participant(self) -> str:
        return self.meta["participant"]

    @property
    def name(self) -> str:
        return self.meta["sequence"]

    @property
    def views(self) -> list[str]:
        return list(self.meta["views"])

    def __len__(self) -> int:
        return self.meta["n_frames"]

    @cached_property
    def labels(self) -> dict[str, np.ndarray]:
        with np.load(self.root / "labels.npz") as z:
            return {k: z[k] for k in z.files}

    @cached_property
    def fingers(self) -> dict[str, np.ndarray] | None:
        p = self.root / "fingers.npz"
        if not p.exists():
            return None
        with np.load(p) as z:
            return {k: z[k] for k in z.files}

    def pressure(self) -> np.ndarray:
        """[N, 105, 185] float32 PV-convention kPa (uncleaned)."""
        with np.load(self.root / "pressure.npz") as z:
            return z["kpa"].astype(np.float32)

    def has_landmarks(self, view: str) -> bool:
        return (self.root / f"landmarks_{view}.npz").exists()

    def landmarks(self, view: str) -> dict[str, np.ndarray]:
        with np.load(self.root / f"landmarks_{view}.npz") as z:
            return {k: z[k] for k in z.files}

    def image_path(self, view: str, i: int) -> Path:
        frame = self.meta["frame_ids"][i]
        return Path(self.meta["views"][view]["image"].format(frame=frame))


def load_catalog(
    root: Path = PROCESSED_ROOT,
    splits_file: Path = SPLITS_FILE,
    source: str | None = None,
    split: str | None = None,
    family: str | None = None,
    hand: str | None = None,
) -> list[Sequence]:
    splits = load_splits(splits_file) if Path(splits_file).exists() else {}
    out = []
    for meta_path in sorted(Path(root).glob("*/*/*/meta.json")):
        meta = json.loads(meta_path.read_text())
        s = Sequence(meta_path.parent, meta, splits.get(meta["participant"]))
        if source and s.source != source:
            continue
        if split and s.split != split:
            continue
        if family and meta["family"] != family:
            continue
        if hand and meta["hand"] != hand:
            continue
        out.append(s)
    return out
