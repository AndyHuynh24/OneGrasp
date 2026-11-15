"""Check every canonical sequence in data/processed for internal consistency.

Checks: array lengths match n_frames, frame ids sorted/unique, pressure grid shape, first/last
image of every view exists, landmark/finger array shapes, every participant has a split.
Exits non-zero if anything fails.

Usage:
    python scripts/validate_processed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from touch.data.canonical import FORMAT_VERSION  # noqa: E402
from touch.data.catalog import load_catalog  # noqa: E402


def check(s) -> list[str]:
    errs = []
    n = len(s)
    m = s.meta
    if m.get("format_version") != FORMAT_VERSION:
        errs.append(f"format_version {m.get('format_version')} != {FORMAT_VERSION}")
    ids = np.asarray(m["frame_ids"])
    if len(ids) != n or (np.diff(ids) <= 0).any():
        errs.append("frame_ids not sorted/unique or wrong length")
    for k, v in s.labels.items():
        if v.shape[:1] != (n,):
            errs.append(f"labels[{k}] has {v.shape[0]} rows, expected {n}")
    with np.load(s.root / "pressure.npz") as z:
        if z["kpa"].shape != (n, 105, 185):
            errs.append(f"pressure shape {z['kpa'].shape}")
    if s.split is None:
        errs.append(f"participant {s.participant} not in configs/splits.yaml")
    for view in s.views:
        for i in (0, n - 1):
            if not s.image_path(view, i).exists():
                errs.append(f"missing image {s.image_path(view, i)}")
        if s.has_landmarks(view):
            z = s.landmarks(view)
            if z["lm_px"].shape != (n, 2, 21, 3) or z["primary"].shape != (n,):
                errs.append(f"landmarks_{view} shapes {z['lm_px'].shape}")
        else:
            errs.append(f"no landmarks for {view}")
    F = s.fingers
    if F is None:
        errs.append("no fingers.npz")
    else:
        for k, shape in (
            ("max_kpa", (n, 5)),
            ("contact", (n, 5)),
            ("valid", (n,)),
            ("tip_height_mm", (n, 5)),
        ):
            if F[k].shape != shape:
                errs.append(f"fingers[{k}] shape {F[k].shape}")
        if s.meta.get("finger_source", "").startswith("pending"):
            errs.append("finger labels still pending")
    return errs


def main() -> None:
    seqs = load_catalog()
    bad = 0
    for s in seqs:
        errs = check(s)
        if errs:
            bad += 1
            print(f"{s.participant}/{s.name}:")
            for e in errs:
                print(f"    {e}")
    print(f"{len(seqs)} sequences checked, {bad} with problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
