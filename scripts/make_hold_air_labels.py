"""HO-Cap TRAIN + VAL "air" hold labels for training (hold = held AND off the table;
touch.data.hocap.hold_air, docs/DECISIONS.md 2026-11-12): data/hocap_proc/<subject>/<seq>/
hold_<side>.npz, aligned with the prepared <cam>_<side>.npz frames (same eval_hocap.gt_labels).
Test subjects (5, 8) are never prepared for training and are skipped here.

    python scripts/make_hold_air_labels.py [--workers 4]
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]

PROC = Path("data/hocap_proc")


def job(key: str) -> str:
    from eval_hocap import gt_labels

    from touch.data import hocap as H

    subject, name = key.split("/")
    seq = next(s for s in H.list_sequences(split=H.split_of(subject)) if s.name == name)
    msg = []
    for side in seq.sides:
        out = PROC / subject / name / f"hold_{side}.npz"
        if out.exists():
            continue
        L = gt_labels(seq, side)
        np.savez_compressed(out, state=L["state_air"].astype(np.int8))
        moved = int(((L["state_any2"] == 2) & (L["state_air"] == 1)).sum())
        msg.append(f"{side}: {moved} hold->touch")
    return f"{key} " + " ".join(msg)


def main() -> None:
    from touch.data import hocap as H

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    keys = [f"{p.parent.name}/{p.name}" for p in sorted(PROC.glob("*/*")) if (p / "done").exists()]
    assert not any(H.split_of(k.split("/")[0]) == "test" for k in keys)
    with get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(job, keys)):
            print(f"{i + 1}/{len(keys)} {r}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
