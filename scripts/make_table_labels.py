"""Per-frame "hand touches the table" labels for the grasp model's table head (docs/DECISIONS.md
2026-11-12): 1 if the GT hand mesh comes within 1 cm of the table plane (z = 0 in the table
frame), 0 if not, -1 where the GT hand is missing. Independent of object contact (a hand can
touch both). TRAIN + VAL only; DexYCB / HO-Cap test subjects are skipped.

    DexYCB  data/dexycb_proc/<subject>/<seq>/table.npz         (table frame: the AprilTag)
    HO-Cap  data/hocap_proc/<subject>/<seq>/table_<side>.npz    (world frame: table at z = 0)
PVDB / EgoPressure labels come from the pressure pad in the loader (touch.data.grasp_store).

    python scripts/make_table_labels.py [--workers 5]
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

TABLE_M = 0.01


def dex_job(name: str) -> str:
    import sim_replay as SR

    from touch import robot_sim as S
    from touch.data import dexycb as D

    seq = next(s for s in D.list_sequences() if s.name == name)
    out = SR.PROC / seq.subject / seq.name / "table.npz"
    if out.exists() or not out.parent.exists():
        return f"{name} skip"
    v, _, ok = SR.gt_hand(seq, S.extrinsics(seq)["apriltag"])
    lab = np.full(len(ok), -1, np.int8)
    lab[ok] = (v[ok][:, :, 2].min(1) <= TABLE_M).astype(np.int8)
    np.savez_compressed(out, table=lab)
    return f"{name} {float((lab == 1).mean()):.3f}"


def hocap_job(key: str) -> str:
    from eval_hocap import gt_labels

    from touch.data import hocap as H

    subject, name = key.split("/")
    seq = next(s for s in H.list_sequences(split=H.split_of(subject)) if s.name == name)
    msg = []
    for side in seq.sides:
        out = Path("data/hocap_proc") / subject / name / f"table_{side}.npz"
        if out.exists():
            continue
        L = gt_labels(seq, side)
        lab = np.where(L["valid"], (L["verts"][:, :, 2].min(1) <= TABLE_M), -1).astype(np.int8)
        np.savez_compressed(out, table=lab)
        msg.append(f"{side} {float((lab == 1).mean()):.3f}")
    return f"{key} " + " ".join(msg)


def main() -> None:
    from touch.data import dexycb as D
    from touch.data import hocap as H

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    dex = [s.name for s in D.list_sequences() if s.split in ("train", "val")]
    hoc = [f"{p.parent.name}/{p.name}" for p in sorted(Path("data/hocap_proc").glob("*/*"))
           if (p / "done").exists()]  # fmt: skip
    assert not any(H.split_of(k.split("/")[0]) == "test" for k in hoc)
    with get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(hocap_job, hoc)):
            print(f"hocap {i + 1}/{len(hoc)} {r}", flush=True)
        for i, r in enumerate(pool.imap_unordered(dex_job, dex, chunksize=4)):
            if i % 50 == 0:
                print(f"dexycb {i + 1}/{len(dex)} {r}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
