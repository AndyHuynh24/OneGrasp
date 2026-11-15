"""Fit EgoPressure's pad grid -> world mapping from the data (no calibration file needed).

Model (scale fixed at the Sensel pitch, axes aligned — both verified on p_001):
    world_xy_mm = 1.25 * (col, row) + origin
and the pressure centroid of a single-finger press sits at  tip + alpha * (DIP - tip)
on the index distal phalanx. Solves origin (2) and alpha (1) by least squares over every
index_press contact frame (frame max >= 5 kPa, GT joints present, every 2nd frame) of TRAIN
participants only (configs/splits.yaml), then reports the residual on val for the fit and for
the current constant. Test participants are never read.

Usage:
    python scripts/fit_ep_pad_mapping.py      # prints the fit; compare with PAD_ORIGIN_XY_MM
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from touch.data import egopressure as ep  # noqa: E402
from touch.data.catalog import load_splits  # noqa: E402
from touch.data.names import parse_sequence_name  # noqa: E402

RAW = Path("data/egopressure/raw")


def collect(split_of: dict[str, str]) -> dict[str, np.ndarray]:
    A, b, split, hand, variant = [], [], [], [], []
    for done in sorted(RAW.glob("*/.done")):
        d = done.parent
        n = parse_sequence_name(d.name)
        if n.family != "index_press" or n.variant == "no-contact":
            continue
        s = split_of[f"ep_{n.participant:03d}"]
        if s == "test":
            continue
        for f in ep.list_frames(d)[::2]:
            F = ep.load_force(ep.frame_path(d, f, "force.bin"))
            if F.max() < 5:
                continue
            a = ep.load_anno(ep.frame_path(d, f, "anno.pkl"))
            if not a.get("has_annotation"):
                continue
            W = np.where(F >= 1.0, F, 0)
            r, c = np.nonzero(W)
            w = W[r, c]
            cen = np.array([np.sum(c * w), np.sum(r * w)]) / w.sum() * 1.25  # (x, y) mm
            J = np.asarray(a["joint_position"], np.float64) * 1000
            tip, dip = J[8, :2], J[7, :2]
            v = dip - tip
            # cen + origin = tip + alpha (dip - tip)  ->  origin - alpha (dip - tip) = tip - cen
            A.append([[1, 0, -v[0]], [0, 1, -v[1]]])
            b.append(tip - cen)
            split.append(s)
            hand.append(n.hand)
            variant.append(n.variant)
    return {
        "A": np.array(A, np.float64),
        "b": np.array(b, np.float64),
        "split": np.array(split),
        "hand": np.array(hand),
        "variant": np.array(variant),
    }


def main() -> None:
    D = collect(load_splits())
    A, b = D["A"], D["b"]
    train = D["split"] == "train"
    x, *_ = np.linalg.lstsq(A[train].reshape(-1, 3), b[train].reshape(-1), rcond=None)

    # alpha for the current constant: best value with the origin held fixed (train frames)
    o = np.array(ep.PAD_ORIGIN_XY_MM)
    At, bt = A[train], b[train] - o
    alpha0 = float(np.sum(At[:, :, 2] * bt) / np.sum(At[:, :, 2] ** 2))
    current = np.array([o[0], o[1], alpha0])

    print(f"train: {int(train.sum())} frames; val: {int((~train).sum())} frames (test not read)")
    print(f"fit (train)      origin = ({x[0]:.1f}, {x[1]:.1f}) mm   alpha = {x[2]:.2f}")
    print(f"current constant origin = ({o[0]:.1f}, {o[1]:.1f}) mm   alpha = {alpha0:.2f}")
    static = np.isin(D["variant"], ["high", "low"])
    for name, p in (("fit", x), ("current", current)):
        res = np.linalg.norm(A @ p - b, axis=1)
        for s in ("train", "val"):
            for h in ("left", "right"):
                for vname, vm in (("high/low", static), ("pull/push/rotate", ~static)):
                    e = res[(D["split"] == s) & (D["hand"] == h) & vm]
                    if len(e):
                        print(
                            f"  {name:7s} {s:5s} {h:5s} {vname:16s} n={len(e):5d}  "
                            f"median {np.median(e):.1f} mm  p90 {np.percentile(e, 90):.1f} mm"
                        )


if __name__ == "__main__":
    main()
