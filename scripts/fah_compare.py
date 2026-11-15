"""FAH vs MediaPipe as OneTouch's hand tracker, on EgoPressure validation clips with ground truth.

Inputs: scripts/fah_bench.py outputs (results/fah_bench/<seq>__<view>_<model>_<dev>_<set>.npz)
and the sequence's canonical data (GT joints projected into the view, MediaPipe, pressure contact).

Per tracker, on frames with GT joints:
  found    share of GT frames where the tracker returned the sequence's hand
  tip_px   fingertip 2D error vs GT (median, p90), px; tip_rel = same / GT hand size (wrist-MCP3)
  tip_px_offset_removed  the same after removing each tip's constant offset (GT = MANO joints)
  jitter   median |2nd difference| of fingertip px over consecutive found frames, minus GT's own
FAH "contact" runs also: per-finger frame-level contact (>= 0.5, 10 mm geometric) vs pressure GT
(>= 1 kPa) -> precision / recall / F1, and the share of frames called contact (hover-only clips).

Usage:
    python scripts/fah_compare.py results/fah_bench/<seq>__overhead_fah-s_mps_track.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch.data.catalog import load_catalog  # noqa: E402

TIPS = [4, 8, 12, 16, 20]


def parse(path: Path) -> tuple[str, str]:
    seq, rest = path.stem.split("__", 1)
    return seq, rest.split("_", 1)[0]


def tip_stats(pred: np.ndarray, gt: np.ndarray, hand: np.ndarray) -> dict:
    """pred, gt [N, 21, 2] (NaN = not found / no GT), hand [N] GT hand size px."""
    has_gt = np.isfinite(gt[:, TIPS]).all((1, 2))
    found = has_gt & np.isfinite(pred[:, TIPS]).all((1, 2))
    err = np.linalg.norm(pred[found][:, TIPS] - gt[found][:, TIPS], axis=-1)  # [M, 5]
    rel = err / hand[found, None]
    # definitions differ (GT = fitted MANO joints; MediaPipe marks skin points, FAH uses MANO's):
    # also report the error after removing each tip's constant offset over the clip
    d = pred[found][:, TIPS] - gt[found][:, TIPS]
    err_db = np.linalg.norm(d - np.median(d, axis=0, keepdims=True), axis=-1) if len(d) else d

    def jit(x: np.ndarray, ok: np.ndarray) -> float:
        ok3 = ok[2:] & ok[1:-1] & ok[:-2]
        d2 = x[2:, TIPS] - 2 * x[1:-1, TIPS] + x[:-2, TIPS]
        return float(np.median(np.linalg.norm(d2[ok3], axis=-1))) if ok3.any() else float("nan")

    return dict(
        gt_frames=int(has_gt.sum()),
        found=round(float(found.sum() / max(has_gt.sum(), 1)), 3),
        tip_px_median=round(float(np.median(err)), 1) if len(err) else None,
        tip_px_p90=round(float(np.percentile(err, 90)), 1) if len(err) else None,
        tip_rel_median=round(float(np.median(rel)), 3) if len(rel) else None,
        tip_px_offset_removed=round(float(np.median(err_db)), 1) if len(err) else None,
        jitter_px=round(jit(pred, found) - jit(gt, has_gt), 2),
    )


def contact_stats(prob: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    """prob [N, 5] FAH finger contact (NaN = no hand -> no contact), gt [N, 5] bool, valid [N]."""
    p = np.nan_to_num(prob, nan=0.0)[valid] >= 0.5
    g = gt[valid]
    tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(
        finger_frames=int(g.size),
        gt_contact_frames=int(g.sum()),
        precision=round(pr, 3),
        recall=round(rc, 3),
        f1=round(2 * pr * rc / max(pr + rc, 1e-9), 3),
        pred_contact_share=round(float(p.mean()), 3),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fah_npz", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("results/fah_bench/compare.json"))
    args = ap.parse_args()
    cat = {s.name: s for s in load_catalog(source="egopressure", split="val")}
    rows = []
    for f in args.fah_npz:
        seq, view = parse(f)
        s = cat[seq]
        z = np.load(f)
        n = len(z["n"])
        labels, fingers, lm = s.labels, s.fingers, s.landmarks(view)
        gt = labels[f"gt2d_{view}"][:n].astype(np.float64)
        hand = np.linalg.norm(gt[:, 9] - gt[:, 0], axis=-1)
        side = 1 if seq.endswith("_right") else 0  # the sequence's hand
        fah_j = z["j2d"][:, side].astype(np.float64)
        prim = lm["primary"][:n]
        mp = np.full((n, 21, 2), np.nan)
        ok = prim >= 0
        mp[ok] = lm["lm_px"][:n][ok, prim[ok], :, :2]
        row = dict(
            file=f.name,
            seq=seq,
            view=view,
            frames=n,
            fah_ms_median=round(float(np.median(z["ms"])), 1),
            fah=tip_stats(fah_j, gt, hand),
            mediapipe=tip_stats(mp, gt, hand),
        )
        if np.isfinite(z["fingers"][:, side]).any():
            valid = fingers["valid"][:n].astype(bool)
            gt_c = fingers["contact"][:n].astype(bool)
            row["fah_contact"] = contact_stats(z["fingers"][:, side, 1:6], gt_c, valid)
            if "fingers_k" in z.files:  # FAH's 5 / 10 / 20 / 35 mm outputs and its video model
                for i, mm in enumerate((5, 10, 20, 35)):
                    row[f"fah_contact_{mm}mm"] = contact_stats(
                        z["fingers_k"][:, side, i, 1:6], gt_c, valid
                    )
                row["fah_contact_video"] = contact_stats(
                    z["fingers_video"][:, side, 1:6], gt_c, valid
                )
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(args.out.read_text()) if args.out.exists() else []
    keep = [r for r in old if r["file"] not in {r2["file"] for r2 in rows}]
    args.out.write_text(json.dumps(keep + rows, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
