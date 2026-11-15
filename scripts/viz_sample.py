"""Visual sanity check of the canonical data (P1 step 5).

For random frames, draw on each view: the pressure map projected into the image (coloured cells),
GT joints (green dots, EgoPressure only) and the MediaPipe primary hand (red skeleton).
If calibration and the pad mapping are right, the pressure blob sits under the touching finger.

Usage:
    python scripts/viz_sample.py --source egopressure --n 8 --out docs/figures/viz_egopressure.jpg
    python scripts/viz_sample.py --source pvdb --n 8 --out docs/figures/viz_pvdb.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from touch.data import egopressure as ep  # noqa: E402
from touch.data import pvdb as pv  # noqa: E402
from touch.data.catalog import Sequence, load_catalog  # noqa: E402
from touch.data.pressure import FINGER_NAMES, NOISE_FLOOR_KPA, SENSEL_SHAPE  # noqa: E402

BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]
TILE = 360


def cell_pixels(s: Sequence, view: str, i: int, rc: np.ndarray) -> np.ndarray:
    """Pad cells (row, col) [M, 2] -> pixel positions [M, 2] in this view."""
    v = s.meta["views"][view]
    K, dist = np.array(v["K"]), np.array(v["dist"])
    if s.source == "egopressure":
        X = ep.pad_rc_to_world_m(rc)
        if view == "ego":
            R = s.labels["ego_R"][i].astype(np.float64)
            t = s.labels["ego_t_m"][i].astype(np.float64)
        else:
            R, t = np.array(v["R"]), np.array(v["t_m"])
    else:
        X = pv.pad_rc_to_world_m(rc)
        R, t = np.array(v["R"]), np.array(v["t_m"])
    return ep.project(X, R, t, K, dist)


def draw(s: Sequence, view: str, i: int, kpa: np.ndarray) -> np.ndarray:
    img = cv2.imread(str(s.image_path(view, i)))
    rr, cc = np.nonzero(kpa >= NOISE_FLOOR_KPA)
    pts_all = []
    if len(rr):
        uv = cell_pixels(s, view, i, np.stack([rr, cc], 1).astype(np.float64))
        col = cv2.applyColorMap(
            np.clip(kpa[rr, cc] / 20 * 255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_JET
        )[:, 0]
        for (u, w), c in zip(uv, col, strict=True):
            if np.isfinite(u):
                cv2.circle(img, (int(u), int(w)), 3, tuple(int(x) for x in c), -1)
        pts_all.append(uv)
    gt = s.labels.get(f"gt2d_{view}")
    if gt is not None and np.isfinite(gt[i]).all():
        for u, w in gt[i]:
            cv2.circle(img, (int(u), int(w)), 5, (0, 255, 0), -1)
        pts_all.append(gt[i])
    if s.has_landmarks(view):
        z = s.landmarks(view)
        k = z["primary"][i]
        if k >= 0:
            lm = z["lm_px"][i, k, :, :2]
            for a, b in BONES:
                cv2.line(img, tuple(lm[a].astype(int)), tuple(lm[b].astype(int)), (0, 0, 255), 2)
            pts_all.append(lm)
    # Crop around everything drawn (or the image centre if nothing).
    pts = np.concatenate(pts_all) if pts_all else np.empty((0, 2))
    pts = pts[np.isfinite(pts).all(1)]
    h, w = img.shape[:2]
    if len(pts):
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.75 + 40
    else:
        cx, cy, half = w / 2, h / 2, h / 2
    xa, ya = int(max(cx - half, 0)), int(max(cy - half, 0))
    xb, yb = int(min(cx + half, w)), int(min(cy + half, h))
    crop = img[ya:yb, xa:xb]
    return cv2.resize(crop, (TILE, TILE)) if crop.size else np.zeros((TILE, TILE, 3), np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", default="egopressure", choices=["egopressure", "pvdb"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    seqs = [s for s in load_catalog(source=args.source) if s.labels["contact"].any()]
    rows = []
    for s in rng.choice(seqs, size=min(args.n, len(seqs)), replace=False):
        c = s.labels["contact"]
        # Half the samples during contact, half just before a touch-down (hover).
        idx = (
            np.flatnonzero(c)
            if rng.random() < 0.5
            else np.flatnonzero(np.diff(c.astype(int)) == 1) - 3
        )
        idx = idx[(idx >= 0) & (idx < len(s))]
        i = int(rng.choice(idx)) if len(idx) else int(rng.integers(len(s)))
        kpa = s.pressure()[i]
        assert kpa.shape == SENSEL_SHAPE
        tiles = [draw(s, v, i, kpa) for v in s.views]
        row = np.hstack(tiles)
        F = s.fingers
        fing = ""
        if F is not None and F["valid"][i]:
            fing = " ".join(f"{n[0]}{F['max_kpa'][i, k]:.0f}" for k, n in enumerate(FINGER_NAMES))
        text = (
            f"{s.participant} {s.name} #{s.meta['frame_ids'][i]}  max {s.labels['max_kpa'][i]:.1f}"
            f" contact={bool(c[i])}  {fing}"
        )
        per_line = max(row.shape[1] // 8, 20)
        lines = [text[k : k + per_line] for k in range(0, len(text), per_line)]
        bar = np.full((8 + 18 * len(lines), row.shape[1], 3), 255, np.uint8)
        for k, line in enumerate(lines):
            cv2.putText(bar, line, (6, 20 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        rows.append(np.vstack([bar, row]))
    width = max(r.shape[1] for r in rows)
    rows = [
        np.hstack([r, np.full((r.shape[0], width - r.shape[1], 3), 255, np.uint8)]) for r in rows
    ]
    out = Path(args.out or f"docs/figures/viz_{args.source}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


if __name__ == "__main__":
    main()
