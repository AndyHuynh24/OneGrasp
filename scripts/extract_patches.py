"""Cut aligned fingertip patches (all 5 fingers, primary MediaPipe hand) for every frame and view.

Writes data/patches/<source>/<participant>/<sequence>/<view>.npz:
    jpeg     [B] uint8      all patch JPEGs concatenated
    offset   [N, 5] int64   start of each patch in `jpeg` (-1 where no hand)
    length   [N, 5] int32
    present  [N] bool       primary hand detected (MediaPipe, causal VIDEO mode)
    lm_feat  [N, F] float32 hand-relative landmark features (touch.data.patches.landmark_features)
    L_px     [N, 5] float32 DIP-tip distance in image pixels (fingertip pixel size)
Skips outputs that already exist. Uses `primary` (deployable) hands only. At the end all JPEGs are
packed into data/patches/<source>_store.bin (+ _index.npz) and dropped from the per-sequence files.

Usage:
    python scripts/extract_patches.py --source egopressure --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from touch.data.catalog import load_catalog  # noqa: E402
from touch.data.patch_store import build_store  # noqa: E402
from touch.data.patches import (  # noqa: E402
    PATCH_SIZE,
    finger_patches,
    landmark_features,
    primary_hand,
)
from touch.data.pressure import FINGER_DIPS, FINGER_TIPS  # noqa: E402

OUT_ROOT = Path("data/patches")
JPEG_QUALITY = 92


def out_path(seq, view: str) -> Path:
    return OUT_ROOT / seq.source / seq.participant / seq.name / f"{view}.npz"


def run_one(job) -> tuple[str, int, float]:
    seq, view = job
    out = out_path(seq, view)
    if out.exists():
        return str(out), -1, 0.0
    t0 = time.time()
    n = len(seq)
    hand = primary_hand(seq.landmarks(view))
    feats, present = landmark_features(hand)
    offset = np.full((n, 5), -1, np.int64)
    length = np.zeros((n, 5), np.int32)
    L_px = np.full((n, 5), np.nan, np.float32)
    chunks, pos = [], 0
    for i in np.flatnonzero(present):
        img = cv2.imread(str(seq.image_path(view, i)))
        if img is None:
            present[i] = False
            continue
        h = hand[i, :, :2]
        P = finger_patches(img, h, PATCH_SIZE)
        for f in range(5):
            ok, buf = cv2.imencode(".jpg", P[f], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            assert ok
            offset[i, f], length[i, f] = pos, len(buf)
            chunks.append(buf.reshape(-1))
            pos += len(buf)
        L_px[i] = np.linalg.norm(h[list(FINGER_TIPS)] - h[list(FINGER_DIPS)], axis=1)
    jpeg = np.concatenate(chunks) if chunks else np.zeros(0, np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(
        tmp, jpeg=jpeg, offset=offset, length=length, present=present, lm_feat=feats, L_px=L_px
    )
    tmp.replace(out)
    return str(out), int(present.sum()), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="egopressure")
    ap.add_argument("--split", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    seqs = load_catalog(source=args.source, split=args.split)
    jobs = [(s, v) for s in seqs for v in s.views if s.has_landmarks(v)]
    print(f"{len(jobs)} (sequence, view) jobs", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, (path, n, dt) in enumerate(pool.imap_unordered(run_one, jobs), 1):
            if n >= 0:
                print(f"[{k}/{len(jobs)}] {path} frames={n} {dt:.0f}s", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    build_store(args.source, seqs)
    compact(args.source)
    print("packed into", OUT_ROOT / f"{args.source}_store.bin", flush=True)


def compact(source: str) -> None:
    """Drop the per-sequence jpeg arrays once they are packed (they are duplicates)."""
    for p in (OUT_ROOT / source).glob("*/*/*.npz"):
        with np.load(p) as z:
            if "jpeg" not in z.files:
                continue
            d = {k: z[k] for k in z.files if k != "jpeg"}
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **d)
        tmp.replace(p)


if __name__ == "__main__":
    main()
