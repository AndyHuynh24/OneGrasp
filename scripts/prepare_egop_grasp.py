"""EgoPressure / PressureVisionDB "fingers on a surface, no object" tracks for the grasp model.

The grasp model's state head means "interacting with an OBJECT" (DexYCB / HO-Cap label contact
with objects only), yet g1 calls a hand resting on the table touch / hold (HO-Cap benchmark).
EgoPressure shows hands pressing / drawing / typing on a flat pad with no object, from overhead
and egocentric views: those frames are state 0 (none). Finger / vertex / mesh labels are left out
(the fingertip head keeps meaning "fingertip touching something").

Train-split participants only (frozen split, asserted). Same per-track format as
scripts/prepare_dexycb.py (5 fingertip patches + hand crop, JPEG, packed into
data/patches/egop_grasp_store.bin), read by touch.data.grasp_store.load_egop_tracks.

PVDB (--source pvdb, g4): overhead camera 1.6 deg from vertical over a tabletop pad (the
top-down-webcam view), 15 fps -> upsampled to 30 fps (landmarks interpolated at the half frames,
patches repeated), and its pressure-pad fingertip contact labels are KEPT (fingers.npz contact,
-1 where not valid): the fingertip head learns hover vs press seen from above. Output
data/pvdb_grasp + data/patches/pvdb_grasp_store.bin.

    python scripts/prepare_egop_grasp.py [--source egopressure --tracks 80 --frames 600]
    python scripts/prepare_egop_grasp.py --source pvdb --tracks 160 --frames 300
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch.data.catalog import load_catalog  # noqa: E402
from touch.data.grasp_store import EGOP_PROC, PATCH_ROOT, PVDB_PROC, PVDB_VAL_PROC  # noqa: E402
from touch.data.patches import (  # noqa: E402
    HAND_CROP_SIZE,
    PATCH_SIZE,
    finger_patches,
    hand_crop,
    landmark_features,
    primary_hand,
)

JPEG_QUALITY = 90


def pick_tracks(n_tracks: int, seed: int, source: str = "egopressure",
                split: str = "train") -> list[tuple]:  # fmt: skip
    """Seeded, stratified by (gesture family, view): [(sequence, view)]."""
    seqs = load_catalog(source=source, split=split)
    assert seqs and all(s.split == split for s in seqs)
    groups: dict[tuple, list] = {}
    for s in seqs:
        for v in s.views:
            if (s.root / f"landmarks_{v}.npz").exists():
                groups.setdefault((s.meta["family"], v), []).append(s)
    rng = np.random.default_rng(seed)
    per = max(1, n_tracks // len(groups))
    out = []
    for key in sorted(groups):
        g = groups[key]
        for i in rng.permutation(len(g))[:per]:
            out.append((g[i], key[1]))
    return out


def upsample2(hand: np.ndarray) -> np.ndarray:
    """[N, 21, 3] at 15 fps -> [2N, 21, 3] at 30 fps: odd frames = mean of neighbours (NaN if
    either is missing), the last odd frame repeats."""
    out = np.repeat(hand, 2, axis=0)
    out[1:-1:2] = (hand[:-1] + hand[1:]) / 2
    return out


def process(job) -> str:
    seq, view, max_frames, source, proc = job
    proc = Path(proc)
    out = proc / seq.participant / seq.name / f"{view}.npz"
    if out.exists():
        return f"skip {out}"
    with np.load(seq.root / f"landmarks_{view}.npz") as z:
        lm = {k: z[k] for k in z.files}
    hand = primary_hand(lm)[:max_frames]
    n = len(hand)
    prim = lm["primary"][:n]
    handed = np.where(prim >= 0, lm["handed"][np.arange(n), np.maximum(prim, 0)], -1)
    finger = None
    if source == "pvdb":
        with np.load(seq.root / "fingers.npz") as z:
            finger = np.where(z["valid"][:n, None], z["contact"][:n], -1).astype(np.int8)
    _, present = landmark_features(hand, "v2")
    flip = seq.meta["hand"] == "left"
    offset = np.full((n, 6), -1, np.int64)
    length = np.zeros((n, 6), np.int32)
    chunks, pos = [], 0
    for i in np.flatnonzero(present):
        img = cv2.imread(str(seq.image_path(view, i)))
        if img is None:
            present[i] = False
            continue
        ims = list(finger_patches(img, hand[i, :, :2], PATCH_SIZE))
        ims.append(hand_crop(img, hand[i, :, :2], HAND_CROP_SIZE, flip=flip))
        for f, im in enumerate(ims):
            ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            assert ok
            offset[i, f], length[i, f] = pos, len(buf)
            chunks.append(buf.reshape(-1))
            pos += len(buf)
    mp_right = np.where(present, handed, -1).astype(np.int8)
    extra = {}
    if source == "pvdb":  # 15 -> 30 fps; patches repeat, landmark features from interpolation
        hand = upsample2(hand)
        offset, length = np.repeat(offset, 2, 0), np.repeat(length, 2, 0)
        present, mp_right = np.repeat(present, 2), np.repeat(mp_right, 2)
        extra["finger"] = np.repeat(finger, 2, 0)
    feats, ok = landmark_features(hand, "v2")
    present &= ok
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, jpeg=np.concatenate(chunks) if chunks else np.zeros(0, np.uint8),
             offset=offset, length=length, present=present, lm_px=hand, lm_feat=feats,
             flip=np.bool_(flip), mp_right=mp_right,
             split=np.str_(seq.split), view=np.str_(view), **extra)  # fmt: skip
    return f"{out} {int(present.sum())} frames"


def pack_store(proc: Path, name: str) -> None:
    """Append unpacked JPEG bytes to data/patches/<name>_store.bin (crash-safe per file)."""
    store = PATCH_ROOT / f"{name}_store.bin"
    with open(store, "ab") as fh:
        for p in sorted(proc.glob("*/*/*.npz")):
            with np.load(p) as z:
                if "jpeg" not in z.files:
                    continue
                d = {k: z[k] for k in z.files}
            pos = fh.tell()
            d.pop("jpeg").tofile(fh)
            fh.flush()
            os.fsync(fh.fileno())
            d["offset"][d["offset"] >= 0] += pos
            tmp = p.with_suffix(".tmp.npz")
            np.savez(tmp, **d)
            tmp.replace(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=80)
    ap.add_argument("--frames", type=int, default=600, help="first N frames of each track")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["egopressure", "pvdb"], default="egopressure")
    ap.add_argument("--split", choices=["train", "val"], default="train",
                    help="val: PVDB val fold, model selection only")  # fmt: skip
    args = ap.parse_args()
    assert args.split == "train" or args.source == "pvdb"
    picks = pick_tracks(args.tracks, args.seed, args.source, args.split)
    name = "pvdb_grasp" if args.source == "pvdb" else "egop_grasp"
    proc = PVDB_PROC if args.source == "pvdb" else EGOP_PROC
    if args.split == "val":
        name, proc = "pvdb_grasp_val", PVDB_VAL_PROC
    jobs = [(s, v, args.frames, args.source, str(proc)) for s, v in picks]
    print(f"{len(jobs)} tracks", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, msg in enumerate(pool.imap_unordered(process, jobs), 1):
            print(f"[{time.time() - t0:5.0f}s] {k}/{len(jobs)} {msg}", flush=True)
    pack_store(proc, name)
    print("packed", PATCH_ROOT / f"{name}_store.bin", flush=True)


if __name__ == "__main__":
    main()
