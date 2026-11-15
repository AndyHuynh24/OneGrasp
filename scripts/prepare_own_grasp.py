"""Our own grasp recordings (scripts/collect_labels.py, data/own/grasp*/<session>) -> grasp-model
tracks, processed exactly like the live demo (scripts/live_grasp.py): MediaPipe VIDEO mode via
frames_with_hands, the same-hand HandTracker, 5 fingertip patches + hand crop (JPEG 90, hand crop
mirrored for a left hand). Grasp sessions: hand state only (none / touch / hold from the keys);
finger, vertex and mesh labels are unknown (-1 / no mesh). Touch sessions (data/own/touch*,
fingertip on the bare table, no object): state none (0) wherever labelled, fingertip contact
per finger: the selected finger touches while F or H is down (tap / long press), the other
fingers are not touching (the touch task's label definition, touch.data.own_labels).

Splits are by whole session (data/own/grasp_splits.json; sessions not listed are train). The test
session is never used for training or tuning (project rule).

Label cleaning (state -> -1 = ignored):
  * frames from each H release to +RELEASE_PAD_S (H was let go before the hand: hand opening
    starts a median 174 ms, p75 438 ms after H-up, docs/DECISIONS.md 2026-11-13)
  * "none" frames within EDGE_S of a hold start / end: F (touch) was rarely pressed, so contact
    before lift-off / after set-down is mostly unlabelled there

    python scripts/prepare_own_grasp.py [--workers 5]
Writes data/own_grasp/<split>/<session>/track.npz and data/patches/own_grasp_store.bin.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from touch.data.grasp_store import OWN_PROC, PATCH_ROOT  # noqa: E402
from touch.data.own_labels import labels_from_meta  # noqa: E402
from touch.data.patches import (  # noqa: E402
    HAND_CROP_SIZE,
    PATCH_SIZE,
    finger_patches,
    hand_crop,
    landmark_features,
)

OWN = Path("data/own")  # sessions in data/own/grasp*, data/own/touch* (e.g. grasp_an, touch_an)
SPLITS = Path("data/own/grasp_splits.json")
JPEG_QUALITY = 90
RELEASE_PAD_S = 0.45
EDGE_S = 1.0
MIN_SECONDS = 60.0


def clean_state(label: np.ndarray, valid: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Per-frame key labels -> training state (-1 ignore): invalid frames, and "none" frames
    within EDGE_S of a hold edge."""
    state = np.where(valid, label, -1).astype(np.int8)
    hold = label == 2
    edges = np.flatnonzero(np.diff(hold.astype(np.int8)) != 0) + 1
    near = np.zeros(len(label), bool)
    k = int(round(EDGE_S * fps))
    for e in edges:
        near[max(0, e - k) : e + k] = True
    state[near & (state == 0)] = -1
    return state


def sessions() -> list[Path]:
    return sorted(s for pat in ("grasp*", "touch*") for d in OWN.glob(pat) if d.is_dir()
                  for s in d.iterdir()
                  if (s / "meta.json").exists() and (s / "video.mp4").exists())  # fmt: skip


def touch_labels(label: np.ndarray, valid: np.ndarray, finger: np.ndarray):
    """Touch-task key labels -> (state, finger contact [N, 5]): no object, so state none where
    labelled; the selected finger touches while F (1) or H (2) is down, the others don't."""
    state = np.where(valid, 0, -1).astype(np.int8)
    fc = np.full((len(label), 5), -1, np.int8)
    fc[valid] = 0
    ok = valid & (finger >= 0)
    fc[np.flatnonzero(ok), finger[ok]] = (label[ok] > 0).astype(np.int8)
    return state, fc


def process(sdir: Path) -> str:
    from live_grasp import HandTracker, frames_with_hands

    name = sdir.name
    split = json.loads(SPLITS.read_text()).get(name, "train")
    out = OWN_PROC / split / name / "track.npz"
    if out.exists():
        return f"skip {name}"
    meta = json.loads((sdir / "meta.json").read_text())
    touch = meta["task"] == "touch"
    if not touch:
        meta.setdefault("release_pad_s", {"h": RELEASE_PAD_S})
    ev = json.loads((sdir / "events.json").read_text())
    t = np.load(sdir / "frames.npz")["t"]
    lab = labels_from_meta(t, ev, meta)
    cap = cv2.VideoCapture(str(sdir / "video.mp4"))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    tracker = HandTracker(w)
    hands_px, left, chunks = [], [], []
    rows = []  # per frame: list of 6 (offset, length) or None
    pos = 0
    for frame, hands, handedness, _, _ in frames_with_hands(cap, str(sdir), 30.0,
                                                             threading.Event()):  # fmt: skip
        lm, is_left = tracker.pick(hands, handedness)
        if lm is None:
            hands_px.append(np.full((21, 3), np.nan, np.float32))
            left.append(False)
            rows.append(None)
            continue
        hands_px.append(lm.astype(np.float32))
        left.append(is_left)
        ims = list(finger_patches(frame, lm[:, :2], PATCH_SIZE))
        ims.append(hand_crop(frame, lm[:, :2], HAND_CROP_SIZE, flip=is_left))
        r = []
        for im in ims:
            ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            assert ok
            r.append((pos, len(buf)))
            chunks.append(buf.reshape(-1))
            pos += len(buf)
        rows.append(r)
    n = min(len(rows), len(t))  # video frames == frames.npz rows (one per captured frame)
    assert abs(len(rows) - len(t)) <= 2, (name, len(rows), len(t))
    hand = np.stack(hands_px[:n])
    feats, present = landmark_features(hand, "v2")
    offset = np.full((n, 6), -1, np.int64)
    length = np.zeros((n, 6), np.int32)
    for i in range(n):
        if rows[i] is None:
            present[i] = False
        else:
            offset[i] = [a for a, _ in rows[i]]
            length[i] = [b for _, b in rows[i]]
    left = np.array(left[:n])
    if touch:
        state, fing = touch_labels(lab["label"][:n], lab["valid"][:n], lab["finger"][:n])
    else:
        state, fing = clean_state(lab["label"][:n], lab["valid"][:n]), np.full((n, 5), -1, np.int8)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, jpeg=np.concatenate(chunks) if chunks else np.zeros(0, np.uint8),
             offset=offset, length=length, present=present, lm_px=hand, lm_feat=feats,
             flip=np.bool_(left[present].mean() > 0.5 if present.any() else False),
             mp_right=np.where(present, (~left).astype(np.int8), -1).astype(np.int8),
             state=state, finger=fing, split=np.str_(split),
             object=np.str_(meta.get("object") or ""))  # fmt: skip
    k = state[present]
    if touch:
        c = fing[present]
        return (
            f"{name} [{split}] {n} frames, hand {present.mean():.0%}, touch task: finger "
            f"contact {(c == 1).sum()} / labelled {(c >= 0).sum()} finger-frames"
        )
    return (f"{name} [{split}] {n} frames, hand {present.mean():.0%}, state none "
            f"{(k == 0).sum()} touch {(k == 1).sum()} hold {(k == 2).sum()} ignored "
            f"{(k < 0).sum()}")  # fmt: skip


def pack_store() -> None:
    """Append unpacked JPEG bytes to data/patches/own_grasp_store.bin (crash-safe per file)."""
    store = PATCH_ROOT / "own_grasp_store.bin"
    with open(store, "ab") as fh:
        for p in sorted(OWN_PROC.glob("*/*/track.npz")):
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
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    names = []
    for s in sessions():
        t = np.load(s / "frames.npz")["t"]
        if len(t) / 30.0 < MIN_SECONDS:
            print(f"skip {s.name}: {len(t) / 30:.0f} s", flush=True)
            continue
        names.append(s)
    split = json.loads(SPLITS.read_text())
    assert "test" in split.values() and "val" in split.values(), SPLITS
    print(f"{len(names)} sessions; splits {split}", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        for k, m in enumerate(pool.imap_unordered(process, names), 1):
            print(f"[{(time.time() - t0) / 60:5.1f} min] {k}/{len(names)} {m}", flush=True)
    pack_store()
    print("packed", PATCH_ROOT / "own_grasp_store.bin", flush=True)


if __name__ == "__main__":
    main()
