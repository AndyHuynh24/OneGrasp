"""HO-Cap TRAIN / VAL tracks for the grasp model (g5): the top-down rig the live demo targets.

Subjects 1, 2, 3, 4, 6, 7 = train, 9 = val (model selection). Subjects 5 and 8 stay the HO-Cap
TEST set (refused here). Cameras: whatever scripts/download_hocap_cams.py fetched (2 top-down +
1 oblique). Per (sequence, camera, GT hand): MediaPipe on every frame (all hands), each GT hand
takes its nearest MediaPipe hand (eval_hocap.associate, the same rule as the test benchmark),
5 fingertip patches + hand crop exactly as scripts/prepare_dexycb.py, labels from the GT
hand / object meshes (eval_hocap.gt_labels: finger, state with the any2 hold rule, vertex),
wrist-relative MANO mesh in camera axes (left mirrored), packed into
data/patches/hocap_grasp_store.bin. Read by touch.data.grasp_store.load_hocap_tracks.

    python scripts/prepare_hocap_train.py [--workers 6]
Re-running adds cameras downloaded later (e.g. the oblique 105322251225) to prepared sequences.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from touch.data.grasp_store import HOCAP_PROC, PATCH_ROOT  # noqa: E402
from touch.data.patches import (  # noqa: E402
    HAND_CROP_SIZE,
    PATCH_SIZE,
    finger_patches,
    hand_crop,
    landmark_features,
)

JPEG_QUALITY = 90
TRAIN, VAL, TEST = {1, 2, 3, 4, 6, 7}, {9}, {5, 8}


LEGACY_CAMS = ("105322251564", "043422252387", "108222250342")  # prepared before per-camera marks


def cam_done(out_dir: Path, cam: str) -> bool:
    """A camera is done if marked (done_<cam>), or it is one of the first three cameras of a
    sequence prepared before per-camera marks existed (done)."""
    return (out_dir / f"done_{cam}").exists() or (
        (out_dir / "done").exists() and cam in LEGACY_CAMS
    )


def split_of(subject: str) -> str:
    k = int(subject.split("_")[1])
    assert k not in TEST, f"{subject} is the HO-Cap test set"
    return "train" if k in TRAIN else "val"


def cams_ready(seq) -> list[str]:
    n = seq.n_frames
    out = []
    for cam in seq.serials:
        d = seq.root / cam
        if d.is_dir() and (d / f"color_{n - 1:06d}.jpg").exists():
            if sum(1 for _ in d.glob("color_*.jpg")) >= n:
                out.append(cam)
    return out


def process(name: str) -> str:
    from eval_hocap import associate, gt_cam_mesh, gt_labels, mediapipe_cache, world_to_cam

    from touch.data import dexycb as D
    from touch.data import hocap as H

    seq = next(s for s in H.list_sequences(video_only=True) if s.name == name)
    out_dir = HOCAP_PROC / seq.subject / seq.name
    todo = [c for c in cams_ready(seq) if not cam_done(out_dir, c)]
    if not todo:
        return f"skip {name}"
    split = split_of(seq.subject)
    t0, n = time.time(), seq.n_frames
    L = {side: gt_labels(seq, side) for side in seq.sides}
    out_dir.mkdir(parents=True, exist_ok=True)
    msg = []
    for cam in todo:
        K = H.intrinsics(cam)
        lm = mediapipe_cache(seq, cam, n)["lm"]
        gt_uv, gt_ok = {}, {}
        for side in seq.sides:
            jc = world_to_cam(L[side]["joints"], cam)
            gt_uv[side] = D.project(jc.reshape(-1, 3), K).reshape(n, 21, 2)
            gt_ok[side] = L[side]["valid"] & (jc[:, :, 2] > 0.05).all(1)
        tracks = associate(lm, gt_uv, gt_ok)
        for side, hand in tracks.items():
            flip = side == "left"
            feats, present = landmark_features(hand, "v2")
            if present.sum() < 30:
                continue
            offset = np.full((n, 6), -1, np.int64)
            length = np.zeros((n, 6), np.int32)
            chunks, pos = [], 0
            for i in np.flatnonzero(present):
                img = cv2.imread(str(seq.image(cam, i)))
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
            gj, gv = gt_cam_mesh(L[side], cam, flip)
            Ls = L[side]
            np.savez(out_dir / f"{cam}_{side}.npz",
                     jpeg=np.concatenate(chunks) if chunks else np.zeros(0, np.uint8),
                     offset=offset, length=length, present=present, lm_px=hand, lm_feat=feats,
                     flip=np.bool_(flip),
                     mp_right=np.where(present, int(side == "right"), -1).astype(np.int8),
                     verts=gv.astype(np.float16), joints=gj.astype(np.float16),
                     finger=Ls["finger"].astype(np.int8), state=Ls["state_any2"].astype(np.int8),
                     vert=Ls["vert"].astype(np.int8), valid=Ls["valid"],
                     split=np.str_(split))  # fmt: skip
            msg.append(f"{cam}/{side} {int(present.sum())}")
        (out_dir / f"done_{cam}").touch()
    (out_dir / "done").touch()
    return f"{seq.subject}/{name} [{split}] {' '.join(msg)} | {time.time() - t0:.0f} s"


def pack_store() -> None:
    """Append unpacked JPEG bytes to data/patches/hocap_grasp_store.bin (crash-safe per file)."""
    store = PATCH_ROOT / "hocap_grasp_store.bin"
    with open(store, "ab") as fh:
        for p in sorted(HOCAP_PROC.glob("*/*/*.npz")):
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
    from touch.data import hocap as H

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    seqs = [s for s in H.list_sequences(video_only=True)
            if int(s.subject.split("_")[1]) in TRAIN | VAL and cams_ready(s)]  # fmt: skip
    print(f"{len(seqs)} sequences", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        for k, m in enumerate(pool.imap_unordered(process, [s.name for s in seqs]), 1):
            print(f"[{(time.time() - t0) / 60:5.1f} min] {k}/{len(seqs)} {m}", flush=True)
    pack_store()
    print("packed", PATCH_ROOT / "hocap_grasp_store.bin", flush=True)


if __name__ == "__main__":
    main()
