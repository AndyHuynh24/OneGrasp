"""DexYCB -> grasp-model training data (camera only: labels from DexYCB's 3D hand/object fits).

    python scripts/prepare_dexycb.py --workers 6          # processes every downloaded subject

Per sequence, data/dexycb_proc/<subject>/<sequence>/:
  contact.npz  finger [N,5] / state [N] / vert [N,778] int8 labels (1, 0, -1 = ignore;
               touch.data.dexycb.labels_from_dist, default thresholds), part_dist [N,6],
               tip_dist [N,5], obj_disp [N], valid [N]
  <cam>.npz    per camera: offset/length [N,6] (5 fingertip patches + hand crop, JPEG; packed
               into data/patches/dexycb_store.bin at the end), present [N], lm_px [N,21,3]
               (MediaPipe, the hand used), lm_feat [N,F] (v2), flip (GT hand is left),
               mp_right [N] (MediaPipe says Right; -1 none), hand_err_px [N] (MediaPipe vs GT 2D),
               verts [N,778,3] f16 / joints [N,21,3] f16 (root-relative, camera axes, mirrored to
               a right hand when flip), root [N,3] (wrist, camera frame, m)
Hand choice: train subjects use the MediaPipe hand closest to the GT 2D joints (clean labels);
val/test subjects use the deployable choice (highest-scoring hand), so their scores are honest.
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
from touch.data import dexycb as D  # noqa: E402
from touch.data.landmarks import DEFAULT_MODEL, _make_landmarker  # noqa: E402
from touch.data.patches import (  # noqa: E402
    HAND_CROP_SIZE,
    PATCH_SIZE,
    finger_patches,
    hand_crop,
    landmark_features,
)

OUT = Path("data/dexycb_proc")
PATCH_ROOT = Path("data/patches")
JPEG_QUALITY = 92
GT_MATCH_PX = 40.0  # 640x480 images: max mean MediaPipe-to-GT joint distance to accept a hand
FPS = 30.0


def vertex_labels(vd: np.ndarray, valid: np.ndarray, c_mm=3.0, ign_mm=10.0) -> np.ndarray:
    d = vd * 1000.0
    out = np.where(d <= c_mm, 1, np.where(d > ign_mm, 0, -1)).astype(np.int8)
    out[~valid] = -1
    return out


def mediapipe_track(seq: D.Seq, cam: str):
    import mediapipe as mp

    n = seq.n_frames
    lm = np.full((n, 2, 21, 3), np.nan, np.float32)
    right = np.full((n, 2), -1, np.int8)
    score = np.zeros((n, 2), np.float32)
    with _make_landmarker(DEFAULT_MODEL, 0.3) as det:
        for i in range(n):
            bgr = cv2.imread(str(seq.image(cam, i)))
            h, w = bgr.shape[:2]
            img = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            )
            r = det.detect_for_video(img, int(round(i * 1000 / FPS)))
            for k, (hl, hd) in enumerate(zip(r.hand_landmarks, r.handedness, strict=True)):
                if k >= 2:
                    break
                lm[i, k] = [[p.x * w, p.y * h, p.z * w] for p in hl]
                right[i, k] = int(hd[0].category_name == "Right")
                score[i, k] = hd[0].score
    return lm, right, score


def process(seq: D.Seq) -> tuple[str, float, str]:
    out_dir = OUT / seq.subject / seq.name
    if (out_dir / "done").exists():
        return seq.name, 0.0, "skip"
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    r = D.sequence_distances(seq)
    L = D.labels_from_dist(r["part_dist"], r["obj_disp"], r["valid"])
    np.savez_compressed(
        out_dir / "contact.npz", finger=L["finger"], state=L["state"],
        vert=vertex_labels(r["vert_dist"], r["valid"]), part_dist=r["part_dist"],
        tip_dist=r["tip_dist"], obj_disp=r["obj_disp"], valid=r["valid"],
    )  # fmt: skip
    train = seq.split == "train"
    flip = seq.side == "left"
    betas = D.mano_betas(seq)
    n = seq.n_frames
    for cam in D.CAMS:
        lab = [seq.labels(cam, i) for i in range(n)]
        j2d = np.stack([x["joint_2d"][0] for x in lab])
        pose_m = np.stack([x["pose_m"][0] for x in lab])
        verts, joints = D.mano_forward(np.where(r["valid"][:, None], pose_m, 0), betas, seq.side)
        root = joints[:, 0].copy()
        verts, joints = verts - root[:, None], joints - root[:, None]
        if flip:
            verts[..., 0] *= -1
            joints[..., 0] *= -1
        mp_lm, mp_right, score = mediapipe_track(seq, cam)
        err = np.full((n, 2), np.inf, np.float32)
        for k in range(2):
            ok = np.isfinite(mp_lm[:, k, 0, 0]) & r["valid"]
            err[ok, k] = np.linalg.norm(mp_lm[ok, k, :, :2] - j2d[ok], axis=-1).mean(-1)
        if train:
            pick = err.argmin(1)
            use = err[np.arange(n), pick] <= GT_MATCH_PX
        else:
            pick = score.argmax(1)
            use = np.isfinite(mp_lm[np.arange(n), pick, 0, 0])
        hand = np.full((n, 21, 3), np.nan, np.float32)
        hand[use] = mp_lm[use, pick[use]]
        feats, present = landmark_features(hand, "v2")
        offset = np.full((n, 6), -1, np.int64)
        length = np.zeros((n, 6), np.int32)
        chunks, pos = [], 0
        for i in np.flatnonzero(present):
            img = cv2.imread(str(seq.image(cam, i)))
            ims = list(finger_patches(img, hand[i, :, :2], PATCH_SIZE))
            ims.append(hand_crop(img, hand[i, :, :2], HAND_CROP_SIZE, flip=flip))
            for f, im in enumerate(ims):
                ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                assert ok
                offset[i, f], length[i, f] = pos, len(buf)
                chunks.append(buf.reshape(-1))
                pos += len(buf)
        np.savez(
            out_dir / f"{cam}.npz",
            jpeg=np.concatenate(chunks) if chunks else np.zeros(0, np.uint8),
            offset=offset, length=length, present=present, lm_px=hand, lm_feat=feats,
            flip=np.bool_(flip), mp_right=mp_right[np.arange(n), pick] * present - (~present),
            hand_err_px=np.where(present, err[np.arange(n), pick], np.nan).astype(np.float32),
            verts=verts.astype(np.float16), joints=joints.astype(np.float16), root=root,
        )  # fmt: skip
    (out_dir / "done").touch()
    return seq.name, time.time() - t0, "ok"


def pack_store() -> None:
    """Append every <cam>.npz that still holds `jpeg` to data/patches/dexycb_store.bin and record
    its offsets in the npz (then drop the jpeg). Crash-safe per file: the npz is rewritten only
    after its bytes are in the store and fsynced."""
    import os

    PATCH_ROOT.mkdir(parents=True, exist_ok=True)
    store = PATCH_ROOT / "dexycb_store.bin"
    with open(store, "ab") as fh:
        for p in sorted(OUT.glob("*/*/*.npz")):
            if p.name == "contact.npz":
                continue
            with np.load(p) as z:
                if "jpeg" not in z.files:
                    continue
                d = {k: z[k] for k in z.files}
            pos = fh.tell()
            d.pop("jpeg").tofile(fh)
            fh.flush()
            os.fsync(fh.fileno())
            off = d["offset"]
            off[off >= 0] += pos
            tmp = p.with_suffix(".tmp.npz")
            np.savez(tmp, **d)
            tmp.replace(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="first N sequences only (smoke test)")
    ap.add_argument("--no-pack", action="store_true")
    args = ap.parse_args()
    done = {p.name.split("_")[-1] for p in Path("data/dexycb").glob(".done_subject_*")}
    seqs = [s for s in D.list_sequences() if s.subject[-2:] in done]
    if args.limit:
        seqs = seqs[: args.limit]
    print(f"{len(seqs)} sequences from subjects {sorted(done)}", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, (name, dt, st) in enumerate(pool.imap_unordered(process, seqs), 1):
            if st != "skip" and (k % 25 == 0 or k == len(seqs)):
                print(f"[{time.time() - t0:6.0f}s] {k}/{len(seqs)} {name} {dt:.1f}s", flush=True)
    if not args.no_pack:
        pack_store()
        print("packed", PATCH_ROOT / "dexycb_store.bin", flush=True)


if __name__ == "__main__":
    main()
