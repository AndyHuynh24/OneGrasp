"""MediaPipe hand landmarks for a canonical sequence -> landmarks_<view>.npz.

Runs the Hand Landmarker in VIDEO mode (tracking uses past frames only, like live use).
Arrays (N frames, up to 2 hands):
    lm_px     [N, 2, 21, 3] float32  x, y in pixels; z = MediaPipe relative depth * image width
    world_m   [N, 2, 21, 3] float32  MediaPipe world landmarks (metres, hand-centred)
    handed    [N, 2] int8            MediaPipe label: 0 = Left, 1 = Right, -1 = no hand
    score     [N, 2] float32         handedness score
    primary   [N] int8               deployable choice of "the" hand (label matches the sequence's
                                     hand, else highest score); -1 if no hand
    primary_gt [N] int8              EgoPressure only: hand closest to the projected GT joints
                                     (-1 if no GT or no hand within GT_MATCH_PX); else all -1
    gt_err_px [N] float32            mean landmark-to-GT distance of primary_gt (NaN if none)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = Path("checkpoints/mediapipe/hand_landmarker.task")
GT_MATCH_PX = 80.0  # max mean landmark distance to accept a detection as the GT hand
_LABEL = {"Left": 0, "Right": 1}


def _make_landmarker(model_path: Path, min_conf: float):
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    opts = vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(
            model_asset_path=str(model_path), delegate=mpt.BaseOptions.Delegate.CPU
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )
    return vision.HandLandmarker.create_from_options(opts)


def run_sequence(
    seq_dir: Path, view: str, model_path: Path = DEFAULT_MODEL, min_conf: float = 0.3
) -> Path:
    import mediapipe as mp

    seq_dir = Path(seq_dir)
    meta = json.loads((seq_dir / "meta.json").read_text())
    v = meta["views"][view]
    labels = np.load(seq_dir / "labels.npz")
    frames, t = labels["frame_id"], labels["t"]
    n = len(frames)
    lm_px = np.full((n, 2, 21, 3), np.nan, np.float32)
    world = np.full((n, 2, 21, 3), np.nan, np.float32)
    handed = np.full((n, 2), -1, np.int8)
    score = np.zeros((n, 2), np.float32)

    with _make_landmarker(model_path, min_conf) as det:
        last_ms = -1
        for i, f in enumerate(frames):
            bgr = cv2.imread(v["image"].format(frame=int(f)))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ms = max(int(round(t[i] * 1000)), last_ms + 1)
            last_ms = ms
            r = det.detect_for_video(img, ms)
            for k, (hl, hw, hd) in enumerate(
                zip(r.hand_landmarks, r.hand_world_landmarks, r.handedness, strict=True)
            ):
                if k >= 2:
                    break
                lm_px[i, k] = [[p.x * w, p.y * h, p.z * w] for p in hl]
                world[i, k] = [[p.x, p.y, p.z] for p in hw]
                handed[i, k] = _LABEL.get(hd[0].category_name, -1)
                score[i, k] = hd[0].score

    want = _LABEL.get(meta["hand"].capitalize(), -1)
    primary = np.full(n, -1, np.int8)
    for i in range(n):
        present = np.flatnonzero(handed[i] >= 0)
        if len(present) == 0:
            continue
        match = [k for k in present if handed[i, k] == want]
        pool = match if match else list(present)
        primary[i] = max(pool, key=lambda k: score[i, k])

    primary_gt = np.full(n, -1, np.int8)
    gt_err = np.full(n, np.nan, np.float32)
    gt_key = f"gt2d_{view}"
    if gt_key in labels.files:
        gt = labels[gt_key]
        for i in range(n):
            if not np.isfinite(gt[i]).all():
                continue
            errs = [
                np.linalg.norm(lm_px[i, k, :, :2] - gt[i], axis=1).mean()
                if handed[i, k] >= 0
                else np.inf
                for k in range(2)
            ]
            k = int(np.argmin(errs))
            if errs[k] <= GT_MATCH_PX:
                primary_gt[i], gt_err[i] = k, errs[k]

    out = seq_dir / f"landmarks_{view}.npz"
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        lm_px=lm_px,
        world_m=world,
        handed=handed,
        score=score,
        primary=primary,
        primary_gt=primary_gt,
        gt_err_px=gt_err,
        min_conf=np.float32(min_conf),
    )
    tmp.replace(out)
    return out
