"""Offline touch annotation of a recorded video -> per-finger touch events + gestures (JSONL).

For recorded video (e.g. robot-learning data), not live use. Pipeline: MediaPipe (one hand, VIDEO
mode) -> 5 fingertip patches per frame (JPEG round trip, as in training) -> TouchNet -> the
checkpoint's validation-tuned median + hysteresis -> touch.interact events and gestures.

Works with any checkpoint. With an offline checkpoint (temporal: bigru) each frame is also scored
from frames after it (non-causal, see touch.models.touchnet), which only recorded video allows;
with a causal checkpoint the scores equal the live demo's.

Outputs (next to --out):
    <out>.jsonl  one "meta" line, then touch events (down / move / up) and gestures in time order
    <out>.npz    per frame: prob [N, 5], touch [N, 5], landmarks [N, 21, 3] (NaN = no hand), t [N]
    --out-video  optional MP4 with fingertips (green = touching) and gestures drawn

Usage:
    python scripts/annotate_video.py clip.mp4 --ckpt runs/<run>/best.pt
    python scripts/annotate_video.py clip.mp4 --ckpt runs/<run>/best.pt --out-video clip_touch.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from touch.data.landmarks import DEFAULT_MODEL  # noqa: E402
from touch.data.patch_store import represent  # noqa: E402
from touch.data.patches import PATCH_SIZE, finger_patches, landmark_features  # noqa: E402
from touch.data.pressure import FINGER_TIPS  # noqa: E402
from touch.eval.run import binarize, smooth, window_logits  # noqa: E402
from touch.interact.gestures import GestureRecognizer  # noqa: E402
from touch.interact.touches import TouchTracker, hand_points  # noqa: E402
from touch.models.streaming import pad_features  # noqa: E402
from touch.train import build_model, load_config, pick_device  # noqa: E402

JPEG_QUALITY = 92  # as scripts/extract_patches.py


def detect(video: Path, max_frames: int) -> tuple[np.ndarray, list, float, tuple[int, int]]:
    """-> landmarks [N, 21, 3] px (z = rel. depth * width, NaN = no hand), per-frame JPEG patches
    (list of 5 encoded patches or None), fps, (w, h)."""
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(
                model_asset_path=str(DEFAULT_MODEL), delegate=mpt.BaseOptions.Delegate.CPU
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    )  # same settings as scripts/live_touch.py
    lms, patches, last_ms, size = [], [], -1, (0, 0)
    while not max_frames or len(lms) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        size = (w, h)
        ms = max(round(len(lms) * 1000 / fps), last_ms + 1)
        last_ms = ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = det.detect_for_video(Image(image_format=ImageFormat.SRGB, data=rgb), ms)
        if r.hand_landmarks:
            lm = np.array([[q.x * w, q.y * h, q.z * w] for q in r.hand_landmarks[0]], np.float32)
            P = finger_patches(frame, lm[:, :2])
            enc = [cv2.imencode(".jpg", p, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1] for p in P]
            lms.append(lm)
            patches.append(enc)
        else:
            lms.append(np.full((21, 3), np.nan, np.float32))
            patches.append(None)
        if len(lms) % 300 == 0:
            print(f"  mediapipe {len(lms)} frames", flush=True)
    cap.release()
    return np.stack(lms), patches, fps, size


@torch.no_grad()
def score(model, cfg, lms: np.ndarray, patches: list, device, batch: int = 64) -> np.ndarray:
    """Touch probabilities [N, 5]; 0 where there is no hand (as in live / evaluation)."""
    n, T, rep = len(lms), cfg["T"], cfg["input_repr"]
    lm_feat, present = landmark_features(lms, cfg["lm_feats"])  # causal, as at extraction
    fingers = torch.arange(5, device=device)
    held = np.zeros((5, PATCH_SIZE, PATCH_SIZE, 3), np.float32)  # no hand: hold the last patch
    prev = held
    feats = []
    for a in range(0, n, batch):
        imgs = []
        for i in range(a, min(n, a + batch)):
            if patches[i] is not None:
                held = np.stack([cv2.imdecode(e, 1) for e in patches[i]])[..., ::-1] / 255.0
                held = held.astype(np.float32)
            imgs.append(represent(held, prev, rep))
            prev = held
        b = len(imgs)
        img = torch.from_numpy(np.stack(imgs)).float().to(device).flatten(0, 1)  # [b*5, C, S, S]
        lm = torch.from_numpy(lm_feat[a : a + b]).to(device).repeat_interleave(5, 0)
        pres = torch.from_numpy(present[a : a + b]).to(device).repeat_interleave(5, 0)
        f = fingers.repeat(b)
        out = model.frame_features(img[:, None], lm[:, None], pres[:, None], f)[:, 0]
        feats.append(out.view(b, 5, -1))
    feats = torch.cat(feats)  # [N, 5, d]
    pad = pad_features(model.cpu(), rep, torch.arange(5)).to(device)
    model.to(device)
    probs = np.zeros((n, 5), np.float32)
    for f in range(5):
        logits = window_logits(model, feats[:, f], pad[f], T)
        probs[:, f] = torch.sigmoid(logits).float().cpu().numpy()
    probs[~present] = 0.0
    return probs


def write_video(video: Path, out: Path, lms, prob, touch, gestures, fps, size) -> None:
    cap = cv2.VideoCapture(str(video))
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(len(lms)):
        ok, frame = cap.read()
        if not ok:
            break
        if np.isfinite(lms[i]).all():
            r = max(6, size[0] // 120)
            for f, t in enumerate(FINGER_TIPS):
                c = (int(lms[i][t, 0]), int(lms[i][t, 1]))
                if touch[i, f]:
                    cv2.circle(frame, c, r, (0, 220, 0), -1)
                else:
                    cv2.circle(frame, c, r, (0, int(255 * prob[i, f]), 255), 2)
        now = i / fps
        recent = [g for g in gestures if 0 <= now - g.t < 1.5][-4:]
        for k, g in enumerate(reversed(recent)):
            text = " ".join([g.kind, *(str(g.info[x]) for x in ("direction",) if x in g.info)])
            cv2.putText(
                frame, text, (20, 60 + 45 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 3
            )
        wr.write(frame)
    cap.release()
    wr.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="output stem (default: next to video)")
    ap.add_argument("--out-video", type=Path, default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    out = args.out or args.video.with_name(args.video.stem + "_touch")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = load_config(ck["config"])
    assert "post" in ck, "checkpoint has no validation-tuned post-processing"
    device = pick_device()
    model = build_model({**cfg, "pretrained": False}).eval()
    model.load_state_dict(ck["model"])
    model.to(device)

    print(f"{args.video}: hand landmarks + patches", flush=True)
    lms, patches, fps, size = detect(args.video, args.max_frames)
    n = len(lms)
    print(f"{n} frames at {fps:.1f} fps, hand in {np.mean([p is not None for p in patches]):.0%}")
    prob = score(model, cfg, lms, patches, device)
    p_s = smooth(prob, ck["post"]["k"])
    touch = binarize(prob, ck["post"])

    tracker, rec = TouchTracker(), GestureRecognizer()
    t = np.arange(n) / fps
    lines, gestures = [], []
    for i in range(n):
        tips, hand_size = hand_points(lms[i] if np.isfinite(lms[i]).all() else None)
        evs = tracker.update(float(t[i]), touch[i], tips, hand_size, p_s[i])
        gs = rec.update(evs)
        gestures += gs
        lines += [x.to_dict() for x in [*evs, *gs]]
    meta = {
        "type": "meta",
        "video": str(args.video),
        "ckpt": str(args.ckpt),
        "temporal": cfg["temporal"],
        "causal": model.causal,
        "fps": fps,
        "n_frames": n,
        "size": list(size),
        "post": ck["post"],
        "coords": "image px; distances for gestures in hand sizes (no surface calibration)",
    }
    jsonl = out.with_suffix(".jsonl")
    with open(jsonl, "w") as fh:
        for x in [meta, *lines]:
            fh.write(json.dumps(x) + "\n")
    np.savez_compressed(out.with_suffix(".npz"), prob=prob, touch=touch, landmarks=lms, t=t)
    kinds = [g.kind for g in gestures]
    n_down = sum(x["type"] == "touch" and x["phase"] == "down" for x in lines)
    print(f"{n_down} touches, gestures: { {k: kinds.count(k) for k in sorted(set(kinds))} }")
    print("wrote", jsonl, "and", out.with_suffix(".npz"))
    if args.out_video:
        write_video(args.video, args.out_video, lms, prob, touch, gestures, fps, size)
        print("wrote", args.out_video)


if __name__ == "__main__":
    main()
