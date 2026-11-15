"""Run a touch checkpoint over a video exactly like scripts/live_touch.py (MediaPipe VIDEO mode ->
LiveTouch prep / model / postprocess) and save what it saw, to debug live failures:

    python scripts/touch_probe.py --ckpt runs/<run>/best.pt --video clip.mp4 [--start 0]
writes <out>/<clip>_<run>.npz (t, prob [N, 5] raw, p [N, 5] smoothed, touch [N, 5], lm [N, 21, 3],
tip_px [N, 5] = tip-DIP length in pixels) and <out>/<clip>_<run>_patches.jpg (model inputs at the
frames with the highest index-finger probability and at random hand frames).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from live_touch import LiveTouch  # noqa: E402

from touch.data.landmarks import DEFAULT_MODEL  # noqa: E402
from touch.data.pressure import FINGER_TIPS  # noqa: E402

FINGER_DIPS = (3, 7, 11, 15, 19)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--start", type=float, default=0.0, help="s")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = to the end")
    ap.add_argument("--flip", default="", choices=["", "h", "v", "hv"])
    ap.add_argument("--out", type=Path, default=Path("results/touch_probe"))
    args = ap.parse_args()
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(DEFAULT_MODEL),
                                     delegate=mpt.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_hands=1, min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3, min_tracking_confidence=0.3))  # fmt: skip
    live = LiveTouch(args.ckpt, threads=4)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    n_max = int(args.seconds * fps) if args.seconds else 10**9
    T, PR, P, TO, LM, TIP, PATCH = [], [], [], [], [], [], []
    i = 0
    while i < n_max:
        ok, frame = cap.read()
        if not ok:
            break
        if args.flip:
            fh, fv = "h" in args.flip, "v" in args.flip
            frame = cv2.flip(frame, -1 if fh and fv else 1 if fh else 0)
        h, w = frame.shape[:2]
        t = args.start + i / fps
        r = det.detect_for_video(
            Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            int(t * 1000),
        )
        lm = None
        if r.hand_landmarks:
            lm = np.array([[q.x * w, q.y * h, q.z * w] for q in r.hand_landmarks[0]], np.float32)
        inputs = live.prep(frame, lm)
        prob = live.model(inputs)
        p = live.postprocess(prob)
        T.append(t)
        PR.append(prob)
        P.append(p)
        TO.append(live.touch.copy())
        LM.append(lm if lm is not None else np.full((21, 3), np.nan, np.float32))
        TIP.append([np.hypot(*(lm[a, :2] - lm[b, :2])) if lm is not None else np.nan
                    for a, b in zip(FINGER_TIPS, FINGER_DIPS, strict=True)])  # fmt: skip
        PATCH.append((live.held * 255).astype(np.uint8) if lm is not None else None)
        i += 1
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.video.stem}_{args.ckpt.parent.name}"
    np.savez_compressed(args.out / f"{stem}.npz", t=np.array(T), prob=np.array(PR),
                        p=np.array(P), touch=np.array(TO), lm=np.array(LM), tip_px=np.array(TIP),
                        on=live.post["on"], off=live.post["off"])  # fmt: skip
    pres = np.isfinite(np.array(LM)[:, 0, 0])
    PRa = np.array(PR)
    idx = list(np.argsort(np.where(pres, PRa[:, 1], -1))[::-1][:4])  # highest index prob
    rng = np.random.default_rng(0)
    idx += list(rng.choice(np.flatnonzero(pres), min(4, pres.sum()), replace=False))
    rows = []
    for k in idx:
        row = np.hstack([cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in PATCH[k]])
        cv2.putText(row, f"t {T[k]:.1f}s  p " + " ".join(f"{v:.2f}" for v in PRa[k]), (4, 14), 0,
                    0.45, (0, 255, 255), 1)  # fmt: skip
        rows.append(row)
    if rows:
        cv2.imwrite(str(args.out / f"{stem}_patches.jpg"), np.vstack(rows))
    tip = np.array(TIP)
    pa = PRa[pres]
    print(f"{len(T)} frames, hand in {pres.mean():.0%}; index tip-DIP median "
          f"{np.nanmedian(tip[:, 1]):.1f} px; raw prob per finger: median "
          f"{np.round(np.median(pa, 0), 3)}, p99 {np.round(np.percentile(pa, 99, 0), 3)}, "
          f"max {np.round(pa.max(0), 3)}; touch frames {np.array(TO)[pres].mean(0).round(3)} "
          f"(on >= {live.post['on']:.2f})")  # fmt: skip


if __name__ == "__main__":
    main()
