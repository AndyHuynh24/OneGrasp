"""Live fingertip-touch demo and real-time check: camera -> MediaPipe -> patches -> ONNX TouchNet.

Per frame: one hand (MediaPipe Hand Landmarker, VIDEO mode), 5 fingertip patches, one
StreamingStep for all 5 fingers (ONNX Runtime, CPU), then the checkpoint's validation-tuned
causal median + hysteresis. Live scores equal touch.eval.run scores (see touch.models.streaming).
Touch states then go through touch.interact: per-finger touch events (down / move / up) and
gestures (tap, double tap, long press, drag, swipe, pinch), shown on screen and optionally logged
as JSON lines (--log). Times every stage and writes results/live_latency_<run>.json when it stops.

Usage:
    python scripts/live_touch.py --ckpt runs/<run>/best.pt                  # webcam 0, window
    python scripts/live_touch.py --ckpt runs/<run>/best.pt --log events.jsonl
    python scripts/live_touch.py --ckpt runs/<run>/best.pt --source clip.mp4 --no-show
Keys: q / Esc stop, h = flip left-right, v = flip upside down (or start with --flip h|v|hv).

Pipeline (default): a worker thread does capture + MediaPipe for frame t while a second thread
runs patches + model for frame t-1, so a frame costs max(stages), not their sum.
Every frame reaches the model in order (identical causal state to --serial). The window runs
on the main thread as a third stage (cv2.waitKey costs ~13 ms per call on macOS); if it falls
behind it skips frames, the model never does.
The model was trained at 30 fps: the fps text turns red below 28.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from touch.data.landmarks import DEFAULT_MODEL  # noqa: E402
from touch.data.patch_store import represent  # noqa: E402
from touch.data.patches import (  # noqa: E402
    LM_HISTORY,
    PATCH_SIZE,
    clamp_tip_segments,
    finger_patches,
    landmark_features,
)
from touch.data.pressure import FINGER_TIPS  # noqa: E402
from touch.interact.gestures import GestureRecognizer  # noqa: E402
from touch.interact.touches import TouchTracker, hand_points  # noqa: E402
from touch.models.streaming import INPUTS, export_onnx, pad_features  # noqa: E402
from touch.train import build_model, load_config  # noqa: E402

JPEG_QUALITY = 92  # training patches were stored as JPEG at this quality (extract_patches.py)
STAGES = ["capture", "mediapipe", "prep", "model", "post_draw"]


class LiveTouch:
    """Causal per-frame touch state for one hand (5 fingers)."""

    def __init__(self, ckpt: Path, threads: int, jpeg: bool = True, tip_clamp: bool = True):
        self.tip_clamp = tip_clamp
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = load_config(ck["config"])
        model = build_model({**cfg, "pretrained": False}).eval()
        model.load_state_dict(ck["model"])
        self.T, self.repr, self.post, self.jpeg = cfg["T"], cfg["input_repr"], ck["post"], jpeg
        self.lm_feats = cfg["lm_feats"]
        onnx_path = ckpt.parent / f"stream_{ckpt.stem}.onnx"
        export_onnx(model, self.repr, self.T, onnx_path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.fingers = np.arange(5, dtype=np.int64)
        pad = pad_features(model, self.repr, torch.from_numpy(self.fingers)).numpy()
        self.hist = np.repeat(pad[:, None], self.T - 1, axis=1).astype(np.float32)
        self.held = np.zeros((5, PATCH_SIZE, PATCH_SIZE, 3), np.float32)  # last patch, RGB [0, 1]
        self.prev_x = self.held.copy()
        nan = np.full((21, 3), np.nan, np.float32)
        self.lm_hist = deque([nan] * LM_HISTORY[self.lm_feats], maxlen=LM_HISTORY[self.lm_feats])
        self.recent = deque(maxlen=max(1, self.post["k"]))
        self.touch = np.zeros(5, bool)

    def prep(self, frame_bgr: np.ndarray, lm: np.ndarray | None) -> dict:
        """Network inputs for this frame; lm [21, 3] pixels (z = rel. depth * width) or None."""
        cur = lm if lm is not None else np.full((21, 3), np.nan, np.float32)
        self.lm_hist.append(cur)
        feats, _ = landmark_features(np.stack(self.lm_hist), self.lm_feats)
        present = lm is not None
        if present:
            # straight flat fingers look up to 2x longer than any training touch: cap the tip
            # segment so the patch zoom stays in the training range (patches.TIP_RATIO_P90)
            xy = clamp_tip_segments(lm[:, :2]) if self.tip_clamp else lm[:, :2]
            P = finger_patches(frame_bgr, xy)
            if self.jpeg:
                P = np.stack(
                    [
                        cv2.imdecode(
                            cv2.imencode(".jpg", p, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1], 1
                        )
                        for p in P
                    ]
                )
            self.held = P[..., ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        x = self.held  # no hand: hold the last patch with present = 0 (as in training/eval)
        img = represent(x, self.prev_x, self.repr)
        self.prev_x = x
        return {
            "img": np.ascontiguousarray(img, np.float32),
            "lm": np.repeat(feats[-1:], 5, axis=0).astype(np.float32),
            "present": np.full(5, present),
            "finger": self.fingers,
            "hist": self.hist,
        }

    def model(self, inputs: dict) -> np.ndarray:
        prob, feat = self.sess.run(None, {k: inputs[k] for k in INPUTS})
        self.hist = np.concatenate([self.hist[:, 1:], feat[:, None]], axis=1)
        return np.where(inputs["present"], prob, 0.0)  # no hand -> 0, as in evaluation

    def postprocess(self, prob: np.ndarray) -> np.ndarray:
        """Causal median over the last k frames, then hysteresis (on / off thresholds)."""
        self.recent.append(prob)
        p = np.median(np.stack(self.recent), axis=0)
        self.touch = np.where(self.touch, p >= self.post["off"], p >= self.post["on"])
        return p


def draw(frame, lm, p, touch, fps: float, gestures=(), on: float = 0.5, flipped="") -> None:
    if lm is not None:
        for f, t in enumerate(FINGER_TIPS):
            c = (int(lm[t, 0]), int(lm[t, 1]))
            r = max(6, frame.shape[1] // 120)
            if touch[f]:
                cv2.circle(frame, c, r, (0, 220, 0), -1)
            else:
                cv2.circle(frame, c, r, (0, int(255 * p[f]), 255), 2)
    col = (255, 255, 255) if fps >= 28 else (0, 0, 255)  # trained at 30 fps
    cv2.putText(frame, f"{fps:5.1f} fps", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
    hw = frame.shape[1]
    txt = "p " + " ".join(f"{n}{x:.2f}" for n, x in zip("TIMRP", p, strict=True))
    txt += f"  (touch >= {on:.2f})" + (f"  flip {flipped}" if flipped else "")
    cv2.putText(frame, txt, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * hw / 640, (255, 255, 255), 2)  # fmt: skip
    for k, g in enumerate(reversed(gestures)):  # newest first
        text = " ".join([g.kind, *(str(g.info[x]) for x in ("direction",) if x in g.info)])
        y = 100 + 45 * k
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 3)


def threaded(gen, maxsize: int = 2):
    """Run a generator in a worker thread; yields its items in order."""
    q: queue.Queue = queue.Queue(maxsize)

    def work():
        try:
            for item in gen:
                q.put(item)
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30, help="requested camera frame rate")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until q)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--no-jpeg", action="store_true", help="skip the JPEG round trip on patches")
    ap.add_argument("--log", type=Path, default=None, help="write touch events + gestures (JSONL)")
    ap.add_argument("--serial", action="store_true", help="no pipelining (old behaviour)")
    ap.add_argument(
        "--no-tip-clamp",
        action="store_true",
        help="old patch zoom (no cap on the fingertip segment length)",
    )
    ap.add_argument(
        "--flip",
        default="",
        choices=["", "h", "v", "hv"],
        help="start flipped: h left-right, v upside down (toggle live: h / v keys)",
    )
    ap.add_argument("--display-width", type=int, default=0,
                    help="window width (0 = camera size; resizing costs ~4 ms)")  # fmt: skip
    args = ap.parse_args()

    from mediapipe import Image, ImageFormat
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if cam else args.source)
    if cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        sys.exit(f"cannot open source {args.source}")
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
    )
    live = LiveTouch(
        args.ckpt, args.threads, jpeg=not args.no_jpeg, tip_clamp=not args.no_tip_clamp
    )
    tracker, recognizer = TouchTracker(), GestureRecognizer()
    shown = deque(maxlen=4)  # recent gestures on screen
    log = open(args.log, "w") if args.log else None
    times = {s: [] for s in STAGES}
    n_hand, t_start, fps = 0, time.perf_counter(), 0.0
    frame_t = deque(maxlen=30)
    stop = threading.Event()
    flip = {"h": "h" in args.flip, "v": "v" in args.flip}  # toggled by the window keys

    def capture_and_detect():
        """Yields (frame, lm or None, t1 wall time, capture ms, mediapipe ms, video seconds)."""
        last_ms = -1
        while not stop.is_set():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                return
            t1 = time.perf_counter()
            vid_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            fh, fv = flip["h"], flip["v"]  # flip BEFORE MediaPipe: landmarks + patches agree
            if fh or fv:
                frame = cv2.flip(frame, -1 if fh and fv else 1 if fh else 0)
            h, w = frame.shape[:2]
            ms = max(int(t1 * 1000), last_ms + 1)
            last_ms = ms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = det.detect_for_video(Image(image_format=ImageFormat.SRGB, data=rgb), ms)
            lm = None
            if r.hand_landmarks:
                lm = np.array([[q.x * w, q.y * h, q.z * w] for q in r.hand_landmarks[0]],
                              np.float32)  # fmt: skip
            yield frame, lm, t1, (t1 - t0) * 1000, (time.perf_counter() - t1) * 1000, vid_t

    title = "OneTouch live (q to quit)"
    w = h = 0

    def show_now(frame) -> None:  # main thread only (macOS GUI)
        cv2.imshow(title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            stop.set()
        elif key in (ord("h"), ord("v")):
            flip[chr(key)] = not flip[chr(key)]

    display: queue.Queue = queue.Queue(maxsize=1)

    def show_later(frame) -> None:  # model thread: newest frame for the window, never blocks
        try:
            display.get_nowait()  # the window fell behind: drop its old frame (model unaffected)
        except queue.Empty:
            pass
        display.put_nowait(frame)

    def model_loop(show) -> None:
        nonlocal n_hand, fps, w, h
        items = capture_and_detect() if args.serial else threaded(capture_and_detect())
        for frame, lm, t1, cap_ms, mp_ms, vid_t in items:
            if stop.is_set():
                break
            h, w = frame.shape[:2]
            n_hand += lm is not None
            t2 = time.perf_counter()
            inputs = live.prep(frame, lm)
            t3 = time.perf_counter()
            prob = live.model(inputs)
            t4 = time.perf_counter()
            p = live.postprocess(prob)
            # video time for files, wall time for cameras
            t_now = t1 - t_start if cam else vid_t
            tips, hand_size = hand_points(lm)
            evs = tracker.update(t_now, live.touch, tips, hand_size, p)
            gs = recognizer.update(evs)
            shown.extend(gs)
            if log:
                for x in [*evs, *gs]:
                    log.write(json.dumps(x.to_dict()) + "\n")
            frame_t.append(t4)
            if len(frame_t) > 1:
                fps = (len(frame_t) - 1) / (frame_t[-1] - frame_t[0])
            if not args.no_show:
                fl = ("h" if flip["h"] else "") + ("v" if flip["v"] else "")
                draw(frame, lm, p, live.touch, fps, [g for g in shown if t_now - g.t < 1.5],
                     live.post["on"], fl)  # fmt: skip
                if args.display_width and w > args.display_width:
                    dh = round(h * args.display_width / w)
                    frame = cv2.resize(frame, (args.display_width, dh),
                                       interpolation=cv2.INTER_AREA)  # fmt: skip
                show(frame)
            t5 = time.perf_counter()
            stage_ms = [cap_ms, mp_ms, (t3 - t2) * 1000, (t4 - t3) * 1000, (t5 - t4) * 1000]
            for s, dt in zip(STAGES, stage_ms, strict=True):
                times[s].append(dt)
            if args.seconds and t5 - t_start > args.seconds:
                break
        stop.set()

    if args.serial:
        model_loop(show_now)
    else:  # capture+MediaPipe thread -> model thread -> window on the main thread
        worker = threading.Thread(target=model_loop, args=(show_later,), daemon=True)
        worker.start()
        while worker.is_alive():
            try:
                frame = display.get(timeout=0.1)
            except queue.Empty:
                continue
            if not args.no_show:
                show_now(frame)
        worker.join()
    elapsed = time.perf_counter() - t_start
    stop.set()
    cap.release()
    cv2.destroyAllWindows()
    if log:
        log.close()
        print("wrote", args.log)

    n = len(times["capture"])
    if n < 10:
        sys.exit(f"only {n} frames processed")
    skip = min(30, n // 5)  # warm-up frames
    a = {s: np.array(v[skip:]) for s, v in times.items()}
    proc = sum(a[s] for s in STAGES[1:])
    res = {
        "ckpt": str(args.ckpt),
        "source": args.source,
        "resolution": [int(w), int(h)],
        "frames": n,
        "hand_frac": round(n_hand / n, 3),
        "threads": args.threads,
        "stage_ms_median": {s: round(float(np.median(v)), 2) for s, v in a.items()},
        "stage_ms_p95": {s: round(float(np.percentile(v, 95)), 2) for s, v in a.items()},
        "processing_ms_median": round(float(np.median(proc)), 2),
        "processing_ms_p95": round(float(np.percentile(proc, 95)), 2),
        "processing_fps": round(1000 / float(np.median(proc)), 1),
        "sustained_fps": round(n / elapsed, 1),
        "pipelined": not args.serial,
    }
    tag = "" if cam else f"_{Path(args.source).stem}"  # video runs don't overwrite camera runs
    out = Path("results") / f"live_latency_{args.ckpt.parent.name}{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
