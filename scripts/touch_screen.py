"""Make a surface (paper, table area, a display) touchable: camera -> touches in surface px.

1. Calibrate once: ArUco markers 0-3 at the surface corners (TL, TR, BR, BL; DICT_4X4_50, the
   HandFlow layout) are found automatically, or click the 4 corners in the camera window (--click).
   Saved to --calib and reused next time.
2. Live: MediaPipe -> TouchSession (fingertip patches + causal TouchNet) -> surface coordinates ->
   HandFlow-style states (touch_hover / touch / touch_hold) for the index finger.
   A second window draws the surface: cursor = index fingertip, green = touching, taps as rings.
   --mouse drives the OS cursor (pyautogui: down / drag / up), like a touchscreen.

Usage:
    python scripts/touch_screen.py --ckpt runs/<run>/best.pt                  # ArUco calibration
    python scripts/touch_screen.py --ckpt runs/<run>/best.pt --click          # click 4 corners
    python scripts/touch_screen.py --ckpt a/best.pt b/best.pt --post results/ensemble_<tag>.json
    python scripts/touch_screen.py --ckpt runs/<run>/best.pt --mouse --size 1920 1080
Press q or Esc to stop, c to recalibrate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from touch.data.landmarks import DEFAULT_MODEL  # noqa: E402
from touch.interact.handflow import HandFlowAdapter  # noqa: E402
from touch.interact.session import TouchSession, landmarks_px, load_post  # noqa: E402
from touch.interact.surface import SurfaceMap, detect_aruco_corners  # noqa: E402

CAM_WIN, SURF_WIN = "camera (q quit, c recalibrate)", "surface"


def click_corners(cap) -> np.ndarray | None:
    pts: list[tuple[int, int]] = []
    names = ["top-left", "top-right", "bottom-right", "bottom-left"]

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(CAM_WIN)
    cv2.setMouseCallback(CAM_WIN, on_mouse)
    while len(pts) < 4:
        ok, frame = cap.read()
        if not ok:
            return None
        for p in pts:
            cv2.circle(frame, p, 6, (0, 220, 0), -1)
        msg = f"click the surface's {names[len(pts)]} corner"
        cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
        cv2.imshow(CAM_WIN, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return None
    cv2.setMouseCallback(CAM_WIN, lambda *a: None)
    return np.array(pts, np.float32)


def aruco_corners(cap, timeout_s: float = 20.0) -> np.ndarray | None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ok, frame = cap.read()
        if not ok:
            return None
        c = detect_aruco_corners(frame)
        if c is not None:
            return c
        msg = "looking for ArUco markers 0-3 at the corners (or rerun with --click)"
        cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        cv2.imshow(CAM_WIN, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return None
    return None


def calibrate(cap, args) -> SurfaceMap | None:
    c = click_corners(cap) if args.click else aruco_corners(cap)
    if c is None:
        return None
    s = SurfaceMap.from_corners(c, tuple(args.size))
    s.save(args.calib)
    print("saved calibration to", args.calib)
    return s


def draw_surface(size, hf, res, taps, view_w: int = 960) -> np.ndarray:
    w, h = size
    scale = view_w / w
    img = np.full((int(h * scale), view_w, 3), 40, np.uint8)
    for t_tap, xy in taps:
        r = int(10 + 60 * (time.time() - t_tap))
        cv2.circle(img, (int(xy[0] * scale), int(xy[1] * scale)), r, (0, 220, 0), 2)
    if res.tips_surface is not None and res.present:
        for f in range(5):
            u, v = res.tips_surface[f] * scale
            if not np.isfinite([u, v]).all():
                continue
            col = (0, 220, 0) if res.touch[f] else (0, int(255 * res.prob[f]), 255)
            cv2.circle(img, (int(u), int(v)), 12 if f == 1 else 6, col, -1 if res.touch[f] else 2)
    label = f"{hf.gesture or 'no hand'}  p={hf.prob:.2f}"
    cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


class Mouse:
    """touch_hover -> move, touch_hold -> drag, touch -> click (pyautogui, optional)."""

    def __init__(self, size):
        import pyautogui  # optional dependency, only for --mouse

        pyautogui.FAILSAFE = False
        self.pg, self.size, self.dragging = pyautogui, size, False
        self.screen = pyautogui.size()

    def update(self, hf) -> None:
        if hf.xy is None:
            if self.dragging:
                self.pg.mouseUp()
                self.dragging = False
            return
        x = hf.xy[0] / self.size[0] * self.screen[0]
        y = hf.xy[1] / self.size[1] * self.screen[1]
        if hf.gesture == "touch":
            self.pg.click(x, y)
        elif hf.gesture == "touch_hold":
            self.pg.moveTo(x, y)
            if not self.dragging:
                self.pg.mouseDown()
                self.dragging = True
        else:
            if self.dragging:
                self.pg.mouseUp()
                self.dragging = False
            self.pg.moveTo(x, y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--post", type=Path, default=None, help="ensemble post (ensemble_eval json)")
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--size", type=float, nargs=2, default=[1920, 1080], help="surface w h")
    ap.add_argument("--calib", type=Path, default=Path("results/surface_calib.json"))
    ap.add_argument("--click", action="store_true", help="calibrate by clicking 4 corners")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--mouse", action="store_true", help="drive the OS cursor (pyautogui)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--log", type=Path, default=None, help="write surface touch events (JSONL)")
    args = ap.parse_args()

    from mediapipe import Image, ImageFormat
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if cam else args.source)
    if cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"cannot open source {args.source}")
    args.calib.parent.mkdir(parents=True, exist_ok=True)
    if args.calib.exists() and not (args.recalibrate or args.click):
        surface = SurfaceMap.load(args.calib)
        print("loaded calibration", args.calib)
    else:
        surface = calibrate(cap, args)
    if surface is None:
        sys.exit("no calibration")
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
    post = load_post(args.post) if args.post else None
    session = TouchSession(args.ckpt, surface=surface, threads=args.threads, post=post)
    adapter, mouse = HandFlowAdapter(), (Mouse(surface.size) if args.mouse else None)
    log = open(args.log, "w") if args.log else None
    taps, last_ms, t_start = [], -1, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter() - t_start
        h, w = frame.shape[:2]
        ms = max(int(t * 1000), last_ms + 1)
        last_ms = ms
        r = det.detect_for_video(
            Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ms
        )
        lm = landmarks_px(r.hand_landmarks[0], w, h) if r.hand_landmarks else None
        res = session.step(frame, lm, t)
        hf = adapter.update(res)
        if hf.gesture == "touch":
            taps.append((time.time(), hf.xy))
        taps = [x for x in taps if time.time() - x[0] < 0.8]
        if mouse:
            mouse.update(hf)
        if log:
            for p in res.pointer:
                log.write(json.dumps(p) + "\n")
            if hf.gesture == "touch":
                log.write(json.dumps({"type": "click", "t": round(t, 4), "xy": hf.xy}) + "\n")
        quad = np.round(cv2.perspectiveTransform(
            np.array([[[0, 0], [surface.size[0], 0], list(surface.size), [0, surface.size[1]]]],
                     np.float64), np.linalg.inv(surface.H))[0]).astype(np.int32)  # fmt: skip
        cv2.polylines(frame, [quad], True, (255, 200, 0), 2)
        if lm is not None:
            for f in range(5):
                c = tuple(int(v) for v in res.tips[f])
                cv2.circle(frame, c, 7, (0, 220, 0) if res.touch[f] else (0, 160, 255), -1)
        cv2.imshow(CAM_WIN, frame)
        cv2.imshow(SURF_WIN, draw_surface(surface.size, hf, res, taps))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            surface = calibrate(cap, args) or surface
            session.surface = surface
    cap.release()
    cv2.destroyAllWindows()
    if log:
        log.close()


if __name__ == "__main__":
    main()
