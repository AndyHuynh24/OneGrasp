"""Record our own touch / grasp data, labelled live by holding a key (format and keys:
src/touch/data/own_labels.py), then check and fix the timing in the review mode.

Always records 30 fps (the live demos' rate): a camera running faster is decimated to 30.

Record (click the window first so it gets the keys; one hand performs, the other holds the key):
    python scripts/collect_labels.py record --task touch --cam 1   (left / right arrow: finger)
    python scripts/collect_labels.py record --task grasp --object cup --cam 1
  touch: hold F exactly while the finger touches the surface (tap = press + release), H while
         it presses and stays down (hold / long press).
  grasp: hold F from first contact until the hand leaves the object; hold H from lift-off
         until it is set down (hold = in the hand AND off the table, as in the training data).
  X: mark a mistake (+-0.5 s dropped).   Q / Esc: stop and save.
  touch: left / right arrow = previous / next finger (thumb .. pinky; the labelled fingertip is
         ringed in yellow; not while F is down).   1 / 2: flip the view left-right / upside down.
  The preview draws MediaPipe's hand landmarks (blue left, pink right, white fingertips) and
  shows NO HAND when it sees none: keep the hand where it is tracked (--no-landmarks: off).

Review (step through every label change; align the labels with the video):
    python scripts/collect_labels.py review data/own/touch/<session>
  left / right: previous / next label change      , / . : all labels 1 frame earlier / later
  [ / ] : this change 1 frame earlier / later        D : drop / undrop this press
  S : save        Q / Esc : save and quit
The strip shows frames around the change: border = label of each frame (grey none, orange touch,
green hold, red invalid); the centre frame is the first frame of the new label.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch.data.own_labels import (  # noqa: E402
    FINGERS,
    LABEL_NAMES,
    TASKS,
    intervals,
    labels_from_meta,
)

# MediaPipe hand: 21 landmarks; bones as (parent, child)
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
         (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19),
         (19, 20)]  # fmt: skip
HAND_COLORS = {"Left": (80, 200, 255), "Right": (255, 90, 200)}  # RGB, MediaPipe handedness

FPS = 30.0  # recordings are always 30 fps, the live demos' rate (faster cameras are decimated)
COLORS = {0: (150, 150, 150), 1: (255, 160, 0), 2: (0, 200, 80)}  # RGB per label
BAD = (230, 40, 40)


# ------------------------------------------------------------------ video writing
def hw_encoder() -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True,
                             text=True, timeout=10).stdout  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "h264_videotoolbox" in out


class FfmpegWriter:
    """BGR frames -> H.264 mp4 through an ffmpeg pipe, in a background thread. Default: the Mac's
    hardware encoder (VideoToolbox, 20 Mbit/s): same quality as libx264 CRF 18 on a 1080p hand
    clip (PSNR 41.1 vs 41.3 dB) at ~1/8 of the CPU (31% vs 251% of a core), which keeps the CPU
    free for capture. Falls back to libx264 when VideoToolbox is missing."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float, hw: bool = True):
        import queue

        enc = (["-c:v", "h264_videotoolbox", "-b:v", "20M", "-realtime", "1"] if hw
               else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"])  # fmt: skip
        self.encoder = enc[1]
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:.3f}", "-i", "-", *enc,
               "-pix_fmt", "yuv420p", str(path)]  # fmt: skip
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.q: queue.Queue = queue.Queue(maxsize=60)  # ~2 s of 1080p (370 MB)
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while (f := self.q.get()) is not None:
            self.p.stdin.write(f.tobytes())

    def write(self, frame) -> bool:
        """False if the encoder fell 2 s behind (this frame then waited: capture stalled)."""
        full = self.q.full()
        self.q.put(frame)
        return not full

    def close(self):
        self.q.put(None)
        self.th.join()
        self.p.stdin.close()
        self.p.wait()


def keep_30fps(cam, t: float) -> bool:
    """30 Hz time grid: keep a frame if it is not more than 0.4 frame early for the next slot.
    A 60 fps camera keeps every 2nd frame; a 30 fps one keeps all, also a frame that arrives
    early right after a late one (the next slot moves with late frames)."""
    period = 1.0 / cam.fps
    slot = getattr(cam, "next_slot", None)
    if slot is not None and t < slot - 0.4 * period:
        return False
    cam.next_slot = max((slot if slot is not None else t) + period, t + 0.4 * period)
    return True


class Camera(threading.Thread):
    """Reads frames as fast as the camera gives them; stamps each when read and writes it."""

    def __init__(self, source, size, fps, flip, writer_path: Path):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if str(source).isdigit():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        ok, f = self.cap.read()
        if not ok:
            raise SystemExit(f"camera {source} gives no frames")
        self.size = (f.shape[1], f.shape[0])
        self.fps = fps
        self.n_read = self.n_skipped = 0
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or fps  # file source pacing
        self.flip, self.path = flip, writer_path
        self.t: list[float] = []
        self.latest = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.realtime = not str(source).isdigit()  # a video file: pace it at fps
        self.hw = hw_encoder()
        self.encoder, self.encoder_stalls = "", 0

    def run(self):
        w = FfmpegWriter(self.path, self.size, self.fps, hw=self.hw)
        self.encoder = w.encoder
        t_next = time.perf_counter()
        while not self.stop.is_set():
            ok, f = self.cap.read()
            t = time.perf_counter()
            if not ok:
                break
            self.n_read += 1
            if self.realtime:  # a video file: play it at its own rate
                t_next += 1 / self.src_fps
                time.sleep(max(0.0, t_next - time.perf_counter()))
            if not keep_30fps(self, t):  # faster than 30 fps: skip frame
                self.n_skipped += 1
                continue
            if self.flip:
                fh, fv = "h" in self.flip, "v" in self.flip
                f = cv2.flip(f, -1 if fh and fv else 1 if fh else 0)
            if not w.write(f):
                self.encoder_stalls += 1
            with self.lock:
                self.t.append(t)
                self.latest = f
        w.close()
        self.cap.release()


class HandOverlay(threading.Thread):
    """MediaPipe HandLandmarker (VIDEO mode) on the newest camera frame, in its own thread so it
    never slows capture. For the preview only: the recorded video stays clean."""

    def __init__(self, cam: Camera, num_hands: int = 2):
        super().__init__(daemon=True)
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        from touch.data.landmarks import DEFAULT_MODEL

        model = Path(__file__).resolve().parents[1] / DEFAULT_MODEL
        self.det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3))  # fmt: skip
        self.cam = cam
        self.hands: list[tuple[np.ndarray, str, float]] = []  # (xy [21, 2] normalised, side, score)
        self.ms = 0.0  # inference time
        self.n_done = 0
        self.throttled = False
        self.stop = threading.Event()

    def cam_fps(self) -> float:
        with self.cam.lock:
            r = self.cam.t[-31:]
        return (len(r) - 1) / max(r[-1] - r[0], 1e-6) if len(r) > 10 else self.cam.fps

    def run(self):
        from mediapipe import Image, ImageFormat

        last_n, last_ts = -1, -1
        while not self.stop.is_set():
            with self.cam.lock:
                n, frame = len(self.cam.t), self.cam.latest
                t = self.cam.t[-1] if self.cam.t else 0.0
            if frame is None or n == last_n:
                time.sleep(0.003)
                continue
            last_n = n
            ts = max(int(t * 1000), last_ts + 1)  # VIDEO mode: strictly increasing ms
            last_ts = ts
            t0 = time.perf_counter()
            h, w = frame.shape[:2]  # landmarks are normalised: a 640 px copy is enough here
            small = cv2.resize(frame, (640, round(h * 640 / w)), interpolation=cv2.INTER_AREA)
            img = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            r = self.det.detect_for_video(img, ts)
            self.ms = (time.perf_counter() - t0) * 1000
            self.hands = [(np.array([[q.x, q.y] for q in lm], np.float32), h[0].category_name,
                           h[0].score) for lm, h in zip(r.hand_landmarks, r.handedness,
                                                        strict=True)]  # fmt: skip
            self.n_done += 1
            if self.cam_fps() < 0.95 * self.cam.fps:  # capture first: back off to ~10 Hz
                self.throttled = True
                time.sleep(0.1)
            else:
                self.throttled = False
        self.det.close()


def draw_hands(pg, screen, hands, x0, y0, w, h, tip: int | None = None) -> None:
    """tip: landmark index of the labelled fingertip (ringed in yellow)."""
    for xy, side, _ in hands:
        col = HAND_COLORS.get(side, (255, 255, 255))
        pts = [(x0 + float(x) * w, y0 + float(y) * h) for x, y in xy]
        for a, b in BONES:
            pg.draw.line(screen, col, pts[a], pts[b], 3)
        for k, pt in enumerate(pts):
            pg.draw.circle(screen, (255, 255, 255) if k in (4, 8, 12, 16, 20) else col, pt,
                           6 if k in (4, 8, 12, 16, 20) else 4)  # fmt: skip
        if tip is not None:
            pg.draw.circle(screen, (255, 230, 0), pts[tip], 16, 4)


# ------------------------------------------------------------------ helpers
def to_surface(pg, bgr, width):
    h, w = bgr.shape[:2]
    if width and width != w:
        bgr = cv2.resize(bgr, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return pg.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).parent).stdout.strip()  # fmt: skip
    except OSError:
        return ""


def save_labels(sdir: Path) -> dict:
    meta = json.loads((sdir / "meta.json").read_text())
    ev = json.loads((sdir / "events.json").read_text())
    t = np.load(sdir / "frames.npz")["t"]
    lab = labels_from_meta(t, ev, meta)
    np.savez_compressed(sdir / "labels.npz", **lab, t=t)
    return lab


def summary(lab: dict, task: str, fps: float) -> str:
    names = LABEL_NAMES[task]
    v = lab["valid"]
    parts = [f"{n} {(lab['label'][v] == i).sum() / fps:.1f} s" for i, n in enumerate(names)]
    return f"{len(v)} frames, {v.mean():.0%} valid: " + ", ".join(parts)


# ------------------------------------------------------------------ record
def record(args) -> None:
    import pygame as pg

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{args.participant}_{args.task}_{args.object or args.finger}_{stamp}"
    sdir = args.out / args.task / name
    sdir.mkdir(parents=True, exist_ok=True)
    args.fps = FPS
    cam = Camera(args.cam, (args.width, args.height), FPS, args.flip, sdir / "video.mp4")
    keys = {pg.K_f: "f", pg.K_h: "h"}  # F touch, H hold (both tasks)
    names = {"h": "HOLD", "f": "TOUCH"}
    pg.init()
    disp_w = args.display_width or cam.size[0]
    disp_h = round(cam.size[1] * disp_w / cam.size[0])
    screen = pg.display.set_mode((disp_w, disp_h + 40))
    hint = "F = in contact, H = lifted" if args.task == "grasp" else "F = touch, H = hold"
    pg.display.set_caption(f"collect {args.task} - click here, hold {hint}")
    font = pg.font.SysFont("Helvetica", 22, bold=True)
    events, down = [], set()
    finger = args.finger if args.task == "touch" else None
    used = {finger} if finger else set()
    overlay = None
    if not args.no_landmarks:  # load MediaPipe before recording starts (loading stalls capture)
        overlay = HandOverlay(cam)
    cam.start()
    if overlay is not None:
        overlay.start()
    t0 = time.perf_counter()
    last_frame_n, surf, last_down = -1, None, set()
    gap = 1.5 / args.fps  # a read interval longer than this = at least one frame lost
    running = True
    while running and cam.is_alive():
        for e in pg.event.get():  # polled at ~500 Hz: key times within ~2 ms
            t = time.perf_counter()
            if e.type == pg.QUIT:
                running = False
            elif e.type in (pg.KEYDOWN, pg.KEYUP):
                is_down = e.type == pg.KEYDOWN
                if e.key in keys:
                    k = keys[e.key]
                    if is_down != (k in down):
                        events.append({"t": t, "key": k, "down": is_down})
                        (down.add if is_down else down.discard)(k)
                elif is_down and e.key == pg.K_x:
                    events.append({"t": t, "key": "bad", "down": True})
                elif is_down and e.key in (pg.K_q, pg.K_ESCAPE):
                    running = False
                elif is_down and e.key in (pg.K_LEFT, pg.K_RIGHT) and finger and not down:
                    k = FINGERS.index(finger) + (1 if e.key == pg.K_RIGHT else -1)
                    finger = FINGERS[k % len(FINGERS)]
                    used.add(finger)
                    events.append({"t": t, "key": "finger", "down": True, "finger": finger})
                    last_down = None  # redraw
                elif is_down and e.key in (pg.K_1, pg.K_2):
                    c = "h" if e.key == pg.K_1 else "v"
                    cam.flip = cam.flip.replace(c, "") if c in cam.flip else cam.flip + c
        if args.seconds and time.perf_counter() - t0 > args.seconds:
            running = False
        with cam.lock:
            n, frame = len(cam.t), cam.latest
            recent = cam.t[-31:]  # fps over the last ~second of frames
        redraw = down != last_down
        if frame is not None and n != last_frame_n:
            last_frame_n, redraw = n, True
            surf = to_surface(pg, frame, disp_w)
        last_down = set(down)
        if surf is not None and redraw:  # only on a new frame or key change (not 500x / s)
            screen.fill((20, 20, 20))
            screen.blit(surf, (0, 40))
            if overlay is not None:
                tip = 4 + 4 * FINGERS.index(finger) if finger else None
                draw_hands(pg, screen, overlay.hands, 0, 40, disp_w, disp_h, tip)
            lab = "h" if "h" in down else "f" if "f" in down else None
            col = COLORS[2] if lab == "h" else (
                COLORS[1] if lab == "f" else None)  # fmt: skip
            if col:
                pg.draw.rect(screen, col, (0, 40, disp_w, disp_h), 12)
            el = time.perf_counter() - t0
            n_press = sum(1 for e in events if e["down"] and e["key"] != "bad")
            fps = (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6) if len(recent) > 1 else 0.0
            with cam.lock:
                tt = np.asarray(cam.t[1:])
                drops = int((np.diff(cam.t) > gap).sum()) if len(cam.t) > 1 else 0
            low = el > 2 and len(tt) > 30 and fps < 0.97 * args.fps
            fg = f"{finger.upper()} (<- ->)   " if finger else ""
            state = names[lab] if lab else "none"
            txt = (f"REC {el:5.1f} s   {fg}{state:5s}   presses {n_press}   "
                   f"marks {sum(e['key'] == 'bad' for e in events)}   {fps:4.1f} fps   "
                   f"[X mistake, Q stop]")  # fmt: skip
            screen.blit(font.render(txt, True, (255, 60, 60)), (10, 8))
            warn = []
            if low:
                warn.append(f"LOW FPS {fps:.1f}: add light (auto-exposure slows the camera) or "
                            f"set the camera / Camo to {args.fps:.0f} fps")  # fmt: skip
            if drops:
                warn.append(f"frame gaps {drops}")
            if cam.encoder_stalls:
                warn.append(f"encoder behind x{cam.encoder_stalls}")
            if warn:
                ws = font.render("   ".join(warn), True, (255, 255, 255), (200, 0, 0))
                screen.blit(ws, (10, 40 + disp_h - ws.get_height() - 10))
            if overlay is not None:
                nh = len(overlay.hands)
                ht = (f"hands {nh}  ({overlay.ms:.0f} ms{', slowed' if overlay.throttled else ''})"
                      if nh else "NO HAND")  # fmt: skip
                hs = font.render(ht, True, (120, 255, 120) if nh else (255, 220, 0))
                screen.blit(hs, (disp_w - hs.get_width() - 10, 48))
            pg.display.flip()
        time.sleep(0.002)
    t_end = time.perf_counter()
    for k in list(down):  # still held at stop: close the press
        events.append({"t": t_end, "key": k, "down": False})
    if overlay is not None:
        overlay.stop.set()
        overlay.join()
    cam.stop.set()
    cam.join()
    pg.quit()
    t = np.array(cam.t, np.float64)
    np.savez_compressed(sdir / "frames.npz", t=t)
    (sdir / "events.json").write_text(json.dumps(events, indent=0))
    fps_meas = (len(t) - 1) / max(t[-1] - t[0], 1e-6) if len(t) > 1 else 0.0
    d = np.diff(t) if len(t) > 1 else np.zeros(1)
    n_gaps, max_gap = int((d > gap).sum()), float(d.max())
    meta = {"task": args.task, "finger": args.finger if args.task == "touch" else None,
            "fingers_used": [f for f in FINGERS if f in used],
            "object": args.object, "participant": args.participant, "session": name,
            "camera": str(args.cam), "size": list(cam.size), "fps_requested": args.fps,
            "camera_frames_read": cam.n_read, "frames_skipped_to_30fps": cam.n_skipped,
            "fps_measured": round(fps_meas, 2), "frame_gaps": n_gaps,
            "max_gap_ms": round(max_gap * 1000, 1), "encoder": cam.encoder,
            "encoder_stalls": cam.encoder_stalls, "flip": cam.flip, "view": args.view,
            "keys": {k: TASKS[args.task][k] for k in TASKS[args.task]},
            "labels": LABEL_NAMES[args.task], "latency_s": 0.0, "shifts": {}, "dropped": [],
            "reviewed": False, "notes": args.notes, "created": stamp,
            "git": git_hash()}  # fmt: skip
    (sdir / "meta.json").write_text(json.dumps(meta, indent=1))
    lab = save_labels(sdir)
    print(f"saved {sdir}\n  {summary(lab, args.task, fps_meas or args.fps)}"
          f"\n  measured {fps_meas:.1f} fps (requested {args.fps}), {n_gaps} frame gaps "
          f"(longest {max_gap * 1000:.0f} ms), encoder {cam.encoder}"
          + ("" if fps_meas >= 0.97 * args.fps and not n_gaps else
             "\n  WARNING: below the requested fps; more light / camera fps setting")
          + f"\nnext: python scripts/collect_labels.py review {sdir}")  # fmt: skip


# ------------------------------------------------------------------ review
class Frames:
    """Random access to video frames (small LRU cache)."""

    def __init__(self, path: Path, width: int):
        self.cap, self.width, self.cache = cv2.VideoCapture(str(path)), width, {}
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = -1

    def get(self, i: int):
        if i in self.cache:
            return self.cache[i]
        if i != self.pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = self.cap.read()
        self.pos = i
        if not ok:
            return None
        h, w = f.shape[:2]
        f = cv2.resize(f, (self.width, round(h * self.width / w)), interpolation=cv2.INTER_AREA)
        if len(self.cache) > 400:
            self.cache.pop(next(iter(self.cache)))
        self.cache[i] = f
        return f


def review(args) -> None:
    import pygame as pg

    sdir = args.session
    meta = json.loads((sdir / "meta.json").read_text())
    ev = json.loads((sdir / "events.json").read_text())
    t = np.load(sdir / "frames.npz")["t"]
    task, fps = meta["task"], meta["fps_measured"] or meta["fps_requested"]
    meta.setdefault("shifts", {})
    meta.setdefault("dropped", [])
    edges = [i for i, e in enumerate(ev) if e["key"] in TASKS[task]]  # label-change edges
    if not edges:
        raise SystemExit("no key presses in this session")
    press_of = {}
    for r in intervals(ev, float(t[-1]) + 1):
        for i in (r["i0"], r["i1"]):
            if i is not None:
                press_of[i] = (r["i0"], r["i1"])
    n_side, tw = 4, args.thumb_width
    frames = Frames(sdir / "video.mp4", tw)
    th = frames.get(0).shape[0]
    pg.init()
    W, H = (2 * n_side + 1) * (tw + 6) + 6, th + 150
    screen = pg.display.set_mode((W, H))
    pg.display.set_caption(f"review {meta['session']}")
    font = pg.font.SysFont("Helvetica", 18, bold=True)
    small = pg.font.SysFont("Helvetica", 14)
    k, dirty, lab = 0, True, None

    def relabel():
        return labels_from_meta(t, ev, meta)

    def save():
        meta["reviewed"] = True
        (sdir / "meta.json").write_text(json.dumps(meta, indent=1))
        lab = save_labels(sdir)
        print(f"saved: latency {meta['latency_s'] * 1000:.0f} ms, "
              f"{len(meta['shifts'])} shifted, {len(meta['dropped'])} dropped edges; "
              + summary(lab, task, fps))  # fmt: skip

    running = True
    while running:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                running = False
            elif e.type == pg.KEYDOWN:
                i = edges[k]
                step = 1.0 / fps
                if e.key in (pg.K_q, pg.K_ESCAPE):
                    running = False
                elif e.key == pg.K_RIGHT:
                    k = min(k + 1, len(edges) - 1)
                elif e.key == pg.K_LEFT:
                    k = max(k - 1, 0)
                elif e.key == pg.K_COMMA:  # all label changes 1 frame earlier
                    meta["latency_s"] = round(meta["latency_s"] - step, 4)
                elif e.key == pg.K_PERIOD:  # later (frame i shows the keys at t[i] - latency)
                    meta["latency_s"] = round(meta["latency_s"] + step, 4)
                elif e.key == pg.K_LEFTBRACKET:
                    meta["shifts"][str(i)] = round(meta["shifts"].get(str(i), 0.0) - step, 4)
                elif e.key == pg.K_RIGHTBRACKET:
                    meta["shifts"][str(i)] = round(meta["shifts"].get(str(i), 0.0) + step, 4)
                elif e.key == pg.K_d:
                    i0 = press_of.get(i, (i, None))[0]
                    if i0 in meta["dropped"]:
                        meta["dropped"].remove(i0)
                    else:
                        meta["dropped"].append(i0)
                elif e.key == pg.K_s:
                    save()
                dirty = True
        if dirty:
            lab = relabel()
            i = edges[k]
            te = t - meta["latency_s"]
            t_edge = ev[i]["t"] + meta["shifts"].get(str(i), 0.0)
            c = int(np.searchsorted(te, t_edge))  # first frame with the new label
            screen.fill((25, 25, 25))
            for j in range(-n_side, n_side + 1):
                f_i = c + j
                x = 6 + (j + n_side) * (tw + 6)
                if 0 <= f_i < len(t) and (img := frames.get(f_i)) is not None:
                    col = COLORS[int(lab["label"][f_i])] if lab["valid"][f_i] else BAD
                    pg.draw.rect(screen, col, (x - 4, 46, tw + 8, th + 8), 0)
                    screen.blit(to_surface(pg, img, 0), (x, 50))
                    screen.blit(small.render(f"{j:+d}", True, (255, 255, 255)), (x + 4, 52))
                    if j == 0:
                        pg.draw.rect(screen, (255, 255, 255), (x - 6, 44, tw + 12, th + 12), 2)
            e_ = ev[i]
            what = {"space": "SPACE", "h": "H", "f": "F"}[e_["key"]] + (
                " down" if e_["down"] else " up"
            )
            drop = press_of.get(i, (i,))[0] in meta["dropped"]
            if task == "touch" and 0 <= c < len(t) and lab["finger"][c] >= 0:
                what = f"{FINGERS[lab['finger'][c]]} {what}"
            head = (f"change {k + 1}/{len(edges)}: {what}   frame {c}   "
                    f"latency {meta['latency_s'] * 1000:+.0f} ms   "
                    f"this change {meta['shifts'].get(str(i), 0.0) * 1000:+.0f} ms"
                    + ("   DROPPED" if drop else ""))  # fmt: skip
            screen.blit(font.render(head, True, (255, 255, 255)), (8, 12))
            helps = ["<- -> change   , . all labels earlier/later   [ ] this change   "
                     "D drop press   S save   Q save+quit",
                     "centre frame (+0) should be the FIRST frame of the new state (finger "
                     "touching / object held / released)."]  # fmt: skip
            for r, s in enumerate(helps):
                screen.blit(small.render(s, True, (200, 200, 200)), (8, th + 70 + 22 * r))
            pg.display.flip()
            dirty = False
        time.sleep(0.01)
    save()
    pg.quit()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--task", required=True, choices=list(TASKS))
    r.add_argument("--finger", default="index", choices=list(FINGERS),
                   help="touch task: finger labelled at the start (arrows switch)")  # fmt: skip
    r.add_argument("--object", default=None, help="grasp task: what is held (e.g. cup)")
    r.add_argument("--participant", default="andy")
    r.add_argument("--view", default="overhead", help="camera view, e.g. overhead / ego / side")
    r.add_argument("--cam", default="1", help="camera index or a video file (testing)")
    r.add_argument("--width", type=int, default=1920)
    r.add_argument("--height", type=int, default=1080)
    r.add_argument("--flip", default="", choices=["", "h", "v", "hv"])
    r.add_argument("--display-width", type=int, default=1280)
    r.add_argument("--seconds", type=float, default=0.0, help="stop after (0 = until Q)")
    r.add_argument("--notes", default="")
    r.add_argument("--no-landmarks", action="store_true", help="no MediaPipe hand preview")
    r.add_argument("--out", type=Path, default=Path("data/own"))
    v = sub.add_parser("review")
    v.add_argument("session", type=Path)
    v.add_argument("--thumb-width", type=int, default=200)
    args = ap.parse_args()
    if args.cmd == "record":
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        record(args)
    else:
        review(args)


if __name__ == "__main__":
    main()
