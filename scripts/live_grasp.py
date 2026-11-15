"""Live grasp demo: camera -> MediaPipe -> fingertip patches + hand crop -> GraspStep (ONNX, CPU).

Shows per frame: hand state (NONE / TOUCH / HOLD), per-finger contact, and the predicted MANO
mesh's contact points (vertices with contact probability >= 0.5) drawn on the hand. The mesh is
root-relative, so it is placed on the image with a scale + offset fitted to MediaPipe's 2D
landmarks (display only). Left hands (MediaPipe label "Left": matches DexYCB's true side on 96%
of train frames, 2026-11-09) are mirrored, as in training.

    python scripts/live_grasp.py                                                 # g9, webcam 0
    python scripts/live_grasp.py --ckpt runs/<run>/best.pt --source clip.mp4 --out out.mp4 --no-show
    python scripts/live_grasp.py --ckpt runs/<run>/best.pt --bench 200 --no-show  # latency only
Times every stage and writes <run>/live_grasp_latency.json.
Keys: q / Esc stop, h = flip left-right, v = flip upside down (or start with --flip h|v|hv).

Pipeline (default): a worker thread does capture + MediaPipe for frame t while a second thread
runs the model on frame t-1, so a frame costs max(stages) instead of their sum (MediaPipe and
ONNX Runtime release the GIL). The window runs on the main thread as a third stage
(cv2.waitKey costs ~13 ms per call on macOS) and skips frames if it falls behind. Every frame
still reaches the model, in order, so the causal state is identical to --serial. The models were
trained on 30 fps video: the overlay turns red below 28 fps.
"""

from __future__ import annotations

import argparse
import json
import os
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
from touch.data.grasp_store import to_input  # noqa: E402
from touch.data.landmarks import DEFAULT_MODEL  # noqa: E402
from touch.data.patches import (  # noqa: E402
    HAND_CROP_SIZE,
    LM_HISTORY,
    PATCH_SIZE,
    finger_patches,
    hand_crop,
    landmark_features,
)
from touch.interact.grasp import StickyHold  # noqa: E402
from touch.models.graspnet import GraspStep, pad_encoding  # noqa: E402
from touch.train_grasp import build_grasp_model, load_grasp_config  # noqa: E402

JPEG_QUALITY = 92  # training images were stored as JPEG at this quality
STATES = ["NONE", "TOUCH", "HOLD"]
STAGES = ["capture", "mediapipe", "prep", "model", "draw"]
NAMES = ["tips", "hand", "lm", "present", "hist_tok", "hist_he", "hist_present"]
OUTS = ["finger", "state", "vert", "verts", "joints", "tok", "he"]


def _onnx_outputs(path: Path) -> list[str]:
    import onnx

    return [o.name for o in onnx.load(str(path), load_external_data=False).graph.output]


def jpeg(x: np.ndarray) -> np.ndarray:
    ok, b = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


# "coreml": ONNX Runtime's CoreML provider (Mac GPU / Neural Engine; unsupported ops fall back
# to the CPU). Andy 2026-11-14: the GPU may be used live; check against "cpu" before relying on it.
PROVIDERS = {"cpu": ["CPUExecutionProvider"],
             "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"]}  # fmt: skip


class LiveGrasp:
    def __init__(self, ckpt: Path, threads: int, provider: str = "cpu"):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = load_grasp_config(ck["config"])
        model = build_grasp_model({**cfg, "pretrained": False}).eval()
        model.load_state_dict(ck["model"])
        self.T = cfg["T"]
        self.F = model.n_lm
        self.outs = OUTS + (["table"] if model.use_table else []) + ["mano_rot", "mano_betas"]
        pad = pad_encoding(model)
        self.pad_tok, self.pad_he = pad["tok"].numpy(), pad["hand_emb"].numpy()
        onnx_path = ckpt.parent / "grasp_step.onnx"
        if (
            not onnx_path.exists()
            or onnx_path.stat().st_mtime < ckpt.stat().st_mtime
            or _onnx_outputs(onnx_path) != self.outs
        ):  # fmt: skip  (older export)
            ex = self._example(model)
            # parallel workers may export at once: write a private file, then rename (atomic)
            tmp = onnx_path.with_suffix(f".{os.getpid()}.tmp")
            torch.onnx.export(GraspStep(model).eval(), ex, str(tmp), input_names=NAMES,
                              output_names=self.outs, opset_version=17, dynamo=False)  # fmt: skip
            os.replace(tmp, onnx_path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=PROVIDERS[provider])
        self.reset()

    def reset(self) -> None:
        """Fresh causal state (model history, landmark history, held crops, sticky HOLD)."""
        self.sticky = StickyHold()
        self.hist = deque([(self.pad_tok, self.pad_he, False)] * (self.T - 1), maxlen=self.T - 1)
        self.lm_hist: deque = deque(maxlen=LM_HISTORY["v2"])
        self.last_imgs = None

    def _example(self, model):
        T = self.T
        with torch.no_grad():
            p = pad_encoding(model)
        return (
            torch.randn(5, 3, PATCH_SIZE, PATCH_SIZE),
            torch.randn(3, HAND_CROP_SIZE, HAND_CROP_SIZE),
            torch.randn(self.F),
            torch.tensor(True),
            p["tok"][None].expand(T - 1, -1, -1).contiguous(),
            p["hand_emb"][None].expand(T - 1, -1).contiguous(),
            torch.zeros(T - 1, dtype=torch.bool),
        )

    def prep(self, frame: np.ndarray, lm: np.ndarray | None, left: bool) -> dict:
        """lm [21, 3] px (MediaPipe) or None -> ONNX inputs. Causal: holds the last images."""
        return self.inputs(self.crops(frame, lm, left))

    def crops(self, frame: np.ndarray, lm: np.ndarray | None, left: bool) -> tuple:
        """The frame-dependent half of prep (cheap; a recorder runs it live and the model
        later): -> (landmark features [F], present, fingertip patches [5, P, P, 3] uint8 RGB,
        hand crop uint8 RGB). Calling crops for every frame first and inputs + step afterwards,
        in order, gives the same outputs as prep + step per frame (separate states)."""
        self.lm_hist.append(lm if lm is not None else np.full((21, 3), np.nan, np.float32))
        feats, present = landmark_features(np.stack(self.lm_hist), "v2")
        pres = bool(present[-1])
        if pres:
            tips = np.stack([jpeg(p) for p in finger_patches(frame, lm[:, :2], PATCH_SIZE)])
            hand = jpeg(hand_crop(frame, lm[:, :2], HAND_CROP_SIZE, flip=left))
            self.last_imgs = (tips[..., ::-1].copy(), hand[..., ::-1].copy())  # BGR -> RGB
        if self.last_imgs is None:
            tips = np.zeros((5, PATCH_SIZE, PATCH_SIZE, 3), np.uint8)
            hand = np.zeros((HAND_CROP_SIZE, HAND_CROP_SIZE, 3), np.uint8)
        else:
            tips, hand = self.last_imgs
        return (feats[-1] if pres else np.zeros(self.F, np.float32), pres, tips, hand)

    def inputs(self, crops: tuple) -> dict:
        """crops (from .crops) + the model's history -> ONNX inputs."""
        lmf, pres, tips, hand = crops
        return {
            "tips": to_input(torch.from_numpy(tips)).numpy(),
            "hand": to_input(torch.from_numpy(hand)).numpy(),
            "lm": lmf,
            "present": np.array(pres),
            "hist_tok": np.stack([h[0] for h in self.hist]).astype(np.float32),
            "hist_he": np.stack([h[1] for h in self.hist]).astype(np.float32),
            "hist_present": np.array([h[2] for h in self.hist]),
        }

    def step(self, inputs: dict) -> dict:
        out = dict(zip(self.outs, self.sess.run(None, inputs), strict=True))
        self.hist.append((out["tok"], out["he"], bool(inputs["present"])))
        raw = int(out["state"].argmax()) if bool(inputs["present"]) else 0
        out["state_sticky"] = self.sticky.step(raw)  # HOLD stays HOLD until contact ends
        return out


GRASP_RUNS = ("runs/g10_grasp_own_20261113-230008", "runs/g9_grasp_oblique_20261113-093840",
              "runs/g8_grasp_webcam_20261113-011442",
              "runs/g6_grasp_table_20261112-101602")  # fmt: skip


def default_ckpt() -> Path:
    """Live default: g10 (g9 + Andy's own key-labelled grasp sessions; Andy went back to it
    2026-11-14 after g11, which over-fitted one camera spot), else g9, g8, g6 (the weights-v1
    release). g11: --ckpt runs/g11_grasp_own_20261114-075948/best.pt."""
    for r in GRASP_RUNS:
        if (Path(r) / "best.pt").exists():
            return Path(r) / "best.pt"
    return Path(GRASP_RUNS[-1]) / "best.pt"


def draw(frame, lm, out, left: bool, fps: float, decision: bool | None = None,
         mesh: bool = True) -> None:  # fmt: skip
    """decision: the grab gate's closed/open (live_sim); shown as the main label, the model's raw
    per-frame state in small text under it (the raw state flickers during holds).
    mesh=False: no 3D-mesh contact dots (fingertip circles + labels only)."""
    h, w = frame.shape[:2]
    if lm is not None and mesh:
        j = out["joints"].copy()
        v = out["verts"].copy()
        if left:  # undo the mirroring for display
            j[:, 0] *= -1
            v[:, 0] *= -1
        # rows x0, y0, x1, y1, ...: [j, 1, 0] / [j, 0, 1] -> scale, tx, ty (display only)
        A = np.concatenate([j[:, :2].reshape(-1, 1), np.tile(np.eye(2), (21, 1))], 1)
        s, tx, ty = np.linalg.lstsq(A, lm[:, :2].reshape(-1), rcond=None)[0]
        pv = v[:, :2] * s + [tx, ty]
        hot = out["vert"] >= 0.5
        for (x, y), c in zip(pv.astype(int), hot, strict=True):
            if c:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
    if lm is not None:
        for f, tip in enumerate((4, 8, 12, 16, 20)):
            p = out["finger"][f]
            col = (0, 0, 255) if p >= 0.5 else (200, 200, 200)
            cv2.circle(frame, tuple(lm[tip, :2].astype(int)), 7, col, 2)
    k = int(out.get("state_gated", out.get("state_sticky", out["state"].argmax())))
    col = [(200, 200, 200), (0, 200, 255), (0, 0, 255)][k]
    if decision is None:
        cv2.putText(frame, f"{STATES[k]} {out['state'][k]:.2f}", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)  # fmt: skip
    else:
        cv2.putText(frame, "HOLDING" if decision else "not holding", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255) if decision else (200, 200, 200),
                    3)  # fmt: skip
        cv2.putText(frame, f"model this frame: {STATES[k]} {out['state'][k]:.2f}", (16, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)  # fmt: skip
    fcol = (255, 255, 255) if fps >= 28 else (0, 0, 255)  # trained at 30 fps
    cv2.putText(frame, f"{fps:4.1f} fps", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fcol, 2)


class HandTracker:
    """Follow the SAME hand across frames: a second hand in view (e.g. resting near the laptop,
    saying NONE) must not take over. Picks the MediaPipe hand nearest the last tracked one; a
    different hand is accepted only after the tracked one is gone for > `lost_max` frames, or
    farther than `jump` x image width."""

    def __init__(self, width: int, lost_max: int = 15, jump: float = 0.25):
        self.w, self.lost_max, self.jump = width, lost_max, jump
        self.c, self.lost = None, 0

    def pick(self, hands: list, handedness: list) -> tuple[np.ndarray | None, bool]:
        """-> landmarks [21, 3] px of the tracked hand (or None), is_left."""
        k = None
        if hands:
            cen = [hd[:, :2].mean(0) for hd in hands]
            if self.c is None or self.lost > self.lost_max:
                k = int(np.argmax([hd[0].score for hd in handedness]))
            else:
                dist = [float(np.linalg.norm(c - self.c)) for c in cen]
                k = int(np.argmin(dist))
                if dist[k] > self.jump * self.w:
                    k = None
        if k is None:
            self.lost += 1
            return None, False
        self.c, self.lost = cen[k], 0
        return hands[k], handedness[k][0].category_name == "Left"


def frames_with_hands(cap, src, fps_in: float, stop: threading.Event, flip: dict | None = None):
    """Yields (frame, hands [21, 3] px list, handedness, t_capture_ms, t_mediapipe_ms)."""
    from mediapipe import Image, ImageFormat

    from touch.data.landmarks import _make_landmarker

    n = 0
    with _make_landmarker(DEFAULT_MODEL, 0.3) as det:
        while not stop.is_set():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                return
            t1 = time.perf_counter()
            fh, fv = (flip["h"], flip["v"]) if flip else (False, False)
            if fh or fv:  # BEFORE MediaPipe: landmarks, handedness and crops all agree
                frame = cv2.flip(frame, -1 if fh and fv else 1 if fh else 0)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ms = int(round(n * 1000 / fps_in)) if not isinstance(src, int) else int(t0 * 1000)
            r = det.detect_for_video(Image(image_format=ImageFormat.SRGB, data=rgb), ms)
            h, w = frame.shape[:2]
            hands = [np.array([[p.x * w, p.y * h, p.z * w] for p in hl], np.float32)
                     for hl in r.hand_landmarks]  # fmt: skip
            t2 = time.perf_counter()
            n += 1
            yield frame, hands, r.handedness, (t1 - t0) * 1000, (t2 - t1) * 1000


def threaded(gen, maxsize: int = 2):
    """Run a generator in a worker thread; yields its items in order (None-terminated)."""
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
    ap.add_argument("--ckpt", type=Path, default=None, help="default g10, else g9, g8, g6")
    ap.add_argument("--source", default="0", help="webcam index or video file")
    ap.add_argument("--width", type=int, default=1280, help="requested webcam width")
    ap.add_argument("--height", type=int, default=720, help="requested webcam height")
    ap.add_argument("--fps", type=int, default=30, help="requested webcam frame rate")
    ap.add_argument("--out", type=Path, default=None, help="write the annotated video here")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--provider", choices=["cpu", "coreml"], default="cpu",
                    help="coreml = Mac GPU / Neural Engine")  # fmt: skip
    ap.add_argument("--serial", action="store_true", help="no pipelining (old behaviour)")
    ap.add_argument(
        "--flip",
        default="",
        choices=["", "h", "v", "hv"],
        help="start flipped: h left-right, v upside down (toggle live: h / v keys)",
    )
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="stop after N frames")
    ap.add_argument("--gate", action="store_true",
                    help="FIXED camera (top-down rig): contact may only start with an object "
                         "under the hand (touch.interact.object_gate)")  # fmt: skip
    args = ap.parse_args()
    gate = latch = None
    if args.gate:
        from touch.interact.object_gate import ContactLatch, ObjectGate

        gate = ObjectGate(region="hull", min_px=10, arm=2.5, skin_tau=20.0, keep=12, every=4)
        latch = ContactLatch()
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    args.ckpt = args.ckpt or default_ckpt()
    print(f"model: {args.ckpt}")
    live = LiveGrasp(args.ckpt, args.threads, args.provider)
    times = {k: [] for k in STAGES}
    writer, n, t_prev, fps = None, 0, time.perf_counter(), 0.0
    stop = threading.Event()
    flip = {"h": "h" in args.flip, "v": "v" in args.flip}  # toggled by the window keys
    gen = frames_with_hands(cap, src, fps_in, stop, flip)
    t_start = time.perf_counter()
    display: queue.Queue = queue.Queue(maxsize=1)

    def show_now(frame) -> None:  # main thread only (macOS GUI)
        cv2.imshow("grasp (q to stop)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            stop.set()
        elif key in (ord("h"), ord("v")):
            flip[chr(key)] = not flip[chr(key)]

    def show_later(frame) -> None:  # newest frame for the window; the model never waits
        try:
            display.get_nowait()
        except queue.Empty:
            pass
        display.put_nowait(frame)

    def model_loop(show) -> None:
        nonlocal writer, n, t_prev, fps
        items = gen if args.serial else threaded(gen)
        for frame, hands, handedness, t_cap, t_mp in items:
            if stop.is_set():
                break
            t2 = time.perf_counter()
            lm, left, near = None, False, False
            if hands:
                k = int(np.argmax([hd[0].score for hd in handedness]))
                lm = hands[k]
                left = handedness[k][0].category_name == "Left"
            if gate is not None:
                flags = gate.update(frame, hands)
                near = bool(flags[k]) if hands else False
            inputs = live.prep(frame, lm, left)
            t3 = time.perf_counter()
            out = live.step(inputs)
            if latch is not None:
                out["state_gated"] = latch.step(int(out["state_sticky"]), near)
            t4 = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(t4 - t_prev, 1e-6) if n else 0.0
            t_prev = t4
            draw(frame, lm, out, left, fps)
            if args.out is not None:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps_in, (w, h))  # fmt: skip
                writer.write(frame)
            if not args.no_show:
                show(frame)
            t5 = time.perf_counter()
            for k, v in zip(STAGES, (t_cap, t_mp, (t3 - t2) * 1000, (t4 - t3) * 1000,
                                     (t5 - t4) * 1000), strict=True):  # fmt: skip
                times[k].append(v)
            n += 1
            if args.bench and n >= args.bench:
                break
        stop.set()

    if args.serial:
        model_loop(show_now)
    else:  # capture+MediaPipe thread -> model thread -> window on the main thread
        worker = threading.Thread(target=model_loop, args=(show_later,), daemon=True)
        worker.start()
        while worker.is_alive():
            try:
                show_now(display.get(timeout=0.1))
            except queue.Empty:
                continue
        worker.join()
    elapsed = time.perf_counter() - t_start
    stop.set()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    skip = min(20, n // 4)
    res = {k: round(float(np.median(v[skip:])), 2) for k, v in times.items() if len(v) > skip}
    res["total_excl_capture_draw"] = round(res["mediapipe"] + res["prep"] + res["model"], 2)
    res["sustained_fps"] = round(n / max(elapsed, 1e-6), 1)
    res["frames"], res["threads"], res["pipelined"] = n, args.threads, not args.serial
    print(json.dumps(res))
    (args.ckpt.parent / "live_grasp_latency.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
