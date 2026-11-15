"""Real-time touch API: camera frame + hand landmarks -> per-finger touch, events, gestures.

For apps (e.g. HandFlow) that already run MediaPipe: pass the landmarks in, so MediaPipe runs once.

    session = TouchSession(["runs/<run>/best.pt"], surface=SurfaceMap.from_corners(c, (1920, 1080)))
    res = session.step(frame_bgr, lm_px, t)  # lm_px [21, 3] px (z = rel. depth * width) or None
    res.touch, res.prob, res.events, res.gestures, res.pointer  # pointer: surface down/move/up

One TouchSession per hand (its temporal state is per hand). Several checkpoints = an ensemble:
their probabilities are averaged and `post` must then come from scripts/ensemble_eval.py (tuned on
validation for that ensemble). Causal only: StreamingStep refuses the offline (bigru) model.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from touch.data.patch_store import represent
from touch.data.patches import LM_HISTORY, PATCH_SIZE, finger_patches, landmark_features
from touch.interact.gestures import Gesture, GestureConfig, GestureRecognizer
from touch.interact.surface import SurfaceMap
from touch.interact.touches import TouchEvent, TouchTracker, hand_points

JPEG_QUALITY = 92  # training patches were stored as JPEG at this quality (extract_patches.py)
_SESSIONS: dict = {}  # (onnx path, threads) -> onnxruntime session, shared by all hands


def landmarks_px(norm_landmarks, width: int, height: int) -> np.ndarray:
    """MediaPipe normalized landmarks (objects with .x .y .z, or [21, 3]) -> [21, 3] px, z scaled
    by the image width (the convention used for training)."""
    a = np.array(
        [[q.x, q.y, q.z] for q in norm_landmarks]
        if hasattr(norm_landmarks[0], "x")
        else norm_landmarks,
        np.float32,
    )
    return a * np.array([width, height, width], np.float32)


class _Member:
    """One checkpoint: shared ONNX session + this hand's feature history."""

    def __init__(self, ckpt: Path, threads: int):
        import onnxruntime as ort
        import torch

        from touch.models.streaming import export_onnx, pad_features
        from touch.train import build_model, load_config

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = load_config(ck["config"])
        self.T, self.repr, self.lm_feats = cfg["T"], cfg["input_repr"], cfg["lm_feats"]
        self.post = ck.get("post")
        onnx_path = ckpt.parent / f"stream_{ckpt.stem}.onnx"
        key = (str(onnx_path), threads)
        model = build_model({**cfg, "pretrained": False}).eval()
        model.load_state_dict(ck["model"])
        if key not in _SESSIONS:
            if not onnx_path.exists() or onnx_path.stat().st_mtime < ckpt.stat().st_mtime:
                export_onnx(model, self.repr, self.T, onnx_path)
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            _SESSIONS[key] = ort.InferenceSession(
                str(onnx_path), so, providers=["CPUExecutionProvider"]
            )
        self.sess = _SESSIONS[key]
        fingers = torch.arange(5)
        pad = pad_features(model, self.repr, fingers).numpy()
        self.hist = np.repeat(pad[:, None], self.T - 1, axis=1).astype(np.float32)
        self.prev_x = np.zeros((5, PATCH_SIZE, PATCH_SIZE, 3), np.float32)

    def run(self, x: np.ndarray, lm_feat: np.ndarray, present: bool) -> np.ndarray:
        img = represent(x, self.prev_x, self.repr)
        self.prev_x = x
        prob, feat = self.sess.run(
            None,
            {
                "img": np.ascontiguousarray(img, np.float32),
                "lm": np.repeat(lm_feat[None], 5, axis=0).astype(np.float32),
                "present": np.full(5, present),
                "finger": np.arange(5, dtype=np.int64),
                "hist": self.hist,
            },
        )
        self.hist = np.concatenate([self.hist[:, 1:], feat[:, None]], axis=1)
        return prob


class LiveTouch:
    """Causal per-frame touch probabilities + post-processed state for one hand (5 fingers).

    Scores equal touch.eval.run's (see touch.models.streaming). `post` (median k, hysteresis
    on / off) defaults to the checkpoint's validation-tuned setting; required for an ensemble.
    """

    def __init__(self, ckpts, threads: int = 4, jpeg: bool = True, post: dict | None = None):
        ckpts = [Path(c) for c in ([ckpts] if isinstance(ckpts, str | Path) else ckpts)]
        self.members = [_Member(c, threads) for c in ckpts]
        lm_feats = {m.lm_feats for m in self.members}
        assert len(lm_feats) == 1, f"members use different landmark features: {lm_feats}"
        self.lm_feats = lm_feats.pop()
        if post is None:
            assert len(ckpts) == 1, (
                "an ensemble needs its own validation-tuned post (ensemble_eval)"
            )
            post = self.members[0].post
        assert post is not None, "checkpoint has no validation-tuned post-processing"
        self.post, self.jpeg = post, jpeg
        self.held = np.zeros((5, PATCH_SIZE, PATCH_SIZE, 3), np.float32)  # last patch, RGB [0, 1]
        n = LM_HISTORY[self.lm_feats]
        self.lm_hist = deque([np.full((21, 3), np.nan, np.float32)] * n, maxlen=n)
        self.recent = deque(maxlen=max(1, int(self.post["k"])))
        self.touch = np.zeros(5, bool)

    def model(self, frame_bgr: np.ndarray, lm: np.ndarray | None) -> np.ndarray:
        """Raw touch probability [5] this frame; lm [21, 3] px or None (no hand -> 0)."""
        self.lm_hist.append(lm if lm is not None else np.full((21, 3), np.nan, np.float32))
        feats, _ = landmark_features(np.stack(self.lm_hist), self.lm_feats)
        present = lm is not None
        if present:
            P = finger_patches(frame_bgr, lm[:, :2])
            if self.jpeg:  # match the JPEG round trip of the training patches
                q = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                P = np.stack([cv2.imdecode(cv2.imencode(".jpg", p, q)[1], 1) for p in P])
            self.held = P[..., ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        # no hand: hold the last patch with present = 0 (as in training / evaluation)
        prob = np.mean([m.run(self.held, feats[-1], present) for m in self.members], axis=0)
        return np.where(present, prob, 0.0)

    def postprocess(self, prob: np.ndarray) -> np.ndarray:
        """Causal median over the last k frames, then hysteresis -> smoothed prob; sets .touch."""
        self.recent.append(prob)
        p = np.median(np.stack(self.recent), axis=0)
        self.touch = np.where(self.touch, p >= self.post["off"], p >= self.post["on"])
        return p


@dataclass
class FrameResult:
    t: float
    present: bool
    prob: np.ndarray  # [5] smoothed touch probability (thumb..pinky)
    touch: np.ndarray  # [5] bool
    tips: np.ndarray  # [5, 2] fingertip px (NaN without a hand)
    tips_surface: np.ndarray | None  # [5, 2] fingertips in surface units (None without a surface)
    events: list[TouchEvent] = field(default_factory=list)
    gestures: list[Gesture] = field(default_factory=list)
    pointer: list[dict] = field(default_factory=list)  # surface-space down / move / up


class TouchSession:
    """LiveTouch + touch events + gestures (+ surface mapping) for one hand."""

    def __init__(
        self,
        ckpts,
        surface: SurfaceMap | None = None,
        threads: int = 4,
        post: dict | None = None,
        gesture_cfg: GestureConfig | None = None,
        jpeg: bool = True,
    ):
        self.live = LiveTouch(ckpts, threads, jpeg=jpeg, post=post)
        self.surface = surface
        self.tracker, self.recognizer = TouchTracker(), GestureRecognizer(gesture_cfg)
        self.on_surface: set[int] = set()  # touch ids that started inside the surface

    def step(self, frame_bgr: np.ndarray, lm: np.ndarray | None, t: float) -> FrameResult:
        p = self.live.postprocess(self.live.model(frame_bgr, lm))
        tips, hand_size = hand_points(lm)
        events = self.tracker.update(t, self.live.touch, tips, hand_size, p)
        gestures = self.recognizer.update(events)
        uv = self.surface.map(tips) if self.surface is not None else None
        return FrameResult(
            t, lm is not None, p, self.live.touch.copy(), tips, uv, events, gestures,
            self._pointer(events),
        )  # fmt: skip

    def _pointer(self, events: list[TouchEvent]) -> list[dict]:
        """Touch events in surface coordinates. A touch that starts outside the surface is
        ignored; one that starts inside is reported until it ends (even if it slides off)."""
        out = []
        for e in events:
            if self.surface is None:
                xy = e.xy
            else:
                xy = tuple(float(v) for v in self.surface.map(np.array(e.xy)))
                if e.phase == "down":
                    if not self.surface.inside(np.array(xy)):
                        continue
                    self.on_surface.add(e.touch_id)
                elif e.touch_id not in self.on_surface:
                    continue
                if e.phase == "up":
                    self.on_surface.discard(e.touch_id)
            out.append({**e.to_dict(), "xy": [round(v, 1) for v in xy]})
        return out


def load_post(path: Path) -> dict:
    """Validation-tuned post-processing of an ensemble (results/ensemble_<tag>.json)."""
    return json.loads(Path(path).read_text())["post"]
