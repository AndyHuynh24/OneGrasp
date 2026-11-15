"""Object-proximity gate for a FIXED camera (top-down webcam): RGB only, causal, no extra model.

Why: the grasp state head calls a hand resting flat on the table "touch" / "hold" (HO-Cap
top-down: 64% / 66% of table-rest frames for g1 / g3, docs/DECISIONS.md). For a robot, contact
only matters when an object is under the fingers.

How:
1. Background: per-pixel median (Lab, half resolution) of recent frames, sampled every `every`
   frames, `keep` samples, with every hand masked out of the update (convex hull of its
   MediaPipe landmarks, dilated). Objects that rest on the table end up in it; moving hands do not.
2. Object pixels near a hand: background pixels that differ from the local table colour (median
   of the background around the palm) by > `tau`, either within `radius` x hand size of any
   fingertip (region="tips") or inside the hand's outline, i.e. under the hand (region="hull").
3. Latch: contact may only START on a frame where the hand is near an object; once on, it stays
   on while the model keeps predicting contact (a lifted object leaves its resting spot).

Assumes a static camera and a table that is mostly one colour (the overhead-webcam setup).
Parameters are fixed a priori and checked on DexYCB val (scripts/eval_gate.py), never on HO-Cap.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

TIPS = [4, 8, 12, 16, 20]


def hand_size(lm: np.ndarray) -> float:
    """Wrist -> middle MCP distance (px), a scale-free hand unit (~9 cm)."""
    return float(np.linalg.norm(lm[9, :2] - lm[0, :2]))


class ObjectGate:
    def __init__(self, scale: float = 0.5, every: int = 5, keep: int = 20, tau: float = 25.0,
                 radius: float = 0.6, dilate: float = 0.3, min_px: int = 4,
                 table_win: float = 3.0, region: str = "tips", arm: float = 0.0,
                 skin_tau: float = 0.0, work_w: int | None = 320) -> None:  # fmt: skip
        """arm > 0: also mask a forearm band behind the wrist, `arm` x hand size long, from the
        background update (a resting forearm otherwise becomes "object"). skin_tau > 0: pixels
        within skin_tau of the hand's current skin colour never count as object."""
        assert region in ("tips", "hull")
        self.region, self.arm, self.skin_tau = region, arm, skin_tau
        self.cur: np.ndarray | None = None
        self.scale, self.every, self.tau = scale, every, tau
        self.work_w = work_w  # fixed working width (px): scale = work_w / frame width
        self.radius, self.dilate, self.min_px, self.table_win = radius, dilate, min_px, table_win
        self.buf: deque = deque(maxlen=keep)
        self.bg: np.ndarray | None = None
        self.t = 0

    def reset(self) -> None:
        self.buf.clear()
        self.bg, self.t = None, 0

    def _hand_mask(self, shape, hands: list[np.ndarray]) -> np.ndarray:
        m = np.zeros(shape[:2], np.uint8)
        for lm in hands:
            p = lm[:, :2] * self.scale
            hull = cv2.convexHull(p.astype(np.float32)).astype(np.int32)
            cv2.fillConvexPoly(m, hull, 1)
            if self.arm > 0:  # forearm: a band from the wrist away from the fingers
                u = p[0] - p[9]
                u = u / max(np.linalg.norm(u), 1e-6)
                end = p[0] + u * self.arm * hand_size(lm) * self.scale
                width = 1.3 * np.linalg.norm(p[5] - p[17])
                cv2.line(m, tuple(np.round(p[0]).astype(int)), tuple(np.round(end).astype(int)),
                         1, max(2, int(round(width))))  # fmt: skip
            r = max(1, round(self.dilate * hand_size(lm) * self.scale))
            m |= cv2.dilate(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
        return m.astype(bool)

    def update(self, frame_bgr: np.ndarray, hands: list[np.ndarray]) -> list[bool]:
        """frame [H, W, 3] BGR uint8; hands = ALL hands in the frame, each [21, >=2] px
        -> near-object flag per hand (same order)."""
        assert frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3
        if self.work_w:  # constant cost whatever the camera resolution (640 px -> scale 0.5)
            self.scale = self.work_w / frame_bgr.shape[1]
        sample = self.t % self.every == 0 or self.bg is None
        if sample or self.skin_tau > 0:
            small = cv2.resize(frame_bgr, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA)  # fmt: skip
            self.cur = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        if sample:
            lab = self.cur.copy()
            lab[self._hand_mask(lab.shape, hands)] = np.nan
            self.buf.append(lab)
            with np.errstate(all="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    self.bg = np.nanmedian(np.stack(self.buf), axis=0)
        self.t += 1
        return [self._near(lm) for lm in hands]

    def _near(self, lm: np.ndarray) -> bool:
        bg = self.bg
        h, w = bg.shape[:2]
        s = hand_size(lm) * self.scale
        if not np.isfinite(s) or s < 1:
            return False
        c = lm[[0, 5, 9, 13, 17], :2].mean(0) * self.scale
        k = self.table_win * s
        x0, x1 = int(max(0, c[0] - k)), int(min(w, c[0] + k + 1))
        y0, y1 = int(max(0, c[1] - k)), int(min(h, c[1] + k + 1))
        win = bg[y0:y1, x0:x1].reshape(-1, 3)
        win = win[np.isfinite(win).all(1)]
        if len(win) < 20:
            return False
        table = np.median(win, 0)
        if self.region == "hull":  # object pixels UNDER the hand (inside its outline)
            m = np.zeros((h, w), np.uint8)
            hull = cv2.convexHull((lm[:, :2] * self.scale).astype(np.float32)).astype(np.int32)
            cv2.fillConvexPoly(m, hull, 1)
            diff = bg - table
            diff[..., 0] *= 0.5
            dist = np.sqrt((diff**2).sum(-1))
            obj = m.astype(bool) & np.isfinite(dist) & (dist > self.tau)
            obj &= ~self._skin_like(bg, lm)
            return int(obj.sum()) >= self.min_px
        r = self.radius * s
        tips = lm[TIPS, :2] * self.scale
        lo = np.floor(tips.min(0) - r).astype(int).clip(0)
        hi = np.ceil(tips.max(0) + r + 1).astype(int)
        hi = np.minimum(hi, [w, h])
        if (hi <= lo).any():
            return False
        yy, xx = np.mgrid[lo[1] : hi[1], lo[0] : hi[0]]
        d2 = ((xx[..., None] - tips[:, 0]) ** 2 + (yy[..., None] - tips[:, 1]) ** 2).min(-1)
        patch = bg[lo[1] : hi[1], lo[0] : hi[0]]
        diff = patch - table
        diff[..., 0] *= 0.5  # brightness counts half (shadows)
        dist = np.sqrt((diff**2).sum(-1))
        obj = (d2 <= r * r) & np.isfinite(dist) & (dist > self.tau)
        return int(obj.sum()) >= self.min_px

    def _skin_like(self, img: np.ndarray, lm: np.ndarray) -> np.ndarray:
        """bool mask over img (Lab, same size as the background): close to this hand's skin
        colour (median of the current frame at the knuckles / finger joints)."""
        if self.skin_tau <= 0 or self.cur is None:
            return np.zeros(img.shape[:2], bool)
        h, w = self.cur.shape[:2]
        q = np.round(lm[[2, 5, 6, 9, 10, 13, 14, 17, 18], :2] * self.scale).astype(int)
        q = q[(q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)]
        if len(q) < 3:
            return np.zeros(img.shape[:2], bool)
        skin = np.median(self.cur[q[:, 1], q[:, 0]], 0)
        d = img - skin
        d[..., 0] *= 0.5
        return np.sqrt((d**2).sum(-1)) < self.skin_tau


class ContactLatch:
    """Causal: contact (state > 0) may only start where `near`; it then persists while the model
    keeps predicting contact. step(state, near) -> gated state."""

    def __init__(self) -> None:
        self.on = False

    def step(self, state: int, near: bool) -> int:
        if state == 0:
            self.on = False
            return 0
        self.on = self.on or near
        return state if self.on else 0


def latch(states: np.ndarray, near: np.ndarray) -> np.ndarray:
    """Offline helper over a track: states [N] int, near [N] bool -> gated states [N] int8."""
    assert states.shape == near.shape
    lt = ContactLatch()
    return np.array([lt.step(int(s), bool(g)) for s, g in zip(states, near, strict=True)], np.int8)
