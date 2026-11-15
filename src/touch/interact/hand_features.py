"""Hand-shape, hand-motion and object co-motion features for the HOLD decision (causal, per frame).

Shared by the offline study (scripts/hold_features.py, scripts/grab_fusion.py) and the live demo,
so validation numbers carry over exactly.

- `pose_features(joints)`: grasp shape from 21 hand joints (predicted MANO, or MediaPipe
  landmarks): flexion angles, finger curl, thumb-to-finger distances, fingertip-to-palm distances,
  spread, thumb opposition. Distances are divided by the palm length (wrist -> middle MCP), so the
  features do not depend on hand size, camera distance or units.
- `image_motion(lm)`: 2D hand velocity in palm lengths per frame (no calibration needed).
- `CoMotion`: "does something next to the fingers move WITH the hand?" Sparse optical flow
  (Lucas-Kanade) on corners in a band around the hand (hand and forearm masked out), compared
  with the flow on the hand itself and on the background further away. A carried object moves
  with the hand; table / background next to an empty or touching hand stays still. Undefined
  (NaN) while the hand does not move; the last moving value is kept with its age.
Joint order (MANO-21 = MediaPipe): 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring,
17-20 pinky.
"""

from __future__ import annotations

import warnings
from collections import deque

import cv2
import numpy as np

CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16],
          [0, 17, 18, 19, 20]]  # fmt: skip
TIPS = [4, 8, 12, 16, 20]
PALM = [0, 5, 9, 13, 17]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def _norm(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(v, axis=-1)


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle (rad) between vectors a, b [..., 3]."""
    c = (a * b).sum(-1) / np.maximum(_norm(a) * _norm(b), 1e-9)
    return np.arccos(np.clip(c, -1.0, 1.0))


POSE_NAMES = (
    [f"flex_{f}_{j}" for f in FINGERS for j in (1, 2, 3)]
    + [f"curl_{f}" for f in FINGERS]
    + [f"thumb_to_{f}" for f in FINGERS[1:]]
    + [f"tip_palm_{f}" for f in FINGERS]
    + ["spread", "thumb_oppose", "aperture_min"]
)


def pose_features(J: np.ndarray) -> np.ndarray:
    """J [N, 21, 3] (any frame, any units; NaN = no hand) -> [N, len(POSE_NAMES)]."""
    J = np.asarray(J, np.float64)
    with np.errstate(all="ignore"):
        L = _norm(J[:, 9] - J[:, 0])[:, None]  # palm length
        out = []
        for ch in CHAINS:  # flexion at the 3 joints after the root of each finger
            b = J[:, ch[1:]] - J[:, ch[:-1]]  # 4 bones
            out.append(np.stack([_angle(b[:, k], b[:, k + 1]) for k in range(3)], 1))
        for ch in CHAINS:  # curl: tip-to-base distance / finger length (1 straight, small curled)
            seg = _norm(J[:, ch[2:]] - J[:, ch[1:-1]]).sum(1)
            out.append((_norm(J[:, ch[-1]] - J[:, ch[1]]) / seg)[:, None])
        out.append(_norm(J[:, TIPS[1:]] - J[:, 4:5]) / L)  # thumb tip to the 4 fingertips
        palm = J[:, PALM].mean(1, keepdims=True)
        out.append(_norm(J[:, TIPS] - palm) / L)
        tips = J[:, TIPS[1:]]
        d = _norm(tips[:, :, None] - tips[:, None])
        out.append((d.sum((1, 2)) / 12.0 / L[:, 0])[:, None])  # mean pairwise, 4 fingertips
        n = np.cross(J[:, 5] - J[:, 0], J[:, 17] - J[:, 0])  # palm normal
        out.append(_angle(J[:, 4] - J[:, 2], n)[:, None])  # thumb direction vs palm normal
        out.append((_norm(J[:, TIPS[1:]] - J[:, 4:5]).min(1) / L[:, 0])[:, None])
        X = np.concatenate(out, 1)
    return X.astype(np.float32)


def image_motion(lm: np.ndarray, step: int = 3) -> tuple[np.ndarray, list[str]]:
    """lm [N, 21, 2+] px (NaN = none) -> [N, 3]: hand centre velocity over `step` frames in
    palm lengths per frame (x right, y UP in the image) and its magnitude."""
    lm = np.asarray(lm, np.float64)[..., :2]
    with np.errstate(all="ignore"):
        L = _norm(lm[:, 9] - lm[:, 0])
        c = lm[:, PALM].mean(1)
        v = np.full((len(lm), 2), np.nan)
        v[step:] = (c[step:] - c[:-step]) / step / L[step:, None]
        v[:, 1] *= -1
    return np.concatenate([v, _norm(v)[:, None]], 1).astype(np.float32), [
        "img_vx", "img_vy_up", "img_speed"]  # fmt: skip


# ------------------------------------------------------------------ object co-motion (image)
COMOTION_NAMES = ["co_frac", "still_frac", "co_n", "hand_flow", "co_last", "co_age", "co_ema"]


class CoMotion:
    """Streaming co-motion cue. step(gray_frame, landmarks px [21, 2+] or None) -> features [7]:
    co_frac     share of tracked points in the band around the hand moving with the hand
    still_frac  share moving like the background (not with the hand)
    co_n        number of tracked band points / 50
    hand_flow   hand flow relative to background, palm lengths per `stride` frames
    co_last     co_frac of the most recent frame in which the hand moved (latched)
    co_age      frames since that frame / 30 (capped at 3)
    co_ema      exponential average of co_frac over moving frames
    co_frac / still_frac are NaN when the hand moves less than `min_move` palm lengths."""

    def __init__(self, stride: int = 2, size: int = 96, min_move: float = 0.04,
                 band: float = 0.6, ema: float = 0.3):  # fmt: skip
        self.stride, self.size, self.min_move, self.band, self.ema_a = (
            stride, size, min_move, band, ema)  # fmt: skip
        self.reset()

    def reset(self) -> None:
        self.hist: deque = deque(maxlen=self.stride)
        self.last, self.age, self.ema = np.nan, 90.0, np.nan

    def _crop(self, gray: np.ndarray, lm: np.ndarray):
        """Crop window around the hand, scaled so the palm length is `size` / 3.2 px."""
        L = float(np.linalg.norm(lm[9, :2] - lm[0, :2]))
        c = lm[PALM, :2].mean(0)
        half = 2.6 * max(L, 8.0)  # hand + band + background ring
        s = (self.size / 3.2) / max(L, 8.0)
        A = np.array([[s, 0, -s * (c[0] - half)], [0, s, -s * (c[1] - half)]], np.float32)
        W = int(round(2 * half * s))
        return A, W, s * L

    def _masks(self, lm_c: np.ndarray, W: int, Lc: float):
        hand = np.zeros((W, W), np.uint8)
        th = max(2, int(round(0.32 * Lc)))
        pts = np.round(lm_c[:, :2]).astype(np.int32)
        for ch in CHAINS:
            for a, b in zip(ch[:-1], ch[1:], strict=True):
                cv2.line(hand, tuple(pts[a]), tuple(pts[b]), 255, th)
        cv2.fillPoly(hand, [pts[[0, 1, 5, 9, 13, 17]]], 255)
        arm = pts[0] + (lm_c[0, :2] - lm_c[9, :2]) * 1.8  # forearm: away from the fingers
        cv2.line(hand, tuple(pts[0]), tuple(np.round(arm).astype(np.int32)), 255,
                 max(3, int(round(0.9 * Lc))))  # fmt: skip
        dist = cv2.distanceTransform(255 - hand, cv2.DIST_L2, 3)
        band = ((dist > 0) & (dist <= self.band * Lc)).astype(np.uint8) * 255
        bg = ((dist > 1.3 * Lc) & (dist <= 2.3 * Lc)).astype(np.uint8) * 255
        core = cv2.erode(hand, np.ones((3, 3), np.uint8))
        return core, band, bg

    def _flow(self, g0, g1, mask, n):
        p0 = cv2.goodFeaturesToTrack(g0, n, 0.01, 3, mask=mask, blockSize=5)
        if p0 is None or len(p0) < 3:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(11, 11), maxLevel=2)
        ok = st[:, 0] == 1
        return (p1 - p0)[ok, 0] if ok.sum() >= 3 else None

    def step(self, gray: np.ndarray, lm: np.ndarray | None) -> np.ndarray:
        out = np.full(len(COMOTION_NAMES), np.nan, np.float32)
        ok = lm is not None and np.isfinite(lm[:, :2]).all()
        prev = self.hist[0] if len(self.hist) == self.stride else None
        self.hist.append((gray, lm if ok else None))
        self.age = min(self.age + 1, 90.0)
        if ok and prev is not None and prev[1] is not None:
            g0, lm0 = prev
            A, W, Lc = self._crop(g0, lm0)
            if 16 <= W <= 512:
                c0 = cv2.warpAffine(g0, A, (W, W), flags=cv2.INTER_AREA)
                c1 = cv2.warpAffine(gray, A, (W, W), flags=cv2.INTER_AREA)
                lm_c = lm0[:, :2] @ A[:, :2].T + A[:, 2]
                core, band, bg = self._masks(lm_c, W, Lc)
                fh, fb, fg = (self._flow(c0, c1, m, k) for m, k in
                              ((core, 30), (band, 50), (bg, 30)))  # fmt: skip
                if fh is not None and fb is not None:
                    vb = np.median(fg, 0) if fg is not None else np.zeros(2, np.float32)
                    vh = np.median(fh, 0) - vb
                    vn = fb - vb
                    mh = float(np.linalg.norm(vh))
                    out[2], out[3] = len(fb) / 50.0, mh / Lc
                    if mh / Lc >= self.min_move:
                        tol = 0.35 * mh + 0.5
                        out[0] = float((np.linalg.norm(vn - vh, axis=1) < tol).mean())
                        out[1] = float((np.linalg.norm(vn, axis=1) < tol).mean())
                        self.last, self.age = out[0], 0.0
                        self.ema = out[0] if np.isnan(self.ema) else (
                            self.ema_a * out[0] + (1 - self.ema_a) * self.ema)  # fmt: skip
        out[4], out[5], out[6] = self.last, min(self.age / 30.0, 3.0), self.ema
        return out


def comotion_track(frames, lms, **kw) -> np.ndarray:
    """Offline: frames iterable of gray images, lms [N, 21, 2+] (NaN = none) -> [N, 7]."""
    cm = CoMotion(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.stack([cm.step(g, lm if np.isfinite(lm).all() else None)
                         for g, lm in zip(frames, lms, strict=True)])  # fmt: skip


# ------------------------------------------------------------------ 3D motion (calibrated rig)
MOTION3D_NAMES = ["palm_z", "speed", "vz", "rise30"]


def motion3d(Jw: np.ndarray) -> np.ndarray:
    """World hand joints [N, 21, 3] (table frame, z up, m; NaN = none) -> [N, 4]: palm height,
    palm speed and vertical speed (m/s over 3 frames at 30 fps), rise above the lowest palm height
    of the last 30 frames. Same definitions as scripts/grab_features.py."""
    from touch.interact.grab_gate import rolling

    Jw = np.asarray(Jw, np.float64)
    n = len(Jw)
    out = np.full((n, 4), np.nan)
    ok = np.isfinite(Jw).all((1, 2))
    palm = Jw[:, PALM].mean(1)
    out[ok, 0] = palm[ok, 2]
    for i in range(3, n):
        if ok[i] and ok[i - 3]:
            v = (palm[i] - palm[i - 3]) * 10.0
            out[i, 1], out[i, 2] = np.linalg.norm(v), v[2]
    out[:, 3] = out[:, 0] - rolling(out[:, :1], 30, np.nanmin)[:, 0]
    return out.astype(np.float32)


# ------------------------------------------------------------------ fingertip colour (blood)
TIPCOLOR_NAMES = ([f"tip_dL_{f}" for f in FINGERS] + [f"tip_da_{f}" for f in FINGERS]
                  + ["tip_dL_mean", "tip_da_mean", "tip_sat_mean"])  # fmt: skip
_DIP = [3, 7, 11, 15, 19]


def _lab(bgr_mean: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(bgr_mean, np.float32) / 255.0, 0, 1).reshape(-1, 1, 3)
    return cv2.cvtColor(x, cv2.COLOR_BGR2Lab).reshape(-1, 3)


def tip_color(img: np.ndarray, lm: np.ndarray | None) -> np.ndarray:
    """Fingertip colour relative to the palm skin of the same hand (cancels lighting and white
    balance). Pressing pushes blood out of the fingertip: the nail / pad gets paler (L up) and
    less red (a* down). img BGR uint8, lm [21, 2+] px -> [13]: per finger L* and a* of a disc
    between DIP and tip minus the palm's, their means over the 5 fingers, mean tip saturation."""
    out = np.full(len(TIPCOLOR_NAMES), np.nan, np.float32)
    if lm is None or not np.isfinite(lm[:, :2]).all():
        return out
    h, w = img.shape[:2]
    p = lm[:, :2].astype(np.float64)
    L = float(np.linalg.norm(p[9] - p[0]))
    x0, y0 = np.floor(p.min(0) - 0.3 * L).astype(int)
    x1, y1 = np.ceil(p.max(0) + 0.3 * L).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return out
    crop = img[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]

    def disc_mean(c, r):
        m = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r * r
        return crop[m].mean(0) if m.sum() >= 4 else None

    palm = disc_mean(p[[0, 5, 9, 13, 17]].mean(0), 0.25 * L)
    if palm is None:
        return out
    cols = []
    for f in range(5):
        seg = p[TIPS[f]] - p[_DIP[f]]
        c = p[_DIP[f]] + 0.6 * seg
        cols.append(disc_mean(c, max(1.5, 0.4 * float(np.linalg.norm(seg)))))
    if any(c is None for c in cols):
        return out
    lab = _lab(np.stack([*cols, palm]))
    d = lab[:5] - lab[5]
    out[:5], out[5:10] = d[:, 0], d[:, 1]
    out[10], out[11] = d[:, 0].mean(), d[:, 1].mean()
    hsv = cv2.cvtColor(np.clip(np.stack(cols), 0, 255).astype(np.uint8)[None], cv2.COLOR_BGR2HSV)
    out[12] = hsv[0, :, 1].mean() / 255.0
    return out
