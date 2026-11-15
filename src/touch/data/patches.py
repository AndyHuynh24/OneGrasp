"""Fingertip patches (EgoTouch-style alignment) and hand-relative landmark features.

Patch: rotate so DIP->tip points up, scale by the DIP-tip distance L (EgoTouch: "rotated finger-up,
scaled by DIP-tip, ~4x4 cm patch centred on the fingertip"). The patch side is `scale * L` image
pixels and its centre sits `shift * L` from the tip towards the DIP, so it covers the nail, the pad
edge and the surface just beyond the tip. scale/shift are our choices (config), not EgoTouch's.

Landmark features use only hand-relative geometry: wrist-centred, rotated so
wrist->middle-MCP points up, divided by that distance. No absolute image coordinates.
v2 adds kinematics: image-space displacement of the tips and wrist over the last 1/2/4 frames and
tip acceleration (expressed in the current hand axes, divided by hand size, so translation-,
rotation- and scale-invariant -- motion, never position), finger flexion, tip depth vs MCP, and
jitter: the spread of the 1-frame motion over the last JITTER_N frames per tip and wrist (a tip
braced on a surface should shake less than one in the air, and less than its wrist).
Wrist-centred v1 velocity cancels whole-hand motion, so "the finger came down and stopped" is
only visible in v2.
"""

from __future__ import annotations

import cv2
import numpy as np

from touch.data.pressure import FINGER_DIPS, FINGER_TIPS

PATCH_SIZE = 112
PATCH_SCALE = 2.2
PATCH_SHIFT = 0.3
MIN_L_PX = 3.0
N_LM_FEAT = 21 * 3 + 21 * 3 + 2  # coords, velocity, [has_prev, log hand size ratio]
LAGS = (1, 2, 4)  # v2 displacement lags (frames)
KIN_PTS = (*FINGER_TIPS, 0)  # 5 tips + wrist
KIN_SCALE = 10.0  # per-frame displacement is ~0.01-0.1 hand sizes; bring it near unit range
CHAINS = tuple((0, 4 * k + 1, 4 * k + 2, 4 * k + 3, 4 * k + 4) for k in range(5))
JITTER_N = 4  # 1-frame displacements used for jitter
N_KIN = 2 * len(KIN_PTS) * (len(LAGS) + 1) + len(LAGS) + 1 + 15 + 5 + len(KIN_PTS) + 1
LM_FEAT_DIMS = {"v1": N_LM_FEAT, "v2": N_LM_FEAT + N_KIN}
LM_HISTORY = {"v1": 2, "v2": max(*LAGS, JITTER_N) + 1}  # frames (incl. current) a feature uses


def fingertip_matrix(
    tip_xy: np.ndarray,
    dip_xy: np.ndarray,
    size: int = PATCH_SIZE,
    scale: float = PATCH_SCALE,
    shift: float = PATCH_SHIFT,
) -> np.ndarray:
    """2x3 matrix mapping patch pixels -> image pixels (use with cv2.WARP_INVERSE_MAP)."""
    v = np.asarray(tip_xy, np.float64) - np.asarray(dip_xy, np.float64)
    L = max(float(np.hypot(*v)), MIN_L_PX)
    u = v / L  # finger direction in the image
    b = -u  # patch "down" (+y) points from tip towards DIP
    a = np.array([b[1], -b[0]])  # patch +x, keeps a proper rotation (det > 0)
    s = scale * L / size
    centre = np.asarray(tip_xy, np.float64) - shift * L * u
    half = size / 2.0
    M = np.empty((2, 3))
    M[:, 0] = s * a
    M[:, 1] = s * b
    M[:, 2] = centre - s * half * (a + b)
    return M


# Fingertip segment (tip-DIP) / hand size (wrist -> middle MCP), 90th percentile over EgoPressure
# TRAIN touch frames, overhead view (thumb .. pinky; 2026-11-12, docs/DECISIONS.md). Live, a flat
# straight finger seen from above measures up to 2x this, so patches come out 2x zoomed out.
TIP_RATIO_P90 = (0.34, 0.225, 0.231, 0.229, 0.204)


def clamp_tip_segments(lm_xy: np.ndarray, max_ratio=TIP_RATIO_P90) -> np.ndarray:
    """lm_xy [21, 2+] px -> copy with each DIP moved toward its tip so that
    |tip - DIP| <= max_ratio[f] * hand size. Patch zoom then stays in the training range."""
    out = np.array(lm_xy, np.float64, copy=True)
    hs = float(np.hypot(*(out[9, :2] - out[0, :2])))
    for f, (t, d) in enumerate(zip(FINGER_TIPS, FINGER_DIPS, strict=True)):
        v = out[d, :2] - out[t, :2]
        L, cap = float(np.hypot(*v)), max_ratio[f] * hs
        if L > cap > 0:
            out[d, :2] = out[t, :2] + v * (cap / L)
    return out


def extract_patch(img: np.ndarray, M: np.ndarray, size: int = PATCH_SIZE) -> np.ndarray:
    return cv2.warpAffine(
        img,
        M,
        (size, size),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def finger_patches(img: np.ndarray, lm_xy: np.ndarray, size: int = PATCH_SIZE, **kw) -> np.ndarray:
    """img HxWx3, lm_xy [21, 2] pixels -> [5, size, size, 3] (thumb..pinky)."""
    out = np.empty((5, size, size, 3), np.uint8)
    for f, (t, d) in enumerate(zip(FINGER_TIPS, FINGER_DIPS, strict=True)):
        out[f] = extract_patch(img, fingertip_matrix(lm_xy[t], lm_xy[d], size, **kw), size)
    return out


def _hand_frame(lm: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotation R (wrist->middle MCP maps to (0, -1)) and hand size (px) of lm [21, 3]."""
    v = lm[9, :2] - lm[0, :2]
    size = max(float(np.hypot(*v)), 1e-6)
    up = v / size
    R = np.array([[-up[1], up[0]], [-up[0], -up[1]]])
    return R, size


def normalize_hand(lm: np.ndarray) -> tuple[np.ndarray, float]:
    """lm [21, 3] (x, y px, z = rel. depth * width) -> hand-relative [21, 3] and hand size (px)."""
    R, size = _hand_frame(lm)
    p = lm - lm[0]
    q = np.empty_like(p)
    q[:, :2] = p[:, :2] @ R.T
    q[:, 2] = p[:, 2]
    return (q / size).astype(np.float32), size


def primary_hand(lm: dict) -> np.ndarray:
    """landmarks_<view>.npz contents -> primary (deployable) hand [N, 21, 3] px, NaN if none."""
    prim = lm["primary"]
    hand = np.full((len(prim), 21, 3), np.nan, np.float32)
    rows = np.flatnonzero(prim >= 0)
    hand[rows] = lm["lm_px"][rows, prim[rows]]
    return hand


def _flexion(q: np.ndarray) -> np.ndarray:
    """Cosine of the bend at the 3 inner joints of each finger chain -> [15]."""
    out = np.empty(15, np.float32)
    for k, ch in enumerate(CHAINS):
        seg = np.diff(q[list(ch)], axis=0)
        u = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-6)
        out[3 * k : 3 * k + 3] = (u[:-1] * u[1:]).sum(1)
    return out


def _kinematics(lm_seq: np.ndarray, present: np.ndarray, i: int, q: np.ndarray) -> np.ndarray:
    """v2 extras for frame i (present) -> [N_KIN]. Uses frames i-LM_HISTORY["v2"]+1..i only."""
    R, size = _hand_frame(lm_seq[i])

    def c(j):  # tips + wrist of frame j in frame i's hand axes, hand-size units
        return lm_seq[j][list(KIN_PTS), :2].astype(np.float64) @ R.T * (KIN_SCALE / size)

    def ok(j):
        return j >= 0 and bool(present[j])

    out = np.zeros(N_KIN, np.float32)
    m = 2 * len(KIN_PTS)
    ci = c(i)
    for n, k in enumerate(LAGS):
        if ok(i - k):
            out[n * m : (n + 1) * m] = (ci - c(i - k)).ravel()
    o = len(LAGS) * m
    has_acc = ok(i - 1) and ok(i - 2)
    if has_acc:
        out[o : o + m] = (ci - 2 * c(i - 1) + c(i - 2)).ravel()
    o += m
    out[o : o + len(LAGS) + 1] = [ok(i - k) for k in LAGS] + [has_acc]
    o += len(LAGS) + 1
    out[o : o + 15] = _flexion(q)
    o += 15
    out[o : o + 5] = [q[ch[4], 2] - q[ch[1], 2] for ch in CHAINS]
    o += 5
    if all(ok(i - k) for k in range(JITTER_N + 1)):
        pts = np.stack([c(i - k) for k in range(JITTER_N + 1)])  # [J+1, P, 2]
        v = np.diff(pts, axis=0)  # [J, P, 2] 1-frame motion
        dev = v - v.mean(0)  # remove the steady motion, keep the shake
        out[o : o + len(KIN_PTS)] = np.sqrt((dev**2).sum(-1).mean(0))
        out[o + len(KIN_PTS)] = 1.0
    return out


def landmark_features(lm_seq: np.ndarray, version: str = "v1") -> tuple[np.ndarray, np.ndarray]:
    """lm_seq [N, 21, 3] (NaN where no hand) -> feats [N, LM_FEAT_DIMS[version]] float32,
    present [N] bool.

    Causal: frame i uses frames i-LM_HISTORY[version]+1..i only.
    """
    n = len(lm_seq)
    feats = np.zeros((n, LM_FEAT_DIMS[version]), np.float32)
    present = np.isfinite(lm_seq).all(axis=(1, 2))
    prev, prev_size = None, None
    for i in range(n):
        if not present[i]:
            prev = None
            continue
        q, size = normalize_hand(lm_seq[i])
        feats[i, :63] = q.reshape(-1)
        if prev is not None:
            feats[i, 63:126] = (q - prev).reshape(-1)
            feats[i, 126] = 1.0
            feats[i, 127] = np.log(size / prev_size)
        if version == "v2":
            feats[i, N_LM_FEAT:] = _kinematics(lm_seq, present, i, q)
        prev, prev_size = q, size
    return feats, present


HAND_CROP_SIZE = 160
HAND_CROP_EXPAND = 1.8  # crop side = expand * longest side of the landmark box (hand + object)
MIN_CROP_PX = 32.0


def hand_crop_matrix(
    lm_xy: np.ndarray,
    size: int = HAND_CROP_SIZE,
    expand: float = HAND_CROP_EXPAND,
    flip: bool = False,
) -> np.ndarray:
    """2x3 matrix mapping crop pixels -> image pixels (cv2.WARP_INVERSE_MAP) for an axis-aligned
    square crop around the 21 landmarks. flip=True mirrors the crop horizontally (left hands are
    shown as right hands to the mesh head, as HaMeR does)."""
    p = np.asarray(lm_xy, np.float64)[:, :2]
    lo, hi = p.min(0), p.max(0)
    side = max(expand * float((hi - lo).max()), MIN_CROP_PX)
    c = (lo + hi) / 2.0
    s = side / size
    sx = -s if flip else s
    M = np.array([[sx, 0.0, 0.0], [0.0, s, 0.0]])
    M[:, 2] = c - np.array([sx, s]) * (size / 2.0)
    return M


def hand_crop(img: np.ndarray, lm_xy: np.ndarray, size: int = HAND_CROP_SIZE, **kw) -> np.ndarray:
    return extract_patch(img, hand_crop_matrix(lm_xy, size, **kw), size)
