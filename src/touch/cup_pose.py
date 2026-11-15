"""Pose of the demo cylinder (h x d, e.g. CYL_H x CYL_D) resting on the table, from one
camera frame: SAM 2.1 (Hiera-L, Meta's official facebookresearch/sam2) segments it from a
point prompt, then the cylinder's silhouette, projected through the calibrated camera, is
fitted to the mask: standing (position) or lying on its side (position + axis heading mod
180 deg). Model-based, so it handles the oblique view (no top-down assumption).

The silhouette of a convex body under perspective is convex: it is the filled convex hull of
the projected rim points, which makes one evaluation cheap (coarse-to-fine grid search)."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

SAM_CKPT = Path("third_party/sam/sam2.1_hiera_large.pt")
SAM_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
FIT_W = 480  # px: the mask is fitted at this width
MIN_IOU = 0.7  # below: the fit is not trusted


class Segmenter:
    """SAM 2.1 Hiera-L image predictor (loaded once, ~0.9 GB)."""

    def __init__(self, device: str | None = None, ckpt: Path = SAM_CKPT):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.pred = SAM2ImagePredictor(build_sam2(SAM_CFG, str(ckpt), device=device))

    def masks(self, bgr: np.ndarray, pt_xy) -> list[tuple[np.ndarray, float]]:
        """Image + one positive point (px) -> SAM's 3 candidate masks [(bool [H, W], score)]."""
        import torch

        with torch.inference_mode():
            self.pred.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            m, s, _ = self.pred.predict(point_coords=np.array([pt_xy], np.float32),
                                        point_labels=np.array([1]),
                                        multimask_output=True)  # fmt: skip
        return [(m[i] > 0, float(s[i])) for i in range(len(s))]


def scale_K(K, image_size, shape) -> np.ndarray:
    """K for image_size (w, h) -> K for an image of shape (H, W[, 3])."""
    K = np.array(K, np.float64)
    sx, sy = shape[1] / image_size[0], shape[0] / image_size[1]
    return np.diag([sx, sy, 1.0]) @ K


def rim_points(h: float, d: float, n: int = 40) -> np.ndarray:
    """Both rims of a cylinder centred at the origin, axis z -> [2n, 3]."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c = np.c_[d / 2 * np.cos(a), d / 2 * np.sin(a)]
    return np.r_[np.c_[c, np.full(n, -h / 2)], np.c_[c, np.full(n, h / 2)]]


def cup_pose(xy, lying: bool, yaw: float, h: float, d: float):
    """-> (R [3, 3] cup -> world (its axis = column 2), centre [3]). Lying: axis horizontal at
    heading yaw (mod 180 deg), centre d/2 above the table; standing: centre h/2 up."""
    if lying:
        R = Rot.from_euler("z", yaw).as_matrix() @ Rot.from_euler("y", np.pi / 2).as_matrix()
        return R, np.r_[xy, d / 2]
    return np.eye(3), np.r_[xy, h / 2]


def project(K, c2w, pw: np.ndarray) -> np.ndarray:
    """World points [n, 3] -> px [n, 2] (c2w [3, 4]: x_w = R x_c + t)."""
    R, t = c2w[:, :3], c2w[:, 3]
    pc = (pw - t) @ R
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3]


def ray_to_plane(K, c2w, uv, z: float) -> np.ndarray:
    """Pixel -> the point where its ray meets the horizontal plane at height z."""
    R, t = c2w[:, :3], c2w[:, 3]
    r = R @ np.linalg.solve(K, np.r_[uv, 1.0])
    s = (z - t[2]) / r[2]
    return t + s * r


def silhouette(K, c2w, rim, R, c, shape) -> np.ndarray:
    uv = project(K, c2w, rim @ R.T + c)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
    return m.astype(bool)


def _iou(a: np.ndarray, b: np.ndarray, valid=None) -> float:
    if valid is None:
        u = np.count_nonzero(a | b)
        return np.count_nonzero(a & b) / u if u else 0.0
    u = np.count_nonzero((a | b) & valid)
    return np.count_nonzero(a & b & valid) / u if u else 0.0


def fit_mask(mask: np.ndarray, K, c2w, h: float, d: float, ignore=None,
             allow_lying: bool = True) -> dict:  # fmt: skip
    """Mask (bool [H, W]; K for this size) -> the best standing and lying cylinder poses
    (coarse-to-fine grid over x, y [, yaw]) -> {"lying", "xy", "yaw", "iou", "iou_other"}.
    ignore (bool [H, W]): pixels left out of the IoU (e.g. covered by the hand)."""
    valid = None if ignore is None else ~ignore
    c2w = np.asarray(c2w, np.float64)[:3]
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return {"lying": False, "xy": None, "yaw": 0.0, "iou": 0.0, "iou_other": 0.0}
    rim = rim_points(h, d)
    best = {}
    best[True] = (-1.0, None, 0.0)  # (allow_lying False: never chosen)
    for lying in (False, True) if allow_lying else (False,):
        z0 = d / 2 if lying else h / 2
        xy = ray_to_plane(K, c2w, np.array([xs.mean(), ys.mean()]), z0)[:2]
        yaws = np.radians(np.arange(0, 180, 10)) if lying else np.array([0.0])
        top = (-1.0, xy, 0.0)
        for step, span, ystep in ((0.01, 0.05, 10.0), (0.003, 0.012, 3.0), (0.001, 0.003, 1.0)):
            offs = np.arange(-span, span + 1e-9, step)
            for yw in yaws:
                for dx in offs:
                    for dy in offs:
                        p = top[1] + [dx, dy] if step < 0.01 else xy + [dx, dy]
                        R, c = cup_pose(p, lying, yw, h, d)
                        s = _iou(silhouette(K, c2w, rim, R, c, mask.shape), mask, valid)
                        if s > top[0]:
                            top = (s, p, yw)
            if lying:
                yaws = top[2] + np.radians(np.arange(-2, 3) * ystep / 2)
            top = (top[0], np.asarray(top[1]), top[2])
        best[lying] = top
    lying = best[True][0] > best[False][0]
    s, xy, yw = best[lying]
    return {"lying": bool(lying), "xy": np.asarray(xy, float), "yaw": float(yw % np.pi) if lying
            else 0.0, "iou": float(s), "iou_other": float(best[not lying][0])}  # fmt: skip


def estimate(seg: Segmenter, bgr: np.ndarray, K, image_size, c2w, h: float, d: float,
             hint_xy, ignore=None, allow_lying: bool = True) -> dict:  # fmt: skip
    """Frame (any size; K is for image_size) + a rough world xy of the cylinder (prompt) ->
    fit_mask result (+ "mask_score", "prompt_px", "ok"), the best of SAM's 3 masks by fit
    IoU. ignore (bool, frame size): pixels left out of the fit (the hand)."""
    c2w = np.asarray(c2w, np.float64)[:3]
    Ki = scale_K(K, image_size, bgr.shape)
    pt = project(Ki, c2w, np.r_[hint_xy, d / 2][None])[0]
    cands = seg.masks(bgr, pt)
    f = FIT_W / bgr.shape[1]
    small_shape = (int(round(bgr.shape[0] * f)), FIT_W)
    Ks = scale_K(K, image_size, small_shape)
    ign = None
    if ignore is not None:
        ign = cv2.resize(ignore.astype(np.uint8), (small_shape[1], small_shape[0]),
                         interpolation=cv2.INTER_NEAREST).astype(bool)  # fmt: skip
    out = None
    for m, sc in cands:
        ms = cv2.resize(m.astype(np.uint8), (small_shape[1], small_shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)  # fmt: skip
        if ms.mean() > 0.25:  # the table / background, not the cylinder
            continue
        if ign is not None:
            ms &= ~ign
        r = fit_mask(ms, Ks, c2w, h, d, ign, allow_lying)
        r.update({"mask_score": sc, "prompt_px": pt.tolist(), "mask": m})
        if out is None or r["iou"] > out["iou"]:
            out = r
    if out is None:
        out = {"lying": False, "xy": None, "yaw": 0.0, "iou": 0.0, "iou_other": 0.0,
               "mask_score": 0.0, "prompt_px": pt.tolist(), "mask": None}  # fmt: skip
    out["ok"] = out["xy"] is not None and out["iou"] >= MIN_IOU
    return out


def hand_mask(lm_px, shape, image_size, pad: float = 0.04) -> np.ndarray | None:
    """MediaPipe landmarks (px of image_size) -> the hand's convex hull, dilated by pad x the
    image width, as a bool mask of shape (H, W) (None: no hand)."""
    if lm_px is None or not np.isfinite(lm_px).all():
        return None
    f = shape[1] / image_size[0]
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(lm_px[:, :2] * f).astype(np.int32)), 1)
    r = max(1, int(pad * shape[1]))
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0


def pick_frame(lms, before: int, hint_px, r_px: float, shape, image_size) -> tuple[int, float]:
    """The last frame before `before` whose hand (landmarks, full-res px) stays clear of the
    cylinder (a disc of r_px around hint_px, px of an image of shape) -> (index, overlap);
    none clear: the least covered one (overlap = its covered share of the disc)."""
    disc = np.zeros(shape[:2], np.uint8)
    cv2.circle(disc, tuple(int(v) for v in np.round(hint_px)), int(r_px), 1, -1)
    disc = disc > 0
    best = (max(0, before - 1), 1.0)
    for i in range(before - 1, -1, -1):
        hm = hand_mask(lms[i], shape, image_size)
        ov = 0.0 if hm is None else float((hm & disc).sum() / max(disc.sum(), 1))
        if ov == 0.0:
            return i, 0.0
        if ov < best[1]:
            best = (i, ov)
    return best


def draw(bgr: np.ndarray, K, image_size, c2w, est: dict, h: float, d: float) -> np.ndarray:
    """The fitted cylinder's outline (both rims + side lines) and the mask over the frame."""
    img = bgr.copy()
    Ki = scale_K(K, image_size, img.shape)
    c2w = np.asarray(c2w, np.float64)[:3]
    if est.get("mask") is not None:
        m = est["mask"]
        if m.shape != img.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0])) > 0
        img[m] = (0.6 * img[m] + 0.4 * np.array([255, 120, 0])).astype(np.uint8)
    if est.get("xy") is not None:
        R, c = cup_pose(est["xy"], est["lying"], est["yaw"], h, d)
        rim = rim_points(h, d, 48)
        uv = np.round(project(Ki, c2w, rim @ R.T + c)).astype(np.int32)
        col = (0, 255, 0) if est.get("ok") else (0, 0, 255)
        for half in (uv[:48], uv[48:]):
            cv2.polylines(img, [half], True, col, 2, cv2.LINE_AA)
        ax = project(Ki, c2w, np.array([c, c + R[:, 2] * h / 2]))
        cv2.arrowedLine(img, tuple(np.round(ax[0]).astype(int)), tuple(np.round(ax[1]).astype(int)),
                        col, 2, cv2.LINE_AA, tipLength=0.2)  # fmt: skip
        kind = f"lying {math.degrees(est['yaw']):.0f} deg" if est["lying"] else "standing"
        cv2.putText(img, f"{kind}  xy ({est['xy'][0]:.3f}, {est['xy'][1]:.3f}) m  "
                    f"IoU {est['iou']:.2f}", (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 2, cv2.LINE_AA)  # fmt: skip
    return img


def ray_axis_point(o, r, c0, axis, half: float):
    """Camera ray (origin o, unit direction r) vs the cylinder's axis segment (centre c0,
    unit axis, +-half) -> (point on the ray closest to the axis, axis parameter t in
    [-half, half])."""
    w = o - c0
    b, d, e = r @ axis, r @ w, axis @ w
    den = 1.0 - b * b
    t = (e - b * d) / den if den > 1e-9 else 0.0
    t = float(np.clip(t, -half, half))
    q = c0 + t * axis  # the axis point; the ray's point nearest to it
    s = max(float(r @ (q - o)), 0.0)
    return o + s * r, t


def depth_scale(lm_px, J_world, K, c2w, c0, axis, h: float, tips=(4, 8)) -> dict | None:
    """Monocular hand depth fixed with the measured cylinder (metric via the ArUco-calibrated
    camera). While the hand holds the cylinder, the pixel between the thumb and index tips
    looks at the cylinder: the camera ray through it meets the cylinder's axis where the hand
    grips. Scale = that depth / the hand's estimated depth; the hand's size / depth
    ambiguity is one factor per take: x_world' = cam + scale x (x_world - cam) keeps every
    pixel. lm_px [n, 21, >=2] (full-res px of K), J_world [n, 21, 3] over the grab frames.
    -> {"scale" (median), "t" (grip along the axis from the centre, m), "n"} or None."""
    c2w = np.asarray(c2w, np.float64)[:3]
    R, o = c2w[:, :3], c2w[:, 3]
    Kinv = np.linalg.inv(np.asarray(K, np.float64))
    sc, ts = [], []
    for lm, J in zip(lm_px, J_world, strict=True):
        if not (np.isfinite(lm[list(tips), :2]).all() and np.isfinite(J[list(tips)]).all()):
            continue
        uv = lm[list(tips), :2].mean(0)
        r = R @ (Kinv @ np.r_[uv, 1.0])
        r /= np.linalg.norm(r)
        p, t = ray_axis_point(o, r, np.asarray(c0, float), np.asarray(axis, float), h / 2)
        est = np.linalg.norm(J[list(tips)].mean(0) - o)
        if est > 1e-3:
            sc.append(np.linalg.norm(p - o) / est)
            ts.append(t)
    if not sc:
        return None
    return {"scale": float(np.median(sc)), "t": float(np.median(ts)), "n": len(sc)}


def rescale_about_camera(J_world: np.ndarray, c2w, scale: float) -> np.ndarray:
    """World points -> the same pixels at scale x their distance from the camera."""
    o = np.asarray(c2w, np.float64)[:3, 3]
    return o + scale * (np.asarray(J_world, np.float64) - o)
