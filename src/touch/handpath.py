"""Predicted 3D hand path in the world / table frame from ONE calibrated RGB camera.

The grasp model gives a wrist-relative MANO hand (camera axes, left hands mirrored to right) and
MediaPipe gives its 2D keypoints. The camera-frame translation is the least-squares fit of the 3D
joints to the 2D keypoints (touch.robot_sim.fit_translation, uses the predicted hand's metric
size), then the camera extrinsics map it to the world. Depth error grows with the hand-size
error (~1 cm of depth per 1 mm of size at 1 m), so the path is smoothed over time (offline).
"""

from __future__ import annotations

import numpy as np

from touch.robot_sim import fit_translation, smooth

PALM = (0, 5, 9, 13, 17)


MP_BONES = ((0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17), (5, 17))  # palm


def mp_depth(J: np.ndarray, lm: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Per frame, the palm's depth (m, along the camera axis) from MediaPipe: each palm bone's
    metric length (the 3D hand's, median over the take: the hand's size, not its depth) over
    its MediaPipe length in pixels, MediaPipe's relative z (same pixel scale as x / y) undoing
    the foreshortening of a turned hand -> median over the bones. Steady while pouring, where
    the per-frame 3D fit can collapse toward the camera (Andy 2026-11-15: "use the depth from
    MediaPipe"). J [n, 21, 3] (any frame, metres), lm [n, 21, 3] px -> [n] (NaN = no hand)."""
    J, lm = np.asarray(J, float), np.asarray(lm, float)
    ok = np.isfinite(J).all((1, 2)) & np.isfinite(lm).all((1, 2))
    out = np.full(len(J), np.nan)
    if not ok.any():
        return out
    a, b = np.array(MP_BONES).T
    size = np.nanmedian(np.linalg.norm(J[ok][:, a] - J[ok][:, b], axis=2), axis=0)  # [bones]
    px = np.linalg.norm(lm[ok][:, b] - lm[ok][:, a], axis=2)  # [n_ok, bones], z included
    out[ok] = np.median(float(K[0][0]) * size / np.maximum(px, 1e-6), axis=1)
    return out


def clean_depth(J: np.ndarray, cam_to_world: np.ndarray, win: int = 7, tol: float = 0.04,
                z_min: float = -0.03, cam_gap: float = 0.10, win_long: int = 20,
                lm: np.ndarray | None = None, K: np.ndarray | None = None,
                ref_tol: float = 0.06) -> np.ndarray:  # fmt: skip
    """Remove depth spikes of a recorded world-hand track (offline, the whole take known).
    The per-frame depth (fit_translation: hand size vs landmark spread) sometimes blows up for
    a frame or two (foreshortened / partly visible hand): the palm jumps tens of cm along its
    camera ray, which an overhead camera turns into height. Per frame the palm's distance to
    the camera: impossible ones (palm below z_min or within cam_gap of the camera's height)
    are dropped, the rest compared with the median over +-win frames; > tol away -> set to
    that median. Each corrected frame slides along its own camera rays (scale about the
    camera centre: pixels unchanged). Frames with no valid neighbour become NaN.
    Then the same again over +-win_long frames (0 = off): a spike longer than win frames moves
    the short median with it (Andy 2026-11-15: the palm 18 cm "up" for 17 frames while pouring,
    the 2D landmarks steady); the long median ignores spikes up to win_long frames.
    lm + K (MediaPipe px, intrinsics): first, frames whose depth is > ref_tol from MediaPipe's
    (mp_depth, scaled to agree with the fit over the take) are moved to MediaPipe's: a turned
    hand made the fit 20-40 cm too close for over a second, too long for any median.
    J [n, 21, 3] world (NaN = none), cam_to_world [3 or 4, 4] -> cleaned copy."""
    J = np.array(J, np.float64)
    c2w = np.asarray(cam_to_world, np.float64)
    o, fwd = c2w[:3, 3], c2w[:3, 2]  # camera centre, viewing direction (world)
    if lm is not None and K is not None:
        J = _to_mp_depth(J, o, fwd, mp_depth(J, lm, K), ref_tol)
    ok = np.isfinite(J).all((1, 2))
    palm = J[:, list(PALM)].mean(1)
    # a fit behind the camera (negative depth) projects to the same pixels as its mirror
    # through the camera centre: mirror it first (scale -1 about the camera)
    behind = ok & ((palm - o) @ fwd < 0)
    J[behind] = o - (J[behind] - o)
    palm = J[:, list(PALM)].mean(1)
    dist = np.linalg.norm(palm - o, axis=1)
    bad = ok & ((palm[:, 2] < z_min) | (palm[:, 2] > o[2] - cam_gap))
    d_ok = np.where(ok & ~bad, dist, np.nan)
    target = dist.copy()
    for i in np.flatnonzero(ok):
        w = d_ok[max(0, i - win) : i + win + 1]
        if not np.isfinite(w).any():
            target[i] = np.nan
            continue
        med = float(np.nanmedian(w))
        if bad[i] or abs(dist[i] - med) > tol:
            target[i] = med
    out = J.copy()
    sc = target / np.where(dist > 0, dist, np.nan)
    out[ok] = o + sc[ok, None, None] * (J[ok] - o)
    out[ok & ~np.isfinite(target)] = np.nan
    if win_long and win_long > win:
        return clean_depth(out, cam_to_world, win_long, tol, z_min, cam_gap, win_long=0)
    return out


def _palm_frames(P: np.ndarray) -> np.ndarray:
    """Points [n, 21, 3] -> palm frames [n, 3, 3] (columns: wrist -> middle knuckle, index ->
    little knuckle made orthogonal, their cross product); NaN where missing."""
    x = P[:, 9] - P[:, 0]
    y = P[:, 5] - P[:, 17]
    with np.errstate(invalid="ignore", divide="ignore"):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        y = y - (y * x).sum(1, keepdims=True) * x
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], -1)


def _rot_angle(R: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def _mean_rot(Rs: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(Rs.mean(0))
    M = U @ Vt
    if np.linalg.det(M) < 0:
        U[:, -1] *= -1
        M = U @ Vt
    return M


def fuse_orientation(J: np.ndarray, lm: np.ndarray, cam_to_world: np.ndarray,
                     agree: float = 15.0, full: float = 30.0) -> np.ndarray:  # fmt: skip
    """The hand's orientation from the 3D hand and MediaPipe combined (Andy 2026-11-15: the 3D
    hand read a ~100 deg pour as 40 deg on some takes; MediaPipe's 2D + relative z read it).
    Per frame both palm frames (MediaPipe's: its pixel x / y / z, a camera-frame direction,
    turned into the world), MediaPipe's corrected by one fixed offset per take (the average
    over the frames where they roughly agree). Within `agree` deg: halfway between them; the
    further apart, the more MediaPipe, all of it from `full` deg. Each frame's hand is turned
    about its palm centre (position unchanged). J [n, 21, 3] world, lm [n, 21, 3] px."""
    J = np.array(J, np.float64)
    lm = np.asarray(lm, np.float64)
    Rj = _palm_frames(J)
    Rm = np.asarray(cam_to_world, np.float64)[:3, :3] @ _palm_frames(lm)
    ok = np.isfinite(Rj).all((1, 2)) & np.isfinite(Rm).all((1, 2))
    if ok.sum() < 10:
        return J
    off = np.einsum("nji,njk->nik", Rm[ok], Rj[ok])  # Rm^T Rj
    Q = _mean_rot(off)
    res = _rot_angle(np.einsum("nij,jk->nik", Rm[ok], Q) @ np.transpose(Rj[ok], (0, 2, 1)))
    if (res < full).sum() >= 10:
        Q = _mean_rot(off[res < full])
    out = J.copy()
    from scipy.spatial.transform import Rotation as Rot

    for i in np.flatnonzero(ok):
        Rmi = Rm[i] @ Q
        D = Rmi @ Rj[i].T  # Rj -> MediaPipe's
        ang = float(_rot_angle(D))
        w = float(np.clip(0.5 + 0.5 * (ang - agree) / (full - agree), 0.5, 1.0))
        Dw = Rot.from_rotvec(w * Rot.from_matrix(D).as_rotvec()).as_matrix()
        c = J[i, list(PALM)].mean(0)
        out[i] = c + (J[i] - c) @ Dw.T
    return out


def _to_mp_depth(J, o, fwd, d_mp, ref_tol: float) -> np.ndarray:
    """Frames whose palm depth is > ref_tol from MediaPipe's -> slid along their camera rays to
    it. MediaPipe's depth is scaled first to agree with the fit (median ratio over the take:
    its z scale / the hand size are approximate)."""
    ok = np.isfinite(J).all((1, 2)) & np.isfinite(d_mp)
    if ok.sum() < 10:
        return J
    zc = (J[:, list(PALM)].mean(1) - o) @ fwd
    good = ok & (zc > 0)
    ref = d_mp * float(np.median(zc[good] / d_mp[good]))
    fix = good & (np.abs(zc - ref) > ref_tol)
    out = J.copy()
    sc = ref[fix] / zc[fix]
    out[fix] = o + sc[:, None, None] * (J[fix] - o)
    return out


def world_hand(joints: np.ndarray, verts: np.ndarray, lm_px: np.ndarray, K: np.ndarray,
               cam_to_world: np.ndarray, left: bool, smooth_k: int = 5) -> dict:  # fmt: skip
    """joints [N, 21, 3], verts [N, 778, 3] (wrist-relative, camera axes, m; NaN = none),
    lm_px [N, 21, >=2] (MediaPipe px), K [3, 3], cam_to_world [3, 4] (x_w = R x_c + t)
    -> {"joints" [N, 21, 3], "verts" [N, 778, 3] world (m), "ok" [N] bool, "cam_t" [N, 3]}."""
    n = len(joints)
    assert joints.shape == (n, 21, 3) and verts.shape == (n, 778, 3) and lm_px.shape[:2] == (n, 21)
    J, V = joints.astype(np.float64).copy(), verts.astype(np.float64).copy()
    if left:  # undo the mirror: the model works on right-hand-looking crops
        J[..., 0] *= -1
        V[..., 0] *= -1
    ok = np.isfinite(J).all((1, 2)) & np.isfinite(V).all((1, 2)) & np.isfinite(lm_px).all((1, 2))
    T = np.full((n, 3), np.nan)
    for i in np.flatnonzero(ok):
        T[i] = fit_translation(J[i], lm_px[i, :, :2], K)
    if smooth_k > 1 and ok.any():
        T = smooth(T, ok, smooth_k)
    R, t = cam_to_world[:, :3], cam_to_world[:, 3]
    Jw = (J + T[:, None]) @ R.T + t
    Vw = (V + T[:, None]) @ R.T + t
    Jw[~ok], Vw[~ok] = np.nan, np.nan
    return {"joints": Jw, "verts": Vw, "ok": ok, "cam_t": T}


def palm_size(J: np.ndarray) -> float:
    """Mean wrist-knuckle + index-pinky knuckle length of a hand [21, 3] (m): a rigid size."""
    return float(np.mean([np.linalg.norm(J[a] - J[b])
                          for a, b in ((0, 5), (0, 9), (0, 13), (0, 17), (5, 17))]))  # fmt: skip


class StableDepth:
    """Causal (live) world hand with a steady depth, for one camera; call per frame in order.

    world_hand's per-frame depth fits all 21 predicted joints to the 2D keypoints: when the
    predicted finger pose disagrees with MediaPipe's, or the predicted hand size changes, the
    whole hand slides along its camera ray (the overlay still lines up, the path jumps). Here:
      1. the hand is rescaled to a FIXED size: the running median palm size (rigid palm);
      2. depth from the PALM only (wrist + 4 knuckles: no finger curl);
      3. a jump (> jump_tol relative to the median of the last `win` accepted depths) is
         clamped to that median; after `reaccept` frames in a row it is accepted (a real move);
      4. One Euro on the depth (min_cutoff Hz, beta).
    The hand then moves along its camera rays to the filtered depth (scale about the camera
    centre: pixels unchanged). -> the same dict as world_hand for one frame."""

    def __init__(self, K, cam_to_world, win: int = 9, jump_tol: float = 0.06,
                 reaccept: int = 6, min_cutoff: float = 1.5, beta: float = 2.0,
                 size_window: int = 300, fit: str = "all") -> None:  # fmt: skip
        from collections import deque

        from touch.interact.one_euro import OneEuro

        assert fit in ("all", "palm")
        self.fit = fit
        self.K, self.c2w = np.asarray(K, np.float64), np.asarray(cam_to_world, np.float64)
        self.win, self.jump_tol, self.reaccept = win, jump_tol, reaccept
        self.sizes: deque = deque(maxlen=size_window)
        self.depths: deque = deque(maxlen=win)
        self.euro = OneEuro(min_cutoff=min_cutoff, beta=beta, d_cutoff=1.0)
        self.n_out = 0

    def reset(self) -> None:
        """New take: forget the depth history (the hand size is kept: same person)."""
        self.depths.clear()
        self.euro.reset()
        self.n_out = 0

    def step(self, joints, verts, lm_px, left: bool, t: float) -> dict:
        """joints [21, 3], verts [778, 3] (wrist-relative, camera axes, m), lm_px [21, >=2],
        t (s) -> {"joints" [21, 3], "verts" [778, 3] world, "ok", "cam_t" [3], "depth_raw"}."""
        J, V = np.array(joints, np.float64), np.array(verts, np.float64)
        uv = np.asarray(lm_px, np.float64)[:, :2]
        nan = {"joints": np.full((21, 3), np.nan), "verts": np.full((778, 3), np.nan),
               "ok": False, "cam_t": np.full(3, np.nan), "depth_raw": np.nan}  # fmt: skip
        if not (np.isfinite(J).all() and np.isfinite(V).all() and np.isfinite(uv).all()):
            return nan
        if left:
            J[:, 0] *= -1
            V[:, 0] *= -1
        s = palm_size(J)
        if s > 1e-3:
            self.sizes.append(s)
        k = float(np.median(self.sizes)) / s if self.sizes and s > 1e-3 else 1.0
        J, V = J * k, V * k
        P = list(PALM)
        F = P if self.fit == "palm" else list(range(21))
        T = fit_translation(J[F], uv[F], self.K)
        d = float(np.linalg.norm(J[P].mean(0) + T))
        if not np.isfinite(d) or J[P].mean(0)[2] + T[2] <= 0.05:
            return nan
        use = d
        if len(self.depths) >= 3:
            med = float(np.median(self.depths))
            if abs(d - med) > self.jump_tol * med:
                self.n_out += 1
                if self.n_out < self.reaccept:
                    use = med  # a one-off jump: hold the recent depth
                else:
                    self.depths.clear()  # the hand really moved: follow it
                    self.n_out = 0
            else:
                self.n_out = 0
        if use == d:
            self.depths.append(d)
        d_f = float(self.euro(np.array([use]), t)[0])
        f = d_f / d
        Jc, Vc = (J + T) * f, (V + T) * f  # along the camera rays: same pixels
        R, tw = self.c2w[:3, :3], self.c2w[:3, 3]
        return {"joints": Jc @ R.T + tw, "verts": Vc @ R.T + tw, "ok": True,
                "cam_t": T * f, "depth_raw": d}  # fmt: skip
