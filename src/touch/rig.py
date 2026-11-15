"""Camera calibration from 2 ArUco markers lying flat on the table (scripts/make_markers.py:
DICT_4X4_50, ids 0 and 1, known side length). No checkerboard.

Intrinsics: principal point = image centre, square pixels, no lens distortion. Focal length:
initial guess from the markers' plane-to-image homographies (Zhang's constraints with only f
unknown: needs an oblique view; a camera looking straight down cannot see f) and the two markers'
coplanarity, then one joint least-squares fit of f, the table pose and marker 1's in-plane pose
(x, y, yaw) to all 8 corners. With one marker or no usable estimate: a guess from the horizontal
field of view (then only the pose is fitted).

Table frame = marker 0's frame: origin at its centre, x along its top edge (left -> right as
printed), y toward its top edge, z up out of the table (OpenCV ArUco marker convention).
"""

from __future__ import annotations

import cv2
import numpy as np

DICT = cv2.aruco.DICT_4X4_50
IDS = (0, 1)


def marker_object_points(size_m: float) -> np.ndarray:
    """[4, 3] corners in the marker frame, OpenCV order (TL, TR, BR, BL)."""
    h = size_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def detect(gray: np.ndarray) -> dict[int, np.ndarray]:
    """id -> [4, 2] pixel corners (sub-pixel refined), for ids 0 and 1."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT))
    corners, ids, _ = det.detectMarkers(gray)
    out = {}
    if ids is None:
        return out
    for c, i in zip(corners, ids.ravel(), strict=True):
        if int(i) in IDS:
            c = c.reshape(4, 1, 2).astype(np.float32)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
            cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), crit)
            out[int(i)] = c.reshape(4, 2).astype(np.float64)
    return out


def K_of(f: float, w: int, h: int) -> np.ndarray:
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def focal_from_homographies(Hs: list[np.ndarray], w: int, h: int) -> float | None:
    """Least-squares f from Zhang's two constraints per homography, principal point at the
    image centre. Pixels are normalised by the image width for conditioning. None if the views
    are too frontal (no information) or the solution is not positive."""
    s = float(w)
    T = np.array([[1 / s, 0, -w / 2 / s], [0, 1 / s, -h / 2 / s], [0, 0, 1]])
    A, b = [], []
    for H in Hs:
        Hn = T @ H
        h1, h2 = Hn[:, 0], Hn[:, 1]
        # h1' W h2 = 0 and h1' W h1 = h2' W h2 with W = diag(u, u, 1), u = 1 / f_n^2
        A += [h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2]
        b += [-h1[2] * h2[2], -(h1[2] ** 2 - h2[2] ** 2)]
    A, b = np.asarray(A), np.asarray(b)
    if np.abs(A).max() < 1e-12:
        return None
    u = float(A @ b / (A @ A))
    if u <= 0:
        return None
    return float(s / np.sqrt(u))


def marker_pose(corners: np.ndarray, size_m: float, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marker -> camera rotation [3, 3] and translation [3] (IPPE square)."""
    ok, rv, tv = cv2.solvePnP(marker_object_points(size_m), corners, K, None,
                              flags=cv2.SOLVEPNP_IPPE_SQUARE)  # fmt: skip
    assert ok
    return cv2.Rodrigues(rv)[0], tv.ravel()


def coplanarity(c0: np.ndarray, c1: np.ndarray, size_m: float, K: np.ndarray) -> float:
    """Marker 1's distance from marker 0's plane: normal angle (rad) + |height| (m) / size."""
    R0, t0 = marker_pose(c0, size_m, K)
    R1, t1 = marker_pose(c1, size_m, K)
    ang = float(np.arccos(np.clip(R0[:, 2] @ R1[:, 2], -1, 1)))
    dz = float(abs(R0[:, 2] @ (t1 - t0)))
    return ang + dz / size_m


def calibrate(corners: dict[int, np.ndarray], size_m: float, w: int, h: int,
              hfov_deg: float = 70.0, f_fixed: float | None = None) -> dict:  # fmt: skip
    """corners: id -> [4, 2] px (ids 0 and 1, e.g. averaged over frames).
    f_fixed: the camera's focal length (px) measured before (results/rig/intrinsics.json):
    only the pose is fitted (well conditioned from one marker; f from one marker is not).
    -> {"K", "cam_to_world" [3, 4] (camera -> table frame), "f_source", "reproj_px", ...}."""
    assert 0 in corners, "marker 0 (table origin) not visible"
    obj = marker_object_points(size_m)
    Hs = [cv2.findHomography(obj[:, :2], corners[i])[0] for i in sorted(corners)]
    f_guess = w / 2 / np.tan(np.radians(hfov_deg) / 2)
    if f_fixed is not None:
        f, src = float(f_fixed), "fixed"
    else:
        f, src = focal_from_homographies(Hs, w, h), "homography"
        if f is None or not (0.3 * f_guess < f < 3 * f_guess):
            f, src = f_guess, "hfov"
    if 1 in corners and src != "fixed":  # refine: the two markers must be coplanar
        fs = f * np.exp(np.linspace(-0.4, 0.4, 161))
        cost = [coplanarity(corners[0], corners[1], size_m, K_of(x, w, h)) for x in fs]
        k = int(np.argmin(cost))
        if 0 < k < len(fs) - 1:
            f = float(fs[k])
    if src not in ("hfov", "fixed") and 1 in corners:
        src = "joint fit (homography + coplanarity init)"
    K = K_of(f, w, h)
    R0, t0 = marker_pose(corners[0], size_m, K)  # table frame = marker 0
    rv, tv = cv2.Rodrigues(R0)[0].ravel(), t0
    if 1 in corners:
        f, rv, tv, W = _joint_fit(corners, size_m, f, rv, tv, w, h,
                                  fit_f=src not in ("hfov", "fixed"))  # fmt: skip
        P = np.concatenate([corners[0], corners[1]])
    else:
        W, P = obj, corners[0]
    K = K_of(f, w, h)
    R = cv2.Rodrigues(rv)[0]  # table -> camera
    reproj = cv2.projectPoints(W, rv, tv, K, None)[0].reshape(-1, 2)
    c2w = np.concatenate([R.T, (-R.T @ tv).reshape(3, 1)], 1)
    return {"K": K, "cam_to_world": c2w, "f": f, "f_source": src, "image_size": [w, h],
            "marker_size_m": size_m, "markers_seen": sorted(corners),
            "reproj_px": float(np.linalg.norm(reproj - P, axis=1).mean()),
            "camera_height_m": float(c2w[2, 3])}  # fmt: skip


def _joint_fit(corners, size_m, f, rv, tv, w, h, fit_f=True):
    """Least squares over [log f], table->camera pose (6) and marker 1 in the table plane
    (x, y, yaw), on the 8 corner reprojections. Returns f, rvec, tvec, table points [8, 3]."""
    from scipy.optimize import least_squares

    obj = marker_object_points(size_m)
    K = K_of(f, w, h)
    R0 = cv2.Rodrigues(rv)[0]
    R1, t1 = marker_pose(corners[1], size_m, K)
    c1 = (t1 - tv) @ R0  # marker 1 centre in the table frame
    x1 = R0.T @ R1[:, 0]
    m1 = [c1[0], c1[1], np.arctan2(x1[1], x1[0])]
    P = np.concatenate([corners[0], corners[1]])

    def unpack(x):
        ff = float(np.exp(x[0])) if fit_f else f
        c, s = np.cos(x[9]), np.sin(x[9])
        pts1 = obj[:, :2] @ np.array([[c, s], [-s, c]]) + x[7:9]
        W = np.concatenate([obj, np.c_[pts1, np.zeros(4)]])
        return ff, x[1:4], x[4:7], W

    def resid(x):
        ff, r, t, W = unpack(x)
        return (cv2.projectPoints(W, r, t, K_of(ff, w, h), None)[0].reshape(-1, 2) - P).ravel()

    x0 = np.r_[np.log(f), rv, tv, m1]
    x = least_squares(resid, x0, x_scale="jac").x
    ff, r, t, W = unpack(x)
    return ff, np.asarray(r), np.asarray(t), W


# ------------------------------------------------------------------ view quality (calib_rig)
MIN_SIDE = 90  # px at 1920 wide: marker 0's side; smaller -> noisy corners, poor pose
MIN_MARGIN = 0.04  # of the image width: marker corners away from the border (lens edge)
MAX_JITTER = 0.5  # px: corner std over the frames (camera / table shaking)
MAX_REPROJ = 1.0  # px
MIN_TILT_F = 20.0  # deg from straight down: needed to MEASURE the focal length
MIN_SEP_F = 0.15  # m between the two markers' centres when measuring the focal length
MAX_F_SPREAD = 0.02  # relative std of the per-frame focal estimates


def view_quality(frames: list[dict], cal: dict, w: int, h: int, size_m: float,
                 measure_f: bool) -> list[dict]:  # fmt: skip
    """Checks of a calibration view -> [{"name", "ok", "value", "tip"}] (the tip says how
    to fix a failed check). frames: the per-frame detections (id -> [4, 2] px) used;
    measure_f: the focal length is being measured (needs both markers and a tilted view)."""
    c0 = np.array([f[0] for f in frames if 0 in f])
    k = w / 1920
    out = []
    side = float(np.mean(np.linalg.norm(c0 - np.roll(c0, 1, axis=1), axis=2)))
    out.append({"name": "marker 0 size", "ok": side >= MIN_SIDE * k, "value": f"{side:.0f} px",
                "tip": "marker 0 looks too small: move the camera closer / lower, or print "
                       "a bigger marker"})  # fmt: skip
    m = float(min(c0[..., 0].min(), c0[..., 1].min(), w - c0[..., 0].max(),
                  h - c0[..., 1].max()))  # fmt: skip
    out.append({"name": "away from edge", "ok": m >= MIN_MARGIN * w, "value": f"{m:.0f} px",
                "tip": "marker 0 is at the image edge: move it (or the camera) so it sits "
                       "further inside the view"})  # fmt: skip
    jit = float(np.std(c0, axis=0).mean()) if len(c0) > 1 else 0.0
    out.append(
        {
            "name": "steady",
            "ok": jit <= MAX_JITTER,
            "value": f"{jit:.2f} px",
            "tip": "the image is moving: fix the camera mount, don't touch the table",
        }
    )
    out.append({"name": "fit error", "ok": cal["reproj_px"] <= MAX_REPROJ,
                "value": f"{cal['reproj_px']:.2f} px",
                "tip": "the corners don't fit a flat square: flatten the paper, check "
                       "--size-mm (measure the black square)"})  # fmt: skip
    ax = np.asarray(cal["cam_to_world"])[:, 2]  # optical axis in the table frame
    tilt = float(np.degrees(np.arccos(np.clip(-ax[2], -1, 1))))
    if measure_f:
        out.append({"name": "tilt", "ok": tilt >= MIN_TILT_F, "value": f"{tilt:.0f} deg",
                    "tip": f"tilt the camera >= {MIN_TILT_F:.0f} deg from straight down "
                           "(straight down cannot measure the focal length)"})  # fmt: skip
        both = [f for f in frames if 0 in f and 1 in f]
        sep = 0.0
        if both:
            K = np.asarray(cal["K"])
            R0, t0 = marker_pose(np.mean([f[0] for f in both], 0), size_m, K)
            _, t1 = marker_pose(np.mean([f[1] for f in both], 0), size_m, K)
            sep = float(np.linalg.norm(t1 - t0))
        out.append({"name": "marker 1", "ok": len(both) >= 0.8 * len(frames) and sep >= MIN_SEP_F,
                    "value": f"{sep * 100:.0f} cm" if both else "not seen",
                    "tip": f"put marker 1 flat, >= {MIN_SEP_F * 100:.0f} cm from marker 0, "
                           "both fully in view"})  # fmt: skip
        obj = marker_object_points(size_m)[:, :2]
        fs = []
        for f in both:
            Hs = [cv2.findHomography(obj, f[i])[0] for i in (0, 1)]
            v = focal_from_homographies(Hs, w, h)
            if v is not None:
                fs.append(v)
        spread = float(np.std(fs) / np.mean(fs)) if len(fs) > 2 else 1.0
        out.append({"name": "focal steady", "ok": spread <= MAX_F_SPREAD,
                    "value": f"{spread * 100:.1f} %",
                    "tip": "the focal length is not measurable from this view: tilt more, "
                           "spread the markers further apart"})  # fmt: skip
    else:
        out.append({"name": "tilt", "ok": True, "value": f"{tilt:.0f} deg", "tip": ""})
    return out


def expected_corners(cal: dict, size_m: float) -> np.ndarray:
    """Marker 0's corners [4, 2] px where the calibration expects them (a moved camera
    shows them elsewhere)."""
    c2w = np.asarray(cal["cam_to_world"])
    R, t = c2w[:, :3], c2w[:, 3]
    pc = (marker_object_points(size_m) - t) @ R
    K = np.asarray(cal["K"])
    return pc[:, :2] / pc[:, 2:3] * [K[0][0], K[1][1]] + [K[0][2], K[1][2]]
