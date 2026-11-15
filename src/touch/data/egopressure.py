"""EgoPressure pre-release: raw-file access and geometry.

Raw layout (see docs/EGOPRESSURE_FORMAT.md): data/egopressure/raw/<sequence>/
    NNNNNN.cam-{d,4}_rgb.jpeg, NNNNNN.cam-{d,4}_mask.png, NNNNNN.force.bin, NNNNNN.anno.pkl,
    <sequence>.json (camera calibration + participant info), .done (download marker)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from touch.data.io import load_json, load_numpy_pickle
from touch.data.pressure import SENSEL_PITCH_MM, SENSEL_SHAPE

FPS = 30.0
# Our view names -> EgoPressure file camera ids and calibration keys.
VIEW_FILE_CAM = {"ego": "d", "overhead": "4"}
VIEW_CALIB_KEY = {"ego": "0", "overhead": "4"}

# World (x, y) in mm of pad cell (row 0, col 0); world x = +col, world y = +row, 1.25 mm/cell.
# Fitted 2026-10-04 on index_press contact frames of TRAIN participants only (contact centroid vs
# index distal phalanx; scripts/fit_ep_pad_mapping.py), checked on val; see
# docs/EGOPRESSURE_FORMAT.md. Same for every participant.
PAD_ORIGIN_XY_MM = (-115.1, -52.8)


def list_frames(seq_dir: Path) -> list[int]:
    """Frame numbers present in a sequence folder (from the force files), sorted."""
    return sorted(int(p.name.split(".")[0]) for p in Path(seq_dir).glob("*.force.bin"))


def frame_path(seq_dir: Path, frame: int, kind: str) -> Path:
    """kind: 'force.bin', 'anno.pkl', 'cam-d_rgb.jpeg', 'cam-4_mask.png', ..."""
    return Path(seq_dir) / f"{frame:06d}.{kind}"


def image_path(seq_dir: Path, frame: int, view: str) -> Path:
    return frame_path(seq_dir, frame, f"cam-{VIEW_FILE_CAM[view]}_rgb.jpeg")


def load_force(path: Path) -> np.ndarray:
    """(105, 185) float32, PV-convention kPa (already converted by EgoPressure)."""
    data = np.fromfile(path, dtype=np.float32)
    assert data.size == SENSEL_SHAPE[0] * SENSEL_SHAPE[1], (path, data.size)
    return data.reshape(SENSEL_SHAPE)


def load_anno(path: Path) -> dict:
    return load_numpy_pickle(path)


def load_config(seq_dir: Path) -> dict:
    seq_dir = Path(seq_dir)
    return load_json(seq_dir / f"{seq_dir.name}.json")


def camera_intrinsics(config: dict, view: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """K [3, 3], OpenCV distortion [k1, k2, p1, p2, k3, k4, k5, k6], (width, height)."""
    c = config["camera_calibrations"][VIEW_CALIB_KEY[view]]
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], dtype=np.float64)
    dist = np.array([c[k] for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")], np.float64)
    return K, dist, (int(c["ImageSizeX"]), int(c["ImageSizeY"]))


def static_extrinsics_m(config: dict, view: str) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) for a static camera, t converted from mm to metres."""
    M = np.array(config["camera_calibrations"][VIEW_CALIB_KEY[view]]["ModelViewMatrix"], np.float64)
    return M[:3, :3], M[:3, 3] / 1000.0


def ego_extrinsics_m(anno: dict) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) of the head camera for one frame (metres; verified on hand masks)."""
    pose = anno["ego_camera_pose"]
    return np.asarray(pose["R"], np.float64), np.asarray(pose["T"], np.float64)


def project(
    points_m: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> np.ndarray:
    """World points [P, 3] in metres -> pixels [P, 2]. NaN for points behind the camera."""
    pts = np.asarray(points_m, np.float64)
    uv, _ = cv2.projectPoints(pts, cv2.Rodrigues(R)[0], t, K, dist)
    uv = uv.reshape(-1, 2)
    behind = (pts @ R.T + t)[:, 2] <= 0
    uv[behind] = np.nan
    return uv


def world_to_pad_rc(points_m: np.ndarray) -> np.ndarray:
    """World points [..., 3] (metres) -> pad grid (row, col) [..., 2], ignoring height."""
    p = np.asarray(points_m) * 1000.0
    col = (p[..., 0] - PAD_ORIGIN_XY_MM[0]) / SENSEL_PITCH_MM
    row = (p[..., 1] - PAD_ORIGIN_XY_MM[1]) / SENSEL_PITCH_MM
    return np.stack([row, col], axis=-1)


def pad_rc_to_world_m(rc: np.ndarray) -> np.ndarray:
    """Pad grid (row, col) [..., 2] -> world points on the pad plane [..., 3] (metres)."""
    rc = np.asarray(rc, np.float64)
    x = (rc[..., 1] * SENSEL_PITCH_MM + PAD_ORIGIN_XY_MM[0]) / 1000.0
    y = (rc[..., 0] * SENSEL_PITCH_MM + PAD_ORIGIN_XY_MM[1]) / 1000.0
    return np.stack([x, y, np.zeros_like(x)], axis=-1)
