"""PressureVisionDB (full 15 FPS release): raw-file access and geometry.

Raw layout (written by scripts/download_pvdb.py, mirrors PressureVision's own):
    data/pvdb/raw/<split>/<participant>/<sequence>/camera_<c>/#####.jpg,
    force/#####.pkl, metadata.json
Frames are numbered from 00000; metadata["timesteps"][i] is frame i's time in seconds.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from touch.data.io import load_json, load_numpy_pickle
from touch.data.pressure import SENSEL_SHAPE, pv_counts_to_kpa

FPS = 15.0
VIEW_CAMERA = {"overhead": "1"}  # camera_1: optical axis 1.6 deg from the pad normal, ~85 cm up

# Sensel pad corners in PVDB world coordinates (metres), from PressureVision's
# recording/sequence_reader.py get_sensel_homography(). The pad plane is world y = -0.001.
_SENSEL_W, _SENSEL_H = 0.235, 0.135
_SENSEL_OX, _SENSEL_OY, _SENSEL_Z = 0.016, 0.014, -0.001
_CORNERS_COLROW = np.float32([[185, 0], [185, 105], [0, 105], [0, 0]])
_CORNERS_3D = np.float32(
    [
        [_SENSEL_OY + _SENSEL_H, _SENSEL_Z, _SENSEL_OX + _SENSEL_W],
        [_SENSEL_OY, _SENSEL_Z, _SENSEL_OX + _SENSEL_W],
        [_SENSEL_OY, _SENSEL_Z, _SENSEL_OX],
        [_SENSEL_OY + _SENSEL_H, _SENSEL_Z, _SENSEL_OX],
    ]
)


def list_frames(seq_dir: Path) -> list[int]:
    return sorted(int(p.stem) for p in (Path(seq_dir) / "force").glob("*.pkl"))


def image_path(seq_dir: Path, frame: int, view: str) -> Path:
    return Path(seq_dir) / f"camera_{VIEW_CAMERA[view]}" / f"{frame:05d}.jpg"


def load_force_kpa(seq_dir: Path, frame: int) -> np.ndarray:
    """(105, 185) float32 PV-convention kPa (raw files hold Sensel counts)."""
    counts = load_numpy_pickle(Path(seq_dir) / "force" / f"{frame:05d}.pkl")
    assert counts.shape == SENSEL_SHAPE, counts.shape
    return pv_counts_to_kpa(counts)


def load_metadata(seq_dir: Path) -> dict:
    return load_json(Path(seq_dir) / "metadata.json")


def camera_params(
    meta: dict, view: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """K, distortion (8 coeffs), R, t (metres, world->camera), (width, height)."""
    c = meta["camera_calibrations"][VIEW_CAMERA[view]]
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], np.float64)
    dist = np.array([c[k] for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")], np.float64)
    M = np.array(c["ModelViewMatrix"], np.float64)
    return K, dist, M[:3, :3], M[:3, 3] / 1000.0, (int(c["ImageSizeX"]), int(c["ImageSizeY"]))


def sensel_homography(meta: dict, view: str) -> np.ndarray:
    """3x3 homography pad (col, row) -> undistorted image pixels.

    Same construction as PressureVision (projectPoints without distortion), so apply it to
    undistorted pixel coordinates (see pixels_to_pad_rc).
    """
    K, _, R, t, _ = camera_params(meta, view)
    img, _ = cv2.projectPoints(_CORNERS_3D, cv2.Rodrigues(R)[0], t, K, np.array([]))
    H, _ = cv2.findHomography(_CORNERS_COLROW, img[:, 0, :])
    return H


def pixels_to_pad_rc(uv: np.ndarray, meta: dict, view: str) -> np.ndarray:
    """Image pixels [P, 2] (distorted, as in the JPEG) -> pad grid (row, col) [P, 2].

    Treats points as lying on the pad plane; for the overhead camera the parallax of a fingertip
    ~1 cm above the pad is ~1 mm, well under the 20 mm finger-assignment radius.
    """
    K, dist, _, _, _ = camera_params(meta, view)
    pts = np.asarray(uv, np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    colrow = cv2.perspectiveTransform(und, np.linalg.inv(sensel_homography(meta, view)))
    return colrow.reshape(-1, 2)[:, ::-1]


def pad_rc_to_world_m(rc: np.ndarray) -> np.ndarray:
    """Pad grid (row, col) [..., 2] -> PVDB world points on the pad plane [..., 3] (metres).

    Inverse of the corner layout in _CORNERS_COLROW/_CORNERS_3D: world x runs along rows
    (row 0 at x = OY + H), world z along columns, pad plane at y = _SENSEL_Z.
    """
    rc = np.asarray(rc, np.float64)
    x = _SENSEL_OY + _SENSEL_H * (SENSEL_SHAPE[0] - rc[..., 0]) / SENSEL_SHAPE[0]
    z = _SENSEL_OX + _SENSEL_W * rc[..., 1] / SENSEL_SHAPE[1]
    return np.stack([x, np.full_like(x, _SENSEL_Z), z], axis=-1)
