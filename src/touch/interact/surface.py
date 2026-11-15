"""Camera image -> surface coordinates (screen, paper sheet, table area) through a plane homography.

A touch happens on the surface plane, so a fingertip *in contact* maps exactly through the plane
homography. For a hovering fingertip it's only approximate (parallax), which is fine for a cursor.

Calibration: 4 image points of the surface corners, ordered top-left, top-right, bottom-right,
bottom-left. Get them from ArUco markers 0-3 at the corners (same layout as HandFlow,
DICT_4X4_50) or by clicking the corners once in the camera image (scripts/touch_screen.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CORNER_IDS = (0, 1, 2, 3)  # ArUco ids at top-left, top-right, bottom-right, bottom-left


class SurfaceMap:
    def __init__(self, H: np.ndarray, size: tuple[float, float]):
        """H [3, 3] image px -> surface units; size (width, height) of the surface (same units)."""
        self.H = np.asarray(H, np.float64)
        assert self.H.shape == (3, 3)
        self.size = (float(size[0]), float(size[1]))

    @classmethod
    def from_corners(cls, corners_px, size: tuple[float, float]) -> SurfaceMap:
        """corners_px [4, 2]: image px of the surface corners TL, TR, BR, BL."""
        src = np.asarray(corners_px, np.float32).reshape(4, 2)
        w, h = size
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        return cls(cv2.getPerspectiveTransform(src, dst), size)

    def map(self, xy) -> np.ndarray:
        """Image px [..., 2] -> surface units [..., 2] (NaN stays NaN)."""
        xy = np.asarray(xy, np.float64)
        p = np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], -1) @ self.H.T
        return p[..., :2] / p[..., 2:3]

    def inside(self, uv, margin: float = 0.0) -> np.ndarray:
        """Surface coordinates [..., 2] -> bool [...]: within the surface rectangle (+ margin)."""
        uv = np.asarray(uv, np.float64)
        w, h = self.size
        with np.errstate(invalid="ignore"):
            return (
                (uv[..., 0] >= -margin)
                & (uv[..., 0] <= w + margin)
                & (uv[..., 1] >= -margin)
                & (uv[..., 1] <= h + margin)
            )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"H": self.H.tolist(), "size": list(self.size)}))

    @classmethod
    def load(cls, path: Path) -> SurfaceMap:
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["H"]), tuple(d["size"]))


def detect_aruco_corners(frame_bgr: np.ndarray, ids=CORNER_IDS) -> np.ndarray | None:
    """Centres of ArUco markers `ids` (DICT_4X4_50) in order -> [4, 2] px, or None if any is
    missing. Marker centres, not the surface's exact corners: put the markers at the corners."""
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters()
    )
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    corners, found, _ = det.detectMarkers(gray)
    if found is None:
        return None
    centre = {int(i): c.reshape(4, 2).mean(0) for i, c in zip(found.ravel(), corners, strict=True)}
    if not all(i in centre for i in ids):
        return None
    return np.stack([centre[i] for i in ids]).astype(np.float32)
