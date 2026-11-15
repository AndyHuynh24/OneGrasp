import cv2
import numpy as np
import pytest

from touch.rig import DICT, calibrate, detect

W, H, F, SIZE = 1280, 720, 900.0, 0.10


def look_at(eye, target):
    """table -> camera rotation (OpenCV camera: x right, y down, z forward)."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def render(eye, markers, f=F):
    """markers: list of (id, centre xy, yaw rad) on the table (z = 0)."""
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    R = look_at(np.asarray(eye, float), [0.15, 0.0, 0.0])
    t = -R @ np.asarray(eye, float)
    img = np.full((H, W), 200, np.uint8)  # light table
    d = cv2.aruco.getPredefinedDictionary(DICT)
    n = 400
    for mid, (cx, cy), yaw in markers:
        m = cv2.aruco.generateImageMarker(d, mid, n, borderBits=1)
        m = cv2.copyMakeBorder(m, n // 4, n // 4, n // 4, n // 4, cv2.BORDER_CONSTANT, value=255)
        N = m.shape[0]
        c, s = np.cos(yaw), np.sin(yaw)
        scale = SIZE / n
        # marker pixel (u, v) -> table (x, y): centre, flip v, rotate by yaw
        A = np.array([[scale * c, scale * s, 0], [scale * s, -scale * c, 0], [0, 0, 1.0]])
        A[:2, 2] = [cx, cy] - A[:2, :2] @ [N / 2, N / 2]
        Hm = K @ np.c_[R[:, :2], t] @ A
        warped = cv2.warpPerspective(m, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(m, 255), Hm, (W, H), borderValue=0) > 0
        img[mask] = warped[mask]
    return img, K, R, t


@pytest.mark.parametrize("eye", [(-0.35, -0.45, 0.45), (0.15, -0.6, 0.35)])
def test_two_markers_recover_focal_and_pose(eye):
    img, K, R, t = render(eye, [(0, (0.0, 0.0), 0.0), (1, (0.32, 0.06), 0.4)])
    corners = detect(img)
    assert set(corners) == {0, 1}
    cal = calibrate(corners, SIZE, W, H)
    assert cal["K"].shape == (3, 3) and cal["cam_to_world"].shape == (3, 4)
    assert abs(cal["f"] - F) / F < 0.03, cal["f"]
    assert np.linalg.norm(cal["cam_to_world"][:, 3] - eye) < 0.015
    assert cal["reproj_px"] < 1.0


def test_single_marker_falls_back_or_estimates():
    img, *_ = render((-0.35, -0.45, 0.45), [(0, (0.0, 0.0), 0.0)])
    cal = calibrate(detect(img), SIZE, W, H)
    assert cal["markers_seen"] == [0] and cal["f"] > 0
