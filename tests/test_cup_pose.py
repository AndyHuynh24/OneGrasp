"""touch.cup_pose: the silhouette fit recovers a standing / lying cylinder from its own
rendered silhouette through the rig-like camera (no SAM)."""

import math

import numpy as np

from touch import cup_pose as CP

H, D = 0.09, 0.062
K = np.array([[606.0, 0, 400], [0, 606.0, 225], [0, 0, 1]])
# camera 0.5 m above the table, looking straight down-ish (x right, y down in the image)
C2W = np.c_[np.diag([1.0, -1.0, -1.0]), [0.0, 0.0, 0.5]]


def _mask(xy, lying, yaw):
    R, c = CP.cup_pose(np.array(xy), lying, yaw, H, D)
    return CP.silhouette(K, C2W, CP.rim_points(H, D), R, c, (450, 800))


def test_standing():
    m = _mask([0.03, -0.02], False, 0.0)
    assert m.shape == (450, 800) and m.dtype == bool
    r = CP.fit_mask(m, K, C2W, H, D)
    assert not r["lying"] and r["xy"].shape == (2,)
    assert np.linalg.norm(r["xy"] - [0.03, -0.02]) < 0.003 and r["iou"] > 0.9


def test_lying_heading():
    m = _mask([-0.04, 0.05], True, math.radians(130))
    r = CP.fit_mask(m, K, C2W, H, D)
    assert r["lying"] and np.linalg.norm(r["xy"] - [-0.04, 0.05]) < 0.003
    assert abs((math.degrees(r["yaw"]) - 130 + 90) % 180 - 90) < 3


def test_hand_mask_and_pick_frame():
    lm = np.full((3, 21, 3), np.nan)
    lm[1] = np.c_[np.linspace(700, 900, 21), np.linspace(400, 600, 21), np.zeros(21)]
    i, ov = CP.pick_frame(lm, 3, (100, 100), 30, (450, 800), (1920, 1080))
    assert (i, ov) == (2, 0.0)  # no hand in frame 2
    hm = CP.hand_mask(lm[1], (450, 800), (1920, 1080))
    assert hm.shape == (450, 800) and hm.any()


def test_depth_scale_recovers_hand_depth():
    """A hand gripping a standing cylinder at 3.5 cm, reconstructed 20 % too close (the
    monocular size / depth ambiguity): depth_scale finds 1.25 and the grip height."""
    c0, axis = np.r_[0.02, -0.01, H / 2], np.array([0.0, 0.0, 1.0])
    grip = np.r_[0.02, -0.01, 0.035]  # on the axis
    rng = np.random.default_rng(1)
    J = grip + rng.normal(0, 0.03, (5, 21, 3))
    J[:, 4], J[:, 8] = grip + [0, -0.03, 0], grip + [0, 0.03, 0]  # tips straddle the axis
    lm = np.stack([np.c_[CP.project(K, C2W, j), np.zeros(21)] for j in J])
    J_bad = CP.rescale_about_camera(J, C2W, 0.8)
    r = CP.depth_scale(lm, J_bad, K, C2W, c0, axis, H)
    assert r is not None and r["n"] == 5
    assert abs(r["scale"] - 1.25) < 0.01 and abs(r["t"] - (0.035 - H / 2)) < 0.002
    J_fix = CP.rescale_about_camera(J_bad, C2W, r["scale"])
    assert np.allclose(J_fix, J, atol=1e-6)
    assert np.allclose(CP.project(K, C2W, J_bad[0]), CP.project(K, C2W, J[0]), atol=1e-6)


def test_clean_depth_removes_spike_keeps_pixels():
    from touch.handpath import clean_depth

    c2w = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0.6]])  # looking down from 60 cm
    o = c2w[:, 3]
    rng = np.random.default_rng(0)
    hand = rng.normal(0, 0.03, (21, 3)) + [0.05, 0.02, 0.10]
    J = np.repeat(hand[None], 40, 0) + np.linspace(0, 0.05, 40)[:, None, None] * [0, 0, 1]
    spike = J.copy()
    spike[20] = o + 0.3 * (J[20] - o)  # depth blown up: palm jumps toward the camera
    out = clean_depth(spike, c2w)
    assert out.shape == (40, 21, 3)
    assert np.abs(out[20] - J[20]).max() < 0.01  # back on the track
    assert np.allclose(out[np.arange(40) != 20], J[np.arange(40) != 20])  # others untouched
    ray_in = (spike[20] - o) / np.linalg.norm(spike[20] - o, axis=1, keepdims=True)
    ray_out = (out[20] - o) / np.linalg.norm(out[20] - o, axis=1, keepdims=True)
    assert np.allclose(ray_in, ray_out)  # same pixels
