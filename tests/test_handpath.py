import numpy as np
from scipy.spatial.transform import Rotation as Rot

from touch.handpath import world_hand


def test_world_hand_recovers_world_joints_from_2d():
    rng = np.random.default_rng(0)
    n = 6
    K = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    R = Rot.from_euler("x", 160, degrees=True).as_matrix()  # camera looking down at the table
    t = np.array([0.1, -0.2, 0.9])
    c2w = np.c_[R, t]
    Jw = rng.normal(0, 0.03, (n, 21, 3)) + [0.05, 0.0, 0.1]  # hands ~10 cm above the table
    Jc = (Jw - t) @ R  # world -> camera
    uv = Jc[..., :2] / Jc[..., 2:] * [600, 600] + [320, 240]
    rel = Jc - Jc[:, :1]
    V = np.repeat(rel[:, :1], 778, 1)
    for left in (False, True):
        r = rel.copy()
        v = V.copy()
        if left:  # the model's convention: left hands mirrored
            r[..., 0] *= -1
            v[..., 0] *= -1
        out = world_hand(r, v, uv, K, c2w, left, smooth_k=1)
        assert out["joints"].shape == (n, 21, 3) and out["verts"].shape == (n, 778, 3)
        assert out["ok"].all()
        np.testing.assert_allclose(out["joints"], Jw, atol=1e-6)
