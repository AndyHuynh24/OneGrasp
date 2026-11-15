"""HO-Cap helpers on synthetic data (no download needed)."""

import numpy as np

from touch.data.hocap import episode_displacement


def test_episode_displacement_resets_per_contact_episode():
    n = 8
    pd = np.full((n, 6), 0.05)  # metres
    pd[1:4, 1] = pd[1:4, 2] = 0.001  # grasp object 0, frames 1-3
    pd[6:8, 1] = 0.001  # touch object 1, frames 6-7
    po = np.zeros((n, 6), int)
    po[6:8] = 1
    obj_t = np.zeros((2, n, 3))
    obj_t[0, :, 2] = [0, 0, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05]  # object 0 lifted, then left up
    out = episode_displacement(pd, po, obj_t)
    assert out.shape == (n,)
    np.testing.assert_allclose(out, [0, 0, 0.02, 0.05, 0, 0, 0, 0], atol=1e-6)
