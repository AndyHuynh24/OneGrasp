import numpy as np

from touch.robot_export import keyframes, to_quat_wxyz


def test_keyframes_reach_grasp_lift_release():
    fps, n = 30.0, 120
    pos = np.zeros((n, 3))
    pos[50:80, 2] = np.linspace(0, 0.15, 30)  # lift after the grasp
    pos[80:95, 2] = 0.15
    pos[80:95, 0] = np.linspace(0, 0.2, 15)  # transport
    pos[95:, :] = [0.2, 0, 0.15]  # held still, then released
    close = np.zeros(n, bool)
    close[45:100] = True
    state = np.zeros(n, np.int8)
    state[40:45] = 1
    state[45:100] = 2
    ok = np.ones(n, bool)
    ok[:5] = False
    k = keyframes(pos, close, state, ok, fps)
    got = {d["type"]: d["frame"] for d in k}
    assert [d["frame"] for d in k] == sorted(d["frame"] for d in k)
    assert got["reach_start"] == 5 and got["retreat_end"] == n - 1
    assert got["pregrasp"] == 45 - 12 and got["contact"] == 40 and got["grasp"] == 45
    assert 50 < got["lift"] < 60
    assert got["transport_end"] == 95 and got["release"] == 100
    assert k[0]["t"] == round(5 / fps, 3)


def test_quat_wxyz_identity():
    q = to_quat_wxyz(np.tile(np.eye(3), (4, 1, 1)))
    assert q.shape == (4, 4)
    np.testing.assert_allclose(q, [[1, 0, 0, 0]] * 4)
