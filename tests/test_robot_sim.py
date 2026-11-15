"""robot_sim: geometry helpers on synthetic data, plus a MuJoCo pick of a plain box (no DexYCB)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from touch import robot_sim as S


def fake_hand(seed=0):
    """21 joints of a rough right hand (metres), palm in the x-y plane."""
    rng = np.random.default_rng(seed)
    j = np.zeros((21, 3))
    for f, y in enumerate([-0.04, -0.02, 0.0, 0.02, 0.04]):
        for k in range(4):
            j[1 + 4 * f + k] = [0.04 + 0.025 * k, y, 0.0]
    return j + rng.normal(0, 0.002, j.shape)


def test_fit_translation_recovers_exact_T():
    K = np.array([[600.0, 0, 320], [0, 610, 240], [0, 0, 1]])
    J = fake_hand() - fake_hand()[0]
    T = np.array([0.05, -0.03, 0.8])
    X = J + T
    uv = (X[:, :2] / X[:, 2:]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    assert uv.shape == (21, 2)
    np.testing.assert_allclose(S.fit_translation(J, uv, K), T, atol=1e-9)


def test_master_to_table_inverts_tag_pose():
    R = Rot.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    tag = np.concatenate([R, [[0.1], [0.2], [0.9]]], 1)  # tag -> master: x_m = R x_t + t
    x_t = np.random.default_rng(1).normal(size=(7, 3))
    x_m = S.cam_to_master(x_t, tag)
    np.testing.assert_allclose(S.master_to_table(x_m, tag), x_t, atol=1e-12)


def test_debounce_on_off():
    raw = np.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0], bool)
    out = S.debounce(raw, on=2, off=3)
    assert out.shape == raw.shape
    assert out.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]


def test_grasp_signals_carry_over_gaps():
    state = np.array([[1, 0, 0], [0, 1, 0], [0.9, 0, 0.1], [0, 0, 1]], float)
    present = np.array([1, 1, 0, 1], bool)
    assert S.grasp_signal(state, present).tolist() == [0, 1, 1, 1]
    assert S.gt_grasp_signal(np.array([0, 1, -1, 2, 0])).tolist() == [0, 1, 1, 1, 0]


def test_hand_frame_is_rigid_pose():
    H = S.hand_frame(fake_hand())
    assert H.shape == (4, 4)
    np.testing.assert_allclose(H[:3, :3].T @ H[:3, :3], np.eye(3), atol=1e-9)
    assert np.linalg.det(H[:3, :3]) == pytest.approx(1.0)


def test_contact_grasp_axis_and_width():
    j = fake_hand()
    verts = np.array([[0.1, -0.03, -0.05]] * 4 + [[0.1, 0.03, -0.05]] * 4)
    parts = np.array([1] * 4 + [2] * 4)
    G, w = S.contact_grasp(j, verts, np.ones(8, bool), parts)
    assert G.shape == (4, 4)
    assert w == pytest.approx(0.06)
    np.testing.assert_allclose(G[:3, 1], [0, 1, 0], atol=1e-9)  # closing axis thumb -> others
    np.testing.assert_allclose(G[:3, 3], [0.1, 0, -0.05], atol=1e-9)
    assert S.contact_grasp(j, verts, np.r_[np.ones(4), np.zeros(4)].astype(bool), parts) is None


def test_retarget_follows_hand_rigidly():
    n = 12
    R_t = Rot.from_euler("z", np.linspace(0, 40, n)[:, None], degrees=True).as_matrix()
    j0 = fake_hand()
    joints = np.stack([j0 @ R_t[t].T + [0.01 * t, 0, 0.02 * t] for t in range(n)])
    ok = np.ones(n, bool)
    close = np.zeros(n, bool)
    close[5:9] = True
    pos, R, W, hows = S.retarget(joints, ok, close)
    assert pos.shape == (n, 3) and R.shape == (n, 3, 3) and W.shape == (n,)
    assert hows == ["tips"]
    for t in range(n):
        np.testing.assert_allclose(R[t].T @ R[t], np.eye(3), atol=1e-9)
        # pad expressed in the hand frame is constant (rigid attachment)
        rel = np.linalg.inv(S.hand_frame(joints[t]))[:3] @ np.r_[pos[t], 1]
        rel0 = np.linalg.inv(S.hand_frame(joints[0]))[:3] @ np.r_[pos[0], 1]
        np.testing.assert_allclose(rel, rel0, atol=1e-9)


def test_lift_outcome():
    gt = np.zeros((10, 3))
    gt[:, 2] = np.r_[np.zeros(5), np.linspace(0.02, 0.1, 5)]
    assert S.lift_outcome(gt, gt)["success"]
    assert not S.lift_outcome(gt, np.zeros((10, 3)))["success"]
    assert S.lift_outcome(np.zeros((10, 3)), gt) is None  # human never lifted


def box_scene():
    body = (
        '<body name="obj0" pos="0.5 0.3 0.0405"><freejoint/>'
        '<geom type="box" size="0.02 0.02 0.04" density="300" friction="1.0 0.02 0.001" '
        'condim="4"/></body>'
    )
    return S.Scene(S.scene_xml("", body), ["box"], 0)


def top_down_pick(close_from: int):
    """Descend onto a 4 cm box (frames 0-29), wait (30-39), lift 11 cm (40-69)."""
    z = np.r_[np.linspace(0.20, 0.04, 30), np.full(10, 0.04), np.linspace(0.04, 0.15, 30)]
    n = len(z)
    pos = np.stack([np.full(n, 0.5), np.full(n, 0.3), z], 1)
    R = np.repeat(np.diag([1.0, -1.0, -1.0])[None], n, 0)  # approach axis points down
    close = np.arange(n) >= close_from
    out = S.replay(box_scene(), pos, R, np.full(n, 0.04), np.ones(n, bool), close)
    assert out["obj"].shape == (n, 1, 3) and out["opening"].shape == (n,)
    return out["obj"][-1, 0, 2] - out["obj"][0, 0, 2]


def test_sim_pick_on_time_lifts_and_early_close_fails():
    pytest.importorskip("mujoco")
    assert top_down_pick(close_from=32) > 0.08  # closes around the box: lifted with the gripper
    assert top_down_pick(close_from=0) < 0.02  # closes in the air: the box stays down


def test_object_between_pads_and_gate():
    n = 4
    pos = np.zeros((n, 3))
    R = np.repeat(np.eye(3)[None], n, 0)
    pts = np.zeros((n, 50, 3))
    pts[:, :, 1] = np.linspace(-0.02, 0.02, 50)  # an object 4 cm wide along the closing axis
    pts[2:] += [0, 0, 0.2]  # frames 2-3: object far beyond the fingertips
    ready = S.object_between_pads(pos, R, np.full(n, 0.04), pts)
    assert ready.tolist() == [True, True, False, False]
    close = np.array([0, 1, 1, 1, 0, 1], bool)
    ok = np.array([0, 0, 1, 0, 0, 0], bool)
    assert S.gate_close(close, ok).tolist() == [0, 0, 1, 1, 0, 0]


def test_pregrasp_approaches_along_gripper_axis():
    from touch.robot_sim import pregrasp

    n = 60
    pos = np.zeros((n, 3))
    pos[:, 0] = np.linspace(-0.3, 0.0, n)  # the hand sweeps in sideways along x
    R = np.tile(np.eye(3), (n, 1, 1))  # approach axis (z) = world z
    close = np.zeros(n, bool)
    close[40:50] = True
    p2, R2 = pregrasp(pos, R, close, n_app=10, back=0.08, n_tr=10)
    assert p2.shape == (n, 3) and R2.shape == (n, 3, 3)
    np.testing.assert_allclose(p2[40], pos[40])  # at the grasp
    np.testing.assert_allclose(p2[30], pos[40] - [0, 0, 0.08])  # pre-grasp, 8 cm back along z
    assert np.allclose(p2[30:40, :2], pos[40, :2])  # straight in along z
    np.testing.assert_allclose(p2[:20], pos[:20])  # untouched before the blend
    np.testing.assert_allclose(p2[45:], pos[45:])


def test_fill_gaps_merges_short_openings_only():
    from touch.robot_sim import fill_gaps

    c = np.zeros(40, bool)
    c[5:10] = c[14:20] = c[35:38] = True  # gap 4 (merge), gap 15 (keep)
    out = fill_gaps(c, max_gap=10)
    assert out.shape == (40,) and out.dtype == bool
    assert out[5:20].all() and not out[20:35].any() and out[35:38].all()


def test_object_grasp_spans_box_and_clears_table():
    from touch import robot_sim as S

    rng = np.random.default_rng(0)
    half = np.array([0.03, 0.05, 0.06])  # 6 x 10 x 12 cm box standing on the table
    p = rng.uniform(-1, 1, (6000, 3))
    ax = rng.integers(0, 3, 6000)
    p[np.arange(6000), ax] = np.sign(p[np.arange(6000), ax])  # points on the faces
    pts = p * half + [0.4, 0.2, half[2]]
    G0 = np.eye(4)  # human grasp: from the side, closing along world x, slightly off-centre
    G0[:3, :3] = np.stack([np.cross([1.0, 0, 0], [0, 1.0, 0]), [1.0, 0, 0], [0, 1.0, 0]], 1)
    G0[:3, 3] = [0.41, 0.17, 0.08]
    g = S.object_grasp(G0, pts)
    assert g is not None
    G, w = g
    assert G.shape == (4, 4)
    assert np.allclose(G[:3, :3].T @ G[:3, :3], np.eye(3), atol=1e-6)
    assert 0.05 < w < 0.13  # spans one of the box's side widths
    tip = G[:3, 3] + S.TIP_PAST * G[:3, 2]
    assert tip[2] > 0.0  # fingertips above the table
