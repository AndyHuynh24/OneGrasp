import numpy as np

from touch import live_robot as LR
from touch.live_robot import horizontal_grasp

# these tests were tuned on the 9 x 6.2 cm cylinder (the default object is now a 13.4 cm can)
CYL_H, CYL_D = 0.09, 0.062


def _robot(*a, **k):
    return LR.LiveRobot(*a, h=CYL_H, d=CYL_D, **k)


def fake_hand(pad: np.ndarray, off: float = 0.0) -> np.ndarray:
    """21 joints [21, 3] whose top-down grasp (robot_sim.hand_to_gripper) centres on `pad`."""
    j = np.zeros((21, 3))
    P = pad + [off, 0, 0.08]  # palm 8 cm above the pad centre
    j[[0, 5, 9, 13, 17]] = P + np.array([[-0.04, 0, 0], [0.02, 0.03, 0], [0.04, 0, 0],
                                         [0.02, -0.01, 0], [0.02, -0.03, 0]])  # fmt: skip
    j[4] = pad + [0, -0.045, 0]  # thumb tip
    j[8] = j[12] = j[16] = pad + [0, 0.045, 0]  # index / middle / ring tips
    for k in range(21):
        if not j[k].any():
            j[k] = P
    return j


def test_horizontal_grasp_is_a_rotation_with_level_closing_axis():
    R = horizontal_grasp(np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0])
    assert R.shape == (3, 3) and np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert abs(R[2, 1]) < 1e-9


def test_reach_grab_lift_moves_the_cylinder_up():
    rob = _robot((0.0, 0.2))
    z0 = rob.object_pos()[2]
    none, hold = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    obj = np.array([0.0, 0.2])
    # hand 2 cm off the axis (assist pulls the gripper onto it), down, grab, lift
    for z in np.linspace(0.25, 0.045, 40):
        info = rob.step(fake_hand(np.array([*obj, z]), off=0.02), none, True)
    for _ in range(30):
        info = rob.step(fake_hand(np.array([*obj, 0.045]), off=0.02), hold, True)
    assert info["close"] and info["assisted"]
    for z in np.linspace(0.045, 0.20, 40):
        info = rob.step(fake_hand(np.array([*obj, z]), off=0.02), hold, True)
    assert set(info) >= {"close", "object", "opening", "pad_target"}
    assert rob.object_pos()[2] - z0 > 0.08, rob.object_pos()
    # release: the cylinder drops back toward the table
    for _ in range(60):
        rob.step(fake_hand(np.array([*obj, 0.20]), off=0.02), none, True)
    assert rob.object_pos()[2] < CYL_H


def test_release_needs_10_non_hold_frames_and_close_needs_hold():
    rob = _robot((0.0, 0.2), release_frames=10, release_on="not_hold")
    st = {n: np.eye(3)[k] for k, n in enumerate(("none", "touch", "hold"))}
    rob._gate(1)
    rob._gate(1)
    assert not rob.close  # touch alone does not close (close_on="hold")
    rob._gate(2)
    assert not rob.close
    rob._gate(2)
    assert rob.close  # 2 hold frames
    for _ in range(9):
        rob._gate(1 if _ % 2 else 0)  # touch / none flicker
    rob._gate(None)  # no hand: does not count
    assert rob.close
    rob._gate(2)  # hold resets the release count
    for _ in range(9):
        rob._gate(0)
    assert rob.close
    rob._gate(1)
    assert not rob.close  # 10th consecutive none / touch frame releases
    assert set(st) == {"none", "touch", "hold"}


def test_fast_release_on_confident_none_only():
    rob = _robot((0.0, 0.2), fast_p=0.95, fast_frames=3)
    rob._gate(2)
    rob._gate(2)
    assert rob.close
    for _ in range(3):
        rob._gate(1, p_none=0.0)  # confident TOUCH: no fast release
    rob._gate(0, 0.96)
    rob._gate(0, 0.90)  # not sure enough: the streak restarts
    rob._gate(0, 0.96)
    rob._gate(0, 0.97)
    assert rob.close
    rob._gate(0, 0.99)
    assert not rob.close  # 3 sure NONE frames in a row


def test_default_release_ignores_touch():
    rob = _robot((0.0, 0.2), fast_p=1.01)  # slow only: NONE x 20
    rob._gate(2)
    rob._gate(2)
    for _ in range(50):
        rob._gate(1, 0.0)  # a long mid-carry "touch": stays closed
    assert rob.close
    for _ in range(19):
        rob._gate(0, 0.5)
    assert rob.close
    rob._gate(0, 0.5)
    assert not rob.close


def test_fast_release_defaults_need_10_frames_at_099():
    rob = _robot((0.0, 0.2))  # fast 0.99 x 10, slow NONE x 20
    rob._gate(2)
    rob._gate(2)
    for _ in range(9):
        rob._gate(0, 0.995)
    rob._gate(0, 0.98)  # below 0.99: fast streak restarts
    for _ in range(9):
        rob._gate(0, 0.995)
    assert rob.close
    rob._gate(0, 0.995)
    assert not rob.close  # 10th sure frame in a row (19 NONE frames: before the slow rule)


def test_fusion_gate_drives_gripper():
    from touch.interact.grab_gate import Ensemble, Fusion, FusionGate

    nf = 20
    # a "model" whose grab score is sigmoid(10 * (p_hold - 0.5)): weight on feature 2 (p_hold)
    W0 = np.zeros((2 * nf, 1), np.float32)
    W0[2, 0] = 10.0
    model = Ensemble([Fusion(np.zeros(nf, np.float32), np.ones(nf, np.float32),
                             [(W0, np.array([-5.0], np.float32))])])  # fmt: skip
    gate = FusionGate(model, {"on": 0.6, "off": 0.3, "k_on": 2, "k_off": 3})
    r = _robot([0.2, 0.0], fusion=gate)
    hold, none = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    f = np.zeros(5)
    closes = [r.step(None, hold, True, f)["close"] for _ in range(3)]
    assert closes == [False, True, True]
    closes = [r.step(None, none, True, f)["close"] for _ in range(3)]
    assert closes == [True, True, False]
    r.step(None, hold, True, f)
    r.reset()
    assert not r.close and len(gate.buf) == 0


def _tilt_deg(rob) -> float:
    from scipy.spatial.transform import Rotation as Rot

    q = rob.data.xquat[rob.oid]
    return float(np.degrees(np.arccos(np.clip(Rot.from_quat(q[[1, 2, 3, 0]]).as_matrix()[2, 2],
                                              -1, 1))))  # fmt: skip


def _grab_and_lift(rob, obj, z_up=0.12, tilt_deg=0.0):
    from scipy.spatial.transform import Rotation as Rot

    none, hold = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    for z in np.linspace(0.25, 0.045, 40):
        rob.step(fake_hand(np.array([*obj, z])), none, True)
    for _ in range(20):
        rob.step(fake_hand(np.array([*obj, 0.045])), hold, True)
    for k, z in enumerate(np.linspace(0.045, z_up, 30)):
        pad = np.array([*obj, z])
        R = Rot.from_euler("x", tilt_deg * (k + 1) / 30, degrees=True).as_matrix()
        rob.step((fake_hand(pad) - pad) @ R.T + pad, hold, True)  # wrist tilts while lifting
    return none


def test_release_sets_the_cylinder_down_in_place():
    """Released in the air (hand still up), the gripper lowers it onto the table, then opens and
    backs off: upright, where it was carried to, not dropped or flung."""
    rob = _robot((0.0, 0.2))
    obj = np.array([0.0, 0.2])
    none = _grab_and_lift(rob, obj)
    xy = rob.object_pos()[:2].copy()
    assert rob.object_pos()[2] - CYL_H / 2 > 0.05  # carried up
    for _ in range(60):
        info = rob.step(fake_hand(np.array([*obj, 0.12])), none, True)
    assert not info["close"]
    assert abs(rob.object_pos()[2] - CYL_H / 2) < 0.005  # on the table
    assert _tilt_deg(rob) < 10 and np.linalg.norm(rob.object_pos()[:2] - xy) < 0.015


def test_held_gripper_follows_hand_rotation_by_default_upright_keeps_it_standing():
    rob = _robot((0.0, 0.2))  # default: copies the full hand rotation (e.g. a pour)
    _grab_and_lift(rob, np.array([0.0, 0.2]), tilt_deg=40)
    assert _tilt_deg(rob) > 25
    rob = _robot((0.0, 0.2), upright=True)
    _grab_and_lift(rob, np.array([0.0, 0.2]), tilt_deg=40)
    assert rob.object_pos()[2] - CYL_H / 2 > 0.03 and _tilt_deg(rob) < 10


def test_open_gripper_follows_hand_rotation():
    from scipy.spatial.transform import Rotation as Rot

    rob = _robot((0.0, 0.2))
    pad = np.array([0.3, 0.0, 0.15])  # far from the cylinder: no assist
    none = np.array([1.0, 0, 0])
    for _ in range(20):
        rob.step(fake_hand(pad), none, True)
    R0 = rob.data.xmat[rob.gid].reshape(3, 3).copy()
    for k in range(30):  # 60 deg roll in 1 s, then hold
        R = Rot.from_euler("y", 60 * min(1.0, (k + 1) / 20), degrees=True).as_matrix()
        rob.step((fake_hand(pad) - pad) @ R.T + pad, none, True)
    R1 = rob.data.xmat[rob.gid].reshape(3, 3)
    ang = np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))
    assert 50 < ang < 70, ang
