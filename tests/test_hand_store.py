"""touch.store.GraspStore (hand-data store, Tiger Data / SQLite): episodes, raw columns, scenes,
video references, schema upgrade."""

import json

import numpy as np

from touch.store import GraspStore, gripper_pose


def _hand(z: float, width: float = 0.05) -> np.ndarray:
    """21 joints of a pinching right hand, palm above the grasp point, at height z (m)."""
    j = np.zeros((21, 3))
    j[[0, 5, 9, 13, 17]] = [[0, 0.0, z + 0.10], [0.01, 0.02, z + 0.08], [0, 0.02, z + 0.08],
                            [-0.01, 0.02, z + 0.08], [-0.02, 0.02, z + 0.08]]  # fmt: skip
    j[4] = [0, -width / 2, z]  # thumb tip
    j[8] = j[12] = j[16] = [0, width / 2, z]  # index / middle / ring tips
    for k in range(21):  # remaining joints between palm and tips (unused by the pose)
        if not j[k].any():
            j[k] = [0, 0, z + 0.05]
    return j


def test_gripper_pose_axes_and_width():
    pos, q, w = gripper_pose(_hand(0.10, 0.06))
    assert pos.shape == (3,) and q.shape == (4,)
    assert abs(w - 0.06) < 1e-9
    assert abs(np.linalg.norm(q) - 1) < 1e-9
    assert np.allclose(pos[:2], 0, atol=1e-9)  # midway between thumb and fingers
    assert gripper_pose(None) is None
    assert gripper_pose(np.full((21, 3), np.nan)) is None


def test_store_episode_roundtrip(tmp_path):
    st = GraspStore(f"sqlite:///{tmp_path / 'g.sqlite'}")
    st.create_schema()
    sid = st.start_session("unit_test", "table", model="none", meta={"object": "cylinder"})
    n, fps = 60, 30.0
    for i in range(n):
        t = i / fps
        closed = 10 <= i < 40
        z = 0.05 + (0.004 * (i - 10) if 10 <= i < 40 else (0.12 if i >= 40 else 0.0))
        st.add_frame(sid, t, "right", True, [0.1, 0.1, 0.8] if closed else [0.8, 0.1, 0.1],
                     [0.9] * 5, closed, _hand(z), object_name="cylinder",
                     keypoints_2d=np.full((21, 3), 100.0))  # fmt: skip
        st.add_frame(sid, t, "left", False)
    st.end_session(sid)

    gs = st.grasps()
    assert len(gs) == 1
    g = gs[0]
    assert g["hand"] == "right" and g["object"] == "cylinder"
    assert abs(g["t_grab"] - 10 / fps) < 1e-9 and abs(g["t_release"] - 40 / fps) < 1e-9
    assert abs(g["lift_m"] - np.percentile(0.004 * np.arange(1, 30), 90)) < 1e-6  # robust lift
    assert st.grasps(object_like="CYL") and not st.grasps(object_like="tube")

    ep = st.episode(g["grasp_id"])
    m = len(ep["t"])
    assert ep["grasp_pos"].shape == (m, 3) and ep["quat"].shape == (m, 4)
    assert ep["joints"].shape == (m, 21, 3) and ep["finger_p"].shape == (m, 5)
    assert ep["width"].shape == (m,) and ep["gripper_closed"].shape == (m,)
    assert ep["t"][0] < 0 and abs(ep["t"][ep["gripper_closed"].argmax()]) < 1e-9  # 0 = grab
    assert ep["info"]["frame"] == "table" and ep["info"]["quat_order"] == "w x y z"

    p = st.export_episode(g["grasp_id"], tmp_path / "ep.npz")
    z = np.load(p)
    assert z["joints"].shape == (m, 21, 3)
    assert json.loads(p.with_suffix(".json").read_text())["object"] == "cylinder"
    st.close()


def test_open_grasp_kept_at_session_end(tmp_path):
    st = GraspStore(f"sqlite:///{tmp_path / 'g.sqlite'}")
    st.create_schema()
    sid = st.start_session("unit_test", "camera")
    for i in range(5):
        st.add_frame(sid, i / 30, "left", True, [0, 0, 1], None, i >= 2, _hand(0.1))
    st.add_object(sid, 0.1, "left", "PCR tube")
    st.end_session(sid)
    (g,) = st.grasps()
    assert g["t_release"] is None and g["object"] == "PCR tube"
    st.close()


def test_raw_columns_scene_video(tmp_path):
    import cv2

    from touch.store import VideoRecorder

    st = GraspStore(f"sqlite:///{tmp_path / 'g.sqlite'}")
    st.create_schema()
    cam = {
        "K": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
        "cam_to_world": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0.6]],
        "image_size": [640, 480],
    }
    obj = {"name": "cylinder", "shape": "cylinder", "size_m": [0.031, 0.045],
           "pos": [0.1, 0.0, 0.045], "quat_wxyz": [1, 0, 0, 0]}  # fmt: skip
    scid = st.add_scene("unit", "table", camera=cam, objects=[obj])
    sid = st.start_session("unit_test", "table", scene_id=scid)
    rec = VideoRecorder(tmp_path / "v.mp4", fps=30)
    rng = np.random.default_rng(0)
    rot = np.stack([np.eye(3)] * 16)
    contact = rng.random(778)
    for i in range(12):
        idx = rec.write(np.full((48, 64, 3), i * 10, np.uint8))
        raw = {"frame_idx": idx, "hand_score": 0.9, "mp_landmarks": rng.random((21, 3)),
               "joints_model": rng.random((21, 3)), "cam_t": [0, 0, 0.5], "mano_pose": rot,
               "mano_betas": np.zeros(10), "contact_p": contact, "gate_score": 0.7}  # fmt: skip
        st.add_frame(sid, i / 30, "right", True, [0.2, 0.2, 0.6], [0.5] * 5, i >= 3,
                     _hand(0.05), raw=raw)  # fmt: skip
    assert rec.close() == 12
    st.add_video(sid, str(tmp_path / "v.mp4"), fps=30, n_frames=12, width=64, height=48)
    st.end_session(sid)

    f = st.frames(sid, "right")
    n = len(f["t"])
    assert n == 12 and (f["frame_idx"] == np.arange(12)).all()
    assert f["mp_landmarks"].shape == (n, 21, 3) and f["joints_model"].shape == (n, 21, 3)
    assert f["mano_pose"].shape == (n, 16, 3) and np.allclose(f["mano_pose"], 0)  # identity
    assert f["mano_betas"].shape == (n, 10) and f["cam_t"].shape == (n, 3)
    assert f["contact_p"].shape == (n, 778) and np.abs(f["contact_p"][0] - contact).max() < 0.003
    assert np.isnan(f["table_p"]).all() and np.allclose(f["gate_score"], 0.7)

    (v,) = st.videos(sid)
    assert v["n_frames"] == 12 and len(v["sha256"]) == 64 and v["size_bytes"] > 0
    cap = cv2.VideoCapture(v["uri"])
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12

    sc = st.scene(st.session(sid)["scene_id"])
    assert sc["objects"][0]["name"] == "cylinder" and sc["table_plane"]["normal"] == [0, 0, 1]
    import mujoco

    m = mujoco.MjModel.from_xml_string(sc["mjcf"])  # robot-free scene loads
    assert m.nbody == 2 and m.ncam == 1

    (g,) = st.grasps()
    ep = st.episode(g["grasp_id"])
    assert (
        ep["contact_p"].shape == (len(ep["t"]), 778) and ep["info"]["videos"][0]["n_frames"] == 12
    )
    assert ep["info"]["scene"]["scene_id"] == scid
    st.close()


def test_schema_upgrade_adds_columns(tmp_path):
    """A database made before the raw columns existed gets them on create_schema()."""
    import sqlite3

    p = tmp_path / "old.sqlite"
    con = sqlite3.connect(p)
    con.execute(
        "CREATE TABLE hand_frames (time TEXT NOT NULL, session_id TEXT NOT NULL, "
        "t DOUBLE PRECISION NOT NULL, hand TEXT NOT NULL, present BOOLEAN NOT NULL)"
    )
    con.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, "
                "source TEXT NOT NULL, video TEXT, frame TEXT NOT NULL, model TEXT, git TEXT, "
                "meta TEXT)")  # fmt: skip
    con.commit()
    con.close()
    st = GraspStore(f"sqlite:///{p}")
    st.create_schema()
    assert {"frame_idx", "contact_p", "mano_pose"} <= st._columns("hand_frames")
    assert "scene_id" in st._columns("sessions")
    st.close()


def test_mp_depth_recovers_known_depth():
    """touch.handpath.mp_depth: a hand at a known depth, projected (with MediaPipe-style
    relative z in pixels), turned about its long axis -> the same depth every frame."""
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot

    from touch.handpath import mp_depth

    rng = np.random.default_rng(0)
    hand = rng.normal(0, 0.04, (21, 3))
    hand[0] = 0
    K = np.array([[1400.0, 0, 960], [0, 1400.0, 540], [0, 0, 1]])
    n, depth = 12, 0.45
    J, lm = np.zeros((n, 21, 3)), np.zeros((n, 21, 3))
    for i in range(n):
        R = Rot.from_euler("y", 70 * i / n, degrees=True).as_matrix()
        Jc = hand @ R.T + [0.02, -0.01, depth]
        J[i] = hand @ R.T
        uv = (Jc @ K.T)[:, :2] / Jc[:, 2:]
        lm[i, :, :2] = uv
        lm[i, :, 2] = (Jc[:, 2] - Jc[0, 2]) * K[0, 0] / depth  # MediaPipe z: px scale
    d = mp_depth(J, lm, K)
    assert d.shape == (n,)
    assert np.allclose(d, depth, atol=0.03), d


def test_fuse_orientation_follows_mediapipe_when_3d_disagrees():
    """touch.handpath.fuse_orientation: the 3D hand frozen while MediaPipe's turns 90 deg ->
    the result turns with MediaPipe (shape and palm centre kept)."""
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot

    from touch.handpath import PALM, _palm_frames, _rot_angle, fuse_orientation

    rng = np.random.default_rng(1)
    hand = rng.normal(0, 0.04, (21, 3))
    n = 30
    J = np.repeat(hand[None] + [0.0, 0.3, 0.1], n, 0)  # 3D hand: never turns
    lm = np.stack([hand @ Rot.from_euler("x", 90 * max(0, i - 15) / 14, degrees=True).as_matrix().T
                   for i in range(n)]) * 1000  # fmt: skip
    out = fuse_orientation(J, lm, np.eye(4))
    assert out.shape == J.shape
    assert np.allclose(out[:, list(PALM)].mean(1), J[:, list(PALM)].mean(1), atol=1e-9)
    R = _palm_frames(out)
    assert _rot_angle(R[0] @ R[-1].T) > 80
    assert _rot_angle(R[0] @ R[5].T) < 1
