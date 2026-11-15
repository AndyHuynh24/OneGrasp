"""Live, causal robot copy of a human grasp in MuJoCo: one cylinder on the table (the hackathon
demo object: 9 cm tall, 6.2 cm diameter), the parallel gripper of touch.robot_sim, driven one
camera frame at a time by the grasp model's 3D hand (table frame) and hand state.

Per frame (`LiveRobot.step`):
  close    closes after `on_frames` consecutive HOLD frames (close_on="hold", default) or TOUCH /
           HOLD frames (close_on="contact"); once closed it opens after `release_frames`
           (default 20) consecutive NONE frames, or P(NONE) >= fast_p for fast_frames frames in a
           row (default 0.99 x 10). TOUCH never counts toward release (release_on="none"): mid-
           carry "touch" is the model's usual mistake. HO-Cap val (108 holds, g6, 2026-11-12):
           NONE-or-TOUCH x 20 releases mid-hold in 16% of holds, NONE-only x 10 in 1%.
           Frames without a hand count for neither (the last decision holds).
  open     the pad centre follows the hand (robot_sim.hand_to_gripper, One Euro filter on
           capture times; smoothing="exp" for the old fixed exponential smoothing).
           assist (default): within ASSIST_R of the cylinder axis and below its top + 6 cm, the
           pad is pulled onto the axis and the closing axis turned horizontal, so the open
           fingers straddle the cylinder instead of hitting it. This assumes the robot knows the
           cylinder pose (as robot v2's object_refine offline), it is not camera-only.
  closing  the grasp is fixed at the rising edge (on the cylinder when assisted) and then moves
           rigidly with the hand frame: pad(t) = H(t) H(t0)^-1 G, as robot_sim.retarget offline.
           Assisted grasps are taken at mid-height. upright=True (off by default) keeps the
           grasp's tilt and follows only the hand's turn about the vertical.
  release  like a robot placing an object, not following the hand at once: lower until the
           object rests on the table, hold still RELEASE_STOP frames with the fingers closed,
           open, back off RETREAT along the approach axis, then follow the hand again.
           Following the hand while the fingers open is what flung the object (HO-Cap val
           replay of g9 grasps, 2026-11-13: moved > 10 cm after opening in 46% of releases
           before, 0% after).
Physics: robot_sim's force-driven gripper (clipped PD on a free body), stiff contacts,
fingers close at 0.2 m/s (robot v2 settings), robot_sim.SUB steps of 2 ms per 30 fps frame.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from touch import robot_sim as S

CYL_H, CYL_D = 0.134, 0.052  # m: the demo object, a drink can (Andy 2026-11-14; before: the
# black cylinder, 0.09 x 0.062 -> --obj-h 0.09 --obj-d 0.062)
ASSIST_R = 0.10  # m: assist starts this close to the cylinder axis (full within half of it)
SMOOTH = 0.5  # exponential smoothing weight of the newest hand pose (smoothing="exp")
# smoothing="one_euro" (default): touch.interact.one_euro, strong at rest, light when moving.
# position: cutoff Hz = min_cutoff + beta * speed (m/s); rotation: per quaternion speed
# Position tuned on g9's HO-Cap val hand tracks (2026-11-13): vs SMOOTH 0.5, 15% less jitter at
# about equal lag (the predicted 3D hand is noisy, so beta is small); rotation: same, untuned.
ONE_EURO_POS = {"min_cutoff": 3.0, "beta": 0.3, "d_cutoff": 0.3}
ONE_EURO_ROT = {"min_cutoff": 3.0, "beta": 0.3, "d_cutoff": 0.3}
# release like a robot: hold still while the fingers open, back off along the approach axis,
# then follow the hand again (following at once drags the object along with the open fingers)
PLACE_TOL, PLACE_STEP, PLACE_MAX = 0.004, 0.02, 0.35  # m: set the object down before opening
LEVEL_FRAMES = 6  # frames to turn a held cylinder upright before setting it down
RELEASE_STOP = 3  # frames: brake with the fingers still closed (opening mid-swing throws it)
RELEASE_CLEAR = 0.02  # m: fingers open to the object width + this before backing off
RELEASE_MAX = 10  # frames: back off anyway after this many (e.g. the fingers are blocked)
RETREAT, RETREAT_FRAMES = 0.05, 5  # m back along the approach axis, over this many frames


def cylinder_xml(xy, h: float = CYL_H, d: float = CYL_D, camera: str = "") -> str:
    body = (f'<body name="obj0" pos="{xy[0]} {xy[1]} {h / 2 + 0.0005}"><freejoint/>'
            f'<geom name="cyl" type="cylinder" size="{d / 2} {h / 2}" density="300" '
            f'rgba="0.2 0.45 0.85 1" friction="1.0 0.02 0.001" condim="4"/></body>')  # fmt: skip
    xml = S.scene_xml("", body, stiff=True).replace("<worldbody>", "<worldbody>" + camera, 1)
    old = 'pos="0.5 0.3 0" size="1 1 0.01"'  # robot_sim's table: centre it on marker 0, 3 x 3 m
    assert old in xml
    return xml.replace(old, 'pos="0 0 0" size="1.5 1.5 0.01"')


def camera_xml(K: np.ndarray, cam_to_world: np.ndarray, height_px: int, name: str = "rec") -> str:
    """MuJoCo camera at the calibrated webcam pose (OpenCV axes -> MuJoCo: y and z flipped)."""
    R, p = np.asarray(cam_to_world)[:, :3], np.asarray(cam_to_world)[:, 3]
    x, y = R[:, 0], -R[:, 1]
    fovy = np.degrees(2 * np.arctan(height_px / 2 / K[1, 1]))
    return (f'<camera name="{name}" pos="{p[0]} {p[1]} {p[2]}" '
            f'xyaxes="{x[0]} {x[1]} {x[2]} {y[0]} {y[1]} {y[2]}" fovy="{fovy}"/>')  # fmt: skip


def horizontal_grasp(R: np.ndarray) -> np.ndarray:
    """Pad rotation (columns x y z; y = closing axis, z = approach) -> the nearest one whose
    closing axis is horizontal (fingers squeeze a vertical cylinder from the sides)."""
    y = R[:, 1].copy()
    y[2] = 0.0
    if np.linalg.norm(y) < 1e-6:
        y = np.array([0.0, 1.0, 0.0])
    y /= np.linalg.norm(y)
    z = R[:, 2] - (R[:, 2] @ y) * y
    if np.linalg.norm(z) < 1e-6:
        z = np.array([0.0, 0.0, -1.0])
    z /= np.linalg.norm(z)
    return np.stack([np.cross(y, z), y, z], 1)


def _yaw(R: np.ndarray) -> float:
    """Heading of a frame about the world vertical (from its x axis, or y if x is vertical)."""
    v = R[:, 0] if np.hypot(R[0, 0], R[1, 0]) > 0.2 else R[:, 1]
    return float(np.arctan2(v[1], v[0]))


def _blend(Ra: np.ndarray, Rb: np.ndarray, w: float) -> np.ndarray:
    """Rotation w of the way from Ra to Rb."""
    rel = Rot.from_matrix(Ra.T @ Rb).as_rotvec()
    return Ra @ Rot.from_rotvec(w * rel).as_matrix()


def _pose(p: np.ndarray, R: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


class LiveRobot:
    view_camera = "rec"  # MuJoCo camera live_sim renders (the calibrated webcam pose)

    def __init__(self, obj_xy, assist: bool = True, camera: str = "", close_speed: float = 0.2,
                 h: float = CYL_H, d: float = CYL_D, close_on: str = "hold", on_frames: int = 2,
                 release_frames: int = 20, fast_p: float = 0.99,
                 fast_frames: int = 10, release_on: str = "none", fusion=None,
                 upright: bool = False, smoothing: str = "one_euro"):  # fmt: skip
        """fusion: a touch.interact.grab_gate.FusionGate (learned gate) replacing the rule gate
        (close_on / on_frames / release_* / fast_* are then unused).
        upright (off by default: the gripper copies the full hand rotation, e.g. a pour): an
        assisted (cylinder) grasp keeps its grasp-time tilt and turns with the hand about the
        vertical only (keeps it standing despite wrist tilt and pose noise)."""
        self.fusion = fusion
        self.upright = upright
        assert smoothing in ("one_euro", "exp")
        self.smoothing = smoothing
        if smoothing == "one_euro":
            from touch.interact.one_euro import OneEuro, OneEuroRotation

            pos, rot = (lambda: OneEuro(**ONE_EURO_POS)), (lambda: OneEuroRotation(**ONE_EURO_ROT))
            self.filt = {"pad": pos(), "R": rot(), "Hp": pos(), "HR": rot()}
        assert release_on in ("none", "not_hold")
        self.release_on = release_on
        self.fast_p, self.fast_frames = fast_p, fast_frames
        assert close_on in ("hold", "contact")
        self.close_on, self.on_frames, self.release_frames = close_on, on_frames, release_frames
        import mujoco

        self.mj = mujoco
        self.h, self.d, self.assist, self.close_speed = h, d, assist, close_speed
        self._build(obj_xy, camera)
        self.reset(obj_xy)

    def _build(self, obj_xy, camera: str) -> None:
        """MuJoCo model + data: the floating gripper and the cylinder "obj0"."""
        self.m = self.mj.MjModel.from_xml_string(cylinder_xml(obj_xy, self.h, self.d, camera))
        self.data = self.mj.MjData(self.m)
        self.data.eq_active[:] = 0  # force drive, not the weld
        self.gid = self.m.body("gripper").id
        self.oid = self.m.body("obj0").id
        self.g_adr = self.m.jnt_qposadr[self.m.joint("gfree").id]
        self.o_adr = self.m.jnt_qposadr[self.m.body_jntadr[self.oid]]
        self.jl = self.m.jnt_qposadr[self.m.joint("jl").id]
        self.jr = self.m.jnt_qposadr[self.m.joint("jr").id]

    # ------------------------------------------------------------------ state
    def reset(self, obj_xy=None) -> None:
        if obj_xy is not None:
            self.obj_xy = np.asarray(obj_xy, np.float64)
        self._reset_sim()
        self.close = False
        if self.fusion is not None:
            self.fusion.reset()
        self.sure_none = 0
        self.run = 0
        self.pad = self.R = self.H = None  # smoothed hand pad pose / hand frame
        for f in getattr(self, "filt", {}).values():
            f.reset()
        self.rel = None  # closed episode: H(t0)^-1 G
        self.assisted = False
        self.release = None  # running release: held pose, frames, distance backed off

    def _reset_sim(self) -> None:
        """Physics back to the start: cylinder at obj_xy, gripper parked above it, open."""
        mj, m, d = self.mj, self.m, self.data
        mj.mj_resetData(m, d)
        d.eq_active[:] = 0
        d.qpos[self.o_adr : self.o_adr + 3] = [*self.obj_xy, self.h / 2 + 0.0005]
        d.qpos[self.o_adr + 3 : self.o_adr + 7] = [1, 0, 0, 0]
        R0 = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # pointing down
        p0 = np.array([*self.obj_xy, 0.30])  # parked above the cylinder
        q0 = Rot.from_matrix(R0).as_quat()[[3, 0, 1, 2]]
        d.qpos[self.g_adr : self.g_adr + 3], d.qpos[self.g_adr + 3 : self.g_adr + 7] = p0, q0
        d.mocap_pos[0], d.mocap_quat[0] = p0, q0
        d.ctrl[:] = S.MAX_HALF
        mj.mj_forward(m, d)
        for _ in range(100):
            mj.mj_step(m, d)
        self.target = (p0 - 0.0, q0)  # gripper origin, wxyz

    def grasp_z(self, hand_z: float) -> float:
        """Grasp height on the cylinder (assisted grasps): mid-height, its centre of mass, so it
        does not swing in the fingers when the hand accelerates or turns."""
        return self.h / 2

    def object_pos(self) -> np.ndarray:
        return self.data.xpos[self.oid].copy()

    def opening(self) -> float:
        return float(self.data.qpos[self.jl] + self.data.qpos[self.jr])

    # ------------------------------------------------------------------ control
    def _gate(self, state: int | None, p_none: float = 0.0) -> bool:
        """state 0 none / 1 touch / 2 hold, None = no hand; p_none = P(NONE) -> close command."""
        if state is None:
            return self.close  # no evidence either way
        self.sure_none = self.sure_none + 1 if p_none >= self.fast_p else 0
        if not self.close:
            grab = state == 2 or (self.close_on == "contact" and state == 1)
            self.run = self.run + 1 if grab else 0
            if self.run >= self.on_frames:
                self.close, self.run = True, 0
        else:
            rel = state == 0 if self.release_on == "none" else state != 2
            self.run = self.run + 1 if rel else 0
            if self.run >= self.release_frames or self.sure_none >= self.fast_frames:
                self.close, self.run = False, 0
        return self.close

    def _smooth(self, pad, R, H, t=None):
        if self.smoothing == "one_euro":  # t: capture time (s), else 30 fps assumed
            f = self.filt
            self.pad, self.R = f["pad"](pad, t), f["R"](R, t)
            self.H = _pose(f["Hp"](H[:3, 3], t), f["HR"](H[:3, :3], t))
        elif self.pad is None:
            self.pad, self.R, self.H = pad, R, H
        else:
            self.pad = (1 - SMOOTH) * self.pad + SMOOTH * pad
            self.R = _blend(self.R, R, SMOOTH)
            Hp = (1 - SMOOTH) * self.H[:3, 3] + SMOOTH * H[:3, 3]
            self.H = _pose(Hp, _blend(self.H[:3, :3], H[:3, :3], SMOOTH))

    def step(self, joints_world: np.ndarray | None, state_probs: np.ndarray, present: bool,
             finger: np.ndarray | None = None, t: float | None = None,
             **frame) -> dict:  # fmt: skip
        """One camera frame: hand joints [21, 3] (table frame, m; None / NaN = no 3D hand),
        the model's hand-state probabilities [3], hand present, per-finger contact probabilities
        [5] (needed by the fusion gate); frame: extra inputs of a richer fusion gate
        (joints_pred, lm, gray, bgr; see touch.interact.grab_gate.FusionGate.step)
        -> controller info."""
        st = int(np.argmax(state_probs)) if present else None
        raw = st is not None and st >= 1
        was_closed = self.close
        if self.fusion is not None:
            fi = np.zeros(5) if finger is None else finger
            if self.fusion.needs_frames:
                frame.setdefault("joints_world", joints_world)
                close = self.close = self.fusion.step(state_probs, fi, present, **frame)
            else:
                close = self.close = self.fusion.step(state_probs, fi, present)
        else:
            close = self._gate(st, float(state_probs[0]) if present else 0.0)
        if joints_world is not None and np.isfinite(joints_world).all():
            org, R, _ = S.hand_to_gripper(joints_world)
            self._smooth(org + S.PAD_Z * R[:, 2], R, S.hand_frame(joints_world), t)
        obj = self.object_pos()
        pad_R = None
        if self.pad is not None:
            pad, R = self.pad.copy(), self.R.copy()
            dist = float(np.linalg.norm(pad[:2] - obj[:2]))
            near = self.assist and dist < ASSIST_R and pad[2] < self.h + 0.06
            if close and not was_closed:  # grasp starts: fix G relative to the hand frame
                self.assisted = bool(near)
                if near:
                    G = _pose(np.array([obj[0], obj[1], self.grasp_z(pad[2])]),
                              horizontal_grasp(R))  # fmt: skip
                else:
                    G = _pose(pad, R)
                self.rel = np.linalg.inv(self.H) @ G
                self.G0, self.H0 = G, self.H.copy()
            if close and self.rel is not None:
                T = self.H @ self.rel
                pad_pos, pad_R = T[:3, 3], T[:3, :3]
                if self.upright and self.assisted:  # yaw only: rotate G0 about the vertical
                    yaw = _yaw(self.H[:3, :3]) - _yaw(self.H0[:3, :3])
                    Rz = Rot.from_euler("z", yaw).as_matrix()
                    pad_R = Rz @ self.G0[:3, :3]
                    pad_pos = self.H[:3, 3] + Rz @ (self.G0[:3, 3] - self.H0[:3, 3])
            else:
                self.rel = None
                if near:
                    w = float(np.clip((ASSIST_R - dist) / (ASSIST_R / 2), 0.0, 1.0))
                    pad[:2] = (1 - w) * pad[:2] + w * obj[:2]
                    R = _blend(R, horizontal_grasp(R), w)
                pad_pos, pad_R = pad, R
            half = 0.0 if close else S.MAX_HALF
            pad_pos = S.above_table(pad_pos[None], pad_R[None], np.array([max(half, 0.035)]))[0]
            org = pad_pos - S.PAD_Z * pad_R[:, 2]
            self.target = (org, Rot.from_matrix(pad_R).as_quat()[[3, 0, 1, 2]])
        if was_closed and not close:
            q = np.array(self.target[1])
            R1 = Rot.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            R_up = R1  # assisted: level the cylinder (grasp-time tilt, current heading)
            if self.assisted and getattr(self, "G0", None) is not None:
                yaw = _yaw(self.H[:3, :3]) - _yaw(self.H0[:3, :3])
                R_up = Rot.from_euler("z", yaw).as_matrix() @ self.G0[:3, :3]
            self.release = {"pad": self.target[0] + S.PAD_Z * R1[:, 2], "R1": R1, "R_up": R_up,
                            "n": 0, "down": 0.0, "placed": None, "back": 0.0}  # fmt: skip
        if close:
            self.release = None
        braking = False
        if self.release is not None:  # level, set down, brake, open, back off (then follow)
            r = self.release
            r["n"] += 1
            lv = min(1.0, r["n"] / LEVEL_FRAMES)
            Rr = _blend(r["R1"], r["R_up"], lv)
            q = self.data.xquat[self.oid]
            ax = Rot.from_quat(q[[1, 2, 3, 0]]).as_matrix()[:, 2]  # cylinder axis
            sin = float(np.hypot(ax[0], ax[1]))
            bottom = self.object_pos()[2] - (self.h / 2 * abs(ax[2]) + self.d / 2 * sin)
            if r["placed"] is None:  # 1. level and lower until the object rests on the table
                if self.assisted and (lv < 1.0 or (bottom > PLACE_TOL
                                                   and r["down"] < PLACE_MAX)):  # fmt: skip
                    r["down"] += min(PLACE_STEP, max(bottom, 0.0))
                else:
                    r["placed"] = r["n"]
            k = r["n"] - (r["placed"] or r["n"])  # frames since placed
            braking = r["placed"] is None or k < RELEASE_STOP  # 2. fingers still closed
            if not braking and (self.opening() >= self.d + RELEASE_CLEAR or k > RELEASE_MAX):
                r["back"] = min(RETREAT, r["back"] + RETREAT / RETREAT_FRAMES)  # 3. back off
            pad = r["pad"] - r["down"] * np.array([0.0, 0.0, 1.0])
            org = pad - (S.PAD_Z + r["back"]) * Rr[:, 2]  # z = approach axis
            self.target = (org, Rot.from_matrix(Rr).as_quat()[[3, 0, 1, 2]])
            if r["back"] >= RETREAT:
                self.release = None
        self._physics(0.0 if close or braking else S.MAX_HALF)
        return {"close": close, "raw": raw, "assisted": self.assisted, "object": self.object_pos(),
                "opening": self.opening(), "pad_target": None if pad_R is None else
                self.target[0] + S.PAD_Z * pad_R[:, 2]}  # fmt: skip

    def _physics(self, half: float) -> None:
        mj, m, d = self.mj, self.m, self.data
        p1, q1 = self.target
        p0, q0 = d.mocap_pos[0].copy(), d.mocap_quat[0].copy()
        if np.dot(q0, q1) < 0:
            q1 = -q1
        dt = S.SUB * m.opt.timestep
        v_t = (p1 - p0) / dt
        w_t = S._quat_err(q1, q0) / dt
        vel6 = np.zeros(6)
        step = self.close_speed * m.opt.timestep
        for s in range(S.SUB):
            d.ctrl[:] = np.clip(half, d.ctrl - step, d.ctrl + 4 * step)  # opens faster
            a = (s + 1) / S.SUB
            d.mocap_pos[0] = (1 - a) * p0 + a * p1
            qa = (1 - a) * q0 + a * q1
            d.mocap_quat[0] = qa / np.linalg.norm(qa)
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, self.gid, vel6, 0)
            f = S.KP * (d.mocap_pos[0] - d.xpos[self.gid]) + S.KD * (v_t - vel6[3:])
            tq = S.KPR * S._quat_err(d.mocap_quat[0], d.xquat[self.gid]) + S.KDR * (w_t - vel6[:3])
            d.xfrc_applied[self.gid, :3] = f * min(1.0, S.F_MAX / max(np.linalg.norm(f), 1e-9))
            d.xfrc_applied[self.gid, 3:] = tq * min(1.0, S.T_MAX / max(np.linalg.norm(tq), 1e-9))
            mj.mj_step(m, d)
