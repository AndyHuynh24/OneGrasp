"""Live demo with a Franka Panda arm instead of touch.live_robot's floating gripper.

The controller is LiveRobot's, unchanged (grab gate, cylinder assist, grasp fixed to the hand
frame, level / set down / brake / open / back off release): each frame it produces a target
gripper pose. Here that pose's pad centre is reached with the Panda (touch.panda): per-frame
IK warm-started from the last command (joint limits clamped: best effort out of reach), joint
steps capped at touch.panda.VMAX (the arm lags a fast hand instead of jumping), joint servos
with velocity feed-forward, fingers squeezing at the Franka Hand's 70 N.

Base placement (once, at start): candidate spots 0.40-0.70 m from the cylinder every 20 deg,
facing it; score = over typical human grasps of the cylinder (side grasps fanned +-60 deg around
the camera -> cylinder direction, as the hand reaches in from the camera side; tilted 45 deg and
top-down grasps; each at grasp height, 10 cm back along the approach and +15 cm lifted): IK
position error + 0.05 x orientation error + 5 x the arm's penetration into the table / cylinder
with the fingers open (m). The sim is shown from its own third-person camera ("view", beside the
base -> cylinder line, on the webcam's side), so the base may be anywhere, including where the
person's shoulder is. The cylinder can be moved later (click): the base stays (the Panda reaches
~0.85 m).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from touch import robot_sim as S
from touch.live_robot import LiveRobot
from touch.panda import (
    FINGER_MAX,
    KV_OVER_KP,
    MAX_OPEN,
    Q_HOME,
    VMAX,
    PandaIK,
    finger_ctrl,
    panda_mjcf,
)

IK_ITERS = 30  # per frame, warm started (converges in a few when the hand moves slowly)
CLEAR = 0.01  # m: the arm keeps this far from the table, the palm / links from the cylinder
MARGIN = 'margin="0.03" gap="0.03"'  # IK model only: contacts reported within 3 cm
# (in the physics model these near-contacts pushed the cylinder up ~6 cm despite the gap)


def _frame(z: np.ndarray) -> np.ndarray:
    """Pad frame with approach axis z and a horizontal closing axis y."""
    z = z / np.linalg.norm(z)
    y = np.cross([0.0, 0.0, 1.0], z)
    y = np.array([1.0, 0, 0]) if np.linalg.norm(y) < 1e-6 else y / np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], 1)


def typical_grasps(obj_xy, cam_xy, h: float):
    """Pad poses a human reaching in from the camera side would use on the cylinder."""
    u = np.asarray(obj_xy, float) - np.asarray(cam_xy, float)
    u = u / max(np.linalg.norm(u), 1e-9)
    out = []
    for lift in (0.0, 0.15):
        c = np.array([obj_xy[0], obj_xy[1], h / 2 + lift])
        for phi in np.radians([-60, -30, 0, 30, 60]):
            a = np.array([u[0] * math.cos(phi) - u[1] * math.sin(phi),
                          u[0] * math.sin(phi) + u[1] * math.cos(phi), 0.0])  # fmt: skip
            for p, R in ((c, _frame(a)), (c + [0, 0, h / 4], _frame(a + [0, 0, -1.0]))):
                out += [(p, R), (p - 0.10 * R[:, 2], R)]  # grasp, 10 cm back (approach)
        out.append(
            (np.array([obj_xy[0], obj_xy[1], h + 0.01 + lift]), _frame(np.array([0, 0, -1.0])))
        )
    return out


def scene_world(obj_xy, h: float, d: float, extra: str = "", margin: bool = False,
                quat=None, z: float | None = None) -> str:  # fmt: skip
    """Worldbody XML: lights, table plane (z = 0), the cylinder "obj0", extra (cameras).
    margin=True: for the IK's collision checks only, never for physics. quat (w x y z) / z: the
    cylinder's orientation / centre height (default: standing, h / 2)."""
    mg = MARGIN if margin else ""
    z = h / 2 + 0.0005 if z is None else z
    qa = "" if quat is None else f' quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"'

    return (
        '<light pos="0.3 0.3 1.6" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>'
        '<light pos="-0.3 -0.5 1.2" dir="0.3 0.5 -1" diffuse="0.4 0.4 0.4"/>'
        '<geom name="table" type="plane" pos="0 0 0" size="1.5 1.5 0.01" '
        f'rgba="0.32 0.3 0.28 1" friction="0.8 0.02 0.001" {mg}/>'
        f'<body name="obj0" pos="{obj_xy[0]} {obj_xy[1]} {z}"{qa}><freejoint/>'
        f'<geom name="cyl" type="cylinder" size="{d / 2} {h / 2}" density="300" '
        'rgba="0.2 0.45 0.85 1" friction="1.0 0.02 0.001" condim="4" '
        f'solref="0.004 1" solimp="0.95 0.99 0.001" {mg}/>'
        # visual only: a white stripe down the side, so turning the cup is visible
        f'<geom name="stripe" type="box" size="0.0015 0.008 {h / 2 * 0.92}" '
        f'pos="{d / 2} 0 0" rgba="1 1 1 1" contype="0" conaffinity="0" mass="0.0001" '
        'group="1"/></body>' + extra
    )


def robot_bodies(m) -> set[int]:
    names = ["hand", "left_finger", "right_finger"] + [f"link{k}" for k in range(8)]
    return {m.body(n).id for n in names}


def penetration(m, d, robot: set[int]) -> float:
    """Summed penetration depth (m) of robot geoms into anything else (table, objects)."""
    pen = 0.0
    for c in d.contact[: d.ncon]:
        a, b = m.geom_bodyid[c.geom1] in robot, m.geom_bodyid[c.geom2] in robot
        if a != b and c.dist < 0:
            pen -= c.dist
    return pen


def place_base(obj_xy, cam_xy, h: float, d: float) -> tuple[np.ndarray, float, dict]:
    """-> base xy, yaw (facing the cylinder), score parts (see the module docstring)."""
    import mujoco

    poses = typical_grasps(obj_xy, cam_xy, h)
    c = np.asarray(obj_xy, float)
    world = scene_world(obj_xy, h, d, margin=True)
    best = None
    for r in (0.40, 0.50, 0.60, 0.70):
        for a in np.radians(np.arange(0, 360, 20)):
            dirn = np.array([math.cos(a), math.sin(a)])
            b = c + r * dirn
            yaw = math.atan2(-dirn[1], -dirn[0])
            m = mujoco.MjModel.from_xml_string(panda_mjcf(b, yaw, world))
            dd = mujoco.MjData(m)
            ik = PandaIK(m)
            robot = robot_bodies(m)
            fin = [m.jnt_qposadr[m.joint(f"finger_joint{k}").id] for k in (1, 2)]
            ep = er = pen = 0.0
            for p, R in poses:
                q, e1, e2 = ik.solve(p, R, Q_HOME, iters=80, avoid=CLEAR)
                dd.qpos[ik.qadr], dd.qpos[fin] = q, FINGER_MAX
                mujoco.mj_forward(m, dd)
                ep, er, pen = ep + e1, er + e2, pen + penetration(m, dd, robot)
            n = len(poses)
            parts = {"ik_mm": 1000 * ep / n, "rot_deg": math.degrees(er / n),
                     "pen_mm": 1000 * pen / n}  # fmt: skip
            score = (ep + 0.05 * er + 5 * pen) / n
            if best is None or score < best[3]:
                best = (b, yaw, parts, score)
    return best[0], best[1], best[2]


class LivePanda(LiveRobot):
    view_camera = "view"  # MuJoCo camera live_sim renders

    def __init__(self, obj_xy, cam_xy=(0.0, -0.6), **kw):
        """cam_xy: the webcam's position on the table plane (table frame, m), for the base
        placement. Other arguments: LiveRobot's."""
        self.cam_xy = np.asarray(cam_xy, float)[:2]
        super().__init__(obj_xy, **kw)

    def _build(self, obj_xy, camera: str) -> None:
        mj = self.mj
        self.base_xy, self.base_yaw, parts = place_base(obj_xy, self.cam_xy, self.h, self.d)
        print(f"Panda base at {self.base_xy.round(3)} yaw {math.degrees(self.base_yaw):.0f} deg; "
              f"typical grasps: IK {parts['ik_mm']:.1f} mm, {parts['rot_deg']:.1f} deg, "
              f"collision {parts['pen_mm']:.1f} mm")  # fmt: skip
        # third-person view: beside the base -> cylinder line (on the webcam's side), raised
        o = np.asarray(obj_xy, float)
        mid = (o + self.base_xy) / 2
        line = o - self.base_xy
        line /= max(np.linalg.norm(line), 1e-9)
        side = np.array([-line[1], line[0]])
        if side @ (self.cam_xy - mid) < 0:
            side = -side
        cpos = np.r_[mid + 0.95 * side - 0.15 * line, 0.75]
        world = scene_world(obj_xy, self.h, self.d,
                            f'<body name="look" pos="{mid[0]} {mid[1]} 0.15"/>'
                            '<camera name="view" mode="targetbody" target="look" fovy="50" '
                            f'pos="{cpos[0]} {cpos[1]} {cpos[2]}"/>' + camera)  # fmt: skip
        self.m = mj.MjModel.from_xml_string(panda_mjcf(self.base_xy, self.base_yaw, world))
        self.data = mj.MjData(self.m)
        ik_world = scene_world(obj_xy, self.h, self.d, margin=True)  # same bodies, + margins
        self.ik = PandaIK(mj.MjModel.from_xml_string(panda_mjcf(self.base_xy, self.base_yaw,
                                                                ik_world)))  # fmt: skip
        self.oid = self.m.body("obj0").id
        self.o_adr = self.m.jnt_qposadr[self.m.body_jntadr[self.oid]]
        self.fin = [self.m.jnt_qposadr[self.m.joint(f"finger_joint{k}").id] for k in (1, 2)]

    def _reset_sim(self) -> None:
        mj, m, d = self.mj, self.m, self.data
        mj.mj_resetData(m, d)
        d.qpos[self.o_adr : self.o_adr + 3] = [*self.obj_xy, self.h / 2 + 0.0005]
        d.qpos[self.o_adr + 3 : self.o_adr + 7] = [1, 0, 0, 0]
        R0 = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # pointing down
        pad0 = np.array([*self.obj_xy, 0.30])  # parked above the cylinder
        q0, _, _ = self.ik.solve(pad0, R0, Q_HOME, iters=300)
        d.qpos[self.ik.qadr] = q0
        d.qpos[self.fin] = FINGER_MAX
        d.ctrl[:7] = q0
        d.ctrl[7] = finger_ctrl(MAX_OPEN)
        self.q_cmd, self.finger = q0.copy(), FINGER_MAX
        mj.mj_forward(m, d)
        for _ in range(100):
            mj.mj_step(m, d)
        self.target = (pad0 - S.PAD_Z * R0[:, 2], Rot.from_matrix(R0).as_quat()[[3, 0, 1, 2]])

    def opening(self) -> float:
        return float(self.data.qpos[self.fin[0]] + self.data.qpos[self.fin[1]])

    def tcp(self) -> np.ndarray:
        return self.data.site_xpos[self.ik.sid].copy()

    def _physics(self, half: float) -> None:
        """Target gripper pose (LiveRobot convention: origin = pad - PAD_Z z) -> IK -> one
        camera frame of physics (robot_sim.SUB steps)."""
        mj, m, d = self.mj, self.m, self.data
        org, q = self.target
        R = Rot.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()
        pad = org + S.PAD_Z * R[:, 2]
        q_ik, _, _ = self.ik.solve(pad, R, self.q_cmd, iters=IK_ITERS, avoid=CLEAR)
        dt = S.SUB * m.opt.timestep
        dq = np.clip(q_ik - self.q_cmd, -VMAX * dt, VMAX * dt)
        qd = dq / dt
        width = min(MAX_OPEN, 2 * half)
        step = self.close_speed * m.opt.timestep
        q_prev = self.q_cmd
        for s in range(S.SUB):
            self.finger = float(np.clip(width / 2, self.finger - step, self.finger + 4 * step))
            d.ctrl[7] = finger_ctrl(2 * self.finger)
            d.ctrl[:7] = q_prev + dq * (s + 1) / S.SUB + KV_OVER_KP * qd
            mj.mj_step(m, d)
        self.q_cmd = q_prev + dq
        if os.environ.get("PANDA_LOG"):  # debugging: per-frame tracking / contacts (JSON lines)
            robot = robot_bodies(m)
            name = lambda g: m.body(m.geom_bodyid[g]).name or "table"  # noqa: E731
            hits = sorted({f"{name(g)}-{name(o)}" for c in d.contact[: d.ncon]
                           for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1))
                           if m.geom_bodyid[g] in robot
                           and m.geom_bodyid[o] not in robot})  # fmt: skip
            rec = {"pad_err_mm": round(float(np.linalg.norm(self.tcp() - pad)) * 1000, 1),
                   "ik_err_mm": round(float(np.linalg.norm(self.ik.fk(q_ik)[0] - pad)) * 1000, 1),
                   "capped": bool((np.abs(q_ik - q_prev) > VMAX * dt + 1e-9).any()),
                   "half": round(half, 3), "open": round(self.opening(), 4),
                   "obj": self.object_pos().round(3).tolist(), "hits": hits}  # fmt: skip
            with open(os.environ["PANDA_LOG"], "a") as fh:
                fh.write(json.dumps(rec) + "\n")
