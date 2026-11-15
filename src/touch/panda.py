"""Franka Panda (MuJoCo Menagerie panda.xml, third_party/mujoco_panda) for robot replays: scene
assembly and damped-least-squares IK on the finger-pad centre ("tcp" site).

Changes to the Menagerie model (all in panda_mjcf):
  * base body at (x, y, 0) with a yaw, arm links gravity compensated
  * "tcp" site at the finger-pad centre, TCP m along the hand's z (approach) axis; the fingers
    close along the hand's y axis (= robot_sim's pad frame: y closing, z approach)
  * finger tendon servo kp FINGER_KP, force capped at 2 x GRIP_N (Franka Hand: 70 N continuous
    per finger); Menagerie's kp 100 squeezed a 3.5 cm object with ~2 N
  * elliptic friction cones, impratio 10, pad friction 1.5 (as robot_sim's gripper)
IK collision avoidance (solve(avoid=clearance)): robot geoms (not the base) within the
clearance of the table or an object are pushed out along the contact normal, as extra
higher-weighted rows; fingers may touch objects. A geom margin on the table / objects (with an
equal gap: detected, but no contact force) makes MuJoCo report them before they touch.
Joint servos: Menagerie's PD (force = kp (ctrl - q) - kv qdot, kv / kp = 0.1 on every joint);
command ctrl = q_target + 0.1 qdot_target to remove the 0.1 s velocity lag.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

PANDA = Path(__file__).resolve().parents[2] / "third_party/mujoco_panda/panda.xml"
TCP = 0.1034  # hand origin -> centre of the finger pads along hand z (m)
MAX_OPEN = 0.08  # Panda stroke (both fingers)
FINGER_MAX = 0.04  # per finger
VMAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]) * 0.8  # rad/s, 80% of limits
KV_OVER_KP = 0.1  # Menagerie joint servos
Q_HOME = np.array([0, -0.3, 0, -2.2, 0, 2.0, 0.785])
FINGER_KP = 4000.0  # N/m on the finger tendon (Menagerie: 100): squeezes at the force cap
GRIP_N = 70.0  # per finger: Franka Hand continuous grasp force (tendon cap = 2 x GRIP_N)
PAUSE = 8  # 30 fps steps the arm waits while the fingers close / open (retime)
W_AVOID = 3.0  # IK weight of the collision-avoidance rows (pose task: 1 per metre)


def finger_ctrl(width: float) -> float:
    """Opening width (both fingers, m) -> Menagerie gripper command (0..255)."""
    return float(np.clip(width, 0.0, MAX_OPEN)) / 2 / FINGER_MAX * 255


def panda_mjcf(base_xy, base_yaw: float, world: str = "", assets: str = "") -> str:
    """MJCF string: the Panda (changes in the module docstring) + `world` (worldbody XML: table,
    objects, lights, cameras) + `assets` (asset XML, e.g. object meshes)."""
    txt = PANDA.read_text()
    meshdir = PANDA.parent / "assets"
    txt = txt.replace('meshdir="assets"', f'meshdir="{meshdir}"')
    for b in ("link", "hand", "left_finger", "right_finger"):  # gravity compensated arm
        txt = txt.replace(f'<body name="{b}', f'<body gravcomp="1" name="{b}')
    hand = '<body gravcomp="1" name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">'
    assert hand in txt
    txt = txt.replace(hand, hand + f'<site name="tcp" pos="0 0 {TCP}" size="0.005"/>')
    gain = FINGER_KP * FINGER_MAX / 255
    kp, kv = FINGER_KP, FINGER_KP / 10
    old = 'gainprm="0.01568627451 0 0" biasprm="0 -100 -10"'
    assert old in txt
    txt = txt.replace(old, f'gainprm="{gain} 0 0" biasprm="0 -{kp} -{kv}"')
    fr = 'forcerange="-100 100" ctrlrange="0 255"'
    assert fr in txt
    txt = txt.replace(fr, f'forcerange="-{2 * GRIP_N} {2 * GRIP_N}" ctrlrange="0 255"')
    col = '<geom type="mesh" group="3"/>'  # collision default: rubber-like pads
    assert col in txt
    txt = txt.replace(col, '<geom type="mesh" group="3" friction="1.5 0.05 0.002" condim="4"/>')
    opt = '<option integrator="implicitfast"/>'
    assert opt in txt
    txt = txt.replace(opt, '<option integrator="implicitfast" timestep="0.002" '
                      'cone="elliptic" impratio="10"/>'
                      '<visual><global offwidth="960" offheight="720"/>'
                      '<quality shadowsize="2048"/></visual>')  # fmt: skip
    q = Rot.from_euler("z", base_yaw).as_quat()[[3, 0, 1, 2]]
    link0 = '<body gravcomp="1" name="link0" childclass="panda">'
    assert link0 in txt
    txt = txt.replace(link0, f'<body name="base" pos="{base_xy[0]} {base_xy[1]} 0" '
                      f'quat="{q[0]} {q[1]} {q[2]} {q[3]}">' + link0)  # fmt: skip
    i = txt.index("</worldbody>")
    txt = txt[:i] + "</body>" + txt[i:]  # close "base" after the arm
    txt = txt.replace("<worldbody>", "<worldbody>" + world, 1)
    return txt.replace("<asset>", "<asset>" + assets, 1)


def _rotvec(M: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis x angle), the log map; scipy's Rotation for
    angles near 180 deg, a closed form otherwise (the IK's hot loop: ~10x faster)."""
    c = (M[0, 0] + M[1, 1] + M[2, 2] - 1.0) / 2.0
    v = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    if c > 0.9999999:
        return v / 2.0
    if c < -0.99:
        return Rot.from_matrix(M).as_rotvec()
    a = math.acos(c)
    return v * (a / (2.0 * math.sin(a)))


FLIP = np.diag([-1.0, -1.0, 1.0])  # the gripper turned 180 deg about its approach axis


def chain_symmetric(Rs, R_ref) -> np.ndarray:
    """Gripper target rotations [n, 3, 3] -> the same poses, each either R or R FLIP (the
    gripper is symmetric under that turn), chosen consistently along the path: the one closer
    to the previous frame's choice (the first: to R_ref). Choosing per frame by the arm's
    current pose flipped mid-motion during a pour and swung the wrist half a turn (Andy
    2026-11-14: "two jitter points")."""
    out = np.array(Rs, float)
    prev = np.asarray(R_ref, float)
    for k in range(len(out)):
        Rf = out[k] @ FLIP
        if np.linalg.norm(_rotvec(Rf @ prev.T)) < np.linalg.norm(_rotvec(out[k] @ prev.T)):
            out[k] = Rf
        prev = out[k]
    return out


class PandaIK:
    """IK on a private MjData of model m (the arm's joints joint1..7, site "tcp")."""

    def __init__(self, m):
        import mujoco

        self.mj, self.m = mujoco, m
        self.d = mujoco.MjData(m)
        jid = [m.joint(f"joint{k}").id for k in range(1, 8)]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([m.jnt_dofadr[j] for j in jid])
        self.lo = np.array([m.jnt_range[j][0] for j in jid]) + 0.02
        self.hi = np.array([m.jnt_range[j][1] for j in jid]) - 0.02
        self.sid = m.site("tcp").id
        names = ["hand", "left_finger", "right_finger"] + [f"link{k}" for k in range(2, 8)]
        self.robot = {m.body(n).id for n in names}  # link0 / link1 sit on the table
        self.fingers = {m.body("left_finger").id, m.body("right_finger").id}
        self.base = {m.body(n).id for n in ("link0", "link1")}
        # per geom, for the vectorised contact filter in _avoid_rows
        gb = np.asarray(m.geom_bodyid)
        self.g_robot = np.isin(gb, list(self.robot))
        self.g_base = np.isin(gb, list(self.base))
        self.g_finger = np.isin(gb, list(self.fingers))
        self.g_world = gb == 0

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        self.mj.mj_kinematics(self.m, self.d)
        self.mj.mj_comPos(self.m, self.d)  # mj_jacSite needs cdof
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def _avoid_rows(self, clear: float, objects: bool = True):
        """Contacts (MuJoCo collision on the IK pose) of robot geoms closer than `clear` to the
        table or an object -> rows (J_n [k, nv], e [k]): move the robot point out along the
        contact normal by clear - dist. Fingers may touch objects (the grasp), not the table."""
        mj, m, d = self.mj, self.m, self.d
        mj.mj_collision(m, d)
        if d.ncon == 0:
            return None, None
        con = d.contact
        g1, g2, dist = con.geom1, con.geom2, con.dist  # vectorised filter (same rules)
        r1, r2 = self.g_robot[g1], self.g_robot[g2]
        keep = (r1 != r2) & ~self.g_base[g1] & ~self.g_base[g2] & (dist < clear)
        rg, og = np.where(r1, g1, g2), np.where(r1, g2, g1)
        keep &= ~(self.g_finger[rg] & ~self.g_world[og])  # finger on an object: allowed
        if not objects:  # the table only (the IK model's objects stay at their start poses)
            keep &= self.g_world[og]
        idx = np.flatnonzero(keep)
        if not len(idx):
            return None, None
        rows, errs = [], []
        jp = np.zeros((3, m.nv))
        frame, pos = con.frame, con.pos
        for i in idx:
            n = frame[i, :3] * (-1.0 if r1[i] else 1.0)  # frame normal: geom1 -> geom2
            mj.mj_jac(m, d, jp, None, pos[i], m.geom_bodyid[rg[i]])
            rows.append(n @ jp)
            errs.append(clear - dist[i])
        return np.array(rows), np.array(errs)

    def solve(self, p, R, q0, iters: int = 60, w_rot: float = 0.3, lam: float = 0.03,
              avoid: float = 0.0, flip: bool = True, objects: bool = True):  # fmt: skip
        """Pad-centre target p [3], R [3, 3] (columns: x, y closing, z approach) from q0 ->
        q [7], position error (m), orientation error (rad). Joint limits clamped (best effort
        out of reach); null-space pull to Q_HOME. The gripper is symmetric under a 180 deg turn
        about its approach axis: the closer of R and R Rz(pi) to the current pose is used."""
        _, Rc = self.fk(q0)
        Rf = R @ FLIP
        if flip and np.linalg.norm(_rotvec(Rf @ Rc.T)) < np.linalg.norm(_rotvec(R @ Rc.T)):
            R = Rf  # flip=False: R as given (a path already chained, chain_symmetric)
        q = np.array(q0, float)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            pc, Rc = self.fk(q)
            ep = p - pc
            er = _rotvec(R @ Rc.T)
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 5e-3 and not avoid:
                break
            self.mj.mj_jacSite(self.m, self.d, jp, jr, self.sid)
            J = np.vstack([jp[:, self.dadr], w_rot * jr[:, self.dadr]])
            e = np.r_[ep, w_rot * er]
            if avoid > 0:  # collision rows, weighted above the pose task
                Jc, ec = self._avoid_rows(avoid, objects)
                if Jc is not None:
                    J = np.vstack([J, W_AVOID * Jc[:, self.dadr]])
                    e = np.r_[e, W_AVOID * ec]
            JJ = J @ J.T + lam**2 * np.eye(len(e))
            X = np.linalg.solve(JJ, np.c_[e, J])  # one factorisation for both
            dq = J.T @ X[:, 0]
            N = np.eye(7) - J.T @ X[:, 1:]  # null space: drift to neutral
            dq += N @ (0.05 * (Q_HOME - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        pc, Rc = self.fk(q)
        er = float(np.linalg.norm(_rotvec(R @ Rc.T)))
        return q, float(np.linalg.norm(p - pc)), er

    def track(self, P, Rs, frames, iters=60, avoid: float = 0.0, carried_from=None):
        """Sequential IK over the given frames (warm start) -> q [n, 7], pos err [n], rot err.
        carried_from (frame, held_away): from it on only the table is avoided, not the objects:
        the IK model's objects never move, so the carried cup's start spot is empty by then."""
        frames = np.asarray(frames)
        Rc = chain_symmetric(np.asarray(Rs)[frames], np.asarray(Rs)[frames[0]])  # as given at t0
        q, _, _ = self.solve(P[frames[0]], Rc[0], Q_HOME.copy(), iters=300, avoid=avoid,
                             flip=False)  # fmt: skip
        Q, ep, er = [], [], []
        for t, Rt in zip(frames, Rc, strict=True):
            free = carried_from is not None and t >= carried_from
            q, a, b = self.solve(P[t], Rt, q, iters=iters, avoid=avoid, flip=False,
                                 objects=not free)  # fmt: skip
            Q.append(q)
            ep.append(a)
            er.append(b)
        return np.array(Q), np.array(ep), np.array(er)


HELD_AWAY = 0.05  # m: the pad this far from where it closed -> the cup has left its start spot


def held_away(P, close, away: float = HELD_AWAY):
    """First frame after the grasp where the pad is `away` from where it closed (the cup carried
    clear of its start spot) -> PandaIK.track(carried_from=...); None if never. Andy 2026-11-15:
    carrying the can back over its start spot, the arm avoided the (empty) spot, fell 13 cm
    behind his steady hand and then rushed to catch up."""
    close = np.asarray(close, bool)
    if not close.any():
        return None
    k = int(np.argmax(close))
    far = np.flatnonzero(np.linalg.norm(np.asarray(P)[k:] - np.asarray(P)[k], axis=1) > away)
    return k + int(far[0]) if len(far) else None


# ------------------------------------------------------------------ timing + simulation
SUB = 17  # physics steps of 2 ms per 30 fps step (= robot_sim.SUB)


def retime(Q, close, pause: int = PAUSE):
    """Human frames -> robot steps: each frame takes as many 1/30 s steps as the fastest joint
    needs at VMAX, plus `pause` steps at each close / open. -> list of (human frame, q target)."""
    out = [(0, Q[0])]
    for h in range(1, len(Q)):
        n = max(1, math.ceil(np.max(np.abs(Q[h] - Q[h - 1]) / (VMAX / 30.0))))
        for k in range(1, n + 1):
            out.append((h, Q[h - 1] + (Q[h] - Q[h - 1]) * k / n))
        if close[h] != close[h - 1]:
            out += [(h, Q[h])] * pause
    return out


VMAX_RT = VMAX / 0.8  # rad/s: the real-time follower may use the Panda's full joint speeds


def retime_realtime(Q, close=None):
    """Human frames -> robot steps one to one (Andy 2026-11-14: the robot in sync with the hand
    at normal speed, no slow-down, no fast-forward, no pauses): each step's joint target moves
    towards the frame's IK solution by at most the joint speed limits (VMAX_RT / 30 per step),
    zero-phase (see below). -> list of (human frame, q target)."""
    Q = np.asarray(Q, float)
    lim = VMAX_RT / 30.0

    def follow(Qs):
        q, out = Qs[0].copy(), [Qs[0].copy()]
        for h in range(1, len(Qs)):
            q = q + np.clip(Qs[h] - q, -lim, lim)
            out.append(q.copy())
        return np.array(out)

    # the take is recorded in full: the mean of a forward and a backward speed-limited follow is
    # still within the limits and centred on the hand's timing (where the hand is too fast
    # the arm starts a little early and ends a little late instead of only lagging)
    Qz = 0.5 * (follow(Q) + follow(Q[::-1])[::-1])
    return [(h, Qz[h].copy()) for h in range(len(Q))]


RELEASE_AHEAD = 12  # human frames: a planned open this close may happen early on contact
HOLD_FOREVER = 10**6  # frames: "until the end" for the hold after the last release
RELEASE_N = 0.3  # N: cup-table normal force that counts as "set down" (cup weight ~0.8 N)


def simulate(xml, steps, close, width, gi, n_obj, render=None, until=None, contacts=False,
             contact_release: bool = True):  # fmt: skip
    """Joint position servos with velocity feed-forward (the Menagerie PD force is
    kp (ctrl - q) - kv qdot, kv / kp = 0.1: ctrl = q_target + 0.1 qdot_target removes the
    0.1 s lag). until: stop after this human frame. contacts: log arm-link contacts.
    contact_release: a guarded release: while the gripper holds the object and the plan opens
    within RELEASE_AHEAD frames, the object touching the table (normal force >= RELEASE_N)
    opens the gripper at once and holds the arm still until the planned open (after the
    last release: until the end; each release is guarded, not only the first) (the set-down
    height planned from a monocular estimate is off by a few mm to cm: no drop from above,
    no pushing into the table).
    log: per step, the state before the step's command + the command (dataset recording)."""
    import mujoco

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    ik = PandaIK(m)
    d.qpos[ik.qadr] = steps[0][1]
    d.ctrl[:7] = steps[0][1]
    d.ctrl[7] = 255
    mujoco.mj_forward(m, d)
    for _ in range(300):  # objects settle, arm holds its first pose
        mujoco.mj_step(m, d)
    obj_ids = [m.body(f"obj{i}").id for i in range(n_obj)]
    fin = [m.jnt_qposadr[m.joint(f"finger_joint{k}").id] for k in (1, 2)]
    fingers = {m.body("left_finger").id, m.body("right_finger").id}
    arm = {m.body(n).id for n in ["hand"] + [f"link{k}" for k in range(8)]}
    traj, frames, human, opening, tcp, hits = [], [], [], [], [], []
    log = {k: [] for k in ("q", "qd", "fingers", "tcp", "tcp_R", "obj_pos", "obj_quat",
                           "state", "q_cmd", "grip_open")}  # fmt: skip
    vadr = np.array([m.jnt_dofadr[m.joint(f"joint{k}").id] for k in range(1, 8)])
    obj_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in obj_ids}
    table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
    close = np.asarray(close, bool)
    nxt_open = np.full(len(close), -1)  # next planned open frame from each frame
    k_open = -1
    for i in range(len(close) - 1, -1, -1):
        k_open = i if not close[i] else k_open
        nxt_open[i] = k_open
    held_until, q_hold = -1, None  # contact release: open (and hold the arm) up to this frame
    ci_ = np.flatnonzero(close)
    final_open = int(ci_[-1]) + 1 if len(ci_) else -1  # the last planned open
    first_close = int(ci_[0]) if len(ci_) else len(close)  # fully open before it
    f6 = np.zeros(6)
    q_prev = steps[0][1]
    sub = SUB
    dt = sub * m.opt.timestep
    for h, q in steps:
        if until is not None and h > until:
            break
        if (contact_release and h >= held_until and table >= 0 and close[h]
                and 0 <= nxt_open[h] - h <= RELEASE_AHEAD):  # fmt: skip
            for c_i in range(d.ncon):
                c = d.contact[c_i]
                if {c.geom1, c.geom2} & obj_geoms and table in (c.geom1, c.geom2):
                    mujoco.mj_contactForce(m, d, c_i, f6)
                    if f6[0] >= RELEASE_N:
                        held_until, q_hold = int(nxt_open[h]), d.qpos[ik.qadr].copy()
                        if held_until == final_open:  # the last release: the arm stays where
                            # the object touched down (Andy 2026-11-14: "after release don't
                            # move the robot hand"; moving to the planned pose 1-2 cm off
                            # jolted the open fingers into the cup)
                            held_until = len(close) + HOLD_FOREVER
                        break
        early = 0 <= h < held_until
        if early:
            q = q_hold
        is_closed = bool(close[h]) and not early
        open_w = min(MAX_OPEN, width[h] + 0.02)
        if h < first_close:  # fully open on the way in (Andy 2026-11-15)
            open_w = MAX_OPEN
        d.ctrl[7] = 0.0 if is_closed else open_w / 2 / 0.04 * 255
        qd = (q - q_prev) / dt
        # dataset log: the state BEFORE this step's command (observation -> action pairs)
        log["q"].append(d.qpos[ik.qadr].copy())
        log["qd"].append(d.qvel[vadr].copy())
        log["fingers"].append(d.qpos[fin].copy())
        log["tcp"].append(d.site_xpos[ik.sid].copy())
        log["tcp_R"].append(d.site_xmat[ik.sid].reshape(3, 3).copy())
        log["obj_pos"].append(d.xpos[obj_ids].copy())
        log["obj_quat"].append(d.xquat[obj_ids].copy())
        log["state"].append(np.r_[d.qpos, d.qvel])
        log["q_cmd"].append(np.asarray(q, float).copy())
        log["grip_open"].append(not is_closed)
        for s in range(sub):
            d.ctrl[:7] = q_prev + (q - q_prev) * (s + 1) / sub + 0.1 * qd
            mujoco.mj_step(m, d)
        q_prev = q
        traj.append(d.xpos[obj_ids].copy())
        human.append(h)
        opening.append(d.qpos[fin[0]] + d.qpos[fin[1]])
        tcp.append(d.site_xpos[ik.sid].copy())
        if contacts:
            hs = set()
            for c in d.contact[: d.ncon]:
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                for a, b in ((b1, b2), (b2, b1)):
                    if a in arm | fingers and b not in arm | fingers:
                        hs.add(f"{m.body(a).name}-{m.body(b).name or 'world'}")
            hits.append(sorted(hs))
        if render is not None:
            frames.append(render(m, d))
    return {"obj": np.array(traj), "human": np.array(human), "frames": frames,
            "opening": np.array(opening), "tcp": np.array(tcp), "hits": hits,
            "log": {k: np.array(v) for k, v in log.items()}}  # fmt: skip
