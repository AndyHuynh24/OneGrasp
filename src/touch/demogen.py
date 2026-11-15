"""Scale one teleoperated Panda pick-and-place demo into many (MimicGen-style), check each in
MuJoCo, store the successful ones (robomimic-style HDF5).

Generation (MimicGen, Mandlekar et al. 2023, arXiv:2310.17596; transit between object-centric
segments as in SkillMimicGen, Garrett et al. 2024, arXiv:2410.18907):
  * the source demo's executed gripper (TCP) poses are split at the gripper events into a PICK
    segment (the approach + close, ~1.3 s before the close to just after it) expressed in the
    cup's frame, and a PLACE segment (the last ~0.8 s of the carry + release + retreat)
    expressed in the place target's frame
  * a new scene moves the cup and the place target; each segment is re-expressed in its new
    frame. The cup is a vertical cylinder (symmetric), so each frame is also turned about the
    vertical by the change in bearing from the robot base: the grasp keeps its direction
    relative to the robot (better reach than a pure translation)
  * the start pose is joined to the pick segment by a transit motion (minimum jerk, slerped
    orientation); the carry between the segments is the source's own carry (its lift, turns
    and tilts of the cup), each pose blended from the cup frame to the place frame
  * collision-avoiding IK (touch.panda.PandaIK, warm started from the source's joints), retimed
    to the joint speed limits, simulated with the same controller as the teleop rollout

Success (checked at the end of each simulated demo): the cup was lifted >= 3 cm, touches the
green pad (a marker-sized square around the place target), upright (tilt <= 15 deg), resting
on the table, the gripper opened around it (released), and no arm link (fingers excepted)
pressed into the table.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from touch.live_panda import CLEAR, scene_world
from touch.panda import MAX_OPEN, PAUSE, PandaIK, chain_symmetric, panda_mjcf, retime, simulate

SCALE_R = 0.10  # m: cup / place target moved uniformly within +-SCALE_R (x and y)
REACH = (0.32, 0.72)  # m: base -> cup / place distance kept in this range
V_TRANSIT = 0.25  # m/s
SAFE_DZ = 0.06  # m: carry this much above the higher of the two segment ends
PICK_PRE, PICK_POST = 40, PAUSE + 4  # steps around the close in the pick segment
PLACE_PRE, PLACE_POST = 25, 45  # steps around the release in the place segment
HOLD_END = 20  # steps the arm holds its last pose (cup settles) before the check
PLACE_TOL, TILT_MAX, MIN_LIFT = 0.03, 15.0, 0.03
PAD = 0.10  # m: the green destination pad, a square the size of the ArUco marker (Andy
# 2026-11-14); placed = the can touches it ("as long as the object touches the pad")
SETTLE = 12  # steps held after standing a lying cup up, before the demo goes on
STAND_TRIES = 3  # stand-up grasps tried (best first) until the cup ends up upright
TILE = (320, 180)  # rendered tile (w, h)
P_LYING = 0.5  # share of scenes (within LYING_REACH) with the cup lying on its side at start
START_JIT = np.array([0.5, 0.35, 0.5, 0.35, 0.6, 0.4, 0.8])  # rad: random start pose range
LYING_HOVER, LYING_LIFT = 0.10, 0.06  # m: above a lying cup before the grasp / lift after
LYING_REACH = 0.62  # m: lying cups only this close to the base (the tilted grasp's reach)
LYING_JIT = math.radians(30)  # a lying source's cup heading varies this much per scene
UPRIGHT_Z = 0.17  # m: gripper height while a lying cup is swung upright (its end clears)
END_GRIP = 0.022  # m: a lying cup to be stood up is gripped this far in from its end
LYING_MIN, YAW_TOL = 70.0, 25.0  # deg: a lying cup's axis >= 70 deg from vertical; heading


# ------------------------------------------------------------------ source demo
def source_from_rollout(log: dict, base, yaw: float, h: float, d: float) -> dict:
    """Teleop rollout log (touch.panda.simulate "log") -> the source for generation, incl.
    the cup's start / end orientation (lying0 / lying_end: on its side; yaw0 / yaw_end: its
    axis heading)."""
    go = np.asarray(log["grip_open"], bool)
    if go.all() or not go[0]:
        raise ValueError("source demo has no open -> close -> open gripper sequence")
    tc = int(np.argmax(~go))
    if go[tc:].any():
        to = tc + int(np.argmax(go[tc:]))
    else:
        raise ValueError("source demo never releases the cup")
    return {"tcp": np.asarray(log["tcp"]), "R": np.asarray(log["tcp_R"]), "go": go,
            "q": np.asarray(log["q"]), "tc": tc, "to": to, "obj": np.asarray(log["obj_pos"])[:, 0],
            "cup": np.asarray(log["obj_pos"])[0, 0, :2].copy(),
            "place": np.asarray(log["obj_pos"])[-1, 0, :2].copy(),
            "base": np.asarray(base, float)[:2], "yaw": float(yaw), "h": h, "d": d,
            **_lie(log, 0, "0"), **_lie(log, -1, "_end")}  # fmt: skip


def _lie(log: dict, i: int, tag: str) -> dict:
    ax = Rot.from_quat(np.asarray(log["obj_quat"])[i, 0][[1, 2, 3, 0]]).as_matrix()[:, 2]
    return {
        f"lying{tag}": bool(abs(ax[2]) < 0.5),
        f"flip{tag}": bool(ax[2] < -0.5),
        f"yaw{tag}": float(math.atan2(ax[1], ax[0])),
    }


# ------------------------------------------------------------------ success
def check(log: dict, obj_end, place_xy, h: float, d: float, hits=None, lying: bool = False,
          goal_yaw: float | None = None, flipped: bool = False,
          ) -> tuple[bool, str, dict]:  # fmt: skip
    """-> (success, reason, metrics). obj_end: the cup's final position (after the last step).
    lying: the goal is the cup lying on its side (axis horizontal, heading goal_yaw mod 180
    deg within YAW_TOL); else standing: upright, or upside down when flipped (the person
    turned it over on purpose)."""
    z = np.asarray(log["obj_pos"])[:, 0, 2]
    lift = float(z.max() - z[0])
    err = float(np.linalg.norm(np.asarray(obj_end)[:2] - np.asarray(place_xy)[:2]))
    ax = Rot.from_quat(np.asarray(log["obj_quat"])[-1, 0][[1, 2, 3, 0]]).as_matrix()[:, 2]
    tilt = float(np.degrees(np.arccos(np.clip(ax[2], -1, 1))))
    rest = abs(float(obj_end[2]) - (d / 2 if lying else h / 2)) < 0.01
    opening = float(np.asarray(log["fingers"])[-1].sum())
    released = opening > d + 0.004 and bool(np.asarray(log["grip_open"])[-1])
    table = 0
    if hits is not None:  # arm links (not fingers) touching the table, steps
        table = sum(any(x.endswith("-world") and "finger" not in x for x in hs) for hs in hits)
    met = {"lift_cm": round(lift * 100, 1), "place_err_cm": round(err * 100, 1),
           "tilt_deg": round(tilt, 1), "released": released, "arm_table_steps": table,
           "cup_end_z_cm": round(float(obj_end[2]) * 100, 1)}  # fmt: skip
    checks = [(lift < MIN_LIFT, f"not lifted ({lift * 100:.1f} cm)")]
    if lying:
        yerr = 0.0
        if goal_yaw is not None:
            yerr = abs((math.degrees(math.atan2(ax[1], ax[0])) - goal_yaw + 90) % 180 - 90)
        met["yaw_err_deg"] = round(yerr, 1)
        checks += [(abs(tilt - 90) > 90 - LYING_MIN, f"not lying ({tilt:.0f} deg)"),
                   (yerr > YAW_TOL, f"turned wrong ({yerr:.0f} deg)")]  # fmt: skip
    elif flipped:
        met["tilt_from_upside_down_deg"] = round(180 - tilt, 1)
        checks += [(180 - tilt > TILT_MAX, f"not upside down ({tilt:.0f} deg)")]
    else:
        checks += [(tilt > TILT_MAX, f"tipped over ({tilt:.0f} deg)")]
    gap = np.maximum(np.abs(np.asarray(obj_end)[:2] - np.asarray(place_xy)[:2]) - PAD / 2, 0)
    off_pad = float(np.linalg.norm(gap)) - d / 2  # > 0: the cup's footprint misses the pad
    met["off_pad_cm"] = round(max(off_pad, 0.0) * 100, 1)
    checks += [(not rest, "cup not resting on the table"),
               (off_pad > 0.002, f"missed the pad by {off_pad * 100:.1f} cm"),
               (not released, "cup not released"),
               (table > 5, "arm hit the table")]  # fmt: skip
    for bad, why in checks:
        if bad:
            return False, why, met
    return True, "ok", met


# ------------------------------------------------------------------ generation
def _bearing(p, base) -> float:
    v = np.asarray(p, float)[:2] - np.asarray(base, float)[:2]
    return math.atan2(v[1], v[0])


def _xform(P, R, src_xy, new_xy, dyaw):
    """Poses expressed in a vertical frame at src_xy -> the same poses in the frame at new_xy
    turned by dyaw about the vertical."""
    Rz = Rot.from_euler("z", dyaw).as_matrix()
    o = np.r_[src_xy, 0.0]
    return (P - o) @ Rz.T + np.r_[new_xy, 0.0], Rz[None] @ R


def _min_jerk(n: int) -> np.ndarray:
    t = np.linspace(0, 1, n + 1)[1:]
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _line(pa, pb, R, n: int):
    """Straight line pa -> pb (minimum jerk, n steps, excludes pa), constant orientation R."""
    w = _min_jerk(n)[:, None]
    return np.asarray(pa) + (np.asarray(pb) - np.asarray(pa)) * w, np.repeat(R[None], n, 0)


def transit(pa, Ra, pb, Rb, dz: float):
    """pa -> up to z_safe -> across -> down to pb (z_safe = higher end + dz); each leg minimum
    jerk at V_TRANSIT; orientation slerped over the whole transit. Excludes pa, includes pb."""
    zs = max(pa[2], pb[2]) + dz
    way = [np.asarray(pa, float), np.r_[pa[:2], zs], np.r_[pb[:2], zs], np.asarray(pb, float)]
    pts = []
    for a, b in zip(way[:-1], way[1:], strict=True):
        dist = float(np.linalg.norm(b - a))
        if dist < 1e-4:
            continue
        n = max(6, math.ceil(dist / (V_TRANSIT / 30)))
        pts += [a + (b - a) * s for s in _min_jerk(n)]
    if not pts:
        pts = [np.asarray(pb, float)]
    P = np.array(pts)
    sl = Slerp([0, 1], Rot.from_matrix(np.stack([Ra, Rb])))
    return P, sl(np.linspace(0, 1, len(P) + 1)[1:]).as_matrix()


def carry(P, R, src, cup_xy, place_xy, dp: float, dl: float, fix):
    """Standing cup: the source's carry, kept with its lift, turns and tilts: each pose
    expressed in the cup frame and in the place frame, blended from the first to the second
    (minimum jerk), the heading change blended the same way; the in-hand slip fix ramps in."""
    Pp, _ = _xform(P, R, src["cup"], cup_xy, dp)
    Pl, _ = _xform(P, R, src["place"], place_xy, dl)
    w = _min_jerk(len(P) + 1)[:-1, None] if len(P) > 1 else np.zeros((1, 1))
    out_P = (1 - w) * Pp + w * Pl + w * fix
    yaw = (1 - w[:, 0]) * dp + w[:, 0] * dl
    out_R = Rot.from_euler("z", yaw[:, None]).as_matrix() @ R
    return out_P, out_R


def carry_between(P, R, pa, Ra, pb, Rb, dyaw: float, zmin: float):
    """The source's carry shape (its displacements, turns and tilts relative to its first pose)
    moved to start at pose (pa, Ra) and end at (pb, Rb): the displacement (turned by dyaw)
    plus a minimum-jerk correction that lands on pb; the source's rotation relative to its
    start applied to the slerp Ra -> Rb and faded out towards the end. Excludes pa."""
    n = len(P)
    w = _min_jerk(n)[:, None]
    Rz = Rot.from_euler("z", dyaw).as_matrix()
    disp = (P - P[0]) @ Rz.T
    out_P = pa + disp + w * ((pb - pa) - disp[-1])
    out_P[:, 2] = np.maximum(out_P[:, 2], zmin)
    rel = Rot.from_matrix(Rz @ R @ R[0].T @ Rz.T)  # the source's turns, world frame
    base = Slerp([0, 1], Rot.from_matrix(np.stack([Ra, Rb])))(w[:, 0])
    fade = Rot.from_rotvec(rel.as_rotvec() * (1 - w))
    return out_P, (fade * base).as_matrix()


def cup_quat(sc: dict, where: str = "cup") -> np.ndarray:
    """MuJoCo (w x y z) orientation of the cup at the start ("cup") or the goal ("place")."""
    r = Rot.from_euler("z", sc[f"{where}_yaw"])
    if sc["lying" if where == "cup" else "goal_lying"]:
        r = r * Rot.from_euler("y", math.pi / 2)  # axis along (cos yaw, sin yaw, 0)
    return r.as_quat()[[3, 0, 1, 2]]


def lying_grasp(xy, yaw: float, s_off: float, d: float):
    """Top-down grasp across a lying cup (axis heading yaw), s_off along the axis from its
    centre: approach straight down, fingers closing across the diameter -> (pad pos, R)."""
    a = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    y = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    R = np.stack([-a, y, [0.0, 0.0, -1.0]], 1)
    return np.r_[np.asarray(xy) + s_off * a[:2], d / 2 + 0.002], R


STAND_ALPHAS = (30.0, 45.0, 60.0)  # deg: tilt of the approach (from vertical) on a lying cup


def stand_up_grasp(sc: dict, d: float, h: float, base):
    """Lying cup to be stood up: a grasp across it END_GRIP in from its top end, the approach
    tilted by alpha from vertical along the cup's axis (lean +-1: towards its top or bottom
    end), and the gripper orientation after turning the wrist about the closing axis so the
    top end points up (the cup hangs upright below the fingers, the gripper pointing
    diagonally down, not horizontally: a horizontal gripper this low is out of the Panda's
    reach). sc["stand"] = (alpha deg, lean); run_one picks the reachable one.
    -> (pad pos, R grasp, R upright, grip height above the cup's centre) or None."""
    alpha, lean = sc.get("stand", (45.0, 1.0))
    s_mag = h / 2 - END_GRIP
    a = np.array([math.cos(sc["cup_yaw"]), math.sin(sc["cup_yaw"]), 0.0])
    y = np.array([-a[1], a[0], 0.0])
    al = math.radians(alpha)
    z = -math.cos(al) * np.array([0.0, 0.0, 1.0]) + math.sin(al) * lean * a
    R = np.stack([np.cross(y, z), y, z], 1)
    gp = np.r_[np.asarray(sc["cup_xy"]) + s_mag * a[:2], d / 2 + 0.002]
    for th in (math.pi / 2, -math.pi / 2):
        Rr = Rot.from_rotvec(y * th).as_matrix()
        if (Rr @ a)[2] > 0.99:  # the top end points up
            R_up = Rr @ R
            if R_up[2, 2] > -0.2:  # the gripper would point up / level: impossible
                return None
            return gp, R, R_up, s_mag
    return None


def _reuse(src: dict, sc: dict) -> bool:
    """The scene's cup starts and ends like the source's (both standing, or a lying source's
    scenes): its segments are reused (MimicGen)."""
    return sc["lying"] == bool(src.get("lying0")) and (
        not sc["lying"] or sc["goal_lying"] == bool(src.get("lying_end"))
    )


def _wrap_half(a: float) -> float:  # -> (-pi/2, pi/2]: a heading change mod 180 deg
    return (a + np.pi / 2) % np.pi - np.pi / 2


def generate(src: dict, sc: dict, start, fix=None) -> dict:
    """Scene sc (sample_scenes) + the robot's start pose (pos, R) -> gripper targets P [n, 3],
    Rs [n, 3, 3], close [n] (bool), the pick / place segment ranges and where the held cup
    should be at the start of the place segment ("cup_b"). fix [3]: shift of the place part
    (in-hand slip correction).
      cup like the source's (standing, or the source's cup was lying: _reuse): the source's
          pick segment in the cup frame, its carry warped from the cup frame to the place
          frame, its place segment in the place frame (MimicGen)
      lying cup: the source's grasp cannot be reused (a 2-finger grasp of a standing cup does
          not fit a lying one): a top-down grasp across the cup (approach straight down,
          close, lift), the source's carry shape moved between that grasp and the goal, and a
          straight-down place, release, retreat up. Stood up (goal_lying False): gripped
          near one end, lifted to UPRIGHT_Z, the wrist turns 90 deg during the carry so the
          cup hangs upright, set down standing. Else laid at the goal heading."""
    tcp, R, go, tc, to = src["tcp"], src["R"], src["go"], src["tc"], src["to"]
    a0, a1 = max(0, tc - PICK_PRE), min(to, tc + PICK_POST)
    b0, b1 = max(a1, to - PLACE_PRE), min(len(tcp), to + PLACE_POST)
    base = src["base"]
    cup_xy, place_xy = np.asarray(sc["cup_xy"]), np.asarray(sc["place_xy"])
    dp = _bearing(cup_xy, base) - _bearing(src["cup"], base)
    dl = _bearing(place_xy, base) - _bearing(src["place"], base)
    fix = np.zeros(3) if fix is None else np.asarray(fix, float)
    p0, R0 = np.asarray(start[0], float), np.asarray(start[1], float)
    if _reuse(src, sc):
        if src.get("lying0"):  # the lying cup's own heading change (mod 180), not the bearing's
            dp = dp + _wrap_half(sc["cup_yaw"] - src["yaw0"] - dp)
        PA, RA = _xform(tcp[a0:a1], R[a0:a1], src["cup"], cup_xy, dp)
        PB, RB = _xform(tcp[b0:b1], R[b0:b1], src["place"], place_xy, dl)
        cup_b = _xform(src["obj"][b0][None], R[b0 : b0 + 1], src["place"], place_xy, dl)[0][0]
        PB = PB + fix
        T1, R1 = transit(p0, R0, PA[0], RA[0], sc.get("transit_dz", 0.0))  # start -> pick
        if b0 - a1 >= 4:  # the human's carry (lift, turns, tilts), warped pick -> place frame
            T2, R2 = carry(tcp[a1:b0], R[a1:b0], src, cup_xy, place_xy, dp, dl, fix)
        else:
            T2, R2 = transit(PA[-1], RA[-1], PB[0], RB[0], SAFE_DZ)
        cA, cB = ~go[a0:a1], ~go[b0:b1]
    elif not sc["goal_lying"]:  # lying cup -> stood up at the goal
        d, h = src["d"], src["h"]
        gp, gR, R_up, s_mag = stand_up_grasp(sc, d, h, base)
        up = np.array([0.0, 0.0, LYING_HOVER])
        lift = np.r_[gp[:2], UPRIGHT_Z]
        hov = gp - 0.10 * gR[:, 2]  # back off along the (tilted) approach
        Pd, Rd = _line(hov, gp, gR, 25)  # in along the approach onto the cup
        Pc, Rc = np.repeat(gp[None], PAUSE + 6, 0), np.repeat(gR[None], PAUSE + 6, 0)
        Pl, Rl = _line(gp, lift, gR, 20)  # straight up, high enough to swing it
        PA = np.concatenate([hov[None], Pd, Pc, Pl])
        RA = np.concatenate([gR[None], Rd, Rc, Rl])
        cA = np.r_[np.zeros(1 + len(Pd), bool), np.ones(len(Pc) + len(Pl), bool)]
        # at the goal: the gripper horizontal, pointing away from the base, cup hanging upright
        out = place_xy - base
        z_up = R_up[:, 2]
        phi = math.atan2(out[1], out[0]) - math.atan2(z_up[1], z_up[0])
        R_goal = Rot.from_euler("z", phi).as_matrix() @ R_up
        touch = np.r_[place_xy, h / 2 + s_mag + 0.002]  # cup bottom on the table
        pre = touch + [0.0, 0.0, 0.08] + fix
        T1, R1 = transit(p0, R0, PA[0], RA[0], 0.0)
        if sc.get("stand_only"):  # stood up where it lay (run_one's first stage): no carry
            T2, R2 = transit(lift, gR, pre, R_goal, 0.0)
        elif b0 - a1 >= 4:  # the human's carry shape; the wrist turns the cup upright meanwhile
            T2, R2 = carry_between(
                tcp[a1:b0], R[a1:b0], lift, gR, pre, R_goal, dp, UPRIGHT_Z - 0.02
            )
        else:
            T2, R2 = transit(lift, gR, pre, R_goal, 0.0)
        Pdn, Rdn = _line(pre, touch + fix, R_goal, 25)  # straight down, held
        hold = np.repeat(Pdn[-1:], PAUSE + 8, 0)
        Pup, Rup = _line(Pdn[-1], Pdn[-1] + [0, 0, 0.12], R_goal, 15)
        PB = np.concatenate([pre[None], Pdn, hold, Pup])
        RB = np.concatenate([R_goal[None], Rdn, np.repeat(R_goal[None], len(hold), 0), Rup])
        cB = np.r_[np.ones(1 + len(Pdn) + 4, bool), np.zeros(len(hold) - 4 + len(Pup), bool)]
        cup_b = pre - fix - [0.0, 0.0, s_mag]  # held upright: centre s_mag below the grip
    else:  # lying cup -> lying at the goal heading
        d = src["d"]
        gp, gR = lying_grasp(cup_xy, sc["cup_yaw"], sc["s_off"], d)
        lp, lR = lying_grasp(place_xy, sc["place_yaw"], sc["s_off"], d)
        up = np.array([0.0, 0.0, LYING_HOVER])
        lift = gp + [0.0, 0.0, LYING_LIFT]
        Pd, Rd = _line(gp + up, gp, gR, 25)  # straight down onto the cup
        Pc, Rc = np.repeat(gp[None], PAUSE + 6, 0), np.repeat(gR[None], PAUSE + 6, 0)
        Pl, Rl = _line(gp, lift, gR, 15)  # straight up
        PA = np.concatenate([gp[None] + up, Pd, Pc, Pl])
        RA = np.concatenate([gR[None], Rd, Rc, Rl])
        cA = np.r_[np.zeros(1 + len(Pd), bool), np.ones(len(Pc) + len(Pl), bool)]
        pre = lp + up + fix
        T1, R1 = transit(p0, R0, PA[0], RA[0], 0.0)
        if b0 - a1 >= 4:
            T2, R2 = carry_between(tcp[a1:b0], R[a1:b0], lift, gR, pre, lR, dp,
                                   d / 2 + LYING_LIFT * 0.5)  # fmt: skip
        else:
            T2, R2 = transit(lift, gR, pre, lR, SAFE_DZ)
        Pdn, Rdn = _line(pre, lp + fix + [0, 0, 0.001], lR, 25)  # straight down, held
        hold = np.repeat(Pdn[-1:], PAUSE + 8, 0)
        Pup, Rup = _line(Pdn[-1], Pdn[-1] + [0, 0, 0.12], lR, 15)
        PB = np.concatenate([pre[None], Pdn, hold, Pup])
        RB = np.concatenate([lR[None], Rdn, np.repeat(lR[None], len(hold), 0), Rup])
        cB = np.r_[np.ones(1 + len(Pdn) + 4, bool), np.zeros(len(hold) - 4 + len(Pup), bool)]
        a = np.array([math.cos(sc["place_yaw"]), math.sin(sc["place_yaw"]), 0.0])
        cup_b = pre - fix - sc["s_off"] * a  # cup centre when held at the pre-place pose
    P = np.concatenate([p0[None], T1, PA, T2, PB])
    Rs = chain_symmetric(np.concatenate([R0[None], R1, RA, R2, RB]), R0)  # no mid-path flips
    close = np.concatenate([[False], np.zeros(len(T1), bool), cA, np.ones(len(T2), bool), cB])
    i_a = 1 + len(T1)
    i_b = i_a + len(PA) + len(T2)
    return {"P": P, "Rs": Rs, "close": close, "pick": (i_a, i_a + len(PA)),
            "place": (i_b, i_b + len(PB)), "cup_b": cup_b}  # fmt: skip


def sample_scenes(src: dict, n: int, seed: int, r: float = SCALE_R, p_lying: float = P_LYING,
                  p_keep_lying: float = 0.0, n_lying: int | None = None) -> list[dict]:  # fmt: skip
    """n scenes: cup / place target within +-r of the source's (both within REACH of the
    base), the cup standing with a random heading, or (p_lying) lying on its side with a
    random axis heading, to be stood up at the goal (or, p_keep_lying, laid at a random goal
    heading); a seed for the robot's random start pose (drawn in run_one). The source's cup
    lying (measured from the camera): every scene's cup lies, heading = the source's turned
    with the bearing +- LYING_JIT, ending like the source's (lying / stood up). A standing
    source cup that ends on its side (the person laid / flipped it on purpose): every scene's
    goal is lying, heading = the source's end heading turned with the target's bearing (not
    "tipped over"). n_lying: exactly this many scenes start lying (random which; replaces
    p_lying for a standing source cup; Andy 2026-11-14: 6 of 12 lying, every one ending like
    the person left the cup)."""
    rng = np.random.default_rng(seed)
    want = None
    if n_lying is not None:
        want = [True] * min(n_lying, n) + [False] * max(0, n - n_lying)
        rng.shuffle(want)
    out = []
    while len(out) < n:
        c = src["cup"] + rng.uniform(-r, r, 2)
        p = src["place"] + rng.uniform(-r, r, 2)
        dc, dp = np.linalg.norm(c - src["base"]), np.linalg.norm(p - src["base"])
        if not (REACH[0] <= dc <= REACH[1] and REACH[0] <= dp <= REACH[1]):
            continue
        if src.get("lying0"):  # the real cup lay on its side: so do all, heading +-LYING_JIT
            lying, goal_lying = True, bool(src.get("lying_end"))
            db = _bearing(c, src["base"]) - _bearing(src["cup"], src["base"])
            dl = _bearing(p, src["base"]) - _bearing(src["place"], src["base"])
            cup_yaw = (src["yaw0"] + db + rng.uniform(-LYING_JIT, LYING_JIT)) % math.pi
            place_yaw = (src["yaw_end"] + dl) % (math.pi if goal_lying else 2 * math.pi)
        elif src.get("flip_end"):  # turned upside down on purpose: standing starts, same goal
            lying, goal_lying = False, False
            cup_yaw = float(rng.uniform(0, 2 * math.pi))
            place_yaw = float(rng.uniform(0, 2 * math.pi))
        elif src.get("lying_end"):  # standing cup laid down on purpose: every scene ends lying
            lying = want[len(out)] if want else bool(rng.random() < p_lying)
            if lying and dc > LYING_REACH:
                if want:
                    continue  # resample: this one must lie
                lying = False
            goal_lying = True
            cup_yaw = float(rng.uniform(0, math.pi if lying else 2 * math.pi))
            dl = _bearing(p, src["base"]) - _bearing(src["place"], src["base"])
            place_yaw = (src["yaw_end"] + dl) % math.pi  # the source's heading, turned along
        else:
            lying = want[len(out)] if want else bool(rng.random() < p_lying)
            if lying and dc > LYING_REACH:
                if want:
                    continue  # resample: this one must lie
                lying = False
            goal_lying = lying and bool(rng.random() < p_keep_lying)
            cup_yaw = float(rng.uniform(0, math.pi if lying else 2 * math.pi))
            place_yaw = float(rng.uniform(0, math.pi if goal_lying else 2 * math.pi))
        out.append({"cup_xy": c, "place_xy": p, "lying": lying, "goal_lying": goal_lying,
                    "goal_flip": bool(src.get("flip_end")) and not lying,
                    "cup_yaw": float(cup_yaw), "place_yaw": float(place_yaw),
                    "s_off": float(rng.uniform(-0.01, 0.01)),
                    "start_seed": int(rng.integers(1 << 30))})  # fmt: skip
    return out


def random_start(ik, src: dict, seed: int):
    """A random collision-free arm pose near the source's start (each joint within START_JIT
    of it), gripper 12-45 cm above the table -> (q [7], pos, R)."""
    import mujoco

    from touch.live_panda import penetration, robot_bodies

    rng = np.random.default_rng(seed)
    m = ik.m
    d = mujoco.MjData(m)
    robot = robot_bodies(m)
    q0 = src["q"][0]
    for _ in range(200):
        q = np.clip(q0 + rng.uniform(-1, 1, 7) * START_JIT, ik.lo, ik.hi)
        p, R = ik.fk(q)
        if not 0.12 <= p[2] <= 0.45:
            continue
        d.qpos[ik.qadr] = q
        mujoco.mj_forward(m, d)
        if penetration(m, d, robot) == 0.0:
            return q, p, R
    p, R = ik.fk(q0)
    return q0.copy(), p, R


# ------------------------------------------------------------------ scene + rollout
def view_camera(base, center) -> str:
    """Third-person camera beside the base -> scene line (same for every generated scene)."""
    c, b = np.asarray(center, float)[:2], np.asarray(base, float)[:2]
    look = 0.75 * c + 0.25 * b
    line = (c - b) / max(np.linalg.norm(c - b), 1e-9)
    side = np.array([-line[1], line[0]])
    pos = np.r_[look + 0.62 * side + 0.05 * line, 0.45]
    return (f'<body name="look" pos="{look[0]} {look[1]} 0.1"/>'
            '<camera name="view" mode="targetbody" target="look" fovy="55" '
            f'pos="{pos[0]} {pos[1]} {pos[2]}"/>')  # fmt: skip


def scene_xml(src: dict, sc: dict, cam: str = "", margin: bool = False) -> str:
    """Panda + table + the cup in its start pose + the goal drawn flat in green (visual only:
    a disc for a standing goal, the cup's footprint for a lying one) + camera."""
    h, d = src["h"], src["d"]
    px, py = sc["place_xy"]
    vis = 'rgba="0.2 0.85 0.3 0.6" contype="0" conaffinity="0" group="1"'
    if sc["goal_lying"]:
        q = Rot.from_euler("z", sc["place_yaw"]).as_quat()[[3, 0, 1, 2]]
        goal = (f'<geom name="target" type="box" size="{h / 2 + 0.008} {d / 2 + 0.008} 0.0005" '
                f'pos="{px} {py} 0.0006" quat="{q[0]} {q[1]} {q[2]} {q[3]}" {vis}/>')  # fmt: skip
    else:  # the pad: a marker-sized square, axis-aligned like the markers (the table frame)
        goal = (f'<geom name="target" type="box" size="{PAD / 2} {PAD / 2} 0.0005" '
                f'pos="{px} {py} 0.0006" {vis}/>')  # fmt: skip
    z = d / 2 + 0.0005 if sc["lying"] else h / 2 + 0.0005
    world = scene_world(sc["cup_xy"], h, d, ("" if margin else goal + cam), margin,
                        quat=cup_quat(sc), z=z)  # fmt: skip
    return panda_mjcf(src["base"], src["yaw"], world)


def width(src: dict) -> float:
    return min(src["d"], MAX_OPEN - 0.01)


def pick_stand_grasp(ik, src: dict, sc: dict, start, ranked: bool = False):
    """The stand-up grasp (approach tilt, lean) whose key poses the arm reaches best: the
    grasp, the lift, the pre-place and the set-down pose (IK position + 0.05 x orientation
    error). ranked: every candidate, best first."""
    best = (np.inf, (45.0, 1.0))
    allc = []
    q0 = src["q"][max(0, src["tc"] - PICK_PRE)]
    for alpha in STAND_ALPHAS:
        for lean in (1.0, -1.0):
            c = {**sc, "stand": (alpha, lean)}
            if stand_up_grasp(c, src["d"], src["h"], src["base"]) is None:
                continue
            g = generate(src, c, start)
            i_a, i_b = g["pick"][0], g["place"][0]
            q, err = q0.copy(), 0.0
            for t in (i_a, i_a + 26, g["pick"][1] - 1, i_b, i_b + 25):
                q, ep, er = ik.solve(g["P"][t], g["Rs"][t], q, iters=120, avoid=CLEAR, flip=False)
                err = max(err, ep + 0.05 * er)
            allc.append((err, (alpha, lean)))
            if err < best[0]:
                best = (err, (alpha, lean))
    if ranked:
        return [c for _, c in sorted(allc)] or [best[1]]
    return best[1]


REACH_TOL = 0.01  # m: worst IK position error along a planned demo (reach_error)


def _track(ik, g: dict, q: np.ndarray, every: int = 4) -> float:
    """Worst IK position error (m) tracking the targets from the pick on, every 4th step."""
    worst = 0.0
    for t in range(g["pick"][0], len(g["P"]), every):
        q, ep, _ = ik.solve(g["P"][t], g["Rs"][t], q, iters=40, avoid=CLEAR, flip=False)
        worst = max(worst, float(ep))
    return worst


def pick_config(ik, src: dict, g: dict, seed: int, n_rand: int = 8) -> tuple[np.ndarray, float]:
    """Arm configuration at the start of the pick segment: the source's there, or one of
    n_rand random restarts (a single warm start gets stuck in a local minimum when the cup
    moved / turned), the one tracking the whole demo best -> (q, worst error m)."""
    rng = np.random.default_rng(seed)
    i_a = g["pick"][0]
    starts = [src["q"][max(0, src["tc"] - PICK_PRE)]]
    best = (None, np.inf)
    for i in range(1 + n_rand):
        q0 = starts[0] if i == 0 else rng.uniform(ik.lo, ik.hi)
        q, ep, _ = ik.solve(g["P"][i_a], g["Rs"][i_a], q0, iters=200, avoid=CLEAR, flip=False)
        if ep > 0.01:
            continue
        w = _track(ik, g, q.copy())
        if w < best[1]:
            best = (q, w)
        if w < 0.003:  # good enough: keep the earliest (the source's when it works)
            break
    if best[0] is None:
        q, ep, _ = ik.solve(
            g["P"][i_a], g["Rs"][i_a], starts[0], iters=200, avoid=CLEAR, flip=False
        )
        best = (q, np.inf)
    return best


def reach_error(ik, src: dict, sc: dict, start) -> float:
    """Worst IK position error (m) along the generated targets from the pick on."""
    return pick_config(ik, src, generate(src, sc, start), sc["start_seed"])[1]


def run_one(args) -> dict:
    """Pool worker: one generated demo -> random start pose, IK, retime, simulate (streaming
    rendered tiles to `queue` as (k, jpeg bytes)), success check. Writes the scene MJCF to
    scene_path."""
    import cv2
    import mujoco

    from touch.live_panda import penetration, robot_bodies

    src, k, sc, cam, scene_path, queue = args
    t0 = time.time()
    ik = PandaIK(mujoco.MjModel.from_xml_string(scene_xml(src, sc, margin=True)))
    q_start, p_start, R_start = random_start(ik, src, sc["start_seed"])
    if src.get("lying0") and _reuse(src, sc):  # heading jitter the arm cannot follow: less
        db = _bearing(sc["cup_xy"], src["base"]) - _bearing(src["cup"], src["base"])
        base_yaw = src["yaw0"] + db
        extra = _wrap_half(sc["cup_yaw"] - base_yaw)
        for f in (1.0, 0.5, 0.0):
            c = {**sc, "cup_yaw": float((base_yaw + f * extra) % math.pi)}
            ik_c = PandaIK(mujoco.MjModel.from_xml_string(scene_xml(src, c, margin=True)))
            if f == 0.0 or reach_error(ik_c, src, c, (p_start, R_start)) < REACH_TOL:
                sc, ik = c, ik_c
                break
    elif not sc["lying"]:  # an arm that cannot reach the scene knocks the (tall) cup over:
        # pull the cup / target towards the source's spots until it can (2026-11-14, can)
        c0, p0 = np.asarray(sc["cup_xy"], float), np.asarray(sc["place_xy"], float)
        for f in (1.0, 0.5, 0.0):
            c = {**sc, "cup_xy": src["cup"] + f * (c0 - src["cup"]),
                 "place_xy": src["place"] + f * (p0 - src["place"])}  # fmt: skip
            ik_c = PandaIK(mujoco.MjModel.from_xml_string(scene_xml(src, c, margin=True)))
            if f == 0.0 or reach_error(ik_c, src, c, (p_start, R_start)) < REACH_TOL:
                sc, ik = c, ik_c
                break
    xml = scene_xml(src, sc, cam)
    if scene_path is not None:
        Path(scene_path).write_text(xml)
    # a lying cup that ends standing (a standing source): stood up where it lies first, then
    # handled as a standing cup from where it really stands (Andy 2026-11-14: "lift the bottle
    # upright first, then treat it as standing and continue, more realistic")
    two_stage = sc["lying"] and not sc["goal_lying"] and not _reuse(src, sc)
    if two_stage:  # where to stand it: where it lies, or pulled towards the source's spots
        # (with the target) until the standing demo from there is reachable (the carry / pour)
        c0, p0 = np.asarray(sc["cup_xy"], float), np.asarray(sc["place_xy"], float)
        for f in (0.0, 0.5, 1.0):
            st_xy = c0 + f * (src["cup"] - c0)
            pl_xy = p0 + f * (src["place"] - p0)
            c = {**sc, "lying": False, "cup_xy": st_xy, "place_xy": pl_xy}
            ik_c = PandaIK(mujoco.MjModel.from_xml_string(scene_xml(src, c, margin=True)))
            if f == 1.0 or reach_error(ik_c, src, c, (p_start, R_start)) < REACH_TOL:
                break
        sc = {**sc, "place_xy": pl_xy}
        xml = scene_xml(src, sc, cam)  # the target may have moved
        if scene_path is not None:
            Path(scene_path).write_text(xml)
        sc_st = {**sc, "place_xy": st_xy, "stand_only": True}
        stands = pick_stand_grasp(ik, src, sc_st, (p_start, R_start), ranked=True)
    W = np.full(8192, width(src))

    md = mujoco.MjData(ik.m)
    robot = robot_bodies(ik.m)

    def ik_plan(g, q_from, ik=ik, md=md, robot=robot):
        """IK from the pick segment on, from the source's arm configuration there or a
        restart (pick_config; a random start pose must not drag the arm into an odd
        configuration); the
        start -> pick transit in joint space (minimum jerk) when collision-free, else the
        Cartesian transit."""
        P, Rs = g["P"], g["Rs"]
        n, i_a = len(P), g["pick"][0]
        Q, ep = np.zeros((n, 7)), np.zeros(n)
        q, _ = pick_config(ik, src, g, sc["start_seed"])
        q, ep[i_a], _ = ik.solve(P[i_a], Rs[i_a], q, iters=40, avoid=CLEAR, flip=False)
        Q[i_a] = q
        for t in range(i_a + 1, n):
            q, ep[t], _ = ik.solve(P[t], Rs[t], q, iters=40, avoid=CLEAR, flip=False)
            Q[t] = q
        w = np.r_[0.0, _min_jerk(max(1, i_a))][:i_a, None]
        Qj = q_from + (Q[i_a] - q_from) * w
        free = True
        for qq in Qj[::2]:
            md.qpos[ik.qadr] = qq
            mujoco.mj_forward(ik.m, md)
            free &= penetration(ik.m, md, robot) == 0.0
        if free:
            Q[:i_a] = Qj
        else:
            q = q_from.copy()
            Q[0] = q
            for t in range(1, i_a):
                q, ep[t], _ = ik.solve(P[t], Rs[t], q, iters=40, avoid=CLEAR, flip=False)
                Q[t] = q
        return Q, ep

    stage1, stand_tries = None, 0
    if two_stage:  # plan + simulate the stand-up: where does the cup really stand? A stand-up
        # that leaves it on its side: the next grasp (tilt / side), up to STAND_TRIES
        for k_try, stand in enumerate(stands[:STAND_TRIES]):
            g1 = generate(src, {**sc_st, "stand": stand}, (p_start, R_start))
            Q1, ep1 = ik_plan(g1, q_start)
            g1 = {**g1, "P": np.r_[g1["P"], np.repeat(g1["P"][-1:], SETTLE, 0)],
                  "Rs": np.r_[g1["Rs"], np.repeat(g1["Rs"][-1:], SETTLE, 0)],
                  "close": np.r_[g1["close"], np.zeros(SETTLE, bool)]}  # fmt: skip
            Q1, ep1 = np.r_[Q1, np.repeat(Q1[-1:], SETTLE, 0)], np.r_[ep1, np.zeros(SETTLE)]
            s1 = simulate(xml, retime(Q1, g1["close"]), g1["close"], W, 0, 1)
            stood = np.asarray(s1["obj"][-1, 0], float)
            qw = np.asarray(s1["log"]["obj_quat"])[-1, 0][[1, 2, 3, 0]]
            ax = Rot.from_quat(qw).as_matrix()[:, 2]
            upright = (
                ax[2] > math.cos(math.radians(TILT_MAX)) and abs(stood[2] - src["h"] / 2) < 0.01
            )
            # the second part sees the cup standing where it stands (collisions), and the arm
            # leaves the stood-up cup up and over (not through it) towards the pick
            sc2 = {**sc, "lying": False, "cup_xy": stood[:2], "transit_dz": SAFE_DZ}
            stage1 = (g1, Q1, ep1, sc2, ik.fk(Q1[-1]))
            stand_tries = k_try + 1
            if upright:
                break

    if stage1 is not None:
        ik2 = PandaIK(mujoco.MjModel.from_xml_string(scene_xml(src, stage1[3], margin=True)))
        md2, robot2 = mujoco.MjData(ik2.m), robot_bodies(ik2.m)

    def plan(fix):
        if stage1 is None:
            g = generate(src, sc, (p_start, R_start), fix)
            Q, ep = ik_plan(g, q_start)
        else:  # stand-up, then the standing demo from where the cup stands
            g1, Q1, ep1, sc2, (p1, R1) = stage1
            g2 = generate(src, sc2, (p1, R1), fix)
            Q2, ep2 = ik_plan(g2, Q1[-1], ik2, md2, robot2)
            n1 = len(g1["P"])
            g = {"P": np.r_[g1["P"], g2["P"]], "Rs": np.r_[g1["Rs"], g2["Rs"]],
                 "close": np.r_[g1["close"], g2["close"]],
                 "pick": (g2["pick"][0] + n1, g2["pick"][1] + n1),
                 "place": (g2["place"][0] + n1, g2["place"][1] + n1), "cup_b": g2["cup_b"],
                 "stand": g1["pick"]}  # fmt: skip
            Q, ep = np.r_[Q1, Q2], np.r_[ep1, ep2]
        n = len(g["P"])
        return g, Q, ep, retime(Q, g["close"]) + [(n - 1, Q[-1])] * HOLD_END

    # pass 1 (no rendering) up to the place segment: where is the cup in the hand really?
    g, Q, ep, steps = plan(None)
    i_b = g["place"][0]
    sim = simulate(xml, steps, g["close"], W, 0, 1, until=i_b)
    j = int(np.flatnonzero(sim["human"] == i_b)[0]) if (sim["human"] == i_b).any() else -1
    fix = g["cup_b"] - sim["obj"][j, 0]  # the cup slipped in the grasp by -fix
    fix = np.clip(fix, -0.04, 0.04)
    g, Q, ep, steps = plan(fix)
    sim = simulate(xml, steps, g["close"], W, 0, 1, None, contacts=True)
    if queue is not None:  # tiles from the recorded states, from an angle where the arm does
        # not hide the cup (touch.render_view; Andy 2026-11-14), every 2nd step
        from touch import render_view as RV

        mv = mujoco.MjModel.from_xml_string(xml)
        states = sim["log"]["state"]
        cam, _ = RV.view_for(mv, states, TILE[0] / TILE[1])
        for _, img in RV.render_states(mv, states, cam, TILE[0], TILE[1], every=2):
            queue.put((k, cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()))
    ok, why, met = check(sim["log"], sim["obj"][-1, 0], sc["place_xy"], src["h"], src["d"],
                         sim["hits"], sc["goal_lying"], math.degrees(sc["place_yaw"]),
                         flipped=sc.get("goal_flip", False))  # fmt: skip
    met["slip_fix_cm"] = round(float(np.linalg.norm(fix)) * 100, 1)
    if stand_tries:
        met["stand_up_tries"] = stand_tries
    qs = np.asarray(sim["log"]["obj_quat"])[:, 0][:, [1, 2, 3, 0]]
    tz = Rot.from_quat(qs).as_matrix()[:, 2, 2]
    if not ok:  # where it went wrong: the phase of the first step the cup lies past 45 deg
        lie = np.flatnonzero(tz < math.cos(math.radians(45)))
        if stand_tries:  # the cup starts lying: count from when it was first upright
            up = np.flatnonzero(tz > math.cos(math.radians(TILT_MAX)))
            lie = lie[lie > up[0]] if len(up) else lie[:0]
        if len(lie):
            h_ = int(np.asarray(sim["human"])[lie[0]])
            n1 = len(stage1[0]["P"]) if stage1 else 0
            ph = ("stand-up" if h_ < n1 else "transfer" if h_ < g["pick"][0]
                  else "grasp" if h_ < g["pick"][1] else "carry" if h_ < g["place"][0]
                  else "set-down")  # fmt: skip
            met["tipped_in"] = ph
    seg = np.r_[np.arange(*g["pick"]), np.arange(*g["place"])]
    met["ik_err_mm"] = round(float(ep[seg].max()) * 1000, 1)
    kw = int(seg[int(np.argmax(ep[seg]))])
    n_pl = g["place"][1] - g["place"][0]
    met["ik_err_at"] = (f"pick +{kw - g['pick'][0]}" if kw < g["pick"][1]
                        else f"place +{kw - g['place'][0]} of {n_pl}")  # fmt: skip
    eef = np.array([np.r_[p, Rot.from_matrix(R).as_rotvec()]
                    for p, R in (ik.fk(qc) for qc in sim["log"]["q_cmd"])])  # fmt: skip
    return {"k": k, "success": ok, "reason": why, "metrics": met, "log": sim["log"],
            "eef_cmd": eef, "xml": xml, "cup": np.asarray(sc["cup_xy"]),
            "place": np.asarray(sc["place_xy"]), "scene": sc, "q_start": q_start,
            "place_quat": cup_quat(sc, "place")[[1, 2, 3, 0]],
            "seconds": round(time.time() - t0, 1)}  # fmt: skip


# ------------------------------------------------------------------ dataset (robomimic layout)
ENV_META = {
    "env_name": "OneTouchPandaCupPickPlace",
    "type": "mujoco_mjcf",  # not a robosuite env: each demo stores its own MJCF (model_file)
    "env_kwargs": {
        "robot": "Franka Panda (MuJoCo Menagerie)",
        "control_freq": 30,
        "controller": "joint position servo + velocity feed-forward",
        "obs_frame": "table, z up (ArUco marker 0 frame)",
        "quat_order": "xyzw",
    },  # fmt: skip
}


def append_demos(path: Path, demos: list[dict]) -> list[str]:
    """Append demos to an HDF5 file in the robomimic layout -> their group names.
    demo = {"log": simulate log, "eef_cmd": [T, 6], "xml": str, "attrs": {...}}.
      data/demo_k/obs/robot0_joint_pos [T,7], robot0_joint_vel [T,7], robot0_gripper_qpos [T,2],
          robot0_eef_pos [T,3], robot0_eef_quat [T,4] (xyzw), cup_pos [T,3], cup_quat [T,4],
          place_pos [T,3], place_quat [T,4] (goal orientation, xyzw), object [T,14] (cup pos +
          quat + place pos + quat)
      data/demo_k/actions [T,8]: joint position targets (7) + gripper (-1 open, +1 close)
      data/demo_k/actions_eef [T,7]: commanded TCP position (3) + axis-angle (3) + gripper
      data/demo_k/states [T, nq+nv] (MuJoCo qpos, qvel), rewards [T], dones [T]
      attrs: num_samples, model_file (MJCF), source, success, ..."""
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    names = []
    with h5py.File(path, "a") as f:
        g = f.require_group("data")
        g.attrs["env_args"] = json.dumps(ENV_META)
        k = 1 + max((int(n.split("_")[1]) for n in g if n.startswith("demo_")), default=-1)
        for dm in demos:
            L = dm["log"]
            T = len(L["q"])
            dg = g.create_group(f"demo_{k}")
            ob = dg.create_group("obs")
            ob["robot0_joint_pos"] = L["q"].astype(np.float32)
            ob["robot0_joint_vel"] = L["qd"].astype(np.float32)
            ob["robot0_gripper_qpos"] = L["fingers"].astype(np.float32)
            ob["robot0_eef_pos"] = L["tcp"].astype(np.float32)
            ob["robot0_eef_quat"] = Rot.from_matrix(L["tcp_R"]).as_quat().astype(np.float32)
            cup = L["obj_pos"][:, 0].astype(np.float32)
            cq = L["obj_quat"][:, 0][:, [1, 2, 3, 0]].astype(np.float32)  # MuJoCo wxyz -> xyzw
            place = np.tile(np.r_[dm["attrs"]["place_xy"], 0.0], (T, 1)).astype(np.float32)
            pq = np.tile(np.asarray(dm["attrs"].get("place_quat", [0, 0, 0, 1.0])), (T, 1))
            ob["cup_pos"], ob["cup_quat"], ob["place_pos"] = cup, cq, place
            ob["place_quat"] = pq.astype(np.float32)
            ob["object"] = np.concatenate([cup, cq, place, ob["place_quat"][()]], 1)
            grip = np.where(L["grip_open"], -1.0, 1.0)[:, None].astype(np.float32)
            dg["actions"] = np.concatenate([L["q_cmd"].astype(np.float32), grip], 1)
            dg["actions_eef"] = np.concatenate([dm["eef_cmd"].astype(np.float32), grip], 1)
            dg["states"] = L["state"].astype(np.float64)
            rew = np.zeros(T, np.float32)
            rew[-1] = float(dm["attrs"].get("success", False))
            dg["rewards"], dg["dones"] = rew, (np.arange(T) == T - 1).astype(np.int64)
            dg.attrs["num_samples"] = T
            dg.attrs["model_file"] = dm["xml"]
            for a, v in dm["attrs"].items():
                dg.attrs[a] = json.dumps(v) if isinstance(v, dict | list) else v
            names.append(f"demo_{k}")
            k += 1
        g.attrs["total"] = int(sum(g[n].attrs["num_samples"] for n in g if n.startswith("demo_")))
    return names


def count_demos(path: Path) -> int:
    import h5py

    if not Path(path).exists():
        return 0
    with h5py.File(path, "r") as f:
        return sum(n.startswith("demo_") for n in f.get("data", {}))
