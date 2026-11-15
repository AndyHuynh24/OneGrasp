"""OFFLINE: a recorded webcam demo of a person grasping the demo cylinder -> a Franka Panda that
grasps at the person's touch point and lifts the cylinder as the hand did (not live: accuracy
over speed).

  1. perception, exactly as the live demo (scripts/live_sim.py): MediaPipe -> same-hand tracker
     -> grasp model (ONNX) -> metric 3D hand in the table frame (results/rig/calib.json) ->
     learned grab gate. Cached in <out>/perception.npz.
  2. offline cleanup: centred smoothing of the 3D hand (robot_sim.smooth), gate gaps <= 10
     frames filled (robot_sim.fill_gaps)
  3. one episode per grasp (gate closed >= 10 frames): the cylinder stands where the person's
     pad centre is when the gate closes (the hand is on the object then; no object tracking);
     while held, the target object pose follows the hand (its lift = the pad's rise)
  4. robot targets as robot v2 offline (robot_sim.retarget -> object_refine onto the cylinder ->
     pregrasp approach), Panda base searched along this path (IK error + collisions), IK with
     table / cylinder collision avoidance (touch.panda), re-timed to the Panda's joint speeds
  5. "small RL": cross-entropy-method search over a grasp correction (offset, two rotations,
     close timing; scripts/panda_replay.cem), reward = how closely the simulated cylinder
     follows the hand's lift, minus a small cost for leaving the hand's path
  6. render: webcam (hand) | Panda (third-person view) per episode, and all episodes joined

    python scripts/panda_from_video.py --video data/own/grasp/<session>/video.mp4 [--max-episodes 3]
Writes results/robot_sim/panda_video/<video stem>/: episode_<k>.mp4, panda_video.mp4,
summary.json, perception.npz.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from multiprocessing import get_context, parent_process
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
import panda_replay as PR  # noqa: E402

if parent_process() is None:  # the hand models: main process only. A spawn worker imports this
    # module too, and live_grasp loads torch + onnxruntime (~370 MB per worker, unused there)
    from live_grasp import HandTracker, LiveGrasp, default_ckpt, frames_with_hands  # noqa: E402
from render_sim_demo import label  # noqa: E402

import touch.render_view as RV  # noqa: E402
from touch import demogen as DG  # noqa: E402
from touch import robot_sim as S  # noqa: E402
from touch.handpath import world_hand  # noqa: E402
from touch.interact.grab_gate import FusionGate  # noqa: E402
from touch.live_panda import CLEAR, penetration, robot_bodies, scene_world  # noqa: E402
from touch.live_robot import CYL_D, CYL_H, camera_xml  # noqa: E402
from touch.panda import MAX_OPEN, PandaIK, panda_mjcf  # noqa: E402

MIN_CLOSE = 10  # frames: shorter gate closures are not grasp episodes
PRE, POST = 45, 30  # frames of the approach before / of the release after each episode


# ------------------------------------------------------------------ 1. perception
def perceive(video: Path, ckpt: Path, K, c2w, flip: dict, release: str, cache: Path) -> dict:
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    live = LiveGrasp(ckpt, threads=4)
    fdir = Path("runs/grab_fusion") / ckpt.parent.name
    fusion = FusionGate.from_dir(fdir, "hold")
    prof = json.loads((fdir / "fusion_hold.json").read_text()).get("release_profiles", {})
    for k, v in prof.get(release, {}).get("release", {}).items():
        setattr(fusion.hyst, k, v)
    cap = cv2.VideoCapture(str(video))
    tracker = HandTracker(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    rows = {"joints": [], "lm": [], "state": [], "close": [], "present": []}
    t0 = time.time()
    for frame, hands, hd, _, _ in frames_with_hands(cap, str(video), 30.0, threading.Event(),
                                                    flip):  # fmt: skip
        lm, left = tracker.pick(hands, hd)
        out = live.step(live.prep(frame, lm, left))
        jw = np.full((21, 3), np.nan)
        if lm is not None:
            with np.errstate(all="ignore"):
                wh = world_hand(out["joints"][None], out["verts"][None], lm[None], K, c2w, left,
                                smooth_k=1)  # fmt: skip
            if wh["ok"][0]:
                jw = wh["joints"][0]
        extra = {}
        if fusion.needs_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            extra = {"joints_pred": out.get("joints"), "lm": lm, "bgr": frame, "gray": gray,
                     "joints_world": jw}  # fmt: skip
        rows["close"].append(fusion.step(out["state"], out["finger"], lm is not None, **extra))
        rows["joints"].append(jw)
        rows["lm"].append(np.full((21, 3), np.nan) if lm is None else lm[:, :3])
        rows["state"].append(out["state"])
        rows["present"].append(lm is not None)
        if len(rows["close"]) % 900 == 0:
            n = len(rows["close"])
            print(f"  perception {n} frames ({n / (time.time() - t0):.0f} fps)", flush=True)
    res = {k: np.array(v) for k, v in rows.items()}
    np.savez(cache, **res)
    return res


# ------------------------------------------------------------------ 3-4. episodes
def episodes(close: np.ndarray) -> list[tuple[int, int]]:
    c = np.r_[False, close, False]
    d = np.flatnonzero(np.diff(c.astype(np.int8)))
    return [(int(a), int(e)) for a, e in zip(d[::2], d[1::2], strict=True) if e - a >= MIN_CLOSE]


def cylinder_points(xy, h: float, d: float, n: int = 1500) -> np.ndarray:
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(0, h, n)
    side = np.stack([xy[0] + d / 2 * np.cos(a), xy[1] + d / 2 * np.sin(a), z], 1)
    r = d / 2 * np.sqrt(rng.uniform(0, 1, n // 3))
    b = rng.uniform(0, 2 * np.pi, n // 3)
    top = np.stack([xy[0] + r * np.cos(b), xy[1] + r * np.sin(b), np.full(n // 3, h)], 1)
    return np.concatenate([side, top])


def _base_score(args):
    """One base candidate (b, yaw): summed IK + collision score over the candidate paths; with
    a camera position: (score, share of the gripper / cup / spot sight lines from the camera
    that no arm link blocks, over the first path's frames)."""
    import mujoco

    b, yaw, world, paths = args[:4]
    cam = args[4] if len(args) > 4 else None
    m = mujoco.MjModel.from_xml_string(panda_mjcf(b, yaw, world))
    dd = mujoco.MjData(m)
    ik = PandaIK(m)
    robot = robot_bodies(m)
    s = 0.0
    for P, Rs, close in paths:
        frames = np.arange(0, len(P), 3)
        w = np.where(close[frames], 3.0, 1.0)
        Q, ep, er = ik.track(P, Rs, frames, iters=30, avoid=CLEAR)
        pen = np.zeros(len(frames))
        for i, q in enumerate(Q):
            dd.qpos[ik.qadr] = q
            mujoco.mj_forward(m, dd)
            pen[i] = penetration(m, dd, robot)
        s += float(((ep + 0.05 * er + 5 * pen) * w).sum() / w.sum())
        if cam is not None and "vis" not in locals():
            vis = _sight(m, dd, ik, Q, P[frames], cam, args[5])
    return (s, vis) if cam is not None else s


def _sight(m, dd, ik, Q, pads, cam, fixed) -> float:
    """Share of sight lines camera -> (gripper pad, cup start, place spot) per frame that no
    arm link (link0-7; the hand and fingers may cover their own cup) blocks."""
    import mujoco

    links = {m.body(f"link{k}").id for k in range(8)}
    grp = np.array([0, 0, 1, 0, 0, 0], np.uint8)  # the robot's visual meshes
    gid = np.zeros(1, np.int32)
    cam = np.asarray(cam, float)
    seen = tot = 0
    for q, pad in zip(Q, pads, strict=True):
        dd.qpos[ik.qadr] = q
        mujoco.mj_forward(m, dd)
        for pt in (pad, *fixed):
            v = np.asarray(pt, float) - cam
            dist = float(np.linalg.norm(v))
            hit = mujoco.mj_ray(m, dd, cam, v / dist, grp, 1, -1, gid)
            tot += 1
            seen += not (gid[0] >= 0 and hit < dist - 0.02 and m.geom_bodyid[gid[0]] in links)
    return seen / max(tot, 1)


REACH_OK = 1.25  # base spots whose reach score is within this x the best count as reachable;
# among them the one the arm blocks least in the render camera's view wins (Andy 2026-11-14:
# "not blocking the view"; a fixed side spot could not follow a pour + 43 cm lift at all)
VIEW_DIST = 1.2  # m: the render camera's distance from the cup, for the sight lines
SIDE_MIN = 0.5  # base direction . the render's right >= this (within 60 deg of straight right)
VIS_SLACK = 0.05  # spots this close to the clearest view count as clear; of those: the right
ARM_W = 0.004  # score penalty for the base on the person's forearm side (0 .. this)


def view_direction(cam_xml: str) -> np.ndarray | None:
    """Table-plane unit vector the camera looks along (its image top = away; from the MJCF
    camera's xyaxes, touch.live_robot.camera_xml); None if not parsable."""
    import re

    mt = re.search(r'xyaxes="([^"]+)"', cam_xml or "")
    if not mt:
        return None
    v = np.array([float(t) for t in mt.group(1).split()])
    x, y = v[:3], v[3:]
    f = -np.cross(x, y)  # a MuJoCo camera looks along -z
    hd = y[:2] + f[:2]
    n = float(np.linalg.norm(hd))
    return hd / n if n > 1e-6 else None


def camera_axes(cam_xml: str):
    """MJCF camera line -> (rotation [right, up, back], position); None if not parsable."""
    import re

    mt = re.search(r'xyaxes="([^"]+)"', cam_xml or "")
    mp = re.search(r'pos="([^"]+)"', cam_xml or "")
    if not (mt and mp):
        return None
    v = np.array([float(t) for t in mt.group(1).split()])
    x = v[:3] / np.linalg.norm(v[:3])
    y = v[3:] - (v[3:] @ x) * x
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], 1), np.array([float(t) for t in mp.group(1).split()])


def arm_direction(J: np.ndarray, ok: np.ndarray, ka: int) -> np.ndarray | None:
    """Table-plane unit vector from the hand towards the person's forearm (middle-finger base
    -> wrist) over the first 15 grip frames; None when not tracked."""
    v = J[ka : ka + 15][:, 0, :2] - J[ka : ka + 15][:, 9, :2]
    v = v[np.isfinite(v).all(1) & ok[ka : ka + 15]]
    if not len(v):
        return None
    u = np.median(v, 0)
    n = float(np.linalg.norm(u))
    return u / n if n > 1e-6 else None


def place_base(obj_xy, paths, h, d, cam_xy, pool=None, arm_dir=None,
               view_dir=None) -> tuple[np.ndarray, float]:  # fmt: skip
    """Base spots 0.40-0.70 m from the cylinder every 20 deg; score = IK position error + 0.05
    orientation error + 5 x collision depth over every 3rd frame of each candidate path
    [(P, Rs, close)] (grasp frames x 3), summed. arm_dir: the side the person's forearm comes
    from (arm_direction): + up to ARM_W for a base on that side. view_dir: the camera's
    viewing direction (view_direction): among the spots within REACH_OK of the best reach, the
    one whose arm blocks the camera's sight lines least (_sight). Without
    view_dir: any side but straight between the camera and the scene. pool: in parallel."""
    c = np.asarray(obj_xy, float)
    world = scene_world(obj_xy, h, d, margin=True)
    tc = np.asarray(cam_xy) - c
    tc /= max(np.linalg.norm(tc), 1e-9)
    cands = []
    for r in (0.40, 0.50, 0.60, 0.70):
        for a in np.radians(np.arange(0, 360, 15)):
            dirn = np.array([math.cos(a), math.sin(a)])
            if view_dir is None and dirn @ tc > 0.7:  # straight between camera and scene
                continue
            cands.append((c + r * dirn, math.atan2(-dirn[1], -dirn[0]), world, paths))
    if view_dir is not None:  # the render camera (touch.render_view.real_view), roughly
        from touch.render_view import VIEW_ELEV

        e = math.radians(VIEW_ELEV)
        f = np.r_[math.cos(e) * np.asarray(view_dir), math.sin(e)]
        cam = np.r_[c, 0.1] - VIEW_DIST * f
        P0, _, cl0 = paths[0]
        fixed = [np.r_[c, h / 2]] + ([np.r_[P0[-1][:2], 0.02]] if len(P0) else [])
        cands = [cd + (cam, fixed) for cd in cands]
    out = list((pool.map if pool is not None else map)(_base_score, cands))
    vis = np.array([o[1] for o in out]) if view_dir is not None else None
    scores = np.array([o[0] if view_dir is not None else o for o in out])
    k = int(np.argmin(scores))
    dirs = np.array([(cd[0] - c) / np.linalg.norm(cd[0] - c) for cd in cands])
    pref = scores.copy()
    if arm_dir is not None:
        pref += ARM_W * (1 + dirs @ arm_dir) / 2
    if view_dir is not None:
        ang = np.degrees(np.arccos(np.clip(dirs @ view_dir, -1, 1)))
        right = np.array([view_dir[1], -view_dir[0]])  # image right on the table plane
        side = np.flatnonzero(dirs @ right >= SIDE_MIN)  # Andy 2026-11-14: "put the arm to
        # the left or right and choose one for the whole app": always the render's right
        if not len(side):
            side = np.arange(len(cands))
        kr = side[int(np.argmin(scores[side]))]
        ok_ = side[scores[side] <= scores[kr] * REACH_OK]  # reachable on that side
        ok_ = ok_[vis[ok_] >= vis[ok_].max() - VIS_SLACK]  # (nearly) the clearest view
        kf = ok_[np.lexsort((pref[ok_], -np.round(vis[ok_], 2)))[0]]
        print(f"robot base: {ang[kf]:.0f} deg from the view direction, view clear "
              f"{100 * vis[kf]:.0f}% (front spot would be {100 * vis[k]:.0f}%), reach score "
              f"{scores[kf]:.4f} (best {scores[k]:.4f})", flush=True)  # fmt: skip
        k = kf
    return cands[k][0], cands[k][1]


# grasp on the cylinder (the search space of the small RL): azimuth of the approach (rad),
# pitch below horizontal (0 = from the side, pi/2 = from the top), pad-centre height (m),
# close timing shift (frames); the closing axis is horizontal (across the diameter)
N_APP, N_TR = 12, 15  # robot_sim.pregrasp defaults: straight-in / blend frames


def _pose(p, R):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def cylinder_grasp(obj_xy, phi: float, beta: float, zc: float) -> np.ndarray:
    z = np.array([math.cos(beta) * math.cos(phi), math.cos(beta) * math.sin(phi),
                  -math.sin(beta)])  # fmt: skip
    y = np.array([-math.sin(phi), math.cos(phi), 0.0])  # horizontal, across the diameter
    return _pose(np.r_[obj_xy, zc], np.stack([np.cross(y, z), y, z], 1))


def lying_grasp_pose(obj_xy, yaw: float, s: float, beta: float, dz: float, d: float):
    """Grasp across a cylinder lying on its side (axis heading yaw, centre d/2 up): pad
    centre on the axis, s along it from the centre (+ dz along the approach), approach from
    above tilted by beta about the axis, closing across the diameter."""
    a = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    perp = np.array([-a[1], a[0], 0.0])
    z = -math.cos(beta) * np.array([0.0, 0.0, 1.0]) + math.sin(beta) * perp
    y = np.cross(a, z)
    p = np.r_[obj_xy, d / 2] + s * a + dz * z
    return _pose(p, np.stack([np.cross(y, z), y, z], 1))


def cup_frame(pl: dict):
    """-> (cup centre [3], cup axis [3]) at rest at the start (pl["cup"] lying or standing)."""
    cup = pl.get("cup")
    if cup and cup["lying"]:
        y = cup["yaw"]
        return np.r_[pl["obj_xy"], pl["d"] / 2], np.array([math.cos(y), math.sin(y), 0.0])
    return np.r_[pl["obj_xy"], pl["h"] / 2], np.array([0.0, 0.0, 1.0])


def grasp_pose(pl: dict, g: np.ndarray) -> np.ndarray:
    """Grasp parameters -> 4x4 pad pose. Standing: [azimuth, pitch, pad height, close shift];
    lying (pl["cup"]["lying"]): [s along the axis, approach tilt about it, dz, close shift]."""
    cup = pl.get("cup")
    if cup and cup["lying"]:
        return lying_grasp_pose(pl["obj_xy"], cup["yaw"], g[0], g[1], g[2], pl["d"])
    return cylinder_grasp(pl["obj_xy"], g[0], g[1], g[2])


GRASP_TOP = 0.05  # m: the robot's pad centre at least this far below a standing cup's rim
# (was 2 cm; Andy 2026-11-15 "don't grab near the rim too much": <= 8.4 cm on the can)
GRASP_FLOOR = 0.025  # m: and at least this high (the fingers clear the table)
SIDE_PITCH = math.radians(15)  # a standing cup is grasped from the side: approach at most
# this far below horizontal (Andy 2026-11-15: "always grab from the side", never top-down)
GRASP_DOWN, GRASP_UP = 0.01, 0.005  # m: the search's height range around the target: mostly
# lower, hardly higher (Andy 2026-11-14: grasp lower on the body, the hand covers the cup;
# down was 1.5 cm: 5.7 cm grips on the can tipped generated scenes)
HAND_CENTRE = [0, 5, 9, 13, 17, 4, 8, 12, 16, 20]  # palm + fingertips: the grip's centre
CENTRE_W = 0.5  # target = this x hand centre + (1 - this) x the cup's middle


def grip_height(J: np.ndarray, ok: np.ndarray, ka: int, h: float) -> float:
    """Target pad height on a standing cup: its middle, pulled towards the centre of the
    person's hand (palm + fingertips, first 15 grip frames). The camera's hand height can be
    off by a few cm (fingertips "at 10 cm" on a 9 cm cup), so a hand centre outside the cup's
    body counts as unknown -> the cup's middle."""
    mid = h / 2
    tz = J[ka : ka + 15][:, HAND_CENTRE, 2]
    tz = tz[np.isfinite(tz).all(1) & ok[ka : ka + 15]]
    if len(tz):
        hc = float(np.median(tz.mean(1)))
        if GRASP_FLOOR <= hc <= h - GRASP_TOP:
            mid = CENTRE_W * hc + (1 - CENTRE_W) * mid
    return float(np.clip(mid, GRASP_FLOOR, h - GRASP_TOP))


def grasp_bounds(pl: dict):
    h = pl["h"]
    if pl.get("cup") and pl["cup"]["lying"]:  # pads >= 1.5 cm in from either end
        s = h / 2 - 0.015
        return np.array([-s, -1.0, -0.012, -4.0]), np.array([s, 1.0, 0.006, 8.0])
    # pad centre >= GRASP_TOP below the rim: the ~2 cm pads sit on the cup's body (the person's
    # fingertips are often at the rim; a grasp there slips off / pivots when moved)
    lo = np.array([-np.inf, 0.0, GRASP_FLOOR, -4.0])
    hi = np.array([np.inf, SIDE_PITCH, h - GRASP_TOP, 8.0])
    if pl.get("g0") is not None:  # around the target height, hardly higher (Andy 2026-11-14)
        z0 = float(pl["g0"][2])
        lo[2], hi[2] = max(GRASP_FLOOR, z0 - GRASP_DOWN), min(h - GRASP_TOP, z0 + GRASP_UP)
        lo[2] = min(lo[2], hi[2])
    return lo, hi


def grasp_seeds(pl: dict) -> np.ndarray:
    """Search seeds: the person's grasp, + (standing) from the side at the highest allowed
    height and at mid-height, tilted down by SIDE_PITCH; (lying) straight down at the centre,
    straight down where the person grasped."""
    g0, h = pl["g0"], pl["h"]
    if pl.get("cup") and pl["cup"]["lying"]:
        seeds = [g0, [0.0, 0.0, 0.0, 0.0], [g0[0], 0.0, 0.0, 0.0]]
    else:
        seeds = [g0, [g0[0], 0.0, h - GRASP_TOP, 0], [g0[0], 0.0, h / 2, 0],
                 [g0[0], SIDE_PITCH, h - GRASP_TOP, 0]]  # fmt: skip
    return np.clip(np.array(seeds, float), *grasp_bounds(pl))


FOLLOW_R, FOLLOW_DZ = 0.25, 0.02  # m: the pre-approach following of the hand (targets)


def targets(pl: dict, g: np.ndarray):
    """Grasp parameters g -> robot pad poses + close for the episode: the grasp G replaces the
    human's grasp G0 at the close frame and is carried rigidly with the hand (pose(t) G0^-1 G,
    as robot_sim.object_refine); before the approach the robot waits 8 cm above its pre-grasp
    pose, then robot_sim.pregrasp blends to 8 cm behind the grasp and moves straight in."""
    lo_b, hi_b = grasp_bounds(pl)
    g = np.clip(g, lo_b, hi_b)
    ka, n = pl["ka"], len(pl["hand_P"])
    G = grasp_pose(pl, g)
    C = np.linalg.inv(_pose(pl["hand_P"][ka], pl["hand_R"][ka])) @ G
    pos, R = np.zeros((n, 3)), np.zeros((n, 3, 3))
    for t in range(n):
        T = _pose(pl["hand_P"][t], pl["hand_R"][t]) @ C
        pos[t], R[t] = T[:3, 3], T[:3, :3]
    hover = G[:3, 3] - 0.08 * G[:3, 2] + [0, 0, 0.08]
    k = int(round(g[3]))
    close = pl["close0"][np.clip(np.arange(n) - k, 0, n - 1)]
    # the hover-follow runs up to the close (not only to the approach start): pregrasp's blend
    # starts from it. Before, the blend started from the hand-carried grasp pose, up to 12 cm
    # away -> a one-frame jump and a lunge at the can (Andy 2026-11-15)
    n0 = int(np.argmax(close)) if close.any() else n
    # before the approach: the hover pose moved with the person's hand (its displacement from
    # the grasp, at most FOLLOW_R; not below FOLLOW_DZ under the hover height): the robot
    # reaches in as the hand does instead of waiting still (Andy 2026-11-14: in sync)
    dh = pl["hand_P"][:n0] - pl["hand_P"][ka]
    nrm = np.linalg.norm(dh, axis=1, keepdims=True)
    dh = dh * np.minimum(1.0, FOLLOW_R / np.maximum(nrm, 1e-9))
    pos[:n0] = hover + dh
    pos[:n0, 2] = np.maximum(pos[:n0, 2], hover[2] - FOLLOW_DZ)
    R[:n0] = G[:3, :3]
    pos, R = S.pregrasp(pos, R, close, floor=True)
    c0, ax0 = cup_frame(pl)
    return set_down(pos, R, close.copy(), float(G[2, 3]), c0, ax0, pl["h"], pl["d"],
                    pl.get("end_axis"), pl.get("end_xy"),
                    pl.get("pour_angle", POUR_ANGLE), pl.get("pour_lean"))  # fmt: skip


def _mj(n: int) -> np.ndarray:  # minimum-jerk 0 -> 1 in n steps (excludes 0, includes 1)
    t = np.linspace(0, 1, n + 1)[1:]
    return 10 * t**3 - 15 * t**4 + 6 * t**5


PICK_CLEAR = 0.03  # m: after the close the cup is lifted straight up this much first
ROT_SMOOTH = 3  # frames: the hand's rotation is averaged over +-3 frames while held
N_LIFT, N_BLEND = 8, 8  # frames: straight-up lift, then blend back onto the hand's path (xy)
N_ROT0, N_ROT = 2, 4  # frames: the hand's rotation is followed from 2 frames after the close,
#                       fully by 6 (Andy 2026-11-14: waiting for the lift made the robot turn
#                       the cup ~1 s after his hand; only the xy must wait, so it doesn't drag)


RELEASE_DIP = (
    0.01  # m: the set-down aims this far low; touching the table releases (panda.simulate)
)
ND_MIN = 10  # frames: the lowering takes at least this long (gentle: the hand may still be
# turning the cup back; 6 frames was a jolt the arm could not follow, 46 mm off in new scenes)
SET_AHEAD = 10  # frames: the set-down may start this long before the hand opens (no later open)
COS_UP = math.cos(math.radians(45))  # cup axis within 45 deg of vertical: set down standing


def end_pose(axis: np.ndarray, h: float, d: float):
    """The cup's axis when the hand lets go -> (axis it is set down with, centre height):
    standing (upright / upside down) within 45 deg of vertical, else lying (its heading)."""
    if axis[2] >= COS_UP:
        return np.array([0.0, 0.0, 1.0]), h / 2
    if axis[2] <= -COS_UP:
        return np.array([0.0, 0.0, -1.0]), h / 2
    v = np.r_[axis[:2], 0.0]
    return v / max(np.linalg.norm(v), 1e-9), d / 2


POUR_SIDES = {"left": (0, -1), "right": (0, 1), "down": (1, -1), "up": (1, 1)}  # image axis
POUR_ONSET = 15.0  # deg: the pour starts after the last time the estimated tilt is below this
POUR_RATE = 135.0  # deg/s: the robot's pour never tilts faster (the previous pour's peak rate)
POUR_DETECT = 35.0  # deg: a carry whose (smoothed) tilt peaks above this contains a pour
POUR_ANGLE = 135.0  # deg: the pour's peak becomes this (Andy 2026-11-14: "almost upside down,
# 45 deg before"); 0 = copy the tilt as estimated. The 3D hand seen from above reads a pour as
# only ~50-68 deg, and differently every take: a fixed threshold + gain missed one


POUR_SIGMA = 6.0  # frames: the tilt curve is smoothed with this Gaussian
CARRY_SIGMA = 4.0  # frames: the held gripper path is smoothed with this (3D hand jitter)


def _gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-phase Gaussian smoothing along axis 0 (edges held), offline."""
    if sigma <= 0 or len(x) < 3:
        return x
    r = int(3 * sigma)
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    w /= w.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return np.stack([np.convolve(pad[:, c], w, mode="valid") for c in range(x.shape[1])], 1)


def pour_assist(
    Rs: np.ndarray, Ra: np.ndarray, axis0=None, angle: float = POUR_ANGLE, lean=None
) -> np.ndarray:
    """Held gripper rotations [n, 3, 3] -> a simple pour (Andy 2026-11-14: "the pour movement
    is too complex"): the grasp orientation Ra tilted about ONE fixed horizontal axis by the
    hand's tilt from vertical, smoothed (POUR_SIGMA; "my pour is smooth"). lean (table-plane
    vector): the cup top leans that way (Andy: "I pour to the left (it always has been like
    that)"; the 3D hand seen from above gets the pour's direction wrong); None: the hand's
    tilt axis. The biggest tilt, if it peaks above POUR_DETECT, is the pour: from the low point
    before it to the low point after it it is one continuous tilt to `angle` and back, in
    proportion to the hand's (its pace; Andy 2026-11-14: "don't speed up the tilt to catch
    up"; where its rise would be faster than POUR_RATE it starts a little earlier instead); the
    the rise starts where the can starts leaning (POUR_ONSET); other tilts as estimated.
    Twist / wobble not copied."""
    ax0 = np.array([0.0, 0.0, 1.0]) if axis0 is None else np.asarray(axis0, float)
    g = Ra.T @ ax0  # the cup axis in the gripper frame
    z = np.array([0.0, 0.0, 1.0])
    U_ = np.einsum("nij,j->ni", Rs, g)  # the cup axis per frame
    th = np.degrees(np.arccos(np.clip(U_[:, 2], -1, 1)))
    if th[0] > 45.0:  # picked up lying: the hand's rotation as before
        return Rs
    kx = np.cross(z, U_)[:, :2]  # tilt axis (horizontal), weighted by sin(tilt)
    th_s = _gauss(th[:, None], POUR_SIGMA)[:, 0]
    th2 = th_s.copy()
    p = int(np.argmax(th_s))
    peak = float(th_s[p])
    wk = th_s.copy()  # weights for the tilt axis
    if angle > 0 and peak >= POUR_DETECT and peak < angle:
        lo_, hi_ = p, p  # walk down to the low points on either side
        while lo_ > 0 and th_s[lo_ - 1] <= th_s[lo_] + 0.2:
            lo_ -= 1
        while hi_ < len(th_s) - 1 and th_s[hi_ + 1] <= th_s[hi_] + 0.2:
            hi_ += 1
        # one continuous pour (Andy 2026-11-14: "my motion is one continuous action, the robot
        # pauses before it tilts over"): the 3D hand reads the lift as a lean, then a slight
        # straightening, then the pour (in the video the can tips steadily from the lift on;
        # the 3D joints are not late vs the 2D landmarks, the estimate just stalls). From
        # where the can starts leaning for good (POUR_ONSET) the tilt rises without stopping
        # to `angle`: half the estimate's (non-decreasing) rise, half an even ramp
        below = np.flatnonzero(th_s[: p + 1] < POUR_ONSET)
        o = int(below[-1]) + 1 if len(below) else lo_  # the can starts leaning (for good)
        env = np.maximum.accumulate(th_s[o : p + 1])  # the estimate, never backing off
        envn = np.clip((env - env[0]) / max(peak - env[0], 1e-6), 0, 1)
        prog = 0.5 * envn + 0.5 * np.linspace(0.0, 1.0, p - o + 1)  # + an even ramp: no stall
        th2[o : p + 1] = th_s[o] + (angle - th_s[o]) * prog
        for k_ in range(p - 1, -1, -1):  # never faster than POUR_RATE: start a bit earlier
            th2[k_] = max(th2[k_], th2[k_ + 1] - POUR_RATE / 30.0)
        m_ = th_s[hi_]  # back down: to the hand's tilt after the pour, at its pace
        b = np.clip((th_s[p : hi_ + 1] - m_) / max(peak - m_, 1e-6), 0, 1)
        th2[p : hi_ + 1] = th_s[p : hi_ + 1] + (angle - peak) * b
        wk = np.zeros_like(th_s)
        wk[lo_ : hi_ + 1] = th_s[lo_ : hi_ + 1]
    if lean is not None and np.linalg.norm(lean) > 1e-6:
        L = np.r_[np.asarray(lean, float)[:2], 0.0]
        k = np.cross(z, L / np.linalg.norm(L))  # rotating about z x L leans the top to L
    else:
        kw = (kx * wk[:, None]).sum(0)  # one axis for the whole carry (the pour's)
        nk = float(np.linalg.norm(kw))
        if nk < 1e-6:
            return np.repeat(Ra[None], len(Rs), 0)
        k = np.r_[kw / nk, 0.0]
    return np.stack([Rot.from_rotvec(k * math.radians(t - th2[0])).as_matrix() @ Ra for t in th2])


def set_down(pos, R, close, zc: float, c0=None, axis0=None, h: float = 0.0, d: float = 0.0,
             end_axis=None, end_xy=None, pour_angle: float = POUR_ANGLE,
             pour_lean=None):  # fmt: skip
    """Shared-autonomy pick and place (the human gives what / where / when, the robot does the
    contact-critical motions; copying the hand literally drags the cup along the table, which
    tilts it in the grasp, and drops it from where the hand opened):
      pick: after the close the gripper lifts straight up by PICK_CLEAR (xy and orientation
            held at the grasp), then blends back onto the hand's path and rotation (turning
            / tilting the hand turns / tilts the cup); while held the cup's bottom stays
            >= PICK_CLEAR above the table (no dragging)
      place: when the hand releases, the gripper stays closed, turns the cup upright (or,
            tilted > 45 deg by the hand, flat on its side: end_pose) and lowers straight down
            until the cup rests on the table, opens, and holds still (no retreat).
    c0, axis0: the cup's centre and axis at the grasp (None: standing, rest pad height zc).
    end_axis: the axis to set it down with (seen by the camera after the release, plan_episode
    end_obs); None: from the hand's rotation at the release (end_pose). end_xy: where the
    camera saw the cup after the release (table xy): it is set down there."""
    ci = np.flatnonzero(close)
    if not len(ci):
        return pos, R, close
    a, e = int(ci[0]), int(ci[-1]) + 1  # e: first open frame
    p_a = pos[a].copy()  # the pad at the grasp
    t = np.arange(e - a)
    up = np.interp(t, [0, N_LIFT], [0.0, PICK_CLEAR])
    pos[a:e, 2] = np.maximum(pos[a:e, 2], zc + 0.002 + up)
    w = np.clip((t - N_LIFT) / N_BLEND, 0, 1)[:, None]  # 0: held at the grasp xy, 1: the hand's
    pos[a:e, :2] = (1 - w) * pos[a, :2] + w * pos[a:e, :2]
    # smooth the carry (the 3D hand's position jitters frame to frame; Andy 2026-11-14), the
    # grasp point itself unchanged, then the no-dragging floor again
    sm = _gauss(pos[a:e], CARRY_SIGMA)
    wa = np.clip(t / max(1, N_LIFT), 0, 1)[:, None]
    pos[a:e] = (1 - wa) * pos[a:e] + wa * sm
    pos[a:e, 2] = np.maximum(pos[a:e, 2], zc + 0.002 + up)
    # orientation: the hand's rotation (carried by the grasp, smoothed over +-ROT_SMOOTH
    # frames), blended in from the grasp orientation soon after the close (N_ROT0, N_ROT)
    Rh = Rot.from_matrix(R[a:e])
    k = ROT_SMOOTH
    Rh = Rot.concatenate([Rh[max(0, i - k) : i + k + 1].mean() for i in range(len(Rh))])
    Ra = Rot.from_matrix(R[a])
    rel = (Ra.inv() * Rh).as_rotvec()
    wr = np.clip((t - N_ROT0) / N_ROT, 0, 1)[:, None]
    R[a:e] = (Ra * Rot.from_rotvec(wr * rel)).as_matrix()
    R[a:e] = pour_assist(R[a:e], Ra.as_matrix(), axis0, pour_angle, pour_lean)
    n = len(pos)
    ax0 = np.array([0.0, 0.0, 1.0]) if axis0 is None else np.asarray(axis0, float)

    def place_from(k):
        """The set-down starting at frame k: (p0, R0, R_up, target, down, frames)."""
        p0, R0 = pos[k - 1].copy(), R[k - 1].copy()
        # set the cup down upright (or flat): turn the gripper so the cup's axis (axis0 at
        # the grasp, in the gripper frame R[a]^T axis0) is vertical (or horizontal) again
        u = R0 @ (Ra.as_matrix().T @ ax0)
        if end_axis is None:
            target, rest_z = end_pose(u, h, d)
        else:
            target = np.asarray(end_axis, float)
            rest_z = h / 2 if abs(target[2]) > 0.5 else d / 2
        fix = Rot.align_vectors([target], [u])[0]
        R_up = (fix * Rot.from_matrix(R0)).as_matrix()
        if c0 is None:
            down = np.r_[p0[:2], zc + 0.002]
        else:  # the pad where the cup (rigid in the grasp) rests on the table, straight
            # below where the cup was (turning the gripper upright about the pad would swing
            # the cup sideways by its offset from the pad: ~8 cm off on a flip)
            g_c = Ra.as_matrix().T @ (p_a - np.asarray(c0, float))  # pad - cup, gripper frame
            cup_rel = p0 - R0 @ g_c  # the cup's centre
            off = R_up @ g_c
            cxy = cup_rel[:2] if end_xy is None else np.asarray(end_xy, float)
            down = np.r_[cxy + off[:2], rest_z + off[2] - RELEASE_DIP]
        nd = int(np.clip(math.ceil(abs(p0[2] - down[2]) / (0.15 / 30)), ND_MIN, 30))
        return p0, R0, R_up, target, down, min(nd, max(0, n - k - 4))

    # the set-down starts up to SET_AHEAD frames before the hand opens, so the cup is on the
    # table when it does (Andy 2026-11-14: "the release is quite delayed": the cup was held
    # PICK_CLEAR up until the hand opened, then lowered); the gripper still opens only when
    # the cup touches the table (panda.simulate) or at the end of the lowering
    nd0 = place_from(e)[5]
    s0 = e - min(nd0, SET_AHEAD, max(0, e - a - N_LIFT - N_BLEND))
    p0, R0, R_up, target, down, nd = place_from(s0)
    if s0 < e and s0 + nd < e:  # never open before the hand: end the lowering at its release
        s0 = e - nd
        p0, R0, R_up, target, down, nd = place_from(s0)
    R_rel = R[e - 1].copy()  # the hand's rotation as it opens
    for i, w in enumerate(_mj(nd)):
        pos[s0 + i] = p0 + (down - p0) * w
        # the hand's own rotation (it is still turning the cup back as the lowering starts
        # early), blended into upright before the cup lands: no snap the arm cannot follow
        Rh = Rot.from_matrix(R[s0 + i] if s0 + i < e else R_rel)
        R[s0 + i] = Slerp([0, 1], Rot.concatenate([Rh, Rot.from_matrix(R_up)]))(
            min(1.0, 1.6 * w)
        ).as_matrix()
    R0 = R_up
    close[s0 : s0 + nd] = True
    close[s0 + nd :] = False
    rest = n - (s0 + nd)
    # after the release the arm holds still where the cup came to rest (Andy 2026-11-14: "after
    # release don't move the robot hand, it keeps colliding with the object": rising while
    # the fingers still open brushes the cup); the planned pose is RELEASE_DIP below the rest
    # (the lowering aims low, contact opens the gripper): held at the rest height instead
    still = down + np.array([0.0, 0.0, RELEASE_DIP])
    for i in range(rest):
        pos[s0 + nd + i] = still
        R[s0 + nd + i] = R0
    return pos, R, close


SLIP_FREE, SLIP_SCALE, W_SLIP = 5.0, 30.0, 0.5  # deg; in-hand pivot penalty of the search


def in_hand_pivot(log: dict) -> float:
    """Mean angle (deg) the cup's axis turned INSIDE the gripper while held (the cup's axis in
    the gripper frame vs just after the close). A grasp that lets the cup pivot or swing
    loses the hand's rotation and places the cup tilted; turning the cup on purpose (the
    gripper turns with it) is not penalised."""
    held = np.flatnonzero(~np.asarray(log["grip_open"], bool))
    if len(held) < 12:
        return 0.0
    held = held[10:]  # after the fingers settled
    cup = Rot.from_quat(np.asarray(log["obj_quat"])[held, 0][:, [1, 2, 3, 0]]).as_matrix()
    u = np.einsum("nji,nj->ni", np.asarray(log["tcp_R"])[held], cup[:, :, 2])  # R^T up
    return float(np.degrees(np.arccos(np.clip(u @ u[0], -1, 1))).mean())


DROP_DIST, W_DROP = 0.03, 1.0  # m the cup may shift in the gripper; penalty per held share
W_PLACE = 0.5  # penalty: the cup not set down right (tipped, not resting, not released)
W_SPOT, SPOT_FREE, SPOT_SCALE = 0.5, 0.01, 0.05  # the cup set down away from the person's spot
PLACE_FAILS = ("tipped", "not upside down", "not lying", "turned wrong", "cup not resting",
               "cup not released", "arm hit")  # fmt: skip


def dropped_share(log: dict) -> float:
    """Share of the gripper-closed time after the cup left the grasp (its centre > DROP_DIST
    from where it sat in the gripper frame just after the close): a grasp that loses the cup
    late in a long carry still follows most of the lift, so the lift reward alone misses it."""
    held = np.flatnonzero(~np.asarray(log["grip_open"], bool))
    if len(held) < 12:
        return 0.0
    tcp, Rt = np.asarray(log["tcp"])[held], np.asarray(log["tcp_R"])[held]
    obj = np.asarray(log["obj_pos"])[held, 0]
    rel = np.einsum("nji,nj->ni", Rt, obj - tcp)  # cup centre in the gripper frame
    off = np.linalg.norm(rel - rel[10], axis=1)
    lost = np.flatnonzero(off[10:] > DROP_DIST)
    return 0.0 if not len(lost) else float(1 - lost[0] / max(1, len(off) - 10))


def _score(args) -> float:
    """The whole episode simulated (pick, carry, set-down, release): lift accuracy
    (panda_replay.reward) - W_SLIP x the cup's pivot inside the gripper beyond 5 deg (per 30
    deg) - W_DROP x the held time after the cup was lost - W_PLACE if it is not set down
    right (demogen.check): prefers grasps that hold the cup rigidly until a clean release."""
    pl, g = args
    P, Rs, close = targets(pl, g)
    ro = PR.rollout({**pl, "P": P, "Rs": Rs, "close": close}, np.zeros(6), full=True)
    log = ro["sim"]["log"]
    piv = in_hand_pivot(log)
    drop = dropped_share(log)
    obj_end = ro["sim"]["obj"][-1, 0]
    _, why, _ = DG.check(log, obj_end, obj_end[:2], pl["h"], pl["d"],
                         lying=pl.get("end_lying", False),
                         flipped=pl.get("end_flip", False))  # fmt: skip
    bad_place = any(why.startswith(f) for f in PLACE_FAILS)
    spot = 0.0
    if "cup_end_xy" in pl:  # where the person's cup was let go
        spot = float(np.linalg.norm(obj_end[:2] - pl["cup_end_xy"]))
    return (ro["reward"] - W_SLIP * max(0.0, piv - SLIP_FREE) / SLIP_SCALE - W_DROP * drop
            - W_PLACE * bad_place - W_SPOT * max(0.0, spot - SPOT_FREE) / SPOT_SCALE)  # fmt: skip


def your_cup_path(pl: dict) -> np.ndarray:
    """The person's cup centre per frame [n, 3]: at its start until the grasp, carried rigidly
    with the hand while held (rotation included), then set down straight below the release
    point at its rest height."""
    c0, _ = cup_frame(pl)
    ka, P, R = pl["ka"], np.asarray(pl["hand_P"]), np.asarray(pl["hand_R"])
    on = np.flatnonzero(pl["close0"])
    out = np.repeat(np.asarray(c0, float)[None], len(P), 0)
    if not len(on):
        return out
    rel = R[ka].T @ (np.asarray(c0, float) - P[ka])
    for t in range(ka, on[-1] + 1):
        out[t] = P[t] + R[t] @ rel
    out[on[-1] + 1 :] = np.r_[out[on[-1], :2], pl.get("end_z", pl["h"] / 2)]
    return out


JITTER = np.array([[0.0, 0.0], [0.002, 0.0], [-0.001, 0.0017]])  # m: grasp target nudges


def _score_robust(args) -> float:
    """_score averaged over the grasp target nudged by JITTER (common to every candidate): one
    run's score swings by up to ~1.7 for a 1 mm change (contact events), so a single run picks
    grasps that won by luck and fail once anything shifts (generated scenes, the real arm)."""
    pl, g, off = args
    return _score(({**pl, "obj_xy": np.asarray(pl["obj_xy"]) + off}, g))


def _eval(pool, pl: dict, gs: np.ndarray, jitter: np.ndarray = JITTER) -> np.ndarray:
    """Robust scores of grasps gs (all candidate x nudge runs in one parallel batch)."""
    jobs = [(pl, g, off) for g in gs for off in jitter]
    return np.array(pool.map(_score_robust, jobs)).reshape(len(gs), len(jitter)).mean(1)


def search(pl: dict, pool, iters: int, pop: int, good: float = 0.95, n_elite: int = 5,
           budget_s: float = 0.0, progress=None, jitter: np.ndarray | None = None):  # fmt: skip
    """Cross-entropy method over the grasp parameters (episodic policy search). Reward: the
    whole episode (_score: lift following, in-hand pivot, drops, set-down), averaged over the
    grasp target nudged by `jitter` (_score_robust; None = single runs, the default: on
    Andy's takes 02:20 / 02:23 the averaged version, pop 10 x 3 runs, generated 4 + 9 / 20
    vs 9 + 10 / 20 single runs, pop 18, within the same 30 s budget).
    Round 0: the seeds (the person's grasp, from the side near the top,
    mid-height, from the top) filled up to pop with samples near it, so every worker runs.
    pop = workers / len(jitter) x k keeps the pool full. budget_s > 0: stop after this many
    seconds; progress(round, best reward): status callback. (A fixed grasp set + checks in
    shifted scenes was tried 2026-11-14 and did worse in Andy's live use.)"""
    t0 = time.time()
    jitter = np.zeros((1, 2)) if jitter is None else np.asarray(jitter, float)
    rng = np.random.default_rng(0)
    lo_b, hi_b = grasp_bounds(pl)
    seeds = grasp_seeds(pl)
    lying = bool(pl.get("cup") and pl["cup"]["lying"])
    sd0 = np.array([0.01, 0.3, 0.004, 2.0] if lying else [0.6, 0.4, 0.02, 2.0])
    n_fill = max(0, pop - len(seeds))  # round 0 uses every worker: seeds + samples near g0
    extra = np.clip(pl["g0"] + 0.5 * sd0 * rng.standard_normal((n_fill, 4)), lo_b, hi_b)
    seeds = np.r_[seeds, extra]
    rs = _eval(pool, pl, seeds, jitter)
    best_g, best_r = seeds[int(rs.argmax())], float(rs.max())
    hist = [round(best_r, 3)]
    if progress is not None:
        progress(0, best_r)
    mu, sd = best_g.copy(), sd0.copy()
    for it in range(iters):
        if best_r >= good or (budget_s > 0 and time.time() - t0 > budget_s):
            break
        gs = np.clip(mu + sd * rng.standard_normal((pop, 4)), lo_b, hi_b)
        rs = _eval(pool, pl, gs, jitter)
        el = gs[np.argsort(rs)[-n_elite:]]
        mu, sd = el.mean(0), np.maximum(el.std(0), [0.05, 0.05, 0.003, 0.5])
        if rs.max() > best_r:
            best_r, best_g = float(rs.max()), gs[int(rs.argmax())]
        hist.append(round(float(rs.max()), 3))
        if progress is not None:
            progress(it + 1, best_r)
    return best_g, best_r, hist


ROT_OUTLIER_DEG = 25.0  # a frame's hand rotation this far from its neighbours' = a flip


def rotation_outliers(R: np.ndarray, ok: np.ndarray, win: int = 5, passes: int = 2):
    """Frames whose hand rotation is > ROT_OUTLIER_DEG from the mean rotation of the valid
    frames within +-win (excluding itself): single-frame 3D hand flips (a real hand turns
    <= ~8 deg per frame in Andy's takes). Two passes (a big flip skews the first mean)."""
    out = np.zeros(len(R), bool)
    rot = Rot.from_matrix(np.asarray(R))
    for _ in range(passes):
        good = np.asarray(ok, bool) & ~out
        for i in np.flatnonzero(ok):
            nb = [j for j in range(max(0, i - win), min(len(R), i + win + 1)) if j != i and good[j]]
            if len(nb) < 2:
                continue
            m = rot[nb].mean()
            out[i] = np.degrees((m.inv() * rot[i]).magnitude()) > ROT_OUTLIER_DEG
    return out


def plan_episode(per: dict, a: int, e: int, h: float, d: float, cam_xml: str, cam_xy,
                 pool=None, cup: dict | None = None, end_obs: dict | None = None,
                 upright: bool = False, place_xy=None,
                 pour_angle: float = POUR_ANGLE, pour_side: str = "down"):  # fmt: skip
    """cup: the cylinder's pose measured from the camera (touch.cup_pose: {"xy", "lying",
    "yaw"}); None: standing where the person's pad centre is at the grasp. end_obs: how the
    camera saw the cup after the release ({"lying", "yaw"}): the set-down pose (standing:
    upright, or upside down when the hand turned it over); None: from the hand's rotation.
    upright: the cup starts and ends standing upright, whatever the hand / camera say (Andy
    2026-11-14: more reliable; it may still be turned over while carried). place_xy: the
    assigned spot the cup is set down at (table xy, e.g. marker 0's centre = (0, 0); Andy
    2026-11-14), drawn in the scene; overrides where the camera saw it put down."""
    n = len(per["close"])
    lo, hi = max(0, a - PRE), min(n, e + POST)
    J = per["jw"][lo:hi]
    ok = per["ok"][lo:hi]
    close = np.zeros(hi - lo, bool)
    close[a - lo : e - lo] = True
    if ok.sum() < 10 or not ok[a - lo : e - lo].any():
        return None, "hand not tracked"
    pp, RR, _, _ = S.retarget(J, ok, close)
    okr = ok & ~rotation_outliers(RR, ok)  # 3D hand flips: no real hand turns that fast
    hand_P, _ = S._interp_poses(pp, RR, ok)
    _, hand_R = S._interp_poses(pp, RR, okr if okr.sum() >= 2 else ok)
    ka = a - lo + int(np.argmax(ok[a - lo : e - lo]))  # first tracked frame of the grasp
    z_h = hand_R[ka][:, 2]
    if cup is not None:
        obj_xy = np.asarray(cup["xy"], float).copy()
        cup = {"xy": obj_xy, "lying": bool(cup["lying"]), "yaw": float(cup["yaw"])}
    else:
        obj_xy = hand_P[ka, :2].copy()  # the cylinder stands where the person's pad centre is
    if cup is not None and cup["lying"]:
        ax = np.array([math.cos(cup["yaw"]), math.sin(cup["yaw"])])
        perp = np.r_[-ax[1], ax[0], 0.0]
        s_h = float(np.clip((hand_P[ka, :2] - obj_xy) @ ax, -h / 2, h / 2))
        beta_h = float(np.clip(math.atan2(z_h @ perp, -z_h[2]), -1.0, 1.0))
        g0 = np.array([s_h, beta_h, 0.0, 0.0])
    else:
        g0 = np.array([math.atan2(z_h[1], z_h[0]),
                       float(np.clip(math.asin(-z_h[2]), 0, SIDE_PITCH)),
                       grip_height(J, ok, ka, h), 0.0])  # fmt: skip
    pl0 = {"obj_xy": obj_xy, "cup": cup, "h": h, "d": d}
    c0, ax0 = cup_frame(pl0)
    # how the hand leaves the cup: its axis turned with the hand from the grasp to the release
    u_end = hand_R[e - lo - 1] @ hand_R[ka].T @ ax0
    end_ax, end_z = end_pose(u_end, h, d)
    if end_obs is not None:  # the camera saw how the person left it (Andy 2026-11-14)
        if end_obs["lying"]:
            yw = float(end_obs["yaw"])
            end_ax, end_z = np.array([math.cos(yw), math.sin(yw), 0.0]), d / 2
        else:
            end_ax = np.array([0.0, 0.0, -1.0 if u_end[2] < -COS_UP else 1.0])
            end_z = h / 2
    if upright:
        end_ax, end_z = np.array([0.0, 0.0, 1.0]), h / 2
    # target object: at rest, then following the hand's pad while held (lift = its rise)
    gt = np.tile(c0, (len(hand_P), 1))
    seg = slice(ka, e - lo)
    gt[seg, 2] += np.clip(hand_P[seg, 2] - hand_P[ka, 2], 0, None)
    gt[seg, :2] += hand_P[seg, :2] - hand_P[ka, :2]
    if e - lo < len(gt):  # after the release: where it was set down
        gt[e - lo :] = gt[e - lo - 1] * [1, 1, 0] + [0, 0, end_z]
    pl = {"hand_P": hand_P, "hand_R": hand_R, "ka": ka, "close0": close, "h": h, "d": d,
          "obj_xy": obj_xy, "g0": g0, "W": np.full(len(hand_P), min(d, MAX_OPEN - 0.01)),
          "gi": 0, "n_obj": 1, "avoid": CLEAR, "gt_obj": gt, "lo": lo, "cup": cup,
          "end_lying": bool(abs(end_ax[2]) < 0.5), "end_flip": bool(end_ax[2] < -0.5),
          "end_yaw": float(math.atan2(end_ax[1], end_ax[0])),
          "end_axis": end_ax if end_obs is not None or upright else None}  # fmt: skip
    pl["end_z"] = float(end_z)
    pl["pour_angle"] = float(pour_angle)
    pl["cup_end_xy"] = your_cup_path(pl)[-1, :2]
    if end_obs is not None and end_obs.get("xy") is not None and cup is not None:
        pl["end_xy"] = np.asarray(end_obs["xy"], float)[:2]  # seen by the camera
        pl["cup_end_xy"] = pl["end_xy"]
        rel_i = int(np.flatnonzero(close)[-1]) + 1
        pl["gt_obj"][rel_i:, :2] = pl["end_xy"]
    spot = ""
    if place_xy is not None:  # the assigned spot (Andy 2026-11-14: "put it to the assigned
        # location"): the robot sets the cup down there, whatever the hand / camera say
        pl["end_xy"] = np.asarray(place_xy, float)[:2]
        pl["cup_end_xy"] = pl["end_xy"]
        rel_i = int(np.flatnonzero(close)[-1]) + 1
        pl["gt_obj"][rel_i:, :2] = pl["end_xy"]
        spot = (f'<geom name="target" type="box" size="{DG.PAD / 2} {DG.PAD / 2} 0.0005" '
                f'pos="{pl["end_xy"][0]} {pl["end_xy"][1]} 0.0006" rgba="0.2 0.85 0.3 0.6" '
                'contype="0" conaffinity="0" group="1"/>')  # fmt: skip
    ca = camera_axes(cam_xml)
    if ca is not None and pour_side in POUR_SIDES:  # the can top leans that way in the webcam
        # image (Andy 2026-11-14: "the pouring direction is toward the bottom of the video")
        ax, sg = POUR_SIDES[pour_side]
        v = ca[0][:2, ax] / max(np.linalg.norm(ca[0][:2, ax]), 1e-9)
        pl["pour_lean"] = sg * v
    paths = [targets(pl, s) for s in grasp_seeds(pl)[:2]]
    hd = view_direction(cam_xml)
    if ca is not None:  # render heading: the cup's path + the pour look as in the video
        cp = your_cup_path(pl)[:, :2]
        cp = cp[:: max(1, len(cp) // 12)]
        pts = np.c_[cp, np.full(len(cp), h / 2)]
        if pl.get("end_xy") is not None:
            pts = np.r_[pts, [np.r_[pl["end_xy"], h / 2]]]
        lean = pl.get("pour_lean")
        dirs = [(pts[0], np.r_[lean, 0.0])] if lean is not None else []
        hd, mis = RV.match_heading(ca[0], ca[1], pts, dirs)
        print(f"render heading matched to the webcam: mean direction mismatch {mis:.0f} deg",
              flush=True)  # fmt: skip
        pl["view_heading"] = hd
        spot += RV.heading_xml(hd)
    base, yaw = place_base(obj_xy, paths, h, d, cam_xy, pool, arm_direction(J, ok, ka),
                           hd)  # fmt: skip
    o = np.asarray(obj_xy)
    look = 0.6 * o + 0.4 * base
    line = (o - base) / max(np.linalg.norm(o - base), 1e-9)
    side = np.array([-line[1], line[0]])
    if side @ (np.asarray(cam_xy) - look) < 0:
        side = -side
    cpos = np.r_[look + 0.75 * side - 0.1 * line, 0.6]
    quat, z0 = None, None
    if cup is not None and cup["lying"]:
        q = Rot.from_euler("z", cup["yaw"]) * Rot.from_euler("y", np.pi / 2)
        quat, z0 = q.as_quat()[[3, 0, 1, 2]], d / 2 + 0.0005
    world = scene_world(obj_xy, h, d,
                        f'<body name="look" pos="{look[0]} {look[1]} 0.12"/>'
                        '<camera name="view" mode="targetbody" target="look" fovy="50" '
                        f'pos="{cpos[0]} {cpos[1]} {cpos[2]}"/>' + cam_xml + spot,
                        quat=quat, z=z0)  # fmt: skip
    pl.update(
        {
            "xml": panda_mjcf(base, yaw, world),
            "base": base,
            "yaw": yaw,
            "ik_xml": panda_mjcf(
                base, yaw, scene_world(obj_xy, h, d, margin=True, quat=quat, z=z0)
            ),
        }
    )
    return pl, ""


# ------------------------------------------------------------------ 6. render
def draw_hand_lm(img, lm, col):
    if lm is None or not np.isfinite(lm).all():
        return
    pts = [tuple(int(v) for v in p[:2]) for p in lm]
    for i, j in PR.HAND_EDGES:
        cv2.line(img, pts[i], pts[j], col, 3, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 4, (255, 255, 255), -1, cv2.LINE_AA)


def write_episode(clip: Path, video: Path, per: dict, pl: dict, ro: dict, res: dict, fps: int,
                  k: int) -> None:  # fmt: skip
    cap = cv2.VideoCapture(str(video))
    lo = pl["lo"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    frames_cam, last = {}, -1
    raw = clip.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 360))
    sim, close = ro["sim"], ro["close"]
    ok_txt = ("LIFTED", (80, 220, 80)) if res.get("success") else ("NOT LIFTED", (60, 60, 255))
    for i, simf in enumerate(sim["frames"]):
        h = int(sim["human"][i])
        while last < h:  # read the webcam frame of this human frame
            okr, f = cap.read()
            if not okr:
                break
            last += 1
            frames_cam = {"f": f}
        img = frames_cam["f"].copy()
        g = bool(per["close"][lo + h])
        draw_hand_lm(img, per["lm"][lo + h], (0, 0, 255) if g else (0, 255, 0))
        img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
        label(img, f"your hand (grasp {k + 1})", (12, 30), 0.7)
        label(img, f"model: {'HOLD' if g else 'not holding'}", (12, 60), 0.7,
              (0, 0, 255) if g else (255, 255, 255))  # fmt: skip
        a = cv2.resize(simf, (640, 360), interpolation=cv2.INTER_AREA)
        label(a, "Panda (offline plan + small RL)", (12, 30), 0.7)
        label(a, f"gripper: {'CLOSED' if close[h] else 'OPEN'}", (12, 60), 0.7,
              (0, 0, 255) if close[h] else (255, 255, 255))  # fmt: skip
        lift = (sim["obj"][i, 0, 2] - sim["obj"][0, 0, 2]) * 100
        hl = (pl["gt_obj"][h, 2] - pl["gt_obj"][0, 2]) * 100
        label(a, f"lift {lift:4.1f} cm (hand {hl:4.1f} cm)", (12, 345), 0.6)
        frame = np.hstack([img, a])
        vw.write(frame)
    for _ in range(fps):
        f2 = frame.copy()
        label(f2, f"robot: {ok_txt[0]}", (700, 190), 1.3, ok_txt[1], 3)
        vw.write(f2)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(clip)], check=True)  # fmt: skip
    raw.unlink()


# ------------------------------------------------------------------ main
def main() -> None:
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=default_ckpt())
    ap.add_argument("--calib", type=Path, default=Path("results/rig/calib.json"))
    ap.add_argument("--obj-h", type=float, default=CYL_H)
    ap.add_argument("--obj-d", type=float, default=CYL_D)
    ap.add_argument("--release", default="fast", choices=["normal", "fast", "fastest"])
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--iters", type=int, default=10, help="CEM iterations")
    ap.add_argument("--pop", type=int, default=24, help="CEM rollouts per iteration")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fps", type=int, default=15, help="video fps (15 = half speed)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path("results/robot_sim/panda_video") / args.video.parent.name
    out.mkdir(parents=True, exist_ok=True)
    cal = json.loads(args.calib.read_text())
    K, c2w = np.asarray(cal["K"]), np.asarray(cal["cam_to_world"])
    W, H = cal["image_size"]
    flip = {"h": "h" in cal.get("flip", ""), "v": "v" in cal.get("flip", "")}
    per = perceive(args.video, args.ckpt, K, c2w, flip, args.release, out / "perception.npz")
    ok = np.isfinite(per["joints"]).all((1, 2))
    per["ok"] = ok
    per["jw"] = np.where(ok[:, None, None], per["joints"], 0.0)
    per["jw"] = S.smooth(per["jw"], ok, 5)
    close = S.fill_gaps(per["close"].astype(bool), 10)
    per["close"] = close
    eps = episodes(close)
    print(f"{len(close)} frames, hand 3D in {ok.mean():.0%}, {len(eps)} grasp episodes", flush=True)
    if args.max_episodes:
        eps = eps[: args.max_episodes]
    cam_xml = camera_xml(K, c2w, H)
    pool = get_context("spawn").Pool(args.workers)
    summary, clips = [], []
    for k, (a, e) in enumerate(eps):
        t0 = time.time()
        pl, why = plan_episode(per, a, e, args.obj_h, args.obj_d, cam_xml, c2w[:2, 3], pool)
        if pl is None:
            print(f"episode {k}: skip ({why})", flush=True)
            continue
        g, best_r, hist = search(pl, pool, args.iters, args.pop)
        P, Rs, close = targets(pl, g)
        pl.update({"P": P, "Rs": Rs, "close": close})
        rnd = {}

        def render(m, d, rnd=rnd):
            if "r" not in rnd:
                rnd["r"] = mujoco.Renderer(m, 360, 640)
            rnd["r"].update_scene(d, camera="view")
            return rnd["r"].render()[..., ::-1].copy()

        ro = PR.rollout(pl, np.zeros(6), render, full=True)
        z = ro["sim"]["obj"][ro["last"], 0]
        res = S.lift_outcome(pl["gt_obj"], z) or {}
        r = {"episode": k, "frames": [a, e], "obj_xy": pl["obj_xy"].round(3).tolist(),
             "base_xy": pl["base"].round(3).tolist(), "reward": round(ro["reward"], 3),
             "cem_best_per_iter": hist,
             "grasp": {"azimuth_deg": round(math.degrees(g[0]), 1),
                       "pitch_deg": round(math.degrees(g[1]), 1),
                       "height_cm": round(g[2] * 100, 1), "close_shift": int(round(g[3])),
                       "person_azimuth_deg": round(math.degrees(pl["g0"][0]), 1),
                       "person_pitch_deg": round(math.degrees(pl["g0"][1]), 1)},
             "hand_lift_cm": round(float((pl["gt_obj"][:, 2] - pl["gt_obj"][0, 2]).max()) * 100, 1),
             "robot_lift_cm": round(float((z[:, 2] - z[0, 2]).max()) * 100, 1),
             "minutes": round((time.time() - t0) / 60, 1), **res}  # fmt: skip
        summary.append(r)
        print(json.dumps(r), flush=True)
        clip = out / f"episode_{k}.mp4"
        write_episode(clip, args.video, per, pl, ro, res, args.fps, k)
        clips.append(clip)
    pool.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if clips:
        lst = out / "clips.txt"
        lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(lst), "-c", "copy", str(out / "panda_video.mp4")],
                       check=True)  # fmt: skip
        lst.unlink()
    n_ok = sum(bool(r.get("success")) for r in summary)
    print("wrote", out, f"{n_ok}/{len(summary)} lifted")


if __name__ == "__main__":
    main()
