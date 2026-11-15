"""One-command demo: teleoperate the Panda with your hand, record the demo, scale it to 10.

  LIVE      webcam + MediaPipe + grasp model + learned grab gate (as scripts/live_sim.py); every
            frame is buffered (3D hand, landmarks, gate, a small copy of the image)
  E         end the take: the camera stops and every grasp in the take (gate closed >= 10
            frames) is processed as scripts/panda_from_video.py does (cylinder where the hand
            closed, Panda base search, collision-free IK, small-RL grasp search on lift
            accuracy), progress shown in the window
  PLAYBACK  your hand | the Panda (pick and place: the robot lifts the cup straight up after
            the grasp and sets it down before opening). A successful robot run is saved at once
            as a teleop demonstration (robomimic-style HDF5, --dataset)
  SCALE     the demo is turned into 10 new ones (touch.demogen: MimicGen-style object-centric
            segments, cup and place target moved up to 10 cm), simulated side by side in 10
            MuJoCo scenes (10 processes, each scene saved as its own MJCF); each tile shows
            SUCCESS / FAIL at its end and the successful ones are added to the dataset
  R new take   G 10 more from the same demo   P replay the teleop   O open the take's folder
  Q / Esc quit
  Saved automatically per take in results/robot_sim/panda_demo/<time>/: take.mp4 (the camera
  video of your grab), teleop.mp4, scaled_<k>.mp4, scenes_<k>/scene_*.xml, take.npz (hand
  tracking), summary.json

    python scripts/panda_demo.py [--cam 1] [--budget 30]
Needs results/rig/calib.json (scripts/calib_rig.py) and the demo can (13.4 x 5.2 cm; else
--obj-h / --obj-d in m).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue as _queue
import subprocess
import sys
import threading
import time
from multiprocessing import get_context, parent_process
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
import demo_ui as U  # noqa: E402
import panda_from_video as V  # noqa: E402
import panda_replay as PR  # noqa: E402

if parent_process() is None:  # the hand models: main process only. Every spawn worker imports
    # this file as __mp_main__; live_grasp loads torch + onnxruntime, ~370 MB per worker x 12
    # that froze a 16 GB Mac on S (2026-11-15: swap full, app hung right after scaling began)
    from live_grasp import HandTracker, LiveGrasp, frames_with_hands, threaded  # noqa: E402

from touch import cup_pose as CP  # noqa: E402
from touch import demogen as DG  # noqa: E402
from touch import rig  # noqa: E402
from touch import robot_sim as S  # noqa: E402
from touch.handpath import StableDepth, clean_depth, fuse_orientation, world_hand  # noqa: E402
from touch.interact.grab_gate import FusionGate  # noqa: E402
from touch.live_robot import CYL_D, CYL_H, camera_xml  # noqa: E402
from touch.mano_path import MANO_DIR  # noqa: E402

PW, PH = 800, 450  # each panel
MIN_LIFT = 0.03  # m: grasps where the hand lifted less are skipped (touches, slides)
G10 = "runs/g10_grasp_own_20261113-230008/best.pt"
G13 = "runs/g13_all_20261114-123030/best.pt"  # default (Andy 2026-11-14: test g13)
TITLE = "OneTouch -> Panda   (E: process take   R: new take   G: 10 more   Q: quit)"
GREEN, RED, AMBER = (80, 220, 80), (60, 60, 255), (120, 220, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, org, scale=0.8, col=(255, 255, 255), th=2):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, col, th, cv2.LINE_AA)


def draw_live(view, hand: bool, fps: float) -> None:
    """Live preview text: hand tracked or not, the recording fps (no drawing of the hand)."""
    txt, col = ("hand tracked", (0, 255, 0)) if hand else ("no hand", (0, 0, 255))
    cv2.putText(view, txt, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    fc = (255, 255, 255) if fps >= 28 else (0, 0, 255)
    cv2.putText(view, f"{fps:4.1f} fps", (PW - 120, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fc, 2)


def model_step(
    item, live, fusion, K, c2w, depth: StableDepth | None = None, t: float = 0.0
) -> tuple:
    """One recorded frame (crops, landmarks, handedness, gate image features) through the
    grasp model, the 3D hand and the grab gate (causal; call in frame order) -> world joints
    [21, 3] (NaN = none), gripper closed, gate score, P(hold), the model output dict (+
    "verts_w": the 3D hand mesh [778, 3] in the table frame, None = no 3D hand). depth: a
    StableDepth (steady palm-based depth, causal; t = frame time s), else world_hand."""
    crops, lm, left, ff = item
    out = live.step(live.inputs(crops))
    jw = np.full((21, 3), np.nan)
    if lm is not None:
        with np.errstate(all="ignore"):
            if depth is not None:
                wh = depth.step(out["joints"], out["verts"], lm, left, t)
                wh = {k: np.asarray(v)[None] for k, v in wh.items()}
            else:
                wh = world_hand(out["joints"][None], out["verts"][None], lm[None], K, c2w, left,
                                smooth_k=1)  # fmt: skip
        if wh["ok"][0]:
            jw = wh["joints"][0]
            out["verts_w"] = wh["verts"][0]
    extra = {}
    if fusion.needs_frames:
        extra = {"joints_pred": out.get("joints"), "lm": lm, "joints_world": jw, **ff}
    closed = bool(fusion.step(out["state"], out["finger"], lm is not None, **extra))
    p_hold = float(out["state"][2]) if lm is not None else 0.0
    return jw, closed, float(fusion.score), p_hold, out


HOLD_GAP = 15  # frames (0.5 s): an open gap this short inside a hold is a flicker, not a release
RELEASE_GAP = HOLD_GAP + 1  # frames open after a hold = a real release
STOP_AFTER = 45  # frames processed after the release (plan_episode uses POST = 30 of them)


def run_models(take: dict, live, fusion, K, c2w, show=None, depth=None, box=None,
               start: int = 0) -> None:  # fmt: skip
    """The grasp model, the 3D hand and the grab gate over a recorded take (take["raw"]),
    frame by frame in order -> take["joints"], take["close"] (same as running them live).
    To save time (Andy 2026-11-14) only from the first frame with a hand until STOP_AFTER
    frames after the first release; the frames outside are "no hand" (NaN joints, open).
    No release found -> the whole rest of the take (find_holds' fallbacks need it).
    box: the clicked cup's image box -> only a hold AT the cup stops early (Andy 2026-11-14: a
    false hold with the hand on the laptop ended processing before the real grab).
    start: the frame the object was double-clicked; earlier frames are not processed (Andy
    2026-11-14: the hand before the object is set up must not count).
    show(i, n): progress callback."""
    raw = take.pop("raw")
    n = len(raw)
    first = next((i for i in range(start, n) if raw[i][1] is not None), n)
    joints = [np.full((21, 3), np.nan)] * n
    close, score, p_hold = [False] * n, [float("nan")] * n, [0.0] * n
    held, open_run, stop, s0 = False, 0, n, 0
    for i in range(first, n):
        if i >= stop:
            break
        jw, c, sc, ph, _ = model_step(raw[i], live, fusion, K, c2w, depth, i / 30)
        joints[i], close[i], score[i], p_hold[i] = jw, c, sc, ph
        if c and not held:
            held, s0 = True, i
        open_run = 0 if c else open_run + 1
        if held and open_run == RELEASE_GAP and stop == n:
            e0 = i - RELEASE_GAP + 1  # the release frame
            if box is None or at_cup(take["lm"], s0, e0, box):
                stop = e0 + STOP_AFTER
            else:  # a hold away from the cup: keep going, wait for the real one
                held = False
        if show is not None and i % 15 == 0:
            show(i, min(stop, n))
    take["processed"] = (first, min(stop, n))
    take["joints"], take["close"] = joints, close
    take["score"], take["p_hold"] = score, p_hold  # for find_holds' fallbacks


class LiveInference(threading.Thread):
    """During the take: the grasp model, 3D hand and grab gate over take["raw"] in frame
    order (the same causal calls run_models makes after E), while the camera keeps recording.
    Each result goes to `out` (live HOLD label + hand mesh); after E, finish(n) processes
    the frames still waiting and the results replace run_models (no wait after E)."""

    def __init__(self, take: dict, live, fusion, K, c2w, out: _queue.Queue, depth=None,
                 start_fn=None):  # fmt: skip
        super().__init__(daemon=True)
        # start_fn() -> the frame the object was double-clicked (None: not yet). The model
        # waits for it; earlier frames get "no hand" (Andy 2026-11-14)
        self.start_fn = start_fn or (lambda: 0)
        self.depth = depth
        self.take, self.live, self.fusion, self.K, self.c2w, self.out = (
            take, live, fusion, K, c2w, out)  # fmt: skip
        self.lock = threading.Lock()
        self.target: int | None = None  # None: live (follow the recording)
        self.res: dict = {"joints": [], "close": [], "score": [], "p_hold": []}
        self.err: Exception | None = None
        self.stream = None  # HandStream: each processed frame also goes to Tiger Data

    def finish(self, n: int) -> None:
        """Process exactly the first n frames, then stop (0: stop now)."""
        with self.lock:
            self.target = n

    def run(self) -> None:
        i = 0
        try:
            while True:
                with self.lock:
                    if self.target is not None and i >= self.target:
                        return
                    item = self.take["raw"][i] if i < len(self.take["raw"]) else None
                    ended = self.target is not None
                if item is None:
                    time.sleep(0.002)
                    continue
                s0 = self.start_fn()
                if s0 is None and not ended:  # the object is not double-clicked yet
                    time.sleep(0.005)
                    continue
                if i < (s0 or 0):  # before the click: not processed
                    for k, v in zip(self.res, (np.full((21, 3), np.nan), False, float("nan"),
                                               0.0), strict=True):  # fmt: skip
                        self.res[k].append(v)
                    self.out.put((i, False, float("nan"), 0.0, None))
                    i += 1
                    continue
                jw, c, sc, ph, o = model_step(item, self.live, self.fusion, self.K, self.c2w,
                                              self.depth, i / 30)  # fmt: skip
                for k, v in zip(self.res, (jw, c, sc, ph), strict=True):
                    self.res[k].append(v)
                self.out.put((i, c, sc, ph, o.get("verts_w")))
                if self.stream is not None:
                    if i == s0:
                        self.stream.start(time.strftime("%Y%m%d-%H%M%S"), {"click_frame": i})
                    self.stream.frame(i, item[1], bool(item[2]), jw, bool(c), float(sc))
                i += 1
        except Exception as ex:  # re-raised in the main thread
            self.err = ex


def mano_faces() -> np.ndarray:
    """MANO triangle indices [1538, 3] (from the local MANO_RIGHT.pkl; same for both hands)."""
    import pickle

    with open(MANO_DIR / "MANO_RIGHT.pkl", "rb") as f:
        return np.asarray(pickle.load(f, encoding="latin1")["f"], np.int64)


def draw_hand_mesh(view, verts_w, faces, K, c2w, scale: float, hold: bool) -> None:
    """The 3D hand mesh projected into the view (K for the full frame; scale = view / frame
    width), flat shaded, front faces only; red while holding."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    pc = (np.asarray(verts_w, np.float64) - t) @ R  # world -> camera
    if not (pc[:, 2] > 0.05).all():
        return
    uv = (pc[:, :2] / pc[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    tri, tri_c = uv[faces] * scale, pc[faces]
    n = np.cross(tri_c[:, 1] - tri_c[:, 0], tri_c[:, 2] - tri_c[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    view_dir = tri_c.mean(1)
    view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True)
    lit = np.abs((n * view_dir).sum(1))  # both windings (the left hand is mirrored)
    order = np.argsort(-tri_c[:, :, 2].mean(1))  # far -> near (painter's)
    base = np.array((70, 70, 235) if hold else (235, 190, 90), np.float64)
    over = view.copy()
    levels = np.minimum((lit[order] * 6).astype(int), 5)
    pts = (tri[order] * 16).astype(np.int32)  # 4 fractional bits
    for lv in range(6):  # one fillPoly per shade level (fast; order kept within a level)
        m = levels == lv
        if m.any():
            col = tuple(float(c) for c in base * (0.45 + 0.11 * lv))
            cv2.fillPoly(over, list(pts[m]), col, cv2.LINE_AA, shift=4)
    cv2.addWeighted(over, 0.6, view, 0.4, 0, view)


EDGE_FRAC = 0.015  # a hand with a landmark this close to the image edge is cut off: ignored


def whole_hands(hands: list, handedness: list, W: int, H: int) -> tuple[list, list]:
    """Only hands fully inside the frame: a hand cut off by the image edge (e.g. someone
    typing at the top of the view) gives a garbage 3D hand that the robot would copy."""
    mx, my = EDGE_FRAC * W, EDGE_FRAC * H
    keep = [i for i, h in enumerate(hands)
            if (h[:, 0] > mx).all() and (h[:, 0] < W - mx).all()
            and (h[:, 1] > my).all() and (h[:, 1] < H - my).all()]  # fmt: skip
    return [hands[i] for i in keep], [handedness[i] for i in keep]


def cup_box_px(est: dict, h: float, d: float, K, c2w, grow: float = 0.4):
    """The double-clicked cup's image box (full-res px), grown by `grow` of its size."""
    Rc, c0 = CP.cup_pose(est["xy"], est["lying"], est["yaw"], h, d)
    ax, u, v = Rc[:, 2], Rc[:, 0], Rc[:, 1]
    pts = np.array([c0 + sa * ax * h / 2 + su * u * d / 2 + sv * v * d / 2
                    for sa in (-1, 1) for su in (-1, 1) for sv in (-1, 1)])  # fmt: skip
    R, t = np.asarray(c2w)[:3, :3], np.asarray(c2w)[:3, 3]
    pc = (pts - t) @ R
    uv = pc[:, :2] / pc[:, 2:3] * [K[0][0], K[1][1]] + [K[0][2], K[1][2]]
    lo, hi = uv.min(0), uv.max(0)
    g = grow * (hi - lo)
    return lo - g, hi + g


CUP_GROW = 1.0  # the cup box grown by its own size: the hand wraps the cup from the side
FIRST_HOLD = 30  # frames: the shortest hold taken as "the first grab" when none is at the cup


def at_cup(lm: np.ndarray, s: int, e: int, box, n: int = 30) -> bool:
    """A thumb / index / middle tip inside the (grown) cup box in the first n frames of a hold."""
    lo, hi = box
    tips = np.asarray(lm)[s : min(e, s + n)][:, [4, 8, 12], :2]
    ok = np.isfinite(tips).all(-1)
    return bool((((tips >= lo) & (tips <= hi)).all(-1) & ok).any())


def first_grab(eps: list) -> tuple[int, int]:
    """Andy 2026-11-14: the first object he grabs after the double-click is the target -> the
    first hold >= FIRST_HOLD frames (short flickers are not grabs), else the longest."""
    return next(((s, e) for s, e in eps if e - s >= FIRST_HOLD),
                max(eps, key=lambda x: x[1] - x[0]))  # fmt: skip


LOOSE_GATE = {"on": 0.3, "off": 0.15, "k_on": 1, "k_off": 3}  # fallback 2
P_HOLD_MIN = 0.3  # fallback 3: the model's own P(hold)


def find_holds(tk: dict) -> tuple[np.ndarray, list[tuple[int, int]], str]:
    """Grasps of a take -> (close per frame, [(grab, release)], how). The gate's finished holds;
    if none: 1) a hold still on at the end -> released at the take's end; 2) the gate's score
    re-run with a looser hysteresis (LOOSE_GATE); 3) the longest run of the model's P(hold) >=
    P_HOLD_MIN (gaps <= 10 frames filled). Andy 2026-11-14: never stop at "no grasp" when the
    take has one."""
    from touch.interact.grab_gate import Hysteresis

    close = S.fill_gaps(np.array(tk["close"], bool), HOLD_GAP)
    n = len(close)
    eps = V.episodes(close)
    done = [(a, e) for a, e in eps if e < n]
    if done:
        return close, done, "gate"
    if eps:  # still holding when E was pressed
        return close, [(eps[-1][0], n)], "hold until the end of the take"
    score = np.asarray(tk.get("score", []), float)
    if len(score) == n:
        h = Hysteresis(**LOOSE_GATE)
        c2 = S.fill_gaps(np.array([h.step(x, bool(np.isfinite(x))) for x in score]), HOLD_GAP)
        e2 = V.episodes(c2)
        if e2:
            a, e = max(e2, key=lambda x: x[1] - x[0])
            return c2, [(a, min(e, n))], "looser threshold"
    ph = np.asarray(tk.get("p_hold", []), float)
    if len(ph) == n:
        c3 = S.fill_gaps(ph >= P_HOLD_MIN, HOLD_GAP)
        e3 = V.episodes(c3)
        if e3:
            a, e = max(e3, key=lambda x: x[1] - x[0])
            return c3, [(a, min(e, n))], f"model P(hold) >= {P_HOLD_MIN}"
    return close, [], "none"


def count_grasps(close) -> int:
    c = np.r_[False, np.asarray(close, bool)]
    return int((c[1:] & ~c[:-1]).sum())


def panel(lines, col=(255, 255, 255), scale: float = 0.85, step: int = 48) -> np.ndarray:
    img = np.full((PH, PW, 3), 32, np.uint8)
    for i, s in enumerate(lines):
        text(img, s, (30, 70 + step * i), scale, col)
    return img


def render_rollout(args):
    """Pool worker: final rollout of the chosen grasp with the third-person camera -> JPEG
    frames (rendering in a worker process keeps OpenGL off the UI thread)."""
    import mujoco

    from touch import render_view as RV

    pl, g = args
    P, Rs, close = V.targets(pl, g)
    pl = {**pl, "P": P, "Rs": Rs, "close": close, "carry_free": True}  # no rush after a carry

    from touch.panda import PandaIK

    ro = PR.rollout(pl, np.zeros(6), None, full=True)
    z = ro["sim"]["obj"][ro["last"], 0]
    log = ro["sim"]["log"]
    # the camera: the scene's "view", or (the arm hides the cup there) the clearest angle around
    # it (Andy 2026-11-14); frames rendered from the recorded states, one per step
    mv = mujoco.MjModel.from_xml_string(pl["xml"])
    cam, vinfo = RV.view_for(mv, log["state"], PW / PH)
    jpgs = [cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
            for _, img in RV.render_states(mv, log["state"], cam, PW, PH)]  # fmt: skip
    out = {"jpgs": jpgs, "human": ro["sim"]["human"], "close": close, "view": vinfo,
           "target": P[np.asarray(ro["sim"]["human"], int)],  # the pad target per sim frame
           "target_R": Rs[np.asarray(ro["sim"]["human"], int)],
           "obj_z": ro["sim"]["obj"][:, 0, 2], "obj_pos": ro["sim"]["obj"][:, 0],
           "obj_quat": log["obj_quat"][:, 0], "reward": ro["reward"],
           "lift": S.lift_outcome(pl["gt_obj"], z) or {}, "log": log, "src": None}  # fmt: skip
    try:  # the robot's run as a demonstration: did it pick and place the cup?
        src = DG.source_from_rollout(log, pl["base"], pl["yaw"], pl["h"], pl["d"])
        if pl.get("view_heading") is not None:  # the take's render heading, for its scenes
            src["vhead_xml"] = RV.heading_xml(pl["view_heading"])
        ok, why, met = DG.check(log, ro["sim"]["obj"][-1, 0], src["place"], pl["h"], pl["d"],
                                lying=pl.get("end_lying", False),
                                flipped=pl.get("end_flip", False))  # fmt: skip
    except ValueError as ex:
        src, ok, why, met = None, False, str(ex), {}
    ik = PandaIK(mujoco.MjModel.from_xml_string(pl["xml"]))
    eef = np.array([np.r_[p, Rot.from_matrix(R).as_rotvec()]
                    for p, R in (ik.fk(q) for q in log["q_cmd"])])  # fmt: skip
    met = {**met, **path_diff(pl, P, close, ro)}
    try:
        met.update(axis_diff(pl, Rs, close, ro))
    except Exception as ex:  # a metric only: never fail the take
        met["tilt_error"] = f"{type(ex).__name__}: {ex}"[:80]
    out.update({"src": src, "src_ok": ok, "src_why": why, "src_met": met, "eef_cmd": eef})
    return out


def add_path(scn, pts, rgba, width: float = 3.0) -> None:
    """Draw a polyline (world points [n, 3]) into a MuJoCo scene (after update_scene)."""
    import mujoco

    col = np.asarray(rgba, np.float32)
    for a, b in zip(pts[:-1], pts[1:], strict=False):
        if scn.ngeom >= scn.maxgeom:
            return
        gm = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                            np.eye(3).ravel(), col)  # fmt: skip
        mujoco.mjv_connector(gm, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, float),
                             np.asarray(b, float))  # fmt: skip
        scn.ngeom += 1


def axis_diff(pl: dict, Rs, close, ro) -> dict:
    """How far the cup's TILT strays from the person's while held (deg, mean / max), split by
    where it comes from: cmd_vs_hand = the planned gripper turn vs the hand's turn (smoothing,
    lift blend, IK seeds); gripper_vs_cmd = the arm not reaching the planned orientation;
    cup_in_gripper = the cup pivoting / slipping between the fingers; cup_vs_yours = total.
    Axes: the cup's axis at the grasp turned by each frame's rotation since the grasp."""
    last = ro["last"]
    log = ro["sim"]["log"]
    on = np.flatnonzero(np.asarray(close[: len(last)]))
    if len(on) < 2 or "tcp_R" not in log or "obj_quat" not in log:
        return {}
    a, e = int(on[0]), int(on[-1]) + 1
    _, ax0 = V.cup_frame(pl)
    ax0 = np.asarray(ax0, float)
    HR = np.asarray(pl["hand_R"])
    ka = pl["ka"]
    TR = log["tcp_R"][last]
    q = log["obj_quat"][last, 0]  # wxyz
    CR = Rot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    Rs = np.asarray(Rs)

    def turned(Rt, i0):
        return np.einsum("tij,j->ti", Rt[a:e] @ Rt[i0].T, ax0)

    yours = np.einsum("tij,j->ti", HR[a:e] @ HR[ka].T, ax0)
    cmd, grip, cup = turned(Rs, a), turned(TR, a), turned(CR, a)

    def ang(x, y):
        c = np.clip((x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)),
                    -1, 1)  # fmt: skip
        d = np.degrees(np.arccos(c))
        return [round(float(d.mean()), 1), round(float(d.max()), 1)]

    return {
        "tilt_cup_vs_yours_deg": ang(cup, yours),
        "tilt_cmd_vs_hand_deg": ang(cmd, yours),
        "tilt_gripper_vs_cmd_deg": ang(grip, cmd),
        "tilt_cup_in_gripper_deg": ang(cup, grip),
    }


def path_diff(pl: dict, P, close, ro) -> dict:
    """How far the robot strays from the person per phase (approach / held / after release),
    cm: gripper vs the person's hand path, and the cup it moved vs the person's cup path."""
    last = ro["last"]
    tcp, cup = ro["sim"]["tcp"][last], ro["sim"]["obj"][last, 0]
    n = len(tcp)
    hand, gt, c = np.asarray(pl["hand_P"][:n]), V.your_cup_path(pl)[:n], np.asarray(close[:n])
    on = np.flatnonzero(c)
    if not len(on):
        return {}
    out = {}
    for k, sl in (("approach", slice(0, on[0])), ("held", slice(on[0], on[-1] + 1)),
                  ("after_release", slice(on[-1] + 1, n))):  # fmt: skip
        if sl.stop - (sl.start or 0) <= 0:
            continue
        out[f"gripper_vs_hand_{k}_cm"] = round(float(np.linalg.norm(tcp[sl] - hand[sl], axis=1)
                                                     .mean()) * 100, 1)  # fmt: skip
        out[f"cup_vs_your_cup_{k}_cm"] = round(float(np.linalg.norm(cup[sl] - gt[sl], axis=1)
                                                     .mean()) * 100, 1)  # fmt: skip
    if "cup_end_xy" in pl:
        out["cup_vs_your_spot_cm"] = round(float(np.linalg.norm(cup[-1, :2] - pl["cup_end_xy"]))
                                           * 100, 1)  # fmt: skip
    return out


class ResizedCapture:
    """Frames at the calibration's size (same aspect ratio): saved takes, or a camera that
    delivers another size (said once)."""

    def __init__(self, cap, size):
        self.cap, self.size, self.told = cap, tuple(size), False

    def read(self):
        ok, f = self.cap.read()
        if ok and (f.shape[1], f.shape[0]) != self.size:
            assert abs(f.shape[1] / f.shape[0] - self.size[0] / self.size[1]) < 0.01, (
                f"camera / video {f.shape[1]}x{f.shape[0]}: aspect differs from the calibration "
                f"{self.size[0]}x{self.size[1]} (recalibrate: scripts/calib_rig.py)"
            )
            if not self.told:
                self.told = True
                print(f"frames are {f.shape[1]}x{f.shape[0]}, scaled to the calibration's "
                      f"{self.size[0]}x{self.size[1]}", flush=True)  # fmt: skip
            f = cv2.resize(f, self.size, interpolation=cv2.INTER_LINEAR)
        return ok, f

    def set(self, *a):
        return self.cap.set(*a)

    def get(self, *a):
        return self.cap.get(*a)

    def release(self):
        self.cap.release()


class PrefetchCapture:
    """Reads frames in its own thread (every frame, in order) so grabbing / decoding /
    resizing overlaps MediaPipe instead of adding to it: the camera loop's ceiling becomes
    MediaPipe alone (~25 ms at 1080p, ~40 fps) rather than read + MediaPipe (~30 fps)."""

    def __init__(self, cap, depth: int = 4):
        self.cap, self.q = cap, _queue.Queue(depth)
        self.alive = True
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while self.alive:
            ok, f = self.cap.read()
            self.q.put((ok, f))
            if not ok:
                return

    def read(self):
        return self.q.get()

    def release(self):
        """Stop the reader and wait for it BEFORE freeing the camera (releasing it while the
        reader is inside cap.read() segfaults on macOS / AVFoundation)."""
        self.alive = False
        while self.t.is_alive():
            try:  # unblock the reader if it waits on a full queue
                self.q.get_nowait()
            except _queue.Empty:
                pass
            self.t.join(timeout=0.05)
        self.cap.release()


def load_replay(d: Path) -> tuple[dict, str, int]:
    """A saved take folder -> (its double-click selection as `sel` holds it, take.mp4, frames).
    The segmentation is reused (no SAM); older takes without the full fit get it rebuilt from
    click.json's pose."""
    info = json.loads((d / "click.json").read_text())
    n = len(np.load(d / "take.npz", allow_pickle=True)["lm"])
    if not info.get("selected"):
        print(f"replay {d.name}: no object was double-clicked", flush=True)
        return {}, str(d / "take.mp4"), n
    est = dict(info.get("est") or {"lying": info["lying"], "xy": info["xy_m"],
                                   "yaw": math.radians(info["yaw_deg"]), "iou": info["iou"],
                                   "iou_other": 0.0, "ok": info["ok"]})  # fmt: skip
    est["xy"] = None if est.get("xy") is None else np.asarray(est["xy"], float)
    img = cv2.imread(str(d / "click_frame.jpg"))
    ov = cv2.imread(str(d / "click_overlay.jpg"))
    m = cv2.imread(str(d / "click_mask.png"), cv2.IMREAD_GRAYSCALE)
    est["mask"] = None if m is None else m > 127
    sel = {"est": est, "overlay": ov, "image": img, "sam_mask": est["mask"],
           "mask": None if ov is None or img is None else (ov != img).any(2),
           "frame": int(info.get("frame") or 0), "pt": info.get("point_view_px"),
           "t": info.get("clicked_at"), "msg": "object: saved segmentation"}  # fmt: skip
    print(f"replay {d.name}: {n} frames, saved double-click at frame {sel['frame']}", flush=True)
    return sel, str(d / "take.mp4"), n


def save_take_basics(d: Path, take: dict, error: str | None = None) -> None:
    """take.mp4 (the camera, 800 x 450), take.npz (3D hand, landmarks, gate) and the cup
    double-click (click.json: point in the 800 x 450 view, frame, fitted pose; click_frame.jpg,
    click_overlay.jpg, click_mask.png = the mask the fit used) -- also for failed takes."""
    d.mkdir(parents=True, exist_ok=True)
    arrs = {"lm": np.array(take["lm"])}
    for k in ("joints", "close", "score", "p_hold"):
        if k in take and len(take[k]) == len(take["lm"]):
            arrs[k] = np.array(take[k], dtype=float if k != "close" else bool)
    np.savez_compressed(d / "take.npz", **arrs)
    save_video([cv2.imdecode(j, cv2.IMREAD_COLOR) for j in take["jpg"]], d / "take.mp4", 30)
    cc = take.get("click_cup")
    info: dict = {"selected": cc is not None}
    if cc is not None:
        est = cc["est"]
        xy = None if est.get("xy") is None else np.round(est["xy"], 4).tolist()
        info.update(point_view_px=cc.get("pt"), view_size=[PW, PH], frame=cc.get("frame"),
                    clicked_at=cc.get("t"), ok=bool(est.get("ok")), lying=bool(est.get("lying")),
                    yaw_deg=round(math.degrees(est["yaw"]), 1),
                    iou=round(float(est.get("iou", 0)), 3), xy_m=xy,
                    est={k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in est.items() if k != "mask"})  # fmt: skip
        if cc.get("image") is not None:
            cv2.imwrite(str(d / "click_frame.jpg"), cc["image"])
        if cc.get("overlay") is not None:
            cv2.imwrite(str(d / "click_overlay.jpg"), cc["overlay"])
        if cc.get("sam_mask") is not None:
            cv2.imwrite(str(d / "click_mask.png"), np.asarray(cc["sam_mask"], np.uint8) * 255)
    if error is not None:
        info["error"] = error
    (d / "click.json").write_text(json.dumps(info, indent=1))
    if take.get("calib") is not None:  # --replay uses the rig as it was
        (d / "calib.json").write_text(json.dumps(take["calib"], indent=1))


def moved_object_px(first, last, old_mask, hand_lm, near_px, thr: int = 40):
    """Where the moved cup is in `last` (800 x 450): the biggest region that changed since
    `first`, outside the cup's old spot (old_mask, its double-click fit) and the hand (its
    landmarks, grown), within 30 % of the width of near_px -> its centre (px) or None."""
    g0 = cv2.GaussianBlur(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    g1 = cv2.GaussianBlur(cv2.cvtColor(last, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    m = (cv2.absdiff(g0, g1) > thr).astype(np.uint8)
    k = np.ones((15, 15), np.uint8)
    if old_mask is not None:
        m[cv2.dilate(np.asarray(old_mask, np.uint8), k) > 0] = 0
    if hand_lm is not None and np.isfinite(hand_lm).all():
        hull = cv2.convexHull(np.asarray(hand_lm)[:, :2].astype(np.int32))
        hm = np.zeros_like(m)
        cv2.fillConvexPoly(hm, hull, 1)
        m[cv2.dilate(hm, np.ones((41, 41), np.uint8)) > 0] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cen = cv2.connectedComponentsWithStats(m)
    best, area = None, 150  # px: smaller = noise
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > area and (
                np.linalg.norm(cen[i] - near_px) < 0.3 * last.shape[1]):  # fmt: skip
            best, area = cen[i], stats[i, cv2.CC_STAT_AREA]
    return None if best is None else np.asarray(best, float)


RELEASE_WIN = 60  # frames looked back from the gate's release for the set-down
RELEASE_RISE = 0.03  # m: the hand rising this far above where it set the cup down = let go
REFINE_MAX = 45  # frames (1.5 s): the most the release may move earlier; only when the hand is
#                  back down at the height it grabbed at (SETDOWN_TOL), so never mid-carry
SETDOWN_TOL = 0.025  # m: the set-down must be back near the height the hand grabbed at


def refine_release(J: np.ndarray, s: int, e: int) -> int:
    """The gate lets go ~0.5-1 s late (it waits for a low score over several frames): the
    hand has already set the cup down and lifted away. Offline (after E) look back over the
    last RELEASE_WIN frames of the hold: the last frame the fingertips are still near their
    lowest point = the set-down; if the hand then rises > RELEASE_RISE, release right after."""
    a = max(s + 10, e - RELEASE_WIN)
    if e - a < 5:
        return e
    z = np.nanmean(J[a:e][:, [4, 8, 12], 2], axis=1)
    if not np.isfinite(z).any():
        return e
    zmin = np.nanmin(z)
    z0 = np.nanmedian(J[s : s + 10][:, [4, 8, 12], 2])  # fingertip height at the grab
    if not np.isfinite(z0) or zmin > z0 + SETDOWN_TOL:  # not back down on the table: in the air
        return e
    low = np.flatnonzero(z <= zmin + 0.02)
    k = int(low[-1])
    after = z[k:]
    if np.isfinite(after).any() and np.nanmax(after) - z[k] > RELEASE_RISE:
        return max(a + k + 1, e - REFINE_MAX)
    return e


def make_per(J: np.ndarray, close, pad: int) -> dict:
    """World joints [n, 21, 3] + the gate's close -> smoothed perception for plan_episode, the
    last frame repeated pad times."""
    ok = np.isfinite(J).all((1, 2))
    per = {"ok": ok, "jw": S.smooth(np.where(ok[:, None, None], J, 0.0), ok, 5),
           "close": S.fill_gaps(np.array(close, bool), HOLD_GAP)}  # fmt: skip
    return {k: np.concatenate([v, np.repeat(v[-1:], pad, 0)]) for k, v in per.items()}


class Processor(threading.Thread):
    """E pressed: grasp episodes of the take -> planned, searched, rendered (in the pool)."""

    def __init__(self, take: dict, pool, args, cam_xml, cam_xy, cal: dict, seg_box: dict):
        super().__init__(daemon=True)
        self.take, self.pool, self.args = take, pool, args
        self.cam_xml, self.cam_xy = cam_xml, cam_xy
        self.cal, self.seg_box = cal, seg_box
        self.status = ["processing ..."]
        self.result, self.error = None, None
        self.preview = None  # the cup pose overlay, shown while processing
        self.debug_dir = Path("results/robot_sim/panda_demo") / ("_end_" + time.strftime(
            "%Y%m%d-%H%M%S"))  # end-pose fit images (moved into the take folder after)  # fmt: skip

    def cup_pose(self, s: int, e: int):
        """The cylinder's pose from the user's double-click (SAM 2.1 at the clicked point +
        silhouette fit, touch.cup_pose; Andy 2026-11-14: no automatic frame search) -> (cup
        dict for plan_episode or None = standing at the hand, info, overlay image)."""
        a, tk, cal = self.args, self.take, self.cal
        info = {"from": "hand"}
        J = self.J[s:e]
        okj = np.flatnonzero(np.isfinite(J[:, [4, 8]]).all((1, 2)))
        if not len(okj):
            return None, {**info, "why": "hand not tracked at the grasp"}, None
        if a.no_cup_pose:
            return None, {**info, "why": "--no-cup-pose"}, None
        K, c2w = np.asarray(cal["K"]), np.asarray(cal["cam_to_world"])
        lms = np.array(tk["lm"])
        cc = tk.get("click_cup")
        if cc is None:
            return None, {**info, "why": "object not selected (double-click it)"}, None
        est, overlay, i, ov = cc["est"], cc["overlay"], 0, 0.0
        info = {"from": "click" if est["ok"] else "hand", "frame": int(i),
                "hand_cover": round(ov, 2), "iou": round(est["iou"], 3),
                "iou_other": round(est["iou_other"], 3), "lying": est["lying"],
                "yaw_deg": round(math.degrees(est["yaw"]), 1),
                "xy": [] if est["xy"] is None else np.round(est["xy"], 4).tolist()}  # fmt: skip
        if info["from"] == "hand":
            info["why"] = "fit not trusted"
            return None, info, overlay
        # the hand's depth from the measured cylinder: the thumb-index pixel's ray meets the
        # cylinder's axis where the hand grips (CP.depth_scale), over the first grab frames
        Rc, c0 = CP.cup_pose(est["xy"], est["lying"], est["yaw"], a.obj_h, a.obj_d)
        n_g = min(e - s, 10)
        ds = CP.depth_scale(lms[s : s + n_g], J[:n_g], K, c2w, c0, Rc[:, 2], a.obj_h)
        lo_s, hi_s = a.depth_fix
        if ds is not None and lo_s <= ds["scale"] <= hi_s:
            info["depth_scale"] = round(ds["scale"], 3)
            grip = c0 + ds["t"] * Rc[:, 2]
            info["grip_height_cm"] = round(100 * float(grip[2]), 1)
        else:
            info["depth_scale"] = 1.0
            info["why_no_depth"] = "no grip frames" if ds is None else f"scale {ds['scale']:.2f}"
        return {"xy": est["xy"], "lying": est["lying"], "yaw": est["yaw"]}, info, overlay

    def run(self):
        try:
            self.result = self._run()
        except Exception as ex:  # shown in the window, the demo keeps running
            import traceback

            traceback.print_exc()
            self.error = f"{type(ex).__name__}: {ex}"

    def end_pose(self, s: int, e: int):
        """How the person left the cup: SAM 2.1 + cylinder fit on the frame after the release
        where the hand is farthest from the cup (prompt: between the thumb and index tips at
        the release) -> ({"lying", "yaw"} or None = decide from the hand, info)."""
        a, tk, cal = self.args, self.take, self.cal
        seg = self.seg_box.get("seg")
        lms, n = np.array(tk["lm"]), len(tk["jpg"])
        if seg is None or a.no_cup_pose:
            return None, {"from": "hand", "why": "no SAM"}
        rel = lms[min(e, n) - 1, [4, 8], :2]
        if not np.isfinite(rel).all():
            return None, {"from": "hand", "why": "hand not tracked at the release"}
        px = rel.mean(0)  # the cup sits between thumb and index at the release (full-res px)
        best, far = None, 0.0
        for i in range(min(e + 5, n - 1), n):
            L = lms[i, :, :2]
            dist = 1e9 if not np.isfinite(L).all() else float(np.linalg.norm(L - px, axis=1).min())
            if dist >= far:
                best, far = i, dist
        W = cal["image_size"][0]
        if best is None or far < 0.08 * W:  # the hand never left the cup in the take
            return None, {"from": "hand", "why": "hand still at the cup until the take ended"}
        small = cv2.imdecode(tk["jpg"][best], cv2.IMREAD_COLOR)
        K, c2w = np.asarray(cal["K"]), np.asarray(cal["cam_to_world"])
        Ks = CP.scale_K(K, cal["image_size"], small.shape)
        sc = small.shape[1] / W
        old = tk["click_cup"].get("sam_mask") if tk.get("click_cup") else None
        found = moved_object_px(cv2.imdecode(tk["jpg"][0], cv2.IMREAD_COLOR), small, old,
                                lms[best] * sc, px * sc)  # fmt: skip
        prompt = found if found is not None else px * sc
        hint = CP.ray_to_plane(Ks, c2w[:3], prompt, a.obj_d / 2)[:2]
        est = CP.estimate(seg, small, K, cal["image_size"], c2w, a.obj_h, a.obj_d, hint,
                          allow_lying=a.allow_lying)  # fmt: skip
        info = {"from": "camera" if est["ok"] else "hand", "frame": int(best),
                "found_by": "frame difference" if found is not None else "release fingertips",
                "lying": bool(est["lying"]), "yaw_deg": round(math.degrees(est["yaw"]), 1),
                "iou": round(float(est["iou"]), 3),
                "iou_other": round(float(est["iou_other"]), 3)}  # fmt: skip
        try:
            d = Path(self.debug_dir)
            d.mkdir(parents=True, exist_ok=True)
            ov = CP.draw(small, K, cal["image_size"], c2w, est, a.obj_h, a.obj_d)
            cv2.imwrite(str(d / f"end_pose_{s}.jpg"), ov)
        except Exception:
            pass
        if not est["ok"]:
            return None, {**info, "why": "fit not trusted"}
        info["xy"] = np.round(est["xy"], 4).tolist()
        return {"lying": est["lying"], "yaw": est["yaw"], "xy": np.asarray(est["xy"])}, info

    def _run(self):
        a = self.args
        tk = self.take
        # depth spikes out of the hand track (handpath.clean_depth); take.npz keeps the raw one
        J = clean_depth(
            np.array(tk["joints"]),
            self.cal["cam_to_world"],
            lm=np.array(tk["lm"]),
            K=np.array(self.cal["K"]),
        )  # MediaPipe depth
        J = fuse_orientation(J, np.array(tk["lm"]), self.cal["cam_to_world"])  # + its angles
        self.J = J
        # E right after letting go: the hand's last frame is held so the robot has time to set
        # the cup down, open and retreat (V.POST frames after the release)
        pad = V.POST + 15
        close, eps, how = find_holds(tk)
        self.hold_source = how
        if how != "gate":
            print(f"grasp found by fallback: {how} {eps}", flush=True)
        if not eps:
            raise RuntimeError("no grasp found in this take, not even with looser thresholds")
        cc = tk.get("click_cup")
        if cc is not None and cc["est"].get("ok") and cc["est"].get("xy") is not None:
            box = cup_box_px(cc["est"], a.obj_h, a.obj_d, self.cal["K"],
                             self.cal["cam_to_world"], CUP_GROW)  # fmt: skip
            near = [(s, e) for s, e in eps if at_cup(tk["lm"], s, e, box)]
            if not near:  # the hand never looked to be at the cup: the first real grab
                near = [first_grab(eps)]
                print(f"no hold clearly at the cup: taking the first grab {near[0]}", flush=True)
            elif len(near) < len(eps):
                print(f"holds away from the cup ignored: {sorted(set(eps) - set(near))}",
                      flush=True)  # fmt: skip
            keep = np.zeros(len(close), bool)
            for s, e in near:
                keep[s:e] = True
            close, eps = close & keep, near
        refined = []
        for s, e in eps:
            e2 = refine_release(J, s, e)
            if e2 < e:
                close[e2:e] = False
                print(f"release moved earlier: frame {e} -> {e2} (the hand set the cup down "
                      f"and lifted away)", flush=True)  # fmt: skip
            refined.append((s, e2))
        eps = refined
        self.release_fix = [(int(s), int(e)) for s, e in eps]
        tk["close"] = list(close)
        per = make_per(J, tk["close"], pad)
        out = []
        for k, (s, e) in enumerate(eps[::-1]):  # newest first: the last real pick(s)
            if len(out) >= a.max_grasps:
                break
            tag = f"grasp {len(eps) - k}/{len(eps)}"
            t0 = time.time()
            self.status = [tag, "cup pose: SAM 2.1 + cylinder fit ..."]
            cup, cinfo, overlay = self.cup_pose(s, e)
            if overlay is not None:
                self.preview = overlay
            per_k = per
            if cinfo.get("depth_scale", 1.0) != 1.0:  # the whole take at the corrected depth
                Jk = CP.rescale_about_camera(J, self.cal["cam_to_world"], cinfo["depth_scale"])
                per_k = make_per(Jk, tk["close"], pad)
            kind = f"lying {cinfo['yaw_deg']:.0f} deg" if cinfo.get("lying") else "standing"
            dtxt = f", grip {cinfo['grip_height_cm']:.1f} cm" if "grip_height_cm" in cinfo else ""
            cam_ok = cinfo["from"] in ("camera", "click")
            cline = (f"cup: {kind} (your click, IoU {cinfo['iou']:.2f}{dtxt})" if cam_ok
                     else f"cup: standing at the hand ({cinfo.get('why', '')})")  # fmt: skip
            print(f"{tag}: {cline} {cinfo}", flush=True)
            self.status = [tag, cline, "planning: Panda base + approach ..."]
            self.status = [tag, cline, "where you left the cup: standing or lying ..."]
            end_obs, einfo = self.end_pose(s, e)
            cinfo["end_seen"] = einfo
            print(f"{tag}: cup after the release: {einfo}", flush=True)
            self.status = [tag, cline, "planning: Panda base + approach ..."]
            pl, why = V.plan_episode(per_k, s, e, a.obj_h, a.obj_d, self.cam_xml, self.cam_xy,
                                     self.pool, cup=cup, end_obs=end_obs,
                                     upright=not a.allow_lying,
                                     place_xy=None if a.free_place else (0.0, 0.0),
                                     pour_angle=a.pour_angle, pour_side=a.pour_side)  # fmt: skip
            if pl is not None:
                pl["pause"] = TELEOP_PAUSE
            lift = float((pl["gt_obj"][:, 2] - pl["gt_obj"][0, 2]).max()) if pl else 0.0
            if pl is None or lift < MIN_LIFT:
                print(f"{tag}: skipped ({why or f'hand lifted only {lift * 100:.1f} cm'})")
                continue

            def prog(it, best, tag=tag, t0=t0, cline=cline):
                self.status = [tag, cline, f"small RL: round {it}, best lift accuracy {best:.2f}",
                               f"{time.time() - t0:.0f} s"]  # fmt: skip

            g, best, hist = V.search(pl, self.pool, a.iters, a.pop, good=a.good,
                                     budget_s=a.budget, progress=prog)  # fmt: skip
            self.status = [tag, cline, f"rendering (lift accuracy {best:.2f}) ..."]
            res = self.pool.apply(render_rollout, ((pl, g),))
            res.update({"pl": pl, "g": g, "hist": hist, "seconds": time.time() - t0,
                        "cup_pose": cinfo, "cup_overlay": overlay})  # fmt: skip
            out.append(res)
        if not out:
            raise RuntimeError("no pick in this take (lift the cup >= 3 cm, put it down, let go)")
        return out[::-1]


SIM_DT = 17 * 0.002  # s per sim frame (panda.SUB physics steps of 2 ms)
ROBOT_LAG = 2  # sim frames the arm's TCP trails its target (measured 2026-11-14): the hand
#                video is shown that much later so the two move together
TELEOP_PAUSE = 4  # steps the arm waits at the grasp / release (panda.PAUSE = 8 for generated
#                   demos); the sim fingers settle in ~4 (the hand video holds still meanwhile)


def run_telemetry(r: dict) -> dict:
    """Per render frame of a robot run: object speed (cm/s, smoothed), signed tilt (deg, 0 =
    upright) and the events (name, frame): GRASP, LIFT-OFF, SET-DOWN, RELEASE."""
    n = len(r["jpgs"])
    pos = np.asarray(r.get("obj_pos", np.zeros((n, 3))), float)[:n]
    v = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1) / SIM_DT * 100]
    speed = np.convolve(np.pad(v, 3, mode="edge"), np.ones(7) / 7, mode="valid")[:n]
    tilt = np.zeros(n)
    q = r.get("obj_quat")
    if q is not None:
        q = np.asarray(q, float)[:n]
        w, x, y, z = q.T  # MuJoCo (w, x, y, z): the body's z axis in the world
        ax = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)
        side = np.hypot(ax[:, 0], ax[:, 1]) * np.where(ax[:, 0] < 0, -1, 1)
        tilt = np.degrees(np.arctan2(side, ax[:, 2]))
    lift = (pos[:, 2] - pos[0, 2]) * 100
    close = np.asarray(r["close"], bool)[np.asarray(r["human"], int)[:n]]
    ev = []
    gi = np.flatnonzero(close)
    if len(gi):
        ev.append(("GRASP", int(gi[0])))
        up = np.flatnonzero((lift > 1.0) & (np.arange(n) >= gi[0]))
        if len(up):
            ev.append(("LIFT-OFF", int(up[0])))
            pk = int(np.argmax(lift))
            dn = np.flatnonzero((lift < 1.0) & (np.arange(n) > pk))
            if len(dn):
                ev.append(("SET-DOWN", int(dn[0])))
        rel = np.flatnonzero(~close & (np.arange(n) > gi[0]))
        if len(rel):
            ev.append(("RELEASE", int(rel[0])))
    ev.sort(key=lambda e: e[1])
    return {"speed": speed, "tilt": tilt, "events": ev, "lift": lift}


DG_EVERY = 2  # the generated tiles are rendered every 2nd step (touch.demogen.run_one)
HAND_CLOCK_WIN = 11  # hand frames the robot's fast-forward is spread over (hand_clock)


def hand_clock(human: np.ndarray, n: int):
    """Playback on the hand's clock (Andy 2026-11-14: the hand video must not slow down): one
    output frame per hand frame, showing the robot frame that has reached it (the last robot
    frame of that hand frame, + ROBOT_LAG for the arm's tracking lag); where the robot needs
    longer than the hand (joint speed limits, gripper closing) it is fast-forwarded. The robot's
    frames past the hand's last one (its set-down / retreat) follow, the hand's last frame held.
    -> (robot frame, hand frame, playback time s) per output frame."""
    human = np.asarray(human, int)
    h0, h1 = int(human.min()), int(human.max())
    last = np.searchsorted(human, np.arange(h0, h1 + 1), side="right") - 1  # monotone mapping
    # spread the fast-forward over neighbouring frames (a 14-frame catch-up at the release
    # becomes a quick move, not a jump): moving average, kept increasing, end points kept
    w = HAND_CLOCK_WIN
    if len(last) > w:
        pad = np.r_[np.full(w // 2, last[0]), last, np.full(w // 2, last[-1])]
        sm = np.convolve(pad.astype(float), np.ones(w) / w, mode="valid")
        sm[-1] = last[-1]
        last = np.maximum.accumulate(np.round(sm)).astype(int)
    out, k = [], 0
    for j, h in enumerate(range(h0, h1 + 1)):
        i = int(min(n - 1, max(0, last[j]) + ROBOT_LAG))
        out.append((i, h, k / 30.0))
        k += 1
    i_end = out[-1][0] if out else 0
    for i in range(i_end + 1, n):
        out.append((i, h1, k / 30.0))
        k += 1
    return out


def compose(take: dict, results: list, W: int) -> list[tuple]:
    """Playback frames per grasp: (recorded hand with its skeleton, Panda render, meta: grasp,
    hand holding, gripper closed, cup lift / hand lift cm, outcome (last 25 frames))."""
    frames = []
    sc = PW / W
    for k, r in enumerate(results):
        pl, lo = r["pl"], r["pl"]["lo"]
        gt = pl["gt_obj"][:, 2] - pl["gt_obj"][0, 2]
        tele = run_telemetry(r)
        for i, h, t_play in hand_clock(r["human"], len(r["jpgs"])):
            jpg = r["jpgs"][i]
            j = min(lo + h, len(take["jpg"]) - 1)  # past the end of the take: its last frame
            left = cv2.imdecode(take["jpg"][j], cv2.IMREAD_COLOR)
            g = bool(take["close"][j])
            lm = take["lm"][j]
            if np.isfinite(lm).all():
                pts = [tuple(int(v * sc) for v in p[:2]) for p in lm]
                col = U.ACCENT if g else (200, 200, 200)
                for a, b in PR.HAND_EDGES:
                    cv2.line(left, pts[a], pts[b], col, 2, cv2.LINE_AA)
            right = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            meta = {"grasp": k + 1, "hold": g, "closed": bool(r["close"][h]),
                    "lift": (r["obj_z"][i] - r["obj_z"][0]) * 100, "hand_lift": gt[h] * 100,
                    "ok": bool(r["src_ok"]), "why": r["src_why"], "outcome": False,
                    "i": i, "n": len(r["jpgs"]), "t": t_play, "speed": tele["speed"][i],
                    "tilt": tele["tilt"][i], "events": tele["events"]}  # fmt: skip
            frames.append((left, right, meta))
        if frames:
            left, right, meta = frames[-1]
            frames += [(left, right, {**meta, "outcome": True})] * 25
    return frames


def video_frame(left, right, meta, vui) -> np.ndarray:
    """A saved playback frame (teleop.mp4): operator | robot with small labels (vui: a
    1920-wide UI, so sizes match the 800 x 450 panels)."""
    f = np.hstack([left, right])
    vui.shade(f, 0, 60, 0.45, 1.0)
    vui.caps(f, "Operator  ·  " + ("hold" if meta["hold"] else "open"), 16, 14, U.TEXT, 13)
    vui.caps(f, f"Robot  ·  gripper {'closed' if meta['closed'] else 'open'}  ·  lift "
             f"{max(meta['lift'], 0.0):.1f} cm", PW + 16, 14, U.TEXT, 13)  # fmt: skip
    if meta["outcome"]:
        vui.banner(f, U.Rect(PW, 0, PW, PH), "Success" if meta["ok"] else "Failed",
                   U.GOOD if meta["ok"] else U.BAD)  # fmt: skip
    return f


TAKE_CLOCK = {"t0": None, "stop": None}  # the one T+ clock of a take (Andy 2026-11-15): starts
# at the first frame with a hand after the object is selected, runs through processing, motion
# plan, playback and scaling without resetting, stops when scaling is done; reset by a new take


def clock_start() -> None:
    if TAKE_CLOCK["t0"] is None:
        TAKE_CLOCK["t0"] = time.time()


def clock_stop() -> None:
    if TAKE_CLOCK["t0"] is not None and TAKE_CLOCK["stop"] is None:
        TAKE_CLOCK["stop"] = time.time()


def clock_reset() -> None:
    TAKE_CLOCK.update(t0=None, stop=None)


def take_secs() -> float:
    """Seconds on the take clock (0 until it starts, frozen once stopped)."""
    t0 = TAKE_CLOCK["t0"]
    if t0 is None:
        return 0.0
    return (TAKE_CLOCK["stop"] or time.time()) - t0


def hud_y(ui) -> int:
    """Top of the bottom instrument band's arc."""
    return ui.H - ui.px(185)


SLIDE = 0.45  # share of the intro spent sliding the side panels out (the rest: rings sweep)


def instruments(ui, lay, meta, sweep: float, y0: int) -> None:
    """The side panels of the playback (drawn on lay = the rows from y0 down, black elsewhere):
    left wing speed + height rings, right wing operator / gripper state + attitude. sweep:
    0..1 how far the ring arcs have swept up."""
    H = lay.shape[0]
    r = ui.px(78)
    cy = ui.H - ui.px(112) - y0
    ww, sl = ui.px(470), ui.px(70)  # wing width, slant of its inner edge
    fill, edge = (24, 24, 26), (58, 58, 62)
    top = cy - r - ui.px(26)
    lw = np.array([(0, top), (ww, top), (ww + sl, H), (0, H)], np.int32)
    rw = np.array([(ui.W, top), (ui.W - ww, top), (ui.W - ww - sl, H), (ui.W, H)], np.int32)
    for p in (lw, rw):
        cv2.fillPoly(lay, [p], fill, cv2.LINE_AA)
        cv2.polylines(lay, [p[1:3]], False, edge, max(1, ui.px(1.5)), cv2.LINE_AA)
        cv2.line(lay, tuple(p[0]), tuple(p[1]), edge, max(1, ui.px(1.5)), cv2.LINE_AA)
    sp, lift = float(meta.get("speed", 0.0)), max(float(meta["lift"]), 0.0)
    ui.ring(lay, ui.m + r + ui.px(10), cy, r, sweep * sp / 60, "Speed", f"{sp:.0f}", "cm/s")
    ui.ring(lay, ui.m + 3 * r + ui.px(44), cy, r, sweep * lift / 20, "Height", f"{lift:.1f}",
            "cm")  # fmt: skip
    ui.attitude(lay, ui.W - ui.m - r - ui.px(10), cy, r, sweep * float(meta.get("tilt", 0.0)))
    sx = ui.W - ui.m - 2 * r - ui.px(170)
    ui.caps(lay, "Operator", sx, cy - ui.px(62))
    ui.text(lay, "Holding" if meta["hold"] else "Open", sx, cy - ui.px(42), 26,
            U.TEXT if meta["hold"] else U.DIM, "light")  # fmt: skip
    ui.caps(lay, "Gripper", sx, cy + ui.px(8))
    ui.text(lay, "Closed" if meta["closed"] else "Open", sx, cy + ui.px(28), 26,
            U.TEXT if meta["closed"] else U.DIM, "light")  # fmt: skip


def launch_button(ui, img, label: str, key: str = "S") -> None:
    """The big call to action at the top centre: [key] + label, its border pulsing."""
    sz = 22
    tw = ui.measure(label, sz, "semibold", 0.12) + ui.measure(key, sz, "semibold") + ui.px(28)
    bw, bh = tw + ui.px(64), ui.px(52)
    x0, y0 = (ui.W - bw) // 2, ui.px(14)
    roi = img[y0 : y0 + bh, x0 : x0 + bw]
    roi[:] = (roi * 0.25 + np.array(U.GOOD) * 0.12).astype(np.uint8)
    a = 0.55 + 0.45 * math.sin(time.time() * 4.0)
    col = tuple(int(c * a + 40 * (1 - a)) for c in U.GOOD)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), col, max(2, ui.px(3)), cv2.LINE_AA)
    kx = x0 + ui.px(32)
    kw = ui.measure(key, sz, "semibold")
    cv2.rectangle(img, (kx - ui.px(8), y0 + ui.px(10)), (kx + kw + ui.px(8), y0 + bh - ui.px(10)),
                  U.GOOD, -1)  # fmt: skip
    ui.text(img, key, kx, y0 + ui.px(13), sz, (20, 20, 20), "semibold")
    ui.text(img, label, kx + kw + ui.px(28), y0 + ui.px(13), sz, U.TEXT, "semibold", track=0.12)


def playback_screen(ui, left, right, meta, note: str, note_col, t: float = 0.0,
                    intro: float = 1.0, keys=None, launch: str = "") -> np.ndarray:  # fmt: skip
    """Operator | robot side by side; under them the instruments: speed + height rings, the
    clock on the event arc (grasp, lift-off, set-down, release), the object's attitude.
    intro < 1: the opening (videos fade in, the side instruments slide out from behind the
    clock, then their rings sweep up), like a launch webcast at lift-off."""
    e = ease(intro / SLIDE)
    img = ui._blank.copy()
    gap = ui.px(16)
    top = ui.px(96)
    hy = hud_y(ui)
    pw = (ui.W - 2 * ui.m - gap) // 2
    ph = int(pw * PH / PW)
    if top + ph > hy - ui.px(90):
        ph = hy - ui.px(90) - top
        pw = int(ph * PW / PH)
    x0 = (ui.W - 2 * pw - gap) // 2
    top += max(0, (hy - ui.px(110) - top - ph) // 2)  # centred above the instruments
    lr = U.Rect(x0, top, pw, ph)
    rr = U.Rect(x0 + pw + gap, top, pw, ph)
    ui.put(img, cv2.resize(left, (pw, ph), interpolation=cv2.INTER_LINEAR), lr)
    ui.put(img, cv2.resize(right, (pw, ph), interpolation=cv2.INTER_LINEAR), rr)
    if e < 1:
        for q in (lr, rr):
            img[q.y : q.y2, q.x : q.x2] = cv2.convertScaleAbs(img[q.y : q.y2, q.x : q.x2], alpha=e)
    ui.caps(img, f"Operator  ·  grasp {meta['grasp']}", lr.x, top - ui.px(28))
    ui.caps(img, "Robot  ·  simulation", rr.x, top - ui.px(28))
    if note:
        ui.caps(img, note, rr.x2, top - ui.px(28), note_col, anchor="r")
    if meta["outcome"]:
        ui.banner(img, rr, "Pick and place complete" if meta["ok"] else f"Failed: {meta['why']}",
                  U.GOOD if meta["ok"] else U.BAD)  # fmt: skip
    ui.brand(img)
    ui.keys(img, keys or [("R", "New take"), ("Q", "Quit")])
    if launch:
        launch_button(ui, img, launch)
    n = meta.get("n", 1)
    tt = meta.get("t", t)
    span = max(3.0, 0.45 * n * SIM_DT)
    ui.arc(img, ui.W / 2, hy, ui.W * 0.24, [(nm, fi * SIM_DT) for nm, fi in meta.get("events", [])],
           tt, span)  # fmt: skip
    # the side instruments: on a layer, slid out from behind the clock during the intro
    y0 = ui.H - ui.px(250)
    lay = np.zeros((ui.H - y0, ui.W, 3), np.uint8)
    instruments(ui, lay, meta, ease((intro - SLIDE) / (1 - SLIDE)) if intro < 1 else 1.0, y0)
    k = ease(intro / SLIDE)
    band = img[y0:]
    if k >= 1:
        np.maximum(band, lay, out=band)
    elif k > 0:
        half = ui.W // 2
        d = int((1 - k) * ui.W * 0.36)  # how far each side still is from its place
        a8 = cv2.convertScaleAbs(lay, alpha=k)
        if half - d > 0:  # left half moves left into place, right half moves right
            np.maximum(band[:, : half - d], a8[:, d:half], out=band[:, : half - d])
            np.maximum(band[:, half + d :], a8[:, half : ui.W - d], out=band[:, half + d :])
    ui.clock_block(img, ui.W / 2, hy + ui.px(26), U.clock(take_secs()), "Robot playback")
    return img


# ------------------------------------------------------------------ scaling (10 demos)
GW, GH = DG.TILE


def grid_cols(n: int, r) -> int:
    """Columns that make the n tiles largest inside rect r."""
    best, bc = 0.0, 5
    for c in range(1, n + 1):
        rows = -(-n // c)
        k = min(r.w / (c * GW), r.h / (rows * GH))
        if k > best:
            best, bc = k, c
    return bc


def grid(tiles, done, kinds, ui, cols: int = 5, tw: int = GW,
         pending: str = "running") -> np.ndarray:  # fmt: skip
    """Mosaic of the scaled runs, tiles tw px wide (the saved scaled_<k>.mp4 uses GW with a
    1920-wide UI): number, start pose and outcome on each."""
    th = int(tw * GH / GW)
    g = max(4, tw // 40)
    rows = -(-len(tiles) // cols)
    img = np.zeros((rows * th + (rows - 1) * g, cols * tw + (cols - 1) * g, 3), np.uint8)
    k_ = tw / 320  # label sizes relative to a 320 px tile
    for k, t in enumerate(tiles):
        x, y = (k % cols) * (tw + g), (k // cols) * (th + g)
        if t is None:
            tile = np.full((th, tw, 3), U.PANEL[0], np.uint8)
        else:
            tile = cv2.resize(t, (tw, th), interpolation=cv2.INTER_LINEAR)
        ui.shade(tile, 0, int(36 * k_), 0.4, 1.0)
        r = done[k]
        pad = int(10 * k_)
        w = ui.text(tile, f"{k + 1:02d}", pad, int(6 * k_), 15 * k_ / ui.s, U.TEXT, "medium")
        ui.caps(tile, kinds[k] if kinds else "", pad + w + int(8 * k_), int(9 * k_), U.DIM,
                11 * k_ / ui.s)  # fmt: skip
        if r is None:
            ui.caps(tile, pending, tw - pad, int(9 * k_), U.DIM, 11 * k_ / ui.s, "r")
        else:
            col = U.GOOD if r["success"] else U.BAD
            ui.caps(tile, "success" if r["success"] else "failed", tw - pad, int(9 * k_), col,
                    11 * k_ / ui.s, "r")  # fmt: skip
            cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), col, max(1, int(1.5 * k_)))
            ui.shade(tile, th - int(30 * k_), th, 1.0, 0.4)
            if r["success"]:
                e_ = r.get("metrics", {}).get("place_err_cm", 0)
                s_ = f"On the pad · {e_:.1f} cm from its centre"
            else:
                s_ = r["reason"][:40]
            ui.text(tile, s_, pad, th - int(26 * k_), 12 * k_ / ui.s,
                    U.DIM if r["success"] else col)  # fmt: skip
        img[y : y + th, x : x + tw] = tile
    return img


GRID_AT: dict = {}  # the last grid drawn by scale_screen: canvas rect + tile layout (clicks)


def tile_at(x: float, y: float) -> int | None:
    """The grid tile index under canvas point (x, y), None if none."""
    g = GRID_AT
    if not g:
        return None
    mr = g["mr"]
    if not (mr.x <= x < mr.x2 and mr.y <= y < mr.y2):
        return None
    sx, sy = g["mw"] / mr.w, g["mh"] / mr.h  # canvas px -> mosaic px
    mx, my = (x - mr.x) * sx, (y - mr.y) * sy
    c, rw = int(mx // (g["tw"] + g["gap"])), int(my // (g["th"] + g["gap"]))
    k = rw * g["cols"] + c
    return k if 0 <= c < g["cols"] and 0 <= k < g["n"] else None


def scale_screen(ui, tiles, done, kinds, n: int, secs: float, head: str, final: str = "",
                 final_col=U.GOOD, keys=None) -> np.ndarray:  # fmt: skip
    """The generated runs as a grid; instruments under it."""
    img = ui._blank.copy()
    hy = hud_y(ui)
    area = U.Rect(ui.m, ui.px(84), ui.W - 2 * ui.m, hy - ui.px(110) - ui.px(84))
    cols = grid_cols(n, area)
    rows = -(-n // cols)
    tw = int(min(area.w / cols, area.h / rows * GW / GH)) - ui.px(8)
    mosaic = grid(tiles, done, kinds, ui, cols, tw)
    mh, mw = mosaic.shape[:2]
    mosaic, mr = ui.fit(mosaic, area)
    ui.put(img, mosaic, mr)
    GRID_AT.update(mr=mr, mw=mw, mh=mh, cols=cols, tw=tw, th=int(tw * GH / GW),
                   gap=max(4, tw // 40), n=n)  # fmt: skip
    ui.brand(img)
    ui.keys(img, keys or [("Q", "Stop")])
    nd = sum(r is not None for r in done)
    ns = sum(bool(r and r["success"]) for r in done)
    r = ui.px(78)
    cy = ui.H - ui.px(112)
    ui.ring(img, ui.m + r + ui.px(10), cy, r, ns / max(n, 1), "Successful", f"{ns}/{n}", "demos",
            U.GOOD)  # fmt: skip
    ui.ring(img, ui.m + 3 * r + ui.px(44), cy, r, nd / max(n, 1), "Finished", f"{nd}/{n}", "runs")
    ui.phases(img, 5, None, hy)
    sub = "Demonstrations added" if final else "Generating demonstrations"
    ui.clock_block(img, ui.W / 2, hy + ui.px(26), U.clock(secs), sub,
                   U.GOOD if final else U.DIM)  # fmt: skip
    x = ui.W - ui.m - ui.px(420)
    y = cy - ui.px(60)
    for ln in U.wrap(final or head, 46):
        ui.text(img, ln, x, y, 17, final_col if final else U.DIM)
        y += ui.px(26)
    return img


def side_order(scenes: list, cols: int) -> list:
    """Tile order (row-major, cols per row) with the lying-start scenes in the left columns and
    the standing ones in the right (Andy 2026-11-14: each kind on its own side); a kind that
    does not fit its side spills over."""
    lie = [sc for sc in scenes if sc["lying"]]
    up = [sc for sc in scenes if not sc["lying"]]
    rows = -(-len(scenes) // cols)
    k_lie = -(-len(lie) // rows) if lie else 0  # columns for the lying scenes
    out = []
    for i in range(len(scenes)):
        left = i % cols < k_lie
        pool_, other = (lie, up) if left else (up, lie)
        out.append(pool_.pop(0) if pool_ else other.pop(0))
    return out


def scale(pool, queue, src: dict, n: int, seed: int, scene_dir: Path, head: str, ui, vui,
          p_lying: float = 0.0, real_cam: str = "", n_lying: int | None = None):  # fmt: skip
    """real_cam: the webcam camera XML (shown from its perspective, render_view.real_view).
    Generate n demos from src, run them in parallel (tiles streamed live) -> (results,
    mosaic video frames, quit pressed, the last tile images)."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    cam = (
        DG.view_camera(src["base"], (src["cup"] + src["place"]) / 2)
        + real_cam
        + src.get("vhead_xml", "")
    )
    scenes = DG.sample_scenes(src, n, seed, p_lying=p_lying, n_lying=n_lying)
    hy = hud_y(ui)
    cols = grid_cols(n, U.Rect(ui.m, ui.px(84), ui.W - 2 * ui.m, hy - ui.px(110) - ui.px(84)))
    scenes = side_order(scenes, cols)
    kinds = ["lying" if sc["lying"] else "standing" for sc in scenes]
    jobs = [
        pool.apply_async(DG.run_one, ((src, k, sc, cam, str(scene_dir / f"scene_{k}.xml"), queue),))
        for k, sc in enumerate(scenes)
    ]
    tiles, hist, done = [None] * n, [[] for _ in range(n)], [None] * n
    quit_ = False
    t_prog, n_prog = time.time(), 0  # last progress (a finished demo or a new tile)
    cols = grid_cols(n, U.Rect(0, 0, 16, 9))  # the saved video's layout

    def drain():
        while True:
            try:
                k, b = queue.get_nowait()
            except Exception:
                return
            hist[k].append(b)
            tiles[k] = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)

    while any(r is None for r in done):
        drain()
        for k, a in enumerate(jobs):
            if done[k] is None and a.ready():
                try:
                    done[k] = a.get()
                except Exception as ex:  # a crashed rollout is a failed demo, not a crash
                    done[k] = {"k": k, "success": False, "reason": f"error: {ex}"[:40]}
        prog = sum(len(h) for h in hist) + sum(r is not None for r in done)
        if prog != n_prog:
            t_prog, n_prog = time.time(), prog
        elif time.time() - t_prog > STALL_S:  # stuck workers: don't hang the app
            print(f"scaling stalled for {STALL_S:.0f} s: the unfinished demos count as failed",
                  flush=True)  # fmt: skip
            for k in range(n):
                if done[k] is None:
                    done[k] = {"k": k, "success": False, "reason": "timed out"}
        show(scale_screen(ui, tiles, done, kinds, n, take_secs(), head))
        if (cv2.waitKey(15) & 0xFF) in (ord("q"), 27):
            quit_ = True
            break
    drain()
    # the grid as a video: every tile advances together, finished tiles hold their outcome
    frames, T = [], max([len(h) for h in hist] + [1])
    for t in range(T):
        tl = [cv2.imdecode(np.frombuffer(h[min(t, len(h) - 1)], np.uint8), cv2.IMREAD_COLOR)
              if h else None for h in hist]  # fmt: skip
        dn = [r if r is not None and t >= len(hist[k]) - 1 else None for k, r in enumerate(done)]
        frames.append(grid(tl, dn, kinds, vui, cols))
    last = [cv2.imdecode(np.frombuffer(h[-1], np.uint8), cv2.IMREAD_COLOR) if h else None
            for h in hist]  # fmt: skip
    return done, frames, quit_, (last, kinds, hist)


STEPS = ("Hold detection", "Object pose", "Placement", "Motion plan", "Grasp search", "Render")
STEP_PHASE = (1, 2, 2, 2, 3, 3)  # timeline phase per step


def stage_of(status: list, iters: int) -> tuple[int, str, float | None]:
    """Processor.status lines -> (step index, detail, progress or None)."""
    t = " ".join(status).lower()
    if "rendering" in t:
        return 5, "Preparing the playback", None
    if "small rl" in t:
        line = next(x for x in status if "small RL" in x)
        try:
            it = int(line.split("round")[1].split(",")[0])
        except (IndexError, ValueError):
            it = 0
        best = line.split("accuracy")[-1].strip() if "accuracy" in line else ""
        det = f"Round {it} of {iters}" + (f"  ·  best lift accuracy {best}" if best else "")
        return 4, det, it / max(iters, 1)
    if "planning" in t:
        return 3, "Robot base, approach and a collision-free arm path", None
    if "where you left" in t:
        return 2, "Where and how you set the object down", None
    if "cup pose" in t:
        return 1, "Fitting the object with SAM 2.1", None
    return 1, "", None


GO_FADE, GO_LINE, GO_HOLD = 0.5, 0.8, 1.6  # s: "Go for transfer" fade in, line, shown for
INTRO = 1.2  # s: the playback opening (side panels slide out, then the rings sweep up)


def ease(x: float) -> float:
    """Ease-out cubic on 0..1."""
    x = float(np.clip(x, 0.0, 1.0))
    return 1 - (1 - x) ** 3


def go_screen(ui, view, secs: float, t: float) -> np.ndarray:
    """Everything is computed and rendered: "Go for transfer" (the person's skill goes to the
    robot) fades in, a line draws out from the centre under it; then playback starts. t: s
    since it appeared."""
    img, _ = ui.cover(view, U.Rect(0, 0, ui.W, ui.H))
    img = cv2.convertScaleAbs(img, alpha=0.28 * (1 - 0.5 * ease(t / GO_FADE)))
    ui.brand(img)
    a = round(ease(t / GO_FADE) * 8) / 8  # few steps: the text cache stays small
    col = tuple(int(v * a) for v in U.TEXT)
    dim = tuple(int(v * a) for v in U.DIM)
    cy = int(ui.H * 0.33)
    ui.caps(img, "Demonstration captured", ui.W / 2, cy, dim, 15, "c")
    ui.text(img, "GO FOR TRANSFER", ui.W / 2, cy + ui.px(34), 76, col, "light", "c", 0.12)
    w = ui.px(640) * ease(t / GO_LINE)
    ui.line(img, ui.W / 2 - w / 2, cy + ui.px(150), ui.W / 2 + w / 2, cy + ui.px(150), U.TEXT)
    ui.caps(img, "Human skill  →  robot training data", ui.W / 2, cy + ui.px(176), dim, 14, "c")
    hy = hud_y(ui)
    ui.phases(img, 4, None, hy)
    ui.clock_block(img, ui.W / 2, hy + ui.px(26), U.clock(secs), "Go", U.GOOD)
    return img


def process_screen(ui, view, step: int, detail: str, frac, secs: float, failed: bool = False,
                   keys=None, info: dict | None = None) -> np.ndarray:  # fmt: skip
    """The last camera frame, dimmed, full window; the current step large in the middle."""
    img, _ = ui.cover(view, U.Rect(0, 0, ui.W, ui.H))
    img = cv2.convertScaleAbs(img, alpha=0.28)
    ui.brand(img)
    ui.keys(img, keys or [("Q", "Quit")])
    cy = int(ui.H * 0.30)
    if failed:
        ui.caps(img, f"Stopped at step {step + 1} of {len(STEPS)}  ·  {STEPS[step]}", ui.W / 2, cy,
                U.BAD, 15, "c")  # fmt: skip
        ui.text(img, "Could not process this take", ui.W / 2, cy + ui.px(30), 52, U.TEXT, "light",
                "c")  # fmt: skip
        y = cy + ui.px(116)
        for ln in U.wrap(detail, 70):
            ui.text(img, ln, ui.W / 2, y, 22, U.DIM, anchor="c")
            y += ui.px(32)
    else:
        # the step's title and what runs in it, technically (Andy 2026-11-14: nothing else)
        ui.text(img, STEPS[step], ui.W / 2, cy + ui.px(30), 60, U.TEXT, "light", "c")
        y = cy + ui.px(128)
        for ln in U.wrap(HOW[step], 78):
            ui.text(img, ln, ui.W / 2, y, 22, U.DIM, anchor="c")
            y += ui.px(34)
        if frac is not None:
            ui.bar(img, ui.W / 2 - ui.px(260), y + ui.px(24), ui.px(520), frac)
    hy = hud_y(ui)
    ui.phases(img, STEP_PHASE[step], frac, hy)
    ui.clock_block(img, ui.W / 2, hy + ui.px(26), U.clock(secs),
                   "Failed" if failed else "Processing", U.BAD if failed else U.DIM)  # fmt: skip
    return img


def live_screen(ui, frame, lm, W: int, st: dict, sel: dict, secs: float, hs: dict, infer,
                fusion, faces, K, c2w, lag: int):  # fmt: skip
    """The camera fills the window; instruments over it -> (canvas, the rect of the whole
    camera image in canvas px, for the mouse)."""
    # the object's mask only until a hand first shows up after this selection: then gone at
    # once, no overlay over the take (Andy 2026-11-15); a new selection shows its mask again
    if lm is not None and sel.get("est") is not None:
        sel["hand_seen"] = True
    show_mask = sel.get("mask") is not None and not sel.get("hand_seen")
    if show_mask or (faces is not None and hs["verts"] is not None):
        frame = frame.copy()
    H0 = frame.shape[0]
    if show_mask:
        m = cv2.resize(sel["mask"].astype(np.uint8), (W, H0), interpolation=cv2.INTER_NEAREST)
        ov = cv2.resize(sel["overlay"], (W, H0), interpolation=cv2.INTER_LINEAR)
        frame[m > 0] = ov[m > 0]
    mesh_ok = faces is not None and hs["mesh_ms"] < 15 and lag <= 6
    if mesh_ok and hs["verts"] is not None and lm is not None:
        t_m = time.perf_counter()
        draw_hand_mesh(frame, hs["verts"], faces, K, c2w, 1.0, hs["hold"])
        hs["mesh_ms"] = 0.8 * hs["mesh_ms"] + 0.2 * (time.perf_counter() - t_m) * 1e3
    img, cam = ui.cover(frame, U.Rect(0, 0, ui.W, ui.H))
    img = np.ascontiguousarray(img)
    ui.shade(img, 0, ui.px(90), 0.4, 1.0)
    ui.shade(img, ui.H - ui.px(330), ui.H, 1.0, 0.18)
    ui.brand(img)
    keys = [("E", "Send to robot"), ("R", "Restart"), ("Q", "Quit")]
    if st.get("cam_msg"):
        keys.insert(0, ("C", "Recalibrate"))
    ui.keys(img, keys)
    est = sel.get("est")
    picked = est is not None and est.get("ok")
    if picked:
        prompt = "Grab the object, lift it, put it down and let go. Then press E."
    elif sel.get("msg", "").startswith("segmenting"):
        prompt = "Selecting the object ..."
    else:
        prompt = "Double-click the object to select it"
    ui.text(img, prompt, ui.W / 2, ui.px(24), 22, U.TEXT, anchor="c")
    if st.get("cam_msg"):
        ui.text(img, st["cam_msg"], ui.W / 2, ui.px(62), 19, U.BAD, anchor="c")
    # instruments
    r = ui.px(78)
    cy = ui.H - ui.px(112)
    fps = st["fps"]
    ui.ring(img, ui.m + r + ui.px(10), cy, r, fps / 30, "Camera", f"{fps:.0f}", "fps",
            U.TEXT if fps >= 27 else U.BAD)  # fmt: skip
    ui.ring(img, ui.m + 3 * r + ui.px(44), cy, r, (secs % 60) / 60, "Take", f"{secs:.1f}", "sec")
    hy = hud_y(ui)
    ui.phases(img, 0, None, hy)
    ui.clock_block(img, ui.W / 2, hy + ui.px(26), U.clock(secs), "Recording", U.TEXT, U.BAD)
    sx = ui.W - ui.m - ui.px(250)
    rows = [("Hand", "Tracked" if lm is not None else "Not found",
             U.TEXT if lm is not None else U.BAD),
            ("Object", ("Lying" if est["lying"] else "Standing") if picked else "Not selected",
             U.TEXT if picked else U.DIM)]  # fmt: skip
    if infer is not None:
        rows.append(("Grip", "Holding" if hs["hold"] else "Open",
                     U.TEXT if hs["hold"] else U.DIM))  # fmt: skip
    y = cy - ui.px(34 * len(rows))
    for lab, val, col in rows:
        ui.caps(img, lab, sx, y + ui.px(8))
        ui.text(img, val, sx + ui.px(100), y, 24, col, "light")
        y += ui.px(66)
    return img, cam


MOVED_PX = 8.0  # px (1920 wide): marker 0 this far from where the calibration expects it = moved


class CamCheck:
    """Is the camera still where it was calibrated? About once a second (a worker thread)
    marker 0 is found in the live frame and compared with where the calibration projects it
    (Andy 2026-11-14: the camera was moved after calibrating and the take came out wrong).
    Marker hidden (hand over it): no verdict."""

    def __init__(self, cal: dict):
        self.size = float(cal.get("marker_size_m", 0.1))
        self.exp = rig.expected_corners(cal, self.size)
        self.err: float | None = None
        self.busy, self.t = False, 0.0

    @property
    def moved(self) -> bool:
        return self.err is not None and self.err > MOVED_PX

    def feed(self, frame: np.ndarray) -> None:
        now = time.time()
        if self.busy or now - self.t < 1.0:
            return
        self.busy, self.t = True, now
        threading.Thread(target=self._run, args=(frame.copy(),), daemon=True).start()

    def _run(self, frame: np.ndarray) -> None:
        try:
            d = rig.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if 0 in d:
                self.err = float(np.linalg.norm(d[0] - self.exp, axis=1).mean())
        finally:
            self.busy = False


def recalibrate_pose(st: dict, cal: dict, path: Path, n: int = 15) -> str:
    """C pressed: the camera's pose again from marker 0 in the next n live frames, the focal
    length kept (results/rig/intrinsics.json, else the calibration's) -> calib.json rewritten
    (the old one in results/rig/history/). -> a message ("" = done)."""
    W, H = cal["image_size"]
    intr = Path("results/rig/intrinsics.json")
    f = float(cal["f"])
    if intr.exists():
        d = json.loads(intr.read_text())
        if d.get("image_size") == [W, H]:
            f = float(d["f"])
    got, last, t0 = [], None, time.time()
    while len(got) < n and time.time() - t0 < 5:
        pkt = st["pkt"]
        if pkt is not None and pkt is not last:
            last = pkt
            dd = rig.detect(cv2.cvtColor(pkt[0], cv2.COLOR_BGR2GRAY))
            if 0 in dd:
                got.append(dd[0])
        time.sleep(0.01)
    if len(got) < n // 2:
        return "marker 0 not visible: lay it flat in view, hand away, press C again"
    size = float(cal.get("marker_size_m", 0.1))
    new = rig.calibrate({0: np.mean(got, 0)}, size, W, H, f_fixed=f)
    if new["reproj_px"] > rig.MAX_REPROJ:
        return f"marker fit poor ({new['reproj_px']:.1f} px): flatten it, press C again"
    hist = Path("results/rig/history")
    hist.mkdir(parents=True, exist_ok=True)
    (hist / f"calib_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(path.read_text())
    js = {**cal, **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in new.items()}}
    js["calibrated_at"] = time.strftime("%Y-%m-%d %H:%M:%S") + " (demo C: pose, focal kept)"
    path.write_text(json.dumps(js, indent=1))
    print(f"camera re-calibrated (focal {f:.0f} px kept): {path}", flush=True)
    return ""


# ------------------------------------------------------------------ Tiger Data (robot demos)
DEFAULT_DATASET = Path("data/panda_demos/cup_pick_place.hdf5")
H5_LOCK = threading.Lock()  # the demo HDF5: appended by the UI side, read by the uploader
STALL_S = 120.0  # s: scaling with no finished demo and no new tile for this long -> the rest
# count as failed (timed out) and the app goes on, instead of waiting forever
LOG_DIR = Path("results/robot_sim/panda_demo/logs")


class _Tee:
    """Terminal output also written to the session log (to read back what happened)."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def write(self, x):
        self.a.write(x)
        try:
            self.b.write(x)
            self.b.flush()
        except Exception:
            pass
        return len(x)

    def flush(self):
        self.a.flush()

    def __getattr__(self, k):
        return getattr(self.a, k)


def session_log() -> Path:
    """Everything printed (errors and tracebacks too) -> logs/<time>.log, plus a native crash
    / hang dump (faulthandler; `kill -USR1 <pid>` dumps every thread's stack while it runs)."""
    import faulthandler
    import signal

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / time.strftime("%Y%m%d-%H%M%S.log")
    f = open(path, "a", buffering=1)  # noqa: SIM115 (open for the whole session)
    sys.stdout, sys.stderr = _Tee(sys.stdout, f), _Tee(sys.stderr, f)
    faulthandler.enable(f)
    faulthandler.register(signal.SIGUSR1, f, all_threads=True)
    return path


TIGER_EVERY = 2  # Tiger replay: render every 2nd 30 Hz step, play at 15 fps = real time


class TigerSink:
    """Every saved demo goes to Tiger Data (touch.store.demo_store) in a background thread,
    ~0.3 s per demo: the UI never waits. Keeps the store's counts for the screens. Off without
    $TIGER_DATABASE_URL, with --no-tiger, or for another --dataset (test runs)."""

    def __init__(self, dataset: Path, enabled: bool):
        self.url = os.environ.get("TIGER_DATABASE_URL")
        self.dataset = dataset
        self.on = bool(enabled and self.url)
        self.stats: dict = {}
        self.ids: dict[str, list[str]] = {}  # take -> demo ids stored this session
        self.write_ms: list[float] = []
        self.err = ""
        self._q: _queue.Queue = _queue.Queue()
        if self.on:
            threading.Thread(target=self._run, daemon=True).start()

    def add(self, take: str, keys: list[str]) -> None:
        if self.on and keys:
            self._q.put((take, list(keys)))

    def busy(self) -> bool:
        return self.on and self._q.unfinished_tasks > 0

    def _run(self) -> None:
        import h5py

        from touch.store.demo_store import DemoStore, demo_from_group

        try:
            st = DemoStore(self.url)
            st.create_schema()
            self.stats = st.stats()
        except Exception as ex:
            self.on, self.err = False, f"Tiger Data offline ({type(ex).__name__})"
            print(f"{self.err}: {ex}", flush=True)
            return
        name = self.dataset.stem
        while True:
            take, keys = self._q.get()
            try:
                with H5_LOCK, h5py.File(self.dataset, "r") as f:
                    demos = [demo_from_group(f["data"][k], name, k) for k in keys]
                for d in demos:
                    t = time.perf_counter()
                    did = st.add_demo(d) or f"{name}:{d['key']}"
                    self.write_ms.append(1e3 * (time.perf_counter() - t))
                    self.ids.setdefault(take, []).append(did)
                self.stats = st.stats()
                print(f"Tiger Data: stored {len(demos)} demos ({take}); "
                      f"{self.stats.get('demos')} in total", flush=True)  # fmt: skip
            except Exception as ex:
                self.err = f"Tiger Data upload failed ({type(ex).__name__})"
                print(f"{self.err}: {ex}", flush=True)
                try:
                    st.con.rollback()
                except Exception:
                    pass
            finally:
                self._q.task_done()


class HandStream:
    """The live hand data of a take -> Tiger Data hand_frames (touch.store.grasp_store) as it
    is computed (after the double-click): 3D joints in the table frame, grab gate state +
    score, MediaPipe landmarks; one session per take. A writer thread commits every ~0.2 s,
    so the camera loop never waits. Andy 2026-11-14: stored live for the Tiger Data demo."""

    FLUSH_S = 0.2

    def __init__(self, enabled: bool):
        self.url = os.environ.get("TIGER_DATABASE_URL")
        self.on = bool(enabled and self.url)
        self.rows = 0
        self.err = ""
        self._q: _queue.Queue = _queue.Queue()
        if self.on:
            threading.Thread(target=self._run, daemon=True).start()

    def start(self, take_name: str, meta: dict) -> None:
        if self.on:
            self._q.put(("start", take_name, meta))

    def frame(self, i: int, lm, left: bool, jw, closed: bool, score: float) -> None:
        if self.on:
            self._q.put(("frame", i, lm, left, jw, closed, score))

    def end(self) -> None:
        if self.on:
            self._q.put(("end",))

    def take_rows(self, name: str, meta: dict, take: dict, lefts: list, k0: int) -> int:
        """A finished take (models run after E, the default) -> its hand rows, from the
        double-click on, in the background: one session, 3D joints, grab gate, landmarks.
        (--live-hold streams them during the take instead.) -> rows queued."""
        if not self.on:
            return 0
        n = len(take["close"])
        self.start(name, {**meta, "click_frame": int(k0), "mode": "after_e"})
        for i in range(k0, n):
            lm = take["lm"][i]
            lm = None if lm is None or not np.isfinite(lm).all() else lm
            self.frame(i, lm, bool(lefts[i]) if i < len(lefts) else False, take["joints"][i],
                       bool(take["close"][i]), float(take["score"][i]))  # fmt: skip
        self.end()
        return n - k0

    def _run(self) -> None:
        from touch.store.grasp_store import GraspStore

        try:
            st = GraspStore(self.url)
        except Exception as ex:
            self.on, self.err = False, f"hand stream offline ({type(ex).__name__})"
            print(f"{self.err}: {ex}", flush=True)
            return
        sid, t_flush = None, time.perf_counter()
        while True:
            try:
                msg = self._q.get(timeout=self.FLUSH_S)
            except _queue.Empty:
                msg = None
            try:
                if msg is not None and msg[0] == "start":
                    if sid is not None:
                        st.end_session(sid)
                    sid = st.start_session("onetouch_live", "table", meta={"take": msg[1],
                                                                           **msg[2]})  # fmt: skip
                elif msg is not None and msg[0] == "end" and sid is not None:
                    st.end_session(sid)
                    sid = None
                elif msg is not None and msg[0] == "frame" and sid is not None:
                    _, i, lm, left, jw, closed, score = msg
                    ok = jw is not None and np.isfinite(jw).all()
                    raw = {"frame_idx": i, "gate_score": score}
                    if lm is not None:
                        raw["mp_landmarks"] = np.asarray(lm, float)[:, :3]
                    st.add_frame(sid, i / 30, "left" if left else "right", bool(ok),
                                 gate_closed=bool(closed), joints=jw if ok else None,
                                 raw=raw)  # fmt: skip
                    self.rows += 1
                if time.perf_counter() - t_flush >= self.FLUSH_S:
                    st.flush()
                    t_flush = time.perf_counter()
            except Exception as ex:
                self.err = f"hand stream failed ({type(ex).__name__})"
                print(f"{self.err}: {ex}", flush=True)
                try:
                    st.con.rollback()
                    st._buf.clear()
                except Exception:
                    pass


def tiger_lines(sink) -> list[tuple[str, tuple]]:
    """The store's state for the screens, from its real counts."""
    s = sink.stats if sink is not None else {}
    if sink is None or not sink.on or not s:
        why = (sink.err if sink is not None and sink.err else
               "Not connected  ·  demos saved locally (HDF5)")  # fmt: skip
        return [(why, U.DIM)]
    out = [(f"{s['demos']} robot demos  ·  {s['steps']:,} steps of 30 Hz state", U.TEXT),
           (f"{s['teleop']} led by a person  →  {s['generated']} generated", U.DIM)]  # fmt: skip
    if s.get("compression_ratio"):
        out.append((f"Columnstore: {s['bytes_before'] / 1e6:.0f} MB stored in "
                    f"{s['bytes_after'] / 1e6:.0f} MB ({s['compression_ratio']:.1f}x)",
                    U.DIM))  # fmt: skip
    if sink.write_ms:
        out.append((f"Last write {np.median(sink.write_ms[-12:]):.0f} ms per demo, live",
                    U.DIM))  # fmt: skip
    return out


HOW = (  # per processing step: the tools / models that run in it and how (shown while it runs)
    "GraspNet (ours): MobileNetV3 on 5 aligned fingertip patches + RepViT on the hand crop, "
    "causal GRUs over time -> touch / hold per frame; a MANO head regresses the 3D hand "
    "(21 joints, 778-vertex mesh), placed on the table by the ArUco camera pose.",
    "SAM 2.1 (Hiera-L) segments the clicked object; a metric cylinder model is fitted to the "
    "mask (IoU search over position and depth) in the ArUco-calibrated camera frame -> the "
    "MuJoCo scene, matched to the real table.",
    "The release frame defines the goal state: the object set down upright on the assigned "
    "spot (ArUco marker 0), the target the robot's run is checked against.",
    "Your 6-DoF wrist path (position + rotation, 30 Hz) is retargeted to the Franka Panda "
    "gripper; damped least-squares IK with a null-space pull solves its 7 joints per frame, "
    "1 cm clear of table and object, retimed to 80 % of its joint speed limits.",
    "Cross-entropy method (a small black-box RL search): samples grasp angle, height and "
    "close timing, rolls each out in MuJoCo physics, rewards how closely the object follows "
    "your hand, refits around the best and keeps the grasp that really lifts.",
    "MuJoCo replays the robot frame-locked to your video; joints, gripper and object are "
    "logged at 30 Hz as a robomimic HDF5 demo and streamed into Tiger Data (TimescaleDB).",
)


def info_panel(ui, img, step: int, info: dict, y: int) -> None:
    """Three columns under the step: how the tool works now, this demonstration, the data
    engine (Tiger Data)."""
    cw, gap = ui.px(500), ui.px(60)
    x0 = (ui.W - 3 * cw - 2 * gap) // 2
    n, fps = info.get("frames", 0), info.get("fps", 30)
    cols = [
        ("How it works", [(ln, U.TEXT if i == 0 else U.DIM)
                          for i, ln in enumerate(U.wrap(HOW[step], 44))]),  # fmt: skip
        ("This demonstration", [
            (f"{n} camera frames  ·  {n / max(fps, 1):.1f} s", U.TEXT),
            ("3D hand + contacts tracked in every frame", U.DIM),
            (f"→ 1 robot run + {info.get('n_gen', 0)} generated scenes", U.DIM)]),
        ("Data engine  ·  Tiger Data", tiger_lines(info.get("sink"))),
    ]  # fmt: skip
    for i, (head, lines) in enumerate(cols):
        x = x0 + i * (cw + gap)
        ui.line(img, x, y, x + cw, y, U.LINE)
        ui.caps(img, head, x, y + ui.px(14), U.DIM, 13)
        yy = y + ui.px(46)
        for ln, col in lines[:4]:
            ui.text(img, ln, x, yy, 17, col)
            yy += ui.px(28)


def mosaic_screen(ui, mosaic, title: str, sub: str, secs: float, keys, lines=()) -> np.ndarray:
    """A grid video full width (the generated demos, or their Tiger Data replay); title and
    a few lines under it, the clock in the middle."""
    img = ui._blank.copy()
    hy = hud_y(ui)
    area = U.Rect(ui.m, ui.px(84), ui.W - 2 * ui.m, hy - ui.px(110) - ui.px(84))
    m, mr = ui.fit(mosaic, area)
    ui.put(img, m, mr)
    ui.brand(img)
    ui.keys(img, keys)
    x, y = ui.m + ui.px(10), ui.H - ui.px(170)
    ui.caps(img, title, x, y, U.TEXT, 14)
    rows = [(w, col) for ln, col in lines for w in U.wrap(ln, 52)]  # clear of the clock
    for ln, col in rows[:5]:
        y += ui.px(26)
        ui.text(img, ln, x, y, 16, col)
    ui.phases(img, 5, None, hy)
    ui.clock_block(img, ui.W / 2, hy + ui.px(26), U.clock(secs), sub)
    return img


def play_mosaic(ui, frames, fps: float, title: str, sub: str, lines=()) -> str:
    """Play grid frames once (any key but Q returns early) -> "q" | "done"."""
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        show(mosaic_screen(ui, f, title, sub, take_secs(), [("Space", "Back"), ("Q", "Quit")],
                           lines))  # fmt: skip
        wait = t0 + (i + 1) / fps - time.perf_counter()
        k = cv2.waitKey(max(1, int(wait * 1000))) & 0xFF
        if k in (ord("q"), 27):
            return "q"
        if k == ord(" "):
            return "done"
    return "done"


def tiger_replay(ui, vui, pool, sink, take: str) -> str:
    """This take's demos read back from Tiger Data and reproduced in MuJoCo (recorded states,
    in pool workers), then played as a grid in real time -> "q" | "done"."""
    from touch.store.demo_replay import render_worker

    lines = [("Reading demos from Tiger Data and reproducing them in MuJoCo ...", U.DIM)]
    while sink.busy():  # the last uploads first
        show(mosaic_screen(ui, np.zeros((GH, GW, 3), np.uint8), "Storing in Tiger Data",
                           "Tiger Data", take_secs(), [("Q", "Quit")],
                           [("Writing the last demos ...", U.DIM)]))  # fmt: skip
        if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
            return "q"
    ids = sink.ids.get(take, [])
    if not ids:
        return "done"
    jobs = [pool.apply_async(render_worker, ((sink.url, did, k, DG.TILE, TIGER_EVERY,
                                             "states"),)) for k, did in enumerate(ids)]  # fmt: skip
    res = [None] * len(ids)
    kinds = ["teleop" if i == 0 else "generated" for i in range(len(ids))]
    cols = grid_cols(len(ids), U.Rect(0, 0, 16, 9))
    while any(r is None for r in res):
        for k, a in enumerate(jobs):
            if res[k] is None and a.ready():
                try:
                    res[k] = a.get()
                except Exception as ex:
                    res[k] = {"success": False, "reason": f"error: {ex}"[:40], "frames": []}
        nd = sum(r is not None for r in res)
        tiles = [cv2.imdecode(np.frombuffer(r["frames"][0], np.uint8), cv2.IMREAD_COLOR)
                 if r and r["frames"] else None for r in res]  # fmt: skip
        show(mosaic_screen(ui, grid(tiles, [None] * len(ids), kinds, vui, cols, pending="reading"),
                           "Reading from Tiger Data", "Tiger Data", take_secs(),
                           [("Q", "Quit")],
                           [(f"{nd} of {len(ids)} demos read back and reproduced",
                             U.TEXT)]))  # fmt: skip
        if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
            return "q"
    ok = [r for r in res if r["frames"]]
    load = np.median([r["load_ms"] for r in ok]) if ok else 0
    T = max(len(r["frames"]) for r in res) if ok else 0
    dec = [[cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in r["frames"]]
           for r in res]  # fmt: skip
    frames = []
    for t in range(T):
        tl = [fr[min(t, len(fr) - 1)] if fr else None for fr in dec]
        dn = [r if t >= len(dec[k]) - 1 else None for k, r in enumerate(res)]
        frames.append(grid(tl, dn, kinds, vui, cols))
    lines = [(f"{len(ok)} demo{'s' if len(ok) != 1 else ''} read from Tiger Data "
              f"(median {load:.0f} ms each) and "
              "reproduced in MuJoCo from the stored 30 Hz state", U.TEXT),
             ("The same rows a robot policy trains on: SQL in, robot motion out",
              U.DIM)]  # fmt: skip
    return play_mosaic(ui, frames, 30 / TIGER_EVERY, "Reproduced from Tiger Data", "Tiger replay",
                       lines)  # fmt: skip


WIN = {"fill": True, "set": False}


def show(img) -> None:
    """cv2.imshow into the demo window; the first time, size the window to fill the screen
    (a normal window, not macOS full screen)."""
    cv2.imshow(TITLE, img)
    if WIN["set"]:
        return
    WIN["set"] = True
    if WIN["fill"]:
        x, y, w, h = U.window_size()
        cv2.resizeWindow(TITLE, w, h)
        cv2.moveWindow(TITLE, x, y)
    cv2.waitKey(1)


def save_video(frames, path: Path, fps: int) -> None:
    raw = path.with_suffix(".raw.mp4")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(path)], check=True)  # fmt: skip
    raw.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="1", help="webcam index, or a video file (testing)")
    ap.add_argument("--ckpt", type=Path, default=Path(G13),
                    help="default g13 (all own data + public mix); g10: " + G10)  # fmt: skip
    ap.add_argument("--calib", type=Path, default=Path("results/rig/calib.json"))
    ap.add_argument("--obj-h", type=float, default=CYL_H)
    ap.add_argument("--obj-d", type=float, default=CYL_D)
    ap.add_argument("--release", default="normal",
                    choices=["normal", "fast", "fastest", "own_fast"],
                    help="grab gate release speed (fusion_hold.json release_profiles); default "
                         "normal (Andy 2026-11-14: ~0.5 s late is fine, no false releases); "
                         "open gaps <= HOLD_GAP frames inside a hold are filled")  # fmt: skip
    ap.add_argument("--max-grasps", type=int, default=1, help="process the last N grasps")
    ap.add_argument("--budget", type=float, default=30.0, help="RL search seconds per grasp")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--pop", type=int, default=18, help="CEM grasps per round")
    ap.add_argument("--click", type=lambda v: tuple(int(x) for x in v.split(",")), default=None,
                    help="x,y in the 800x450 preview: select the object there at the "
                         "first frame (replays; live: double-click it)")  # fmt: skip
    ap.add_argument(
        "--provider",
        choices=["cpu", "coreml"],
        default="cpu",
        help="grasp model on the CPU or the Mac GPU / Neural Engine (coreml)",
    )
    ap.add_argument("--small-window", action="store_true", help="3/4 size, not filling the screen")
    ap.add_argument("--ui-width", type=int, default=None,
                    help="canvas width (px); default: the screen's native pixels")  # fmt: skip
    ap.add_argument("--live-hold", action="store_true",
                    help="run the grasp model + HOLD detection live during the take (default: "
                         "after E, so the camera keeps a steady 30 fps)")  # fmt: skip
    ap.add_argument("--depth-fix", type=float, nargs=2, default=(0.9, 1.1),
                    metavar=("LO", "HI"),
                    help="rescale the 3D hand's depth from the cup only when the factor is in "
                         "[LO, HI] (Andy 2026-11-14: 1.2-1.6 put the hand under the table); "
                         "'1 1' = never")  # fmt: skip
    ap.add_argument("--stable-depth", action="store_true",
                    help="steadier hand depth (handpath.StableDepth: fixed hand size, jump "
                         "filter, One Euro); default: per-frame depth as last night")  # fmt: skip
    ap.add_argument("--mesh", action="store_true",
                    help="draw the 3D hand mesh on the live view (off by default: 30 fps "
                         "first; it hides itself when the model falls behind)")  # fmt: skip
    ap.add_argument("--good", type=float, default=0.9, help="stop the search at this accuracy")
    ap.add_argument("--workers", type=int, default=12, help=">= --n-gen: all run at once")
    ap.add_argument("--n-gen", type=int, default=12, help="scaled demos per teleop demo")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--pour-angle", type=float, default=V.POUR_ANGLE,
                    help="a pour (the biggest tilt while carrying, > 35 deg as estimated) peaks "
                         "at this angle (deg); 0 = copy the tilt as estimated")  # fmt: skip
    ap.add_argument("--rerun-models", action="store_true",
                    help="--replay: run the grab model / 3D hand again instead of reusing the "
                         "detection saved in take.npz")  # fmt: skip
    ap.add_argument("--pour-side", default="down",
                    choices=["down", "up", "left", "right", "hand"],
                    help="the can top leans this way in the webcam image when pouring (Andy "
                         "pours toward the bottom of the video); hand = the 3D hand's "
                         "estimate")  # fmt: skip
    ap.add_argument("--lying-share", type=float, default=0.5,
                    help="share of the generated scenes whose cup starts lying on its side "
                         "(each still ends the way you left it)")  # fmt: skip
    ap.add_argument("--free-place", action="store_true",
                    help="set the cup down where you put it (default: on marker 0, the "
                         "assigned spot)")  # fmt: skip
    ap.add_argument("--allow-lying", action="store_true",
                    help="the cup may start / end on its side (default: always standing upright "
                         "at the start and the end; it may be turned while carried)")  # fmt: skip
    ap.add_argument("--no-tiger", action="store_true",
                    help="do not upload demos to Tiger Data (off anyway without "
                         "$TIGER_DATABASE_URL or with another --dataset)")  # fmt: skip
    ap.add_argument("--fps", type=int, default=20, help="playback fps (30 = real time)")
    ap.add_argument("--auto-e", type=int, default=0, help="testing: press E after N frames")
    ap.add_argument("--replay", type=Path, default=None,
                    help="a saved take folder (results/robot_sim/panda_demo/<take>): replay its "
                         "take.mp4 with its saved double-click segmentation")  # fmt: skip
    ap.add_argument("--no-cup-pose", action="store_true",
                    help="no SAM cup pose: the cup stands where the hand grasped it")  # fmt: skip
    args = ap.parse_args()
    print(f"session log: {session_log()}  (pid {os.getpid()})", flush=True)
    replay_sel: dict = {}
    if args.replay is not None:
        replay_sel, args.cam, args.auto_e = load_replay(args.replay)
        if (args.replay / "calib.json").exists():  # the rig as it was when recorded
            args.calib = args.replay / "calib.json"
        else:
            print("replay: no calib.json saved with this take (older take): using "
                  f"{args.calib}; if the rig was recalibrated since, the cup is misplaced",
                  flush=True)  # fmt: skip
    cal = json.loads(args.calib.read_text())
    K, c2w = np.asarray(cal["K"]), np.asarray(cal["cam_to_world"])
    W, H = cal["image_size"]
    flip = {"h": "h" in cal.get("flip", ""), "v": "v" in cal.get("flip", "")}
    cam_xml, cam_xy = camera_xml(K, c2w, H), c2w[:2, 3]
    seg_box = {"seg": None, "err": None, "ready": threading.Event()}

    def load_sam():  # in the background: the camera starts right away
        try:
            if not args.no_cup_pose:
                seg_box["seg"] = CP.Segmenter()
                print(f"SAM 2.1 Hiera-L loaded ({seg_box['seg'].device})", flush=True)
        except Exception as ex:  # the demo still runs: cup standing at the hand
            seg_box["err"] = f"{type(ex).__name__}: {ex}"
            print("SAM not loaded:", seg_box["err"], flush=True)
        finally:
            seg_box["ready"].set()

    threading.Thread(target=load_sam, daemon=True).start()
    print("starting worker processes ...", flush=True)
    pool = get_context("spawn").Pool(args.workers)
    pool.map(math.sqrt, range(args.workers))  # start them now, not on the first E
    queue = get_context("spawn").Manager().Queue()  # rendered tiles from the scaled demos
    sink = TigerSink(args.dataset, not args.no_tiger and args.dataset == DEFAULT_DATASET)
    hand_stream = HandStream(sink.on and str(args.cam).isdigit())  # live camera takes only
    print("Tiger Data: " + ("demos are stored live" if sink.on else "off (demos saved locally)"),
          flush=True)  # fmt: skip
    live = LiveGrasp(args.ckpt, 4, args.provider)
    fdir = Path("runs/grab_fusion") / args.ckpt.parent.name
    if not (fdir / "fusion_hold.json").exists():  # the model's own gate not built yet
        fdir = Path("runs/grab_fusion") / Path(G10).parent.name
        print(f"no grab gate for {args.ckpt.parent.name} yet: using g10's ({fdir})", flush=True)
    fusion = FusionGate.from_dir(fdir, "hold")
    prof = json.loads((fdir / "fusion_hold.json").read_text()).get("release_profiles", {})
    for k, v in (prof.get(args.release) or prof.get("fast", {})).get("release", {}).items():
        setattr(fusion.hyst, k, v)
    depth = StableDepth(K, c2w) if args.stable_depth else None
    faces = None
    if args.mesh and args.live_hold:
        try:
            faces = mano_faces()
        except Exception as ex:  # no MANO here: HOLD label only
            print(f"no hand mesh ({type(ex).__name__}: {ex})", flush=True)
    ui = U.UI(args.ui_width)
    vui = U.UI(1920, aspect=16 / 9)  # saved videos (800 x 450 panels)
    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    WIN["fill"] = not args.small_window
    if args.small_window:
        cv2.resizeWindow(TITLE, ui.W * 3 // 4, ui.H * 3 // 4)
    disp: dict = {"cam": None}  # the camera image's rect on screen (mouse -> view px)
    cur = {"st": None, "take": None}  # the running take (for the mouse callback)
    sel: dict = {}  # the clicked cylinder: est, overlay, mask (view px), frame, msg

    def click_cup(frame, pt, k):
        """SAM 2.1 at the clicked point on this frame (background thread) -> sel."""
        try:
            small = cv2.resize(frame, (PW, PH), interpolation=cv2.INTER_AREA)
            seg_box["ready"].wait()
            if seg_box["seg"] is None:
                sel.update(msg=f"SAM not loaded: {seg_box['err']}")
                return
            Ks = CP.scale_K(K, cal["image_size"], small.shape)
            hint = CP.ray_to_plane(Ks, c2w[:3], np.asarray(pt, float), args.obj_d / 2)[:2]
            est = CP.estimate(seg_box["seg"], small, K, cal["image_size"], c2w, args.obj_h,
                              args.obj_d, hint, allow_lying=args.allow_lying)  # fmt: skip
            ov = CP.draw(small, K, cal["image_size"], c2w, est, args.obj_h, args.obj_d)
            kind = f"lying {math.degrees(est['yaw']):.0f} deg" if est["lying"] else "standing"
            msg = (f"object: {kind}, IoU {est['iou']:.2f}" if est["ok"]
                   else f"fit not trusted (IoU {est['iou']:.2f}): click again")  # fmt: skip
            sel.update(est=est, overlay=ov, mask=(ov != small).any(2), frame=k, msg=msg,
                       pt=[int(pt[0]), int(pt[1])], image=small,
                       sam_mask=est.get("mask"), t=time.strftime("%H:%M:%S"),
                       hand_seen=False)  # fmt: skip
        except Exception as ex:  # shown in the panel
            sel.update(msg=f"segmentation failed: {type(ex).__name__}")

    dbl = {"down": 0.0, "fired": 0.0}

    clicks: dict = {"on": False, "tile": None}  # the final grid: a clicked tile replays

    def on_mouse(ev, x, y, *_):
        """Double-click: select the object there (SAM on the current frame); double-click
        again: clear it. (Some OpenCV backends send no DBLCLK event: two clicks < 0.4 s.)"""
        now = time.time()
        if clicks["on"]:
            if ev == cv2.EVENT_LBUTTONDOWN:
                k = tile_at(x, y)
                if k is not None:
                    clicks["tile"] = k
            return
        if ev == cv2.EVENT_LBUTTONDOWN:
            is_dbl, dbl["down"] = now - dbl["down"] < 0.4, now
        else:
            is_dbl = ev == cv2.EVENT_LBUTTONDBLCLK
        r = disp["cam"]
        if not is_dbl or now - dbl["fired"] < 0.5 or r is None:
            return
        if not (r.x <= x < r.x2 and r.y <= y < r.y2):
            return
        dbl["fired"] = now
        select_at((x - r.x) * PW / r.w, (y - r.y) * PH / r.h)

    def select_at(x: float, y: float) -> None:
        """Double-click at (x, y) in the 800 x 450 view: segment the object there (again =
        clear)."""
        if sel:  # second double-click: reset
            sel.clear()
            return
        st_ = cur["st"]
        if st_ is None or st_["pkt"] is None:
            return
        sel["msg"] = "segmenting the object ..."
        k = len(cur["take"]["lm"]) - 1 if cur["take"] is not None else 0
        sel["frame"] = max(k, 0)  # processing starts here
        threading.Thread(target=click_cup, args=(st_["pkt"][0], (x, y), sel["frame"]),
                         daemon=True).start()  # fmt: skip

    cv2.setMouseCallback(TITLE, on_mouse)
    out_root = Path("results/robot_sim/panda_demo")
    quit_ = False
    while not quit_:  # one take per loop
        src = int(args.cam) if str(args.cam).isdigit() else args.cam
        cap = cv2.VideoCapture(src)
        # frames at the calibration size: saved takes (take.mp4 is 800 x 450) and cameras
        # that deliver another size of the same shape (e.g. Camo set to 720p) are scaled;
        # a different shape (aspect) still stops with an error
        cap = ResizedCapture(cap, (W, H))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap = PrefetchCapture(cap)
        stop = threading.Event()
        tracker = HandTracker(W)
        fusion.reset()
        live.reset()
        take = {"raw": [], "lm": [], "jpg": []}
        clock_reset()
        # Live = only what needs the camera frame: MediaPipe (its own thread), the model's
        # fingertip / hand crops, the gate's two image features (flow, fingertip colour) and
        # the take's JPEGs (`encode` thread). The grasp model (ONNX, ~18 ms / frame), the 3D
        # hand and the grab gate (HOLD) run after E over the recording (run_models), in frame
        # order with the same causal state: identical results, and the camera keeps 30 fps
        # (Andy 2026-11-14: HOLD after E is fine). This thread only shows the newest frame and
        # reads keys (cv2.waitKey ~14 ms on macOS: everything in one thread ran at ~21 fps).
        st = {"pkt": None, "fps": 0.0, "done": False, "err": None}
        cur.update(st=st, take=take)
        # a fresh double-click each take (Andy 2026-11-14): after a pick and place the cup is
        # somewhere else, an old segmentation would be drawn and used at the old spot
        sel.clear()
        jq: _queue.Queue = _queue.Queue()

        def encode(take=take, jq=jq):
            while (f := jq.get()) is not None:
                small = cv2.resize(f, (PW, PH), interpolation=cv2.INTER_AREA)
                take["jpg"].append(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1])

        def process(cap=cap, src=src, stop=stop, tracker=tracker, take=take, st=st, jq=jq):
            t_prev = time.perf_counter()
            try:
                # MediaPipe in its own thread: overlaps with the crops / gate features below
                # a saved take (take.mp4) was written after the flip: never flip it again
                fl = flip if isinstance(src, int) else {"h": False, "v": False}
                for frame, hands, hd, _, _ in threaded(frames_with_hands(cap, src, 30.0, stop,
                                                                         fl)):  # fmt: skip
                    assert frame.shape[1] == W and frame.shape[0] == H, "camera size != calib"
                    hands, hd = whole_hands(hands, hd, W, H)
                    lm, left = tracker.pick(hands, hd)
                    crops = live.crops(frame, lm, left)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if fusion.needs_frames else None
                    ff = fusion.frame_features(lm is not None, lm=lm, gray=gray, bgr=frame)
                    jq.put(frame)
                    take["raw"].append((crops, lm, left, ff))
                    take["lm"].append(np.full((21, 3), np.nan) if lm is None else lm[:, :3])
                    now = time.perf_counter()
                    st["fps"], t_prev = 0.9 * st["fps"] + 0.1 / max(now - t_prev, 1e-6), now
                    st["pkt"] = (frame, lm)
            except Exception as ex:  # re-raised in the main thread
                st["err"] = ex
            finally:
                jq.put(None)
                st["done"] = True

        workers = [threading.Thread(target=process, daemon=True),
                   threading.Thread(target=encode, daemon=True)]  # fmt: skip
        for t in workers:
            t.start()
        resq: _queue.Queue = _queue.Queue()
        infer = None
        if args.live_hold:
            if depth is not None:
                depth.reset()
            infer = LiveInference(
                take,
                live,
                fusion,
                K,
                c2w,
                resq,
                depth,
                lambda: sel.get("frame") if sel.get("est") is not None else None,
            )
            infer.stream = hand_stream if hand_stream.on else None
            infer.start()
        hs = {"i": -1, "hold": False, "score": float("nan"), "p_hold": 0.0, "verts": None,
              "mesh_ms": 0.0}  # fmt: skip
        pressed_e, shown = False, None
        # live cameras only: a saved take (replay) was recorded with its own calibration
        camcheck = CamCheck(cal) if isinstance(src, int) else None
        while True:
            while not resq.empty():  # newest live model result
                hs["i"], hs["hold"], hs["score"], hs["p_hold"], hs["verts"] = resq.get_nowait()
            pkt = st["pkt"]
            # replays: a double-click at frame 30 (the first frames can be out of focus)
            if args.click and pkt is not None and not sel and len(take["lm"]) >= 30:
                select_at(*args.click)
            if replay_sel and not sel and len(take["lm"]) - 1 >= replay_sel["frame"]:
                sel.update(replay_sel)  # --replay: the saved segmentation, no SAM
            if pkt is not None and pkt is not shown:
                shown = pkt
                frame, lm = pkt
                if camcheck is not None:
                    camcheck.feed(frame)
                    if camcheck.moved:
                        st["cam_msg"] = (f"Camera moved since calibration ({camcheck.err:.0f} "
                                         "px off): press C to recalibrate")  # fmt: skip
                    elif st.get("cam_msg", "").startswith("Camera moved"):
                        st["cam_msg"] = ""
                lag = len(take["lm"]) - 1 - hs["i"]
                t_ui = time.perf_counter()
                if lm is not None and sel.get("est") is not None:
                    clock_start()  # the first hand after the object is selected
                secs = take_secs()
                canvas, disp["cam"] = live_screen(ui, frame, lm, W, st, sel, secs, hs, infer,
                                                  fusion, faces, K, c2w, lag)  # fmt: skip
                st["ui_ms"] = 0.9 * st.get("ui_ms", 0.0) + 0.1 * (time.perf_counter() - t_ui) * 1e3
                show(canvas)
            key = cv2.waitKey(1) & 0xFF
            if args.auto_e and len(take["lm"]) >= args.auto_e:
                key = ord("e")
            if key in (ord("q"), 27):
                quit_ = True
                break
            if key == ord("r"):
                break
            if key == ord("c") and camcheck is not None:
                st["cam_msg"] = "Recalibrating the camera position ..."
                show(live_screen(ui, *st["pkt"], W, st, sel, take_secs(), hs, infer,
                                 fusion, faces, K, c2w, 0)[0])  # fmt: skip
                cv2.waitKey(1)
                msg = recalibrate_pose(st, cal, args.calib)
                if not msg:  # restart with the new calibration (all derived state with it)
                    stop.set()
                    for t in workers:
                        t.join()
                    cap.release()
                    pool.terminate()
                    os.execv(sys.executable, [sys.executable, *sys.argv])
                st["cam_msg"] = msg
            if key == ord("e") and camcheck is not None and camcheck.moved:
                st["cam_msg"] = "Camera moved since calibration: press C first (E blocked)"
                key = 0
            if key == ord("e"):
                pressed_e = True
                break
            if st["done"]:  # video file ended / camera lost
                break
        stop.set()
        for t in workers:
            t.join()
        take_t = time.strftime("%Y%m%d-%H%M%S")  # the take's name: its folder + Tiger session
        n_keep = min(len(take[k]) for k in ("raw", "lm", "jpg"))
        if args.auto_e:
            n_keep = min(n_keep, args.auto_e)  # replays: exactly the requested frames
        if st["err"] is not None:
            raise st["err"]
        if infer is not None:  # E: finish the frames still waiting; R / Q: stop now
            infer.finish(n_keep if pressed_e and not quit_ else 0)
            if infer.stream is not None:  # close the take's Tiger session once it is done
                threading.Thread(target=lambda f=infer: (f.join(), hand_stream.end()),
                                 daemon=True).start()  # fmt: skip
            with infer.lock:  # never shrink take["raw"] while it reads a frame
                for k in take:
                    del take[k][n_keep:]
            if not (pressed_e and not quit_):
                infer.join()  # before the next take resets the model / gate state
        else:
            for k in take:
                del take[k][n_keep:]
        cap.release()
        cur.update(st=None, take=None)
        if sel.get("est") is not None:
            take["click_cup"] = {k: sel.get(k) for k in ("est", "overlay", "frame", "pt", "image",
                                                         "sam_mask", "t")}  # fmt: skip
        print(f"take: {n_keep} frames, live loop at {st['fps']:.1f} fps", flush=True)
        if quit_ or not pressed_e or n_keep == 0:
            continue
        t_m = time.time()
        last = cv2.imdecode(take["jpg"][-1], cv2.IMREAD_COLOR)
        take["calib"] = cal
        k0 = int(take["click_cup"]["frame"] or 0) if "click_cup" in take else 0
        if k0:
            print(f"object double-clicked at frame {k0}: frames before it not processed",
                  flush=True)  # fmt: skip

        def show_models(i, n, last=last, t_m=t_m):
            show(process_screen(ui, last, 0, f"Reading your take  ·  frame {i} of {n}",
                                             i / max(n, 1), take_secs()))  # fmt: skip
            cv2.waitKey(1)

        if infer is None:
            if depth is not None:
                depth.reset()
            est = (take.get("click_cup") or {}).get("est") or {}
            box = (cup_box_px(est, args.obj_h, args.obj_d, K, c2w, CUP_GROW)
                   if est.get("ok") and est.get("xy") is not None else None)  # fmt: skip
            saved = None
            if args.replay is not None and not args.rerun_models:
                saved = np.load(args.replay / "take.npz", allow_pickle=True)
                if (
                    not all(k in saved for k in ("joints", "close", "score", "p_hold"))
                    or len(saved["close"]) != n_keep
                ):
                    saved = None
            if saved is not None:  # replay: the detection saved with the take (no models)
                for k in ("joints", "close", "score", "p_hold"):
                    take[k] = saved[k].astype(bool if k == "close" else float)
                take.pop("raw", None)
                print("replay: saved detection (take.npz) reused, models not rerun", flush=True)
            else:
                lefts = [bool(x[2]) for x in take["raw"]]  # run_models drops take["raw"]
                run_models(take, live, fusion, K, c2w, show_models, depth, box, k0)
                if args.replay is None:  # a camera take: its hand rows to Tiger Data now
                    q_ = hand_stream.take_rows(take_t, {"take": take_t}, take, lefts, k0)
                    if q_:
                        print(f"Tiger Data: {q_} hand rows of take {take_t} being stored",
                              flush=True)  # fmt: skip
        else:  # the live thread already ran (most of) them, in frame order
            print(f"live model: {len(infer.res['close'])} / {n_keep} frames done at E", flush=True)
            while infer.is_alive():
                show_models(len(infer.res["close"]), n_keep)
                time.sleep(0.03)
            if infer.err is not None:
                raise infer.err
            take.pop("raw")
            for k, v in infer.res.items():
                take[k] = v[:n_keep]
            assert len(take["close"]) == n_keep, (len(take["close"]), n_keep)
            for i in range(min(k0, n_keep)):  # e.g. clicked, cleared, clicked again later
                take["close"][i] = False
        if "processed" in take:
            print(f"processed frames {take['processed'][0]}-{take['processed'][1]} of {n_keep}",
                  flush=True)  # fmt: skip
        print(f"take clock T+{take_secs():.1f} s: models done", flush=True)
        print(f"grab model + gate after E: {time.time() - t_m:.1f} s, "
              f"grasps seen: {count_grasps(take['close'])}", flush=True)  # fmt: skip
        # ---------------------------------------------------------------- processing
        proc = Processor(take, pool, args, cam_xml, cam_xy, cal, seg_box)
        proc.start()
        last = cv2.imdecode(take["jpg"][-1], cv2.IMREAD_COLOR)
        view = last
        info = {"frames": len(take["lm"]) - k0, "fps": 30, "n_gen": args.n_gen, "sink": sink}
        while proc.is_alive():
            view = last if proc.preview is None else proc.preview
            stp, det, frac = stage_of(proc.status, args.iters)
            show(process_screen(ui, view, stp, det, frac, take_secs(), info=info))
            if (cv2.waitKey(50) & 0xFF) in (ord("q"), 27):
                quit_ = True
                break
        if quit_:
            break
        if proc.error is not None:
            d = out_root / (time.strftime("%Y%m%d-%H%M%S") + "_failed")
            save_take_basics(d, take, proc.error)
            for img in Path(proc.debug_dir).glob("end_pose_*.jpg"):
                img.replace(d / img.name)
            print(f"take saved for debugging: {d}", flush=True)
            stp, _, _ = stage_of(proc.status, args.iters)
            while True:
                img = process_screen(
                    ui,
                    last,
                    stp,
                    proc.error,
                    None,
                    take_secs(),
                    failed=True,
                    keys=[("R", "New take"), ("Q", "Quit")],
                )
                show(img)
                k = cv2.waitKey(50) & 0xFF
                if k in (ord("q"), 27) or args.auto_e:
                    k = ord("q")
                    quit_ = True
                if k in (ord("r"), ord("q"), 27):
                    break
            continue
        res = proc.result
        fin: dict = {}

        def finish(res=res, fin=fin, take=take, proc=proc, take_t=take_t):
            """Playback frames, the take folder, videos, summary and the teleop demo (in a
            thread: the UI keeps drawing, the clock keeps running)."""
            try:
                src = None
                frames = compose(take, res, W)
                d = out_root / take_t  # = the take's hand session in Tiger Data
                d.mkdir(parents=True, exist_ok=True)
                save_take_basics(d, take)
                for img in Path(proc.debug_dir).glob("end_pose_*.jpg"):
                    img.replace(d / img.name)
                if Path(proc.debug_dir).exists():
                    Path(proc.debug_dir).rmdir()
                save_video([video_frame(*f, vui) for f in frames], d / "teleop.mp4", args.fps)
                for k, r in enumerate(res):
                    if r["cup_overlay"] is not None:
                        cv2.imwrite(str(d / f"cup_pose_{k + 1}.jpg"), r["cup_overlay"])
                    # hand video vs robot timing (debug): per sim frame the take frame shown, the
                    # robot's target and where its TCP actually was
                    hi_ = np.asarray(r["human"], int)
                    np.savez_compressed(d / f"robot_sync_{k + 1}.npz", human=r["human"],
                                        take_frame=r["pl"]["lo"] + np.asarray(r["human"], int),
                                        target=r["target"], tcp=np.asarray(r["log"]["tcp"]),
                                        target_R=r["target_R"], tcp_R=np.asarray(r["log"]["tcp_R"]),
                                        hand_R=np.asarray(r["pl"]["hand_R"])[hi_],
                                        close=np.asarray(r["close"])[hi_])  # fmt: skip
                summ = {"take": [{"grasp": k + 1, "seconds": round(r["seconds"], 1),
                                  "pick_place_ok": r["src_ok"], "why": r["src_why"],
                                  "metrics": r["src_met"], "lift_accuracy": round(r["reward"], 3),
                                  "grasp_pitch_deg": round(math.degrees(r["g"][1]), 1),
                                  "grasp_height_cm": round(100 * float(r["g"][2]), 1),
                                  "your_grip_height_cm": round(100 * float(r["pl"]["g0"][2]), 1),
                                  "grasp_vs_hand_deg": [
                                      round(math.degrees(r["g"][i] - r["pl"]["g0"][i]), 1)
                                      for i in (0, 1)],  # fmt: skip
                                  "rl_rounds": r["hist"], "cup_pose": r["cup_pose"],
                                  "hold_found_by": getattr(proc, "hold_source", "gate")}
                                 for k, r in enumerate(res)]}  # fmt: skip
                good = [r for r in res if r["src_ok"]]
                teleop = None
                if good:  # record the teleoperated demonstration right away
                    r = good[-1]
                    src = r["src"]
                    with H5_LOCK:
                        teleop = DG.append_demos(args.dataset, [{
                        "log": r["log"], "eef_cmd": r["eef_cmd"], "xml": r["pl"]["xml"],
                        "attrs": {"source": "teleop", "success": True, "take": d.name,
                                  "cup_xy": src["cup"].tolist(),
                                  "place_xy": src["place"].tolist(),
                                  "metrics": r["src_met"], "lying": bool(src["lying0"]),
                                  "cup_yaw": src["yaw0"],
                                  "cup_pose": r["cup_pose"]}}])[0]  # fmt: skip
                    print(f"teleop demo saved: {args.dataset} {teleop}", flush=True)
                (d / "summary.json").write_text(json.dumps(summ, indent=1))
                fin.update(frames=frames, d=d, teleop=teleop, src=src, summ=summ)
            except Exception as ex:  # shown, then re-raised in the main thread
                fin["err"] = ex

        th = threading.Thread(target=finish, daemon=True)
        th.start()
        while th.is_alive():
            show(process_screen(ui, view, len(STEPS) - 1, "Preparing the playback", None,
                                take_secs(), info=info))  # fmt: skip
            cv2.waitKey(30)
        if "err" in fin:
            raise fin["err"]
        frames, d, teleop, src, summ = (fin[k] for k in ("frames", "d", "teleop", "src", "summ"))
        if teleop:
            sink.add(d.name, [teleop])  # to Tiger Data in the background
        # ---------------------------------------------------------------- teleop playback
        note = (f"saved as {teleop}" if teleop else
                "robot run failed, not recorded")  # fmt: skip

        launch = (f"GO FOR SCALE  ·  1 HUMAN DEMO INTO {args.n_gen} ROBOT DEMOS"
                  if args.n_gen > 0 else "")  # fmt: skip
        ready_keys = [("P", "Replay"), ("R", "New take"), ("Q", "Quit")]

        def play(once: bool, frames=frames, note=note, teleop=teleop, intro=False,
                 ready=False, launch=launch, ready_keys=ready_keys) -> str:  # fmt: skip
            """ready: loop the side by side until S (scale) / R / Q; P restarts it."""
            i = 0
            t_start = time.perf_counter()
            while True:
                t_f = time.perf_counter()
                j = i % len(frames)
                k_in = (t_f - t_start) / INTRO if intro else 1.0
                f = playback_screen(ui, *frames[j], note, U.GOOD if teleop else U.BAD,
                                    j / args.fps, k_in, ready_keys if ready else None,
                                    launch if ready else "")  # fmt: skip
                show(f)
                wait = 1000 / args.fps - (time.perf_counter() - t_f) * 1e3
                k = cv2.waitKey(max(1, int(wait))) & 0xFF
                i += 1
                if k in (ord("q"), 27):
                    return "q"
                if k == ord("r"):
                    return "r"
                if ready:
                    if k == ord("p"):
                        i = 0
                    elif k in (ord("s"), 13, 32) or (args.auto_e and i >= len(frames)):
                        return "s"
                    continue
                if (once or args.auto_e) and i >= len(frames):
                    return "done" if teleop else "q"

        print(f"take clock T+{take_secs():.1f} s: plan + grasp search done", flush=True)
        if teleop:  # ready: "Go for transfer", then the playback at once
            t_go = time.time()
            while time.time() - t_go < GO_HOLD:
                show(go_screen(ui, view, take_secs(), time.time() - t_go))
                cv2.waitKey(15)
        act = play(once=bool(teleop), intro=True)
        if act == "q":
            break
        if act == "r" or not teleop:
            continue
        if args.n_gen > 0:  # rewatch the side by side, then launch the multiplier (Andy
            # 2026-11-14: a button, not straight on)
            act = play(once=False, ready=True)
            if act == "q":
                break
            if act == "r":
                continue
        # ---------------------------------------------------------------- scale + run
        rnd = 0
        while args.n_gen > 0:
            head = f"{teleop} (your teleop demo) to {args.n_gen} new demos, cup + target moved"
            t_sc = time.time()
            done, gframes, q_, (last_tiles, kinds, tile_hist) = scale(
                pool,
                queue,
                src,
                args.n_gen,
                rnd,
                d / f"scenes_{rnd}",
                head,
                ui,
                vui,
                DG.P_LYING if args.allow_lying else 0.0,
                cam_xml,
                round(args.lying_share * args.n_gen),
            )
            t_sc = time.time() - t_sc
            clock_stop()  # the take is done: the clock holds the total
            print(f"take clock T+{take_secs():.1f} s: scaling done (stopped)", flush=True)
            if q_:
                quit_ = True
                break
            ok = [r for r in done if r["success"]]
            with H5_LOCK:
                names = DG.append_demos(args.dataset, [{
                "log": r["log"], "eef_cmd": r["eef_cmd"], "xml": r["xml"],
                "attrs": {"source": "mimicgen", "success": True, "src_demo": teleop,
                          "take": d.name, "cup_xy": r["cup"].tolist(),
                          "place_xy": r["place"].tolist(), "metrics": r["metrics"],
                          "lying": r["scene"]["lying"], "cup_yaw": r["scene"]["cup_yaw"],
                          "place_yaw": r["scene"]["place_yaw"],
                          "place_quat": r["place_quat"].tolist(),
                          "q_start": r["q_start"].tolist()}}
                for r in ok]) if ok else []  # fmt: skip
                total = DG.count_demos(args.dataset)
            sink.add(d.name, names)
            summ[f"scaled_{rnd}"] = [{"k": r["k"] + 1, "success": r["success"],
                                      "why": r["reason"], "metrics": r.get("metrics"),
                                      "seconds": r.get("seconds")} for r in done]  # fmt: skip
            (d / "summary.json").write_text(json.dumps(summ, indent=1))
            save_video(gframes, d / f"scaled_{rnd}.mp4", args.fps)
            print(f"scaled: {len(ok)}/{len(done)} successful -> {names}; dataset {total} demos",
                  flush=True)  # fmt: skip
            msg = f"{len(ok)} of {len(done)} added to {args.dataset.name} · {total} demos in total"
            if sink.on:
                msg += " · stored in Tiger Data"
            fin_keys = [("Click", "Replay a tile"), ("P", "Execution"), ("V", "Generated"),
                        ("T", "Tiger replay"),
                        ("G", f"{args.n_gen} more"), ("O", "Folder"), ("R", "New take"),
                        ("Q", "Quit")]  # fmt: skip
            if not sink.on:
                fin_keys = [kk for kk in fin_keys if kk[0] != "T"]
            k = None
            clicks.update(on=True, tile=None)
            play_k, play_i, play_t = None, 0, 0.0  # a clicked tile replaying in place
            while k not in ("g", "r", "q"):
                if clicks["tile"] is not None:  # click: (re)start that tile's run
                    play_k, play_i, play_t = clicks["tile"], 0, time.perf_counter()
                    clicks["tile"] = None
                shown = list(last_tiles)
                if play_k is not None and tile_hist[play_k]:
                    play_i = int((time.perf_counter() - play_t) * args.fps / DG_EVERY)
                    if play_i >= len(tile_hist[play_k]):
                        play_k = None
                    else:
                        shown[play_k] = cv2.imdecode(
                            np.frombuffer(tile_hist[play_k][play_i], np.uint8), cv2.IMREAD_COLOR
                        )
                show(scale_screen(ui, shown, done, kinds, len(done), take_secs(), head, msg,
                                  U.GOOD if ok else U.BAD, fin_keys))  # fmt: skip
                c = cv2.waitKey(20 if play_k is not None else 50) & 0xFF
                if args.auto_e:
                    c = ord("q")
                k = {ord("g"): "g", ord("r"): "r", ord("q"): "q", 27: "q"}.get(c)
                if c == ord("p"):  # your execution: hand video + robot, again
                    if play(once=True) == "q":
                        k = "q"
                if c == ord("v"):  # the generated demos, again
                    if play_mosaic(ui, gframes, args.fps, f"{len(done)} generated demonstrations",
                                   "Generated replay", [(msg, U.DIM)]) == "q":  # fmt: skip
                        k = "q"
                if c == ord("t") and sink.on:  # read back from Tiger Data + reproduced
                    if tiger_replay(ui, vui, pool, sink, d.name) == "q":
                        k = "q"
                if c == ord("o"):  # the take's folder (videos, scenes, summary) in Finder
                    subprocess.run(["open", str(d)], check=False)
            clicks["on"] = False
            if k == "g":
                rnd += 1
                continue
            if k == "q":
                quit_ = True
            break
    pool.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
