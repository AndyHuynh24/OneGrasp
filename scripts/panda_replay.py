"""OFFLINE Franka Panda replay of DexYCB grasps (val subject only): the arm's gripper follows the
human hand's path as closely as its joint limits allow, grasps and lifts the object.

Per sequence:
  1. target gripper poses per frame from the GT 3D hand (robot v2 pipeline of sim_replay:
     retarget -> object_refine -> pregrasp), close / open from the GT hand state
  2. Panda base placed on the table where the path is most reachable (searched)
  3. IK per frame (damped least squares, joint limits clamped, warm start, null-space pull to
     a neutral posture, the gripper's 180 deg symmetry used): best effort where the path is out
     of reach; per-frame position / orientation error reported
  4. re-timed so no joint exceeds 80% of the Panda's velocity limits (offline: the robot may be
     slower than the human), pauses while the fingers close / open
  5. simulated in MuJoCo (Menagerie panda.xml, position-controlled joints, gravity compensated)
     and rendered: DexYCB camera | sim from the same camera | sim overview

    python scripts/panda_replay.py [--n 4] [--fps 15]
Writes results/robot_sim/panda/<seq>_<object>.mp4, panda_dexycb.mp4 (all) and summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
import sim_replay as SR  # noqa: E402
from render_sim_demo import camera_xml, label  # noqa: E402

from touch import robot_sim as S  # noqa: E402
from touch.data import dexycb as D  # noqa: E402
from touch.panda import (  # noqa: E402
    MAX_OPEN,
    PAUSE,
    PandaIK,
    held_away,
    panda_mjcf,
    retime,
    retime_realtime,
    simulate,
)

OBJECTS = ["025_mug", "011_banana", "006_mustard_bottle", "036_wood_block", "035_power_drill",
           "037_scissors", "024_bowl", "007_tuna_fish_can"]  # fmt: skip


# ------------------------------------------------------------------ scene
def panda_scene(seq, p0, R0, base_pos, base_yaw, cam_xml: str, look: np.ndarray) -> str:
    """Menagerie Panda (base at base_pos on the table, yaw base_yaw) + table + the sequence's
    YCB objects (as robot_sim.build_scene) + cameras."""
    models = sorted(p for p in (D.ROOT / "models").iterdir() if p.is_dir())
    names = [models[i - 1].name for i in seq.meta["ycb_ids"]]
    assets, bodies = [], []
    for i, nm in enumerate(names):
        pts = D.object_points(nm, 4000)
        z = (pts @ R0[i].T + p0[i])[:, 2].min()
        pos = p0[i] - [0, 0, z - 0.001]
        q = Rot.from_matrix(R0[i]).as_quat()[[3, 0, 1, 2]]
        col = "0.85 0.3 0.25 1" if i == seq.meta["ycb_grasp_ind"] else "0.55 0.6 0.7 1"
        f = (D.ROOT / "models" / nm).resolve() / "textured_simple.obj"
        assets.append(f'<mesh name="m{i}" file="{f}"/>')
        bodies.append(
            f'<body name="obj{i}" pos="{pos[0]} {pos[1]} {pos[2]}" '
            f'quat="{q[0]} {q[1]} {q[2]} {q[3]}"><freejoint/><geom type="mesh" mesh="m{i}" '
            f'density="300" rgba="{col}" friction="1.0 0.02 0.001" condim="4" '
            f'solref="0.004 1" solimp="0.95 0.99 0.001"/></body>'
        )
    world = (
        '<light pos="0.5 0.3 1.6" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>'
        '<light pos="-0.3 -0.5 1.2" dir="0.3 0.5 -1" diffuse="0.4 0.4 0.4"/>'
        '<geom name="table" type="plane" pos="0 0 0" size="2 2 0.01" rgba="0.32 0.3 0.28 1" '
        'friction="0.8 0.02 0.001"/>'
        f'<body name="look" pos="{look[0]} {look[1]} {look[2]}"/>'
        '<camera name="overview" mode="targetbody" target="look" '
        f'pos="{look[0] + 0.9} {look[1] - 0.9} {look[2] + 0.75}"/>' + cam_xml + "".join(bodies)
    )
    return panda_mjcf(base_pos, base_yaw, world, "".join(assets))


# ------------------------------------------------------------------ base placement
def place_base(seq, p0, R0, P, Rs, close, objs_xy, cam_xy):
    """Search base positions around the path: radius 0.45-0.65 m, every 30 deg, facing the
    path's centre, on the far side from the camera; reject spots within 0.25 m of an
    object; score = mean IK error on every 4th
    frame (grasp frames x3)."""
    import mujoco

    c = P[:, :2].mean(0)
    look = np.array([c[0], c[1], 0.1])
    best = None
    frames = np.arange(0, len(P), 4)
    for r in (0.45, 0.55, 0.65):
        for a in np.radians(np.arange(0, 360, 30)):
            b = c + r * np.array([math.cos(a), math.sin(a)])
            if np.dot(b - c, cam_xy - c) > 0:  # camera side: the arm would block the view
                continue
            if np.min(np.linalg.norm(objs_xy - b, axis=1)) < 0.25:
                continue
            yaw = math.atan2(c[1] - b[1], c[0] - b[0])
            m = mujoco.MjModel.from_xml_string(panda_scene(seq, p0, R0, b, yaw, "", look))
            _, ep, er = PandaIK(m).track(P, Rs, frames, iters=40)
            w = np.where(close[frames], 3.0, 1.0)
            s = float(((ep + 0.05 * er) * w).sum() / w.sum())
            if best is None or s < best[0]:
                best = (s, b, yaw)
    return best[1], best[2], look


# ------------------------------------------------------------------ timing + simulation
# ------------------------------------------------------------------ plan / rollout / search
def plan(seq, cam: str) -> tuple[dict | None, str]:
    """Targets (robot v2 retargeting of the GT hand), base placement and scene for one
    sequence -> (plan dict, "") or (None, reason)."""
    import mujoco

    tag = S.extrinsics(seq)["apriltag"]
    p_obj, R_obj = S.object_poses_table(seq, tag)
    gi = int(seq.meta["ycb_grasp_ind"])
    models = sorted(p for p in (D.ROOT / "models").iterdir() if p.is_dir())
    names = [models[i - 1].name for i in seq.meta["ycb_ids"]]
    obj = names[gi]
    if (p_obj[:, gi, 2] - p_obj[0, gi, 2]).max() < 0.05:
        return None, "not lifted"
    C = np.load(SR.PROC / seq.subject / seq.name / "contact.npz")
    parts = D.vertex_parts(seq.side)
    v, j, ok = SR.gt_hand(seq, tag)
    close = S.debounce(S.gt_grasp_signal(C["state"]))
    if not close.any():
        return None, "no grasp"
    pp, RR, W, _ = S.retarget(j, ok, close, v, C["vert"] == 1, parts)
    pts = [D.object_points(nm, 4000) for nm in names]

    def objects_at(t):
        out = []
        for k, q in enumerate(pts):
            T0, Tt = np.eye(4), np.eye(4)
            z0 = (q @ R_obj[0, k].T + p_obj[0, k])[:, 2].min() - 0.001
            T0[:3, :3], T0[:3, 3] = R_obj[0, k], p_obj[0, k] - [0, 0, z0]
            Tt[:3, :3], Tt[:3, 3] = R_obj[t, k], p_obj[t, k]
            out.append((q @ R_obj[t, k].T + p_obj[t, k], T0 @ np.linalg.inv(Tt)))
        return out

    pp, RR, W, _ = S.object_refine(pp, RR, W, close, objects_at)
    pp, RR = S.pregrasp(pp, RR, close, floor=True)
    P, Rs = S._interp_poses(pp, RR, ok)
    W = np.interp(np.arange(len(W)), np.flatnonzero(ok), W[ok])
    wg = float(np.median(W[close]))
    if wg > MAX_OPEN:
        return None, f"grasp {wg * 100:.1f} cm > Panda stroke {MAX_OPEN * 100:.0f} cm"
    cam_xy = S.master_to_table(S.extrinsics(seq)[cam][:, 3], tag)[:2]
    base, yaw, look = place_base(seq, p_obj[0], R_obj[0], P, Rs, close, p_obj[0, :, :2], cam_xy)
    xml = panda_scene(seq, p_obj[0], R_obj[0], base, yaw, camera_xml(seq, cam, tag), look)
    mujoco.MjModel.from_xml_string(xml)  # fail early
    pl = {"seq": seq.name, "object": obj, "gi": gi, "n_obj": len(names), "P": P, "Rs": Rs,
          "close": close, "W": W, "base": base, "xml": xml, "gt_obj": p_obj[:, gi]}  # fmt: skip
    return pl, ""


# grasp correction theta: offset in the grasp frame (x, y = closing axis, z = approach; m),
# rotation about the approach axis and about the closing axis (rad), close timing shift (frames)
THETA_STD = np.array([0.012, 0.012, 0.015, 0.25, 0.15, 2.0])
THETA_LIM = np.array([0.04, 0.04, 0.04, 0.6, 0.35, 6.0])


def apply_theta(pl: dict, theta: np.ndarray):
    """Corrected targets: the grasp-frame offset / rotation fades in over the approach (from
    27 frames before the close), stays through the hold, fades out over 10 frames after the
    release; the close / open command shifted by round(theta[5]) frames."""
    P, Rs, close = pl["P"].copy(), pl["Rs"].copy(), pl["close"]
    n = len(P)
    k = int(round(theta[5]))
    c = np.zeros(n, bool)
    src = np.clip(np.arange(n) - k, 0, n - 1)
    c[:] = close[src]
    a = int(np.flatnonzero(close)[0])
    e = int(np.flatnonzero(close)[-1])
    t = np.arange(n)
    w = np.clip((t - (a - 27)) / 15, 0, 1) * np.clip(1 - (t - e) / 10, 0, 1)
    Ra = pl["Rs"][a]
    dR = Rot.from_rotvec(Ra[:, 2] * theta[3]) * Rot.from_rotvec(Ra[:, 1] * theta[4])
    off = Ra @ theta[:3]
    for i in np.flatnonzero(w > 0):
        P[i] = P[i] + w[i] * off
        Rs[i] = (Rot.from_rotvec(w[i] * dR.as_rotvec()) * Rot.from_matrix(Rs[i])).as_matrix()
    return P, Rs, c


def reward(gt_z: np.ndarray, sim_z: np.ndarray, theta: np.ndarray) -> float:
    """Lift accuracy: mean over frames where the human lifted the object >= 2 cm of
    max(0, 1 - |sim lift - human lift| / human lift) (too high is as wrong as too low), minus a
    small cost for leaving the human's path."""
    g = gt_z - gt_z[0]
    s = sim_z - sim_z[0]
    m = g >= 0.02
    follow = float(np.clip(1 - np.abs(s[m] - g[m]) / g[m], 0, 1).mean()) if m.any() else 0.0
    dev = np.linalg.norm(theta[:3]) / 0.04 + np.abs(theta[3:5]).sum() / 0.6
    return follow - 0.05 * dev


def rollout(pl: dict, theta: np.ndarray, render=None, full: bool = False) -> dict:
    """full=False: simulate only until 10 frames after the lift's peak (fast; misses the
    carry's end, the set-down and the release)."""
    import mujoco

    P, Rs, close = apply_theta(pl, theta)
    m = mujoco.MjModel.from_xml_string(pl.get("ik_xml", pl["xml"]))  # IK may use margins
    ik = PandaIK(m)
    # carry_free: only the final run that follows the hand (panda_demo.render_rollout); the
    # grasp search keeps the start spot avoided (switching it there changed the chosen grasp
    # and the scaled scenes tipped the can, Andy 2026-11-15)
    cf = held_away(P, close) if pl.get("carry_free") else None
    Q, ep, er = ik.track(P, Rs, np.arange(len(P)), avoid=pl.get("avoid", 0.0), carried_from=cf)
    if pl.get("pour_lean") is not None:  # the gripper is symmetric: the same grasp turned
        # 180 deg about its approach axis can leave the last wrist joint room for a big pour
        Rf = Rs @ np.diag([-1.0, -1.0, 1.0])
        Qf, epf, erf = ik.track(P, Rf, np.arange(len(P)), avoid=pl.get("avoid", 0.0),
                                carried_from=cf)  # fmt: skip
        if (epf + 0.05 * erf).sum() < (ep + 0.05 * er).sum():
            Rs, Q, ep, er = Rf, Qf, epf, erf
    if pl.get("realtime", True):  # one robot step per hand frame (Andy 2026-11-14: in sync)
        steps = retime_realtime(Q, close)
    else:
        steps = retime(Q, close, pl.get("pause", PAUSE))  # pause: steps waited at close / open
    g = pl["gt_obj"][:, 2] - pl["gt_obj"][0, 2]
    until = None if full else min(len(P) - 1, int(np.argmax(g)) + 10)
    sim = simulate(pl["xml"], steps, close, pl["W"], pl["gi"], pl["n_obj"], render, until)
    hs = sim["human"]
    last = np.array([np.flatnonzero(hs == h)[-1] for h in range(hs[-1] + 1)])
    z = sim["obj"][last, pl["gi"], 2]
    r = reward(pl["gt_obj"][: len(z), 2], z, theta)
    return {"reward": r, "sim": sim, "Q": Q, "ep": ep, "er": er, "close": close,
            "steps": len(steps), "last": last}  # fmt: skip


def _score(args):
    pl, theta = args
    return rollout(pl, theta)["reward"]


def cem(pl: dict, pool, iters: int = 8, pop: int = 16, n_elite: int = 4, good: float = 0.9):
    """Cross-entropy method over theta (episodic policy search, reward = lift following).
    Starts from theta = 0 (the human's grasp); stops early once a rollout reaches `good`."""
    rng = np.random.default_rng(0)
    mu, sd = np.zeros(6), THETA_STD.copy()
    best_t, best_r = np.zeros(6), _score((pl, np.zeros(6)))
    hist = [round(best_r, 3)]
    for _ in range(iters):
        if best_r >= good:
            break
        th = np.clip(mu + sd * rng.standard_normal((pop, 6)), -THETA_LIM, THETA_LIM)
        rs = np.array(pool.map(_score, [(pl, t) for t in th]))
        el = th[np.argsort(rs)[-n_elite:]]
        mu, sd = el.mean(0), np.maximum(el.std(0), 0.15 * THETA_STD)
        if rs.max() > best_r:
            best_r, best_t = float(rs.max()), th[int(rs.argmax())]
        hist.append(round(float(rs.max()), 3))
    return best_t, best_r, hist


HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
              (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
              (18, 19), (19, 20)]  # fmt: skip


def draw_hand(img, uv, closed: bool) -> None:
    """DexYCB 2D joints [21, 2] (-1 = no hand) -> skeleton drawn on img; red while grasping."""
    if (uv < 0).all():
        return
    col = (0, 0, 255) if closed else (0, 255, 0)
    pts = [tuple(int(v) for v in p) for p in uv]
    for i, j in HAND_EDGES:
        cv2.line(img, pts[i], pts[j], col, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 3, (255, 255, 255), -1, cv2.LINE_AA)


# ------------------------------------------------------------------ main
def write_video(clip: Path, seq, cam, pl, ro, res, fps: int, tag: str) -> None:
    raw = clip.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 480))
    ok_txt = ("LIFTED", (80, 220, 80)) if res.get("success") else ("NOT LIFTED", (60, 60, 255))
    sim, close = ro["sim"], ro["close"]
    slow = round(ro["steps"] / len(pl["P"]), 2)
    for k, (a, b) in enumerate(sim["frames"]):
        h = int(sim["human"][k])
        img = cv2.imread(str(seq.image(cam, h)))
        draw_hand(img, seq.labels(cam, h)["joint_2d"][0], pl["close"][h])
        label(img, "human hand", (16, 34), 0.8)
        label(img, pl["object"][4:].replace("_", " "), (16, 462), 0.7)
        label(a, f"Panda ({tag})", (16, 34), 0.8)
        g = "CLOSED" if close[h] else "OPEN"
        label(a, f"gripper: {g}", (16, 70), 0.8, (0, 0, 255) if close[h] else (255, 255, 255))
        e = np.linalg.norm(sim["tcp"][k] - pl["P"][h]) * 1000
        col = (80, 220, 80) if e < 20 else (0, 200, 255) if e < 50 else (60, 60, 255)
        label(a, f"off hand path {e:4.0f} mm", (16, 104), 0.6, col)
        label(a, f"robot time x{slow}", (16, 462), 0.6)
        inset = cv2.resize(b, (213, 160))  # overview, bottom right
        cv2.rectangle(inset, (0, 0), (212, 159), (255, 255, 255), 1)
        a[480 - 160 - 8 : 480 - 8, 640 - 213 - 8 : 640 - 8] = inset
        frame = np.hstack([img, a])
        vw.write(frame)
    for _ in range(fps):
        f2 = frame.copy()
        label(f2, f"robot: {ok_txt[0]}", (680, 260), 1.4, ok_txt[1], 3)
        vw.write(f2)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(clip)], check=True)  # fmt: skip
    raw.unlink()


def main() -> None:
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="sequences (one per object)")
    ap.add_argument("--cam", type=int, default=1, help="index into dexycb.CAMS")
    ap.add_argument("--fps", type=int, default=15, help="video fps (15 = half speed)")
    ap.add_argument("--search", action="store_true", help="CEM grasp correction (small RL)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("results/robot_sim/panda"))
    args = ap.parse_args()
    cam = D.CAMS[args.cam]
    args.out.mkdir(parents=True, exist_ok=True)
    seqs = [s for s in D.list_sequences() if s.split == "val"]
    assert seqs and all(s.split == "val" for s in seqs)
    pool = get_context("spawn").Pool(args.workers) if args.search else None
    done, summary, clips = set(), [], []
    models = sorted(p for p in (D.ROOT / "models").iterdir() if p.is_dir())
    for seq in seqs:
        if len(done) >= args.n:
            break
        obj = models[seq.meta["ycb_ids"][int(seq.meta["ycb_grasp_ind"])] - 1].name
        if obj not in OBJECTS or obj in done:
            continue
        pl, why = plan(seq, cam)
        if pl is None:
            print(f"{seq.name} {obj}: skip ({why})", flush=True)
            continue
        theta, hist = np.zeros(6), []
        if args.search:
            theta, _, hist = cem(pl, pool)
        rnd = {}

        def render(m, d, rnd=rnd):
            if "a" not in rnd:
                rnd["a"] = mujoco.Renderer(m, 480, 640)
            rnd["a"].update_scene(d, camera="human")
            a = rnd["a"].render()[..., ::-1].copy()
            rnd["a"].update_scene(d, camera="overview")
            b = rnd["a"].render()[..., ::-1].copy()
            return a, b

        ro = rollout(pl, theta, render, full=True)
        sim, ep = ro["sim"], ro["ep"]
        res = S.lift_outcome(pl["gt_obj"], sim["obj"][ro["last"], pl["gi"]]) or {}
        trk = np.linalg.norm(sim["tcp"] - pl["P"][sim["human"]], axis=1)
        r = {"seq": seq.name, "object": obj, "frames": len(pl["P"]), "robot_steps": ro["steps"],
             "base_xy": pl["base"].round(3).tolist(), "reward": round(ro["reward"], 3),
             "theta": {"offset_mm": (theta[:3] * 1000).round(1).tolist(),
                       "rot_deg": np.degrees(theta[3:5]).round(1).tolist(),
                       "close_shift": int(round(theta[5]))},
             "cem_best_per_iter": hist,
             "ik_err_mm_median": round(float(np.median(ep)) * 1000, 1),
             "sim_track_err_mm": {"median": round(float(np.median(trk)) * 1000, 1),
                                  "max": round(float(trk.max()) * 1000, 1)},
             **res}  # fmt: skip
        summary.append(r)
        print(json.dumps(r), flush=True)
        tag = "small-RL grasp" if args.search else "copying the hand"
        clip = args.out / f"{seq.name}_{obj}{'_rl' if args.search else ''}.mp4"
        write_video(clip, seq, cam, pl, ro, res, args.fps, tag)
        clips.append(clip)
        done.add(obj)
    if pool is not None:
        pool.close()
    sfx = "_rl" if args.search else ""
    (args.out / f"summary{sfx}.json").write_text(json.dumps(summary, indent=1))
    if clips:
        lst = args.out / "clips.txt"
        lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(lst), "-c", "copy", str(args.out / f"panda_dexycb{sfx}.mp4")],
                       check=True)  # fmt: skip
        lst.unlink()
    n_ok = sum(bool(r.get("success")) for r in summary)
    print("wrote", args.out, f"{n_ok}/{len(summary)} lifted")


if __name__ == "__main__":
    main()
