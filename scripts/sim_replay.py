"""Robot replay of DexYCB val (subject 7) grasps in MuJoCo: does a parallel gripper that copies the
human lift the object the human lifted? Val only; the test subjects are never loaded.

Conditions (each sequence where the human lifted the grasped object >= 5 cm):
  A  oracle      GT hand trajectory, GT contact vertices (grasp axis), GT close timing
  B  model-time  GT trajectory + GT contact vertices, close timing from the grasp model (per camera)
  B2 model-time+contact: as B but the grasp axis from the model's predicted vertex contact
  C  shift_k     A with the GT close / open timing shifted by k frames (timing sensitivity)
  D  camera-only everything from one camera: predicted mesh + metric translation fitted to the
                 MediaPipe 2D landmarks (camera intrinsics), predicted contact and timing
Also reports the metric wrist-position error of the translation fit (D's trajectory) vs GT.

    python scripts/sim_replay.py --preds runs/<run>/val_preds.npz [--render 6]
writes results/robot_sim/<run>/sim_val.json (+ demo mp4s with --render).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch import robot_sim as S  # noqa: E402
from touch.data import dexycb as D  # noqa: E402

PROC = Path("data/dexycb_proc")
SHIFTS = [-10, -5, -3, 3, 5, 10]


def shift(close: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(close)
    if k >= 0:
        out[k:] = close[: len(close) - k]
    else:
        out[:k] = close[-k:]
    return out


def gt_hand(seq, tag):
    pm = np.load(seq.root / "pose.npz")["pose_m"][:, 0]
    ok = np.abs(pm).sum(1) > 0
    v = np.full((len(pm), 778, 3), np.nan)
    j = np.full((len(pm), 21, 3), np.nan)
    vv, jj = D.mano_forward(pm[ok], D.mano_betas(seq), seq.side)
    v[ok], j[ok] = S.master_to_table(vv, tag), S.master_to_table(jj, tag)
    return v, j, ok


def predicted_hand(seq, cam, P, ex, tag, flip: bool):
    """Camera-only hand in the table frame: model mesh (root-relative) + translation fitted to the
    MediaPipe 2D landmarks with the camera intrinsics, smoothed offline. Also the per-frame wrist
    error vs GT (camera frame, metres)."""
    key = f"{seq.name}/{cam}"
    present = P[f"{key}/present"]
    J = P[f"{key}/joints"].astype(np.float64)
    V = P[f"{key}/verts"].astype(np.float64)
    if flip:  # the model works on mirrored (right-hand) crops
        J[..., 0] *= -1
        V[..., 0] *= -1
    with np.load(PROC / seq.subject / seq.name / f"{cam}.npz") as z:
        uv = z["lm_px"][..., :2].astype(np.float64)
    K = D.intrinsics(cam)
    n = len(present)
    T = np.zeros((n, 3))
    for t in np.flatnonzero(present):
        T[t] = S.fit_translation(J[t], uv[t], K)
    Ts = S.smooth(T, present, 5)
    err = np.full(n, np.nan)
    err_s = np.full(n, np.nan)
    for t in np.flatnonzero(present):
        lab = seq.labels(cam, t)
        if np.abs(lab["pose_m"]).sum() == 0:
            continue
        g = lab["joint_3d"][0, 0]
        err[t] = np.linalg.norm(T[t] - g)
        err_s[t] = np.linalg.norm(Ts[t] - g)
    Tc = ex[cam]
    jt = np.full((n, 21, 3), np.nan)
    vt = np.full((n, 778, 3), np.nan)
    for t in np.flatnonzero(present):
        jt[t] = S.master_to_table(S.cam_to_master(J[t] + Ts[t], Tc), tag)
        vt[t] = S.master_to_table(S.cam_to_master(V[t] + Ts[t], Tc), tag)
    return vt, jt, present, err, err_s


GATE = {"on": False, "R_obj": None}  # set in main(): robot-side "object between the pads" gate
# robot v2 (set in main()): object-centred grasp on the scene objects (robot_sim.object_refine),
# safe pre-grasp approach, opening = object width + 4 cm, stiff contacts, fingers close at
# 0.2 m/s with the arm waiting for them (docs/ROBOT_SIM.md, 2026-11-12)
ROBOT = {"v2": False, "R_obj": None}


def object_points_world(scene, p_obj, R_obj, n_pts: int = 2000) -> np.ndarray:
    """Every object's surface points per frame [N, n_obj * n_pts, 3] (table frame)."""
    pts = [D.object_points(nm, n_pts)[:n_pts] for nm in scene.names]
    return np.concatenate(
        [
            np.einsum("nij,pj->npi", R_obj[:, k], pts[k]) + p_obj[:, k, None]
            for k in range(len(pts))
        ],
        1,
    )


def run(scene, p_obj, gi, joints, ok, close, verts, contact, parts, render=None):
    if not ok.any():
        return None, None
    pp, RR, W, how = S.retarget(joints, ok, close, verts, contact, parts)
    if ROBOT["v2"]:
        pts = [D.object_points(nm, 4000) for nm in scene.names]
        R_obj = ROBOT["R_obj"]

        def objects_at(t):  # demo points at frame t + map to the scene (objects at frame 0)
            out = []
            for k, q in enumerate(pts):
                T0, Tt = np.eye(4), np.eye(4)
                z0 = (q @ R_obj[0, k].T + p_obj[0, k])[:, 2].min() - 0.001  # as build_scene
                T0[:3, :3], T0[:3, 3] = R_obj[0, k], p_obj[0, k] - [0, 0, z0]
                Tt[:3, :3], Tt[:3, 3] = R_obj[t, k], p_obj[t, k]
                out.append((q @ R_obj[t, k].T + p_obj[t, k], T0 @ np.linalg.inv(Tt)))
            return out

        pp, RR, W, how = S.object_refine(pp, RR, W, close, objects_at)
        pp, RR = S.pregrasp(pp, RR, close, floor=True)
        out = S.replay(scene, pp, RR, W, ok, close, render, **S.V2)
        return S.lift_outcome(p_obj[:, gi], out["obj"][:, gi]), (how, out)
    if GATE["on"]:
        ready = S.object_between_pads(pp, RR, W, object_points_world(scene, p_obj, GATE["R_obj"]))
        close = S.gate_close(close, ready)
    out = S.replay(scene, pp, RR, W, ok, close, render)
    return S.lift_outcome(p_obj[:, gi], out["obj"][:, gi]), (how, out)


def summarise(rows: list[dict]) -> dict:
    rows = [r for r in rows if r is not None]
    if not rows:
        return {"n": 0}
    held = [r["held_end"] for r in rows if r["held_end"] is not None]
    return {
        "n": len(rows),
        "lift_success": round(float(np.mean([r["success"] for r in rows])), 3),
        "held_to_end": round(float(np.mean(held)), 3) if held else None,
        "median_obj_err_cm": round(float(np.median([r["pos_err_peak_m"] for r in rows])) * 100, 1),
    }


def main() -> None:
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--render", type=int, default=0, help="render camera-only demos for N seqs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--on", type=int, default=2, help="close after N consecutive contact frames")
    ap.add_argument(
        "--delay",
        type=int,
        default=3,
        help="close N frames after the contact signal "
        "(GT and model alike; val oracle: 0 -> 0.33, 3 -> 0.48)",
    )
    ap.add_argument("--conditions", default="A,B,B2,D,shift", help="subset to run")
    ap.add_argument("--robot", default="v1", choices=["v1", "v2"],
                    help="v2: object-centred grasp + safe approach + fixed physics")  # fmt: skip
    ap.add_argument("--tag", default="", help="suffix for the output json")
    ap.add_argument(
        "--gate",
        action="store_true",
        help="robot-side gate: start closing only with an object between the pads",
    )
    args = ap.parse_args()
    cond = set(args.conditions.split(","))
    run_name = args.preds.parent.name
    out_dir = Path("results/robot_sim") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    P = np.load(args.preds)
    from touch.data.grasp_store import load_grasp_tracks

    tracks, _, _ = load_grasp_tracks("val")
    flip = {(t.seq, t.cam): bool(t.flip) for t in tracks}
    seqs = [s for s in D.list_sequences() if s.split == "val"]
    assert seqs and all(s.split == "val" for s in seqs)
    if args.limit:
        seqs = seqs[: args.limit]
    res = {k: [] for k in ["A", "B", "B2", "D"] + [f"shift{k:+d}" for k in SHIFTS]}
    per_seq, wrist, wrist_s = [], [], []
    n_render, t0 = 0, time.time()
    for si, seq in enumerate(seqs):
        ex = S.extrinsics(seq)
        tag = ex["apriltag"]
        p_obj, R_obj = S.object_poses_table(seq, tag)
        GATE["on"], GATE["R_obj"] = args.gate, R_obj
        ROBOT["v2"], ROBOT["R_obj"] = args.robot == "v2", R_obj
        gi = int(seq.meta["ycb_grasp_ind"])
        gain = p_obj[:, gi, 2] - p_obj[0, gi, 2]
        C = np.load(PROC / seq.subject / seq.name / "contact.npz")
        parts = D.vertex_parts(seq.side)
        v, j, ok = gt_hand(seq, tag)
        gt_close = shift(S.debounce(S.gt_grasp_signal(C["state"])), args.delay)
        gt_contact = C["vert"] == 1
        scene = S.build_scene(seq, p_obj[0], R_obj[0], stiff=args.robot == "v2")
        lifted = gain.max() >= 0.05
        rec = {"seq": seq.name, "object": scene.names[gi], "side": seq.side, "lifted": bool(lifted)}
        for cam in D.CAMS:  # translation fit error: every sequence
            key = f"{seq.name}/{cam}"
            if f"{key}/present" not in P.files:
                continue
            if "D" in cond:
                fl = flip[(seq.name, cam)]
                vt, jt, pres, err, err_s = predicted_hand(seq, cam, P, ex, tag, fl)
                wrist.append(err[np.isfinite(err)])
                wrist_s.append(err_s[np.isfinite(err_s)])
            else:
                pres = P[f"{key}/present"]
            if not lifted:
                continue
            st = P[f"{key}/state"].astype(np.float32)
            mclose = shift(S.debounce(S.grasp_signal(st, pres), on=args.on), args.delay)
            mcontact = P[f"{key}/vert"] >= 128
            if "B" in cond:
                r, _ = run(scene, p_obj, gi, j, ok, mclose, v, gt_contact, parts)
                res["B"].append(r)
            if "B2" in cond:
                r, _ = run(scene, p_obj, gi, j, ok, mclose, v, mcontact, parts)
                res["B2"].append(r)
            if "D" not in cond:
                continue
            render = None
            if args.render and n_render < args.render and cam == D.CAMS[1]:
                render = Renderer(seq, cam, p_obj[0, gi], mujoco)
            r, extra = run(scene, p_obj, gi, jt, pres, mclose, vt, mcontact, parts, render)
            res["D"].append(r)
            rec[f"D_{cam}"] = r
            if render is not None and extra is not None:
                render.write(out_dir / f"camera_only_{seq.name}_{scene.names[gi]}.mp4", st, mclose)
                n_render += 1
        if lifted and "A" in cond:
            r, _ = run(scene, p_obj, gi, j, ok, gt_close, v, gt_contact, parts)
            res["A"].append(r)
            rec["A"] = r
        if lifted and "shift" in cond:
            for k in SHIFTS:
                r, _ = run(scene, p_obj, gi, j, ok, shift(gt_close, k), v, gt_contact, parts)
                res[f"shift{k:+d}"].append(r)
        per_seq.append(rec)
        print(f"{si + 1}/{len(seqs)} {seq.name} {scene.names[gi]} lifted={lifted} "
              f"A={rec.get('A')} | {(time.time() - t0) / 60:.1f} min", flush=True)  # fmt: skip
    w = np.concatenate(wrist or [np.zeros(1)]) * 1000
    ws = np.concatenate(wrist_s or [np.zeros(1)]) * 1000
    out = {
        "preds": str(args.preds),
        "close_rule": {"on": args.on, "delay": args.delay, "gate": args.gate},
        "robot": args.robot,
        "conditions": {k: summarise(v) for k, v in res.items()},
        "wrist_fit_mm": {"frames": int(len(w)), "median": round(float(np.median(w)), 1),
                         "mean": round(float(w.mean()), 1),
                         "median_smoothed": round(float(np.median(ws)), 1),
                         "p90_smoothed": round(float(np.percentile(ws, 90)), 1)},
        "notes": "lift_success: sim object rose >= 50% of the human's peak lift at that frame. "
                 "Gripper: generic 15 cm stroke, 40 N/finger, force-limited drive (40 N, 3 N m); "
                 "objects density 300 kg/m3 (assumed).",
        "per_seq": per_seq,
    }  # fmt: skip
    (out_dir / f"sim_val{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("conditions", "wrist_fit_mm")}, indent=1))


class Renderer:
    """Side-by-side demo: the human camera frame (with the model's state) | the robot sim."""

    def __init__(self, seq, cam, look_at, mujoco):
        self.seq, self.cam, self.look, self.mj = seq, cam, look_at, mujoco
        self.r = None
        self.sim_frames = []

    def __call__(self, m, d, t):
        mj = self.mj
        if self.r is None:
            self.r = mj.Renderer(m, 480, 640)
            self.c = mj.MjvCamera()
        self.c.lookat[:] = self.look + [0, 0, 0.1]
        self.c.distance, self.c.azimuth, self.c.elevation = 0.8, 200, -25
        self.r.update_scene(d, self.c)
        self.sim_frames.append(self.r.render()[..., ::-1].copy())
        return None

    def write(self, path: Path, state: np.ndarray, close: np.ndarray):
        names = ["NONE", "TOUCH", "HOLD"]
        cols = [(200, 200, 200), (0, 200, 255), (0, 0, 255)]
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 480))
        for t, simf in enumerate(self.sim_frames):
            img = cv2.imread(str(self.seq.image(self.cam, t)))
            k = int(state[t].argmax())
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"human: {names[k]}", (16, 40), font, 1.0, cols[k], 2)
            g = "CLOSE" if close[t] else "OPEN"
            cv2.putText(simf, f"robot gripper: {g}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2)  # fmt: skip
            cv2.putText(simf, "camera-only replay (sim)", (16, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (220, 220, 220), 1)  # fmt: skip
            vw.write(np.hstack([img, simf]))
        vw.release()


if __name__ == "__main__":
    main()
