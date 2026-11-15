"""HO-Cap robot benchmark: one MuJoCo trial per human lift, scene reset before each lift.

For every lift (a GT object rising >= 5 cm), all objects are reset to their true poses 1.5 s before
it; the force-limited gripper follows the GT hand of the lifting side (touch.robot_sim.retarget)
and closes on a timing signal; success = sim lift >= 50% of the human's lift. Timing signals:
  true timing     GT hold (original thumb rule), debounced, +DELAY frames (camera-independent)
  model           model hold (argmax), debounced, +DELAY  (per camera)
  model, no table [GT diag]  model hold only while the GT hand is within 2 cm of an object
  model + object gate        model hold, latched by the RGB object gate (scripts/eval_gate.py)
  ... | pre-grasp            robot approaches along its own axis (touch.robot_sim.pregrasp)
  RGB only (pred hand + gate) | pre-grasp   NO GT hand: the model's own 3D hand path (predicted
      mesh placed by 2D keypoints + calibration, touch.handpath), its predicted contact vertices
      and the gated close signal. Objects still start at their true poses (known scene).
Model signals come from scripts/eval_hocap.py (live causal path). HO-Cap is test-only.

    python scripts/sim_hocap.py --preds runs/hocap/<run>/preds.npz [--workers 5]
writes runs/hocap/<run>/sim.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from eval_hocap import cam_group, gt_labels  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from touch import robot_sim as S  # noqa: E402
from touch.data import dexycb as D  # noqa: E402
from touch.data import hocap as H  # noqa: E402
from touch.handpath import world_hand  # noqa: E402
from touch.interact.object_gate import latch  # noqa: E402

DELAY = 3  # frames, the DexYCB-val close rule (docs/ROBOT_SIM.md)
PRE, POST = 45, 15  # trial window: 1.5 s before the lift starts, 0.5 s after it ends


def delay(x: np.ndarray) -> np.ndarray:
    return np.r_[np.zeros(DELAY, bool), x[:-DELAY]]


def scene(seq: H.Seq, f0: int, stiff: bool = False) -> S.Scene:
    """All objects at their frame-f0 poses, dropped onto the table (z = 0)."""
    R, t = H.object_poses(seq)
    assets, bodies = [], []
    for i, o in enumerate(seq.objects):
        pts = H.object_points(o, 4000)
        z = (pts @ R[i, f0].T + t[i, f0])[:, 2].min()
        p = t[i, f0] - [0, 0, z - 0.001]
        q = Rot.from_matrix(R[i, f0]).as_quat()[[3, 0, 1, 2]]
        mesh = (H.ROOT / "models" / o / "cleaned_mesh_2000.obj").resolve()
        assets.append(f'<mesh name="m{i}" file="{mesh}"/>')
        bodies.append(f'<body name="obj{i}" pos="{p[0]} {p[1]} {p[2]}" '
                      f'quat="{q[0]} {q[1]} {q[2]} {q[3]}"><freejoint/>'
                      f'<geom type="mesh" mesh="m{i}" density="300" friction="1.0 0.02 0.001" '
                      f'condim="4"/></body>')  # fmt: skip
    return S.Scene(S.scene_xml("".join(assets), "".join(bodies), stiff), list(seq.objects), 0)


def lifts(seq: H.Seq) -> list[tuple[int, int, int]]:
    """(object, window start, window end) per human lift >= 5 cm."""
    _, t = H.object_poses(seq)
    n = t.shape[1]
    out = []
    for k in range(len(seq.objects)):
        g = t[k, :, 2] - t[k, 0, 2]
        up = np.flatnonzero(g > 0.01)
        for r in np.split(up, np.flatnonzero(np.diff(up) > 1) + 1) if len(up) else []:
            if g[r].max() >= 0.05:
                out.append((k, max(0, r[0] - PRE), min(n, r[-1] + POST)))
    return out


def trial(seq, k, a0, b0, L, close, pre: bool = False) -> dict:
    ok, j, v = L["valid"], L["joints"], L["verts"]
    side_parts = L["parts"]
    pp, RR, W, _ = S.retarget(j, ok, close, v, L["vert"] == 1, side_parts)
    if pre:
        pp, RR = S.pregrasp(pp, RR, close)
    sl = slice(a0, b0)
    res = S.replay(scene(seq, a0), pp[sl], RR[sl], W[sl], ok[sl], close[sl])
    _, t = H.object_poses(seq)
    human = float((t[k, sl, 2] - t[k, a0, 2]).max())
    robot = float((res["obj"][:, k, 2] - res["obj"][0, k, 2]).max())
    return {
        "human_m": round(human, 3),
        "robot_m": round(robot, 3),
        "success": bool(robot >= 0.5 * human),
    }


def lifter(seq, a0, b0) -> str:
    """The GT hand holding during the lift window."""
    n_hold = {s: int((gt_labels(seq, s)["state_any2"][a0:b0] == 2).sum()) for s in seq.sides}
    return max(n_hold, key=n_hold.get)


def pred_hand(P, key: str, cam: str, left: bool) -> dict:
    """The model's own 3D hand in the table frame (no GT): predicted wrist-relative mesh placed
    by its fit to the MediaPipe keypoints + camera calibration (touch.handpath)."""
    wh = world_hand(P[f"{key}/joints"], P[f"{key}/verts"].astype(np.float32), P[f"{key}/lm"],
                    H.intrinsics(cam), H.cam_to_world(cam), left)  # fmt: skip
    return {"joints": wh["joints"], "verts": wh["verts"], "valid": wh["ok"],
            "vert": (P[f"{key}/vert"] >= 0.5).astype(np.int8)}  # fmt: skip


def work(job) -> list[dict]:
    preds_path, name, cam, gate_file = job
    seq = next(s for s in H.list_sequences(video_only=True) if s.name == name)
    P = np.load(preds_path)
    gp = Path(preds_path).parent / gate_file
    G = np.load(gp) if gp.exists() else None
    rows = []
    for k, a0, b0 in lifts(seq):
        side = lifter(seq, a0, b0)
        L = {**gt_labels(seq, side), "parts": D.vertex_parts(side)}
        key = f"{seq.subject}/{name}/{cam}/{side}"
        pres = P[f"{key}/present"]
        mhold = pres & (P[f"{key}/state"].argmax(1) == 2)
        near = L["part_dist"].min(1) <= 0.02
        sig = {"model": mhold, "model, no table [GT diag]": mhold & near}
        if G is not None:  # RGB object-proximity gate (touch.interact.object_gate), no GT
            ps = np.where(pres, P[f"{key}/state"].argmax(1), 0)
            sig["model + object gate"] = latch(ps, G[f"{key}/near"]) == 2
            sig["model + gate + fill"] = sig["model + object gate"]  # filled after debounce
        if cam == seq.serials[0]:  # camera-independent: once per lift
            sig = {"true timing": L["state"] == 2, **sig}
        base = {"seq": name, "cam": cam, "group": cam_group(cam), "object": seq.objects[k],
                "frames": [int(a0), int(b0)], "side": side}  # fmt: skip
        for v, raw in sig.items():
            for pre in (False, True):
                close = delay(S.debounce(raw))
                if v.endswith("fill"):
                    close = S.fill_gaps(close)
                r = trial(seq, k, a0, b0, L, close, pre)
                rows.append({**base, "variant": v + (" | pre-grasp" if pre else ""), **r})
        # everything from RGB: predicted 3D hand path + predicted contact + gated model signal
        if G is not None:
            Lp = {**L, **pred_hand(P, key, cam, side == "left")}
            sl = slice(a0, b0)
            okp = Lp["valid"][sl] & L["valid"][sl]
            palm = [0, 5, 9, 13, 17]
            err = np.linalg.norm(Lp["joints"][sl][okp][:, palm].mean(1)
                                 - L["joints"][sl][okp][:, palm].mean(1), axis=1)  # fmt: skip
            ext = {"path_err_cm": round(float(err.mean() * 100), 2) if len(err) else None,
                   "path_err_z_cm": round(float(np.abs(
                       Lp["joints"][sl][okp][:, palm].mean(1)[:, 2]
                       - L["joints"][sl][okp][:, palm].mean(1)[:, 2]).mean() * 100), 2)
                   if len(err) else None}  # fmt: skip
            close = S.fill_gaps(delay(S.debounce(sig["model + object gate"])))
            if Lp["valid"][sl].sum() >= 5:
                r = trial(seq, k, a0, b0, Lp, close, True)
            else:
                r = {"human_m": None, "robot_m": None, "success": False}
            rows.append({**base, "variant": "RGB only (pred hand + gate) | pre-grasp", **r, **ext})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--gate", default="gate_hull10_arm_skin_fast.npz",
                    help="eval_gate.py output next to preds")  # fmt: skip
    ap.add_argument("--out", default="sim.json")
    args = ap.parse_args()
    split = "test" if args.preds.parent.parent.name == "hocap" else "val"
    seqs = H.list_sequences(video_only=True, split=split)
    P0 = np.load(args.preds)
    jobs = [
        (str(args.preds), s.name, c, args.gate)
        for s in seqs
        for c in s.serials
        if any(f"{s.subject}/{s.name}/{c}/{sd}/present" in P0.files for sd in s.sides)
    ]
    t0, rows = time.time(), []
    with get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs)):
            rows += r
            print(f"{i + 1}/{len(jobs)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    summary: dict = {}
    for g in ("all", "top-down", "oblique", "low"):
        for v in sorted({r["variant"] for r in rows}):
            sel = [r for r in rows if r["variant"] == v and (g == "all" or r["group"] == g)]
            if v.startswith("true timing") and g != "all":
                continue
            if sel:
                rate = round(float(np.mean([r["success"] for r in sel])), 3)
                summary.setdefault(g, {})[v] = {"trials": len(sel), "success": rate}
                pe = [r["path_err_cm"] for r in sel if r.get("path_err_cm") is not None]
                if pe:
                    pz = [r["path_err_z_cm"] for r in sel if r.get("path_err_z_cm") is not None]
                    summary[g][v]["path_err_cm"] = round(float(np.mean(pe)), 2)
                    summary[g][v]["path_err_z_cm"] = round(float(np.mean(pz)), 2)
    out = args.preds.parent / args.out
    out.write_text(json.dumps({"summary": summary, "trials": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
