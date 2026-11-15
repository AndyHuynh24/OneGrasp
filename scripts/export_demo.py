"""Export recorded human demos as robot data (docs/ROBOT_EXPORT.md), from the model's predictions.

HO-Cap mode: every lift in the HO-Cap test sequences, seen by one camera, using only what the
RGB pipeline produced (scripts/eval_hocap.py preds + scripts/eval_gate.py gate): predicted 3D hand
(touch.handpath), gated contact state, close command, retargeted gripper, keyframes. The GT
object trajectories are added under "gt_" keys as a reference (and to build the MuJoCo scene).

    python scripts/export_demo.py --preds runs/hocap/<run>/preds.npz \\
        [--gate gate_hull10_arm_skin_fast.npz] [--cams 105322251564] \\
        [--out results/robot_export/<run>]
writes <out>/<seq>_<cam>_<side>.npz / .json (one per demo hand that lifts something)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from eval_hocap import cam_group, gt_labels  # noqa: E402
from sim_hocap import DELAY, delay, lifter, lifts  # noqa: E402

from touch import robot_export as X  # noqa: E402
from touch import robot_sim as S  # noqa: E402
from touch.data import dexycb as D  # noqa: E402
from touch.data import hocap as H  # noqa: E402
from touch.handpath import world_hand  # noqa: E402
from touch.interact.object_gate import latch  # noqa: E402


def export_track(P, G, seq, cam: str, side: str, run: str, gate_file: str) -> tuple[dict, dict]:
    key = f"{seq.subject}/{seq.name}/{cam}/{side}"
    pres = P[f"{key}/present"]
    prob = P[f"{key}/state"]
    near = G[f"{key}/near"] if G is not None else np.ones(len(pres), bool)
    state = latch(np.where(pres, prob.argmax(1), 0), near)
    close = S.fill_gaps(delay(S.debounce(state == 2)))
    wh = world_hand(P[f"{key}/joints"], P[f"{key}/verts"].astype(np.float32), P[f"{key}/lm"],
                    H.intrinsics(cam), H.cam_to_world(cam), side == "left")  # fmt: skip
    arrays = X.build(wh["joints"], wh["verts"], wh["ok"], state, prob,
                     (P[f"{key}/finger"] >= 0.5) & pres[:, None], P[f"{key}/vert"] >= 0.5,
                     near, close, D.vertex_parts(side), H.FPS)  # fmt: skip
    R, t = H.object_poses(seq)
    arrays["gt_object_pos"] = t.transpose(1, 0, 2).astype(np.float32)  # [N, n_obj, 3]
    arrays["gt_object_quat"] = X.to_quat_wxyz(R.transpose(1, 0, 2, 3)).astype(np.float32)
    arrays["gt_hand_joints"] = gt_labels(seq, side)["joints"].astype(np.float32)
    meta = {
        "source": "hocap", "subject": seq.subject, "sequence": seq.name, "camera": cam,
        "camera_group": cam_group(cam), "hand": side, "fps": H.FPS, "frames": int(len(pres)),
        "frame": "HO-Cap table tag (tag_1), metres, z up",
        "K": H.intrinsics(cam).tolist(), "cam_to_world": H.cam_to_world(cam).tolist(),
        "objects": [{"id": o, "mesh": str(H.ROOT / "models" / o / "cleaned_mesh_2000.obj")}
                    for o in seq.objects],
        "model": run, "gate": gate_file,
        "close_rule": f"debounce on 2 / off 3, +{DELAY} frames, gaps <= 10 frames filled",
        "gripper_approach": "pre-grasp: 8 cm back along the approach axis, 12 frames straight in",
        "note": "hand / state / gripper are PREDICTED from RGB; gt_* keys are HO-Cap ground truth",
    }  # fmt: skip
    return arrays, meta


def runs(x: np.ndarray) -> list[tuple[int, int]]:
    c = np.r_[False, x.astype(bool), False]
    a, e = np.flatnonzero(~c[:-1] & c[1:]), np.flatnonzero(c[:-1] & ~c[1:])
    return list(zip(a.tolist(), e.tolist(), strict=True))


def grasp_events(close: np.ndarray, gt_hold: np.ndarray) -> dict:
    """Exported grasp episodes vs GT hold episodes (any2): a grasp is correct if it overlaps a GT
    hold; a GT hold is found if some grasp overlaps it."""
    g, h = runs(close), runs(gt_hold)
    tp = sum(any(a < e2 and a2 < e for a2, e2 in h) for a, e in g)
    found = sum(any(a < e2 and a2 < e for a2, e2 in g) for a, e in h)
    return {"grasps": len(g), "correct": tp, "gt_holds": len(h), "found": found}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--gate", default="gate_hull10_arm_skin_fast.npz")
    ap.add_argument("--cams", nargs="*", default=["105322251564", "043422252387"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run = args.preds.parent.name
    out = args.out or Path("results/robot_export") / run
    P = np.load(args.preds)
    G = None if args.gate == "none" else np.load(args.preds.parent / args.gate)
    n, ev = 0, {}
    for seq in H.list_sequences(video_only=True, split="test"):
        sides = {lifter(seq, a0, b0) for _, a0, b0 in lifts(seq)}
        for cam in args.cams:
            for side in sorted(sides):
                if f"{seq.subject}/{seq.name}/{cam}/{side}/present" not in P.files:
                    continue
                arrays, meta = export_track(P, G, seq, cam, side, run, args.gate)
                keys = X.keyframes(arrays["gripper_pos"], arrays["gripper_close"],
                                   arrays["state"], arrays["hand_ok"], H.FPS)  # fmt: skip
                X.save(out / f"{seq.name}_{cam}_{side}", arrays, meta, keys)
                n += 1
                gt = gt_labels(seq, side)["state_any2"] == 2
                e = grasp_events(arrays["gripper_close"], gt)
                for k2, v2 in e.items():
                    ev.setdefault(cam_group(cam), {}).setdefault(k2, 0)
                    ev[cam_group(cam)][k2] += v2
                kinds = [k["type"] for k in keys]
                print(f"{seq.name} {cam} {side}: {len(keys)} keyframes "
                      f"({kinds.count('grasp')} grasps)", flush=True)  # fmt: skip
    for g, e in ev.items():
        e["precision"] = round(e["correct"] / max(e["grasps"], 1), 3)
        e["recall"] = round(e["found"] / max(e["gt_holds"], 1), 3)
        print(g, e)
    import json

    (out / "grasp_events.json").write_text(json.dumps(ev, indent=1))
    print(f"wrote {n} demos to {out}")


if __name__ == "__main__":
    main()
