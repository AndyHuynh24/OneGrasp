"""Why do sim lifts fail? HO-Cap TRAIN + VAL subjects only (test lifts stay untouched), GT hand
path and GT hold timing (sim_hocap "true timing"), so every failure is the gripper / physics.
Variants (VARIANTS): the current human-copied grasp, + pre-grasp, the object-centred grasp
(touch.robot_sim.object_refine: nearest object, its known pose), + safe approach (pre-grasp at or
above the pre-grasp height, opening = object width + 4 cm), fixed physics (stiff 4 ms contacts,
fingers close at 0.2 m/s and the arm waits for them: robot_sim.scene_xml(stiff=True),
replay(close_speed=0.2, wait_close=15)).

Per lift, first matching reason:
  too_wide      grasp width (thumb pad <-> finger pads) > the 15 cm stroke
  knocked       object moved >= 2 cm before the close
  closed_empty  fingers closed to < 5 mm: nothing between them
  slipped       something was held (opening >= 5 mm) but the object rose < 50% of the human's lift
  success       robot lift >= 50% of the human's

    python scripts/sim_diagnose.py [--workers 5] [--max-seqs N]
writes runs/sim_diagnose.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from eval_hocap import gt_labels  # noqa: E402
from sim_hocap import delay, lifter, lifts, scene  # noqa: E402

from touch import robot_sim as S  # noqa: E402
from touch.data import dexycb as D  # noqa: E402
from touch.data import hocap as H  # noqa: E402

# name: (object-centred grasp, safe approach, open margin, fixed physics)
VARIANTS = {
    "human grasp (current)": (False, False, S.MARGIN, False),
    "human grasp + pre-grasp": (False, True, S.MARGIN, False),
    "human grasp, fixed physics": (False, False, S.MARGIN, True),
    "object grasp": (True, False, 0.04, False),
    "object grasp + safe approach": (True, True, 0.04, False),
    "object grasp, fixed physics": (True, False, 0.04, True),
    "object grasp + safe approach, fixed physics": (True, True, 0.04, True),
}


def classify(res, c, W, k, human) -> dict:
    obj = res["obj"][:, k]
    robot = float((obj[:, 2] - obj[0, 2]).max())
    row = {"robot_m": round(robot, 3)}
    if not c.any():
        return row | {"reason": "never_closed"}
    f = int(np.flatnonzero(c)[0])
    w = float(W[f])
    moved = float(np.linalg.norm(obj[f, :2] - obj[0, :2]))
    held = float(np.median(res["opening"][f + 5 : f + 20])) if f + 6 < len(c) else 0.0
    row |= {"grasp_width_m": round(w, 3), "moved_before_close_m": round(moved, 3),
            "opening_after_close_m": round(held, 4)}  # fmt: skip
    if robot >= 0.5 * human:
        return row | {"reason": "success"}
    if w > 2 * S.MAX_HALF:
        return row | {"reason": "too_wide"}
    if moved >= 0.02:
        return row | {"reason": "knocked"}
    return row | {"reason": "closed_empty" if held < 0.005 else "slipped"}


def work(name: str) -> list[dict]:
    seq = next(s for s in H.list_sequences() if s.name == name)
    Ro, to = H.object_poses(seq)
    local = [H.object_points(o, 4000) for o in seq.objects]
    rows = []
    for k, a0, b0 in lifts(seq):
        side = lifter(seq, a0, b0)
        L = {**gt_labels(seq, side), "parts": D.vertex_parts(side)}
        close = delay(S.debounce(L["state"] == 2))
        pp, RR, W, _ = S.retarget(L["joints"], L["valid"], close, L["verts"], L["vert"] == 1,
                                  L["parts"])  # fmt: skip
        drop = [(p @ Ro[i, a0].T + to[i, a0])[:, 2].min() - 0.001 for i, p in enumerate(local)]

        def pose(i, t, dz=0.0):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = Ro[i, t], to[i, t] - [0, 0, dz]
            return T

        def objects_at(t, a0=a0, drop=drop):  # demo points at t + map to the sim scene (at a0)
            return [(p @ Ro[i, t].T + to[i, t], pose(i, a0, drop[i]) @ np.linalg.inv(pose(i, t)))
                    for i, p in enumerate(local)]  # fmt: skip

        human = float((to[k, a0:b0, 2] - to[k, a0, 2]).max())
        sl = slice(a0, b0)
        for v, (objg, pre, margin, phys) in VARIANTS.items():
            p2, R2, W2, how = pp, RR, W, []
            if objg:
                p2, R2, W2, how = S.object_refine(p2, R2, W2, close, objects_at)
            if pre:
                p2, R2 = S.pregrasp(p2, R2, close, floor=objg)
            res = S.replay(scene(seq, a0, phys), p2[sl], R2[sl], W2[sl], L["valid"][sl],
                           close[sl], margin=margin, close_speed=0.2 if phys else None,
                           wait_close=15 if phys else 0)  # fmt: skip
            rows.append({"seq": name, "subject": seq.subject, "object": seq.objects[k],
                         "variant": v, "human_m": round(human, 3), "grasps": how,
                         **classify(res, close[sl], W2[sl], k, human)})  # fmt: skip
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-seqs", type=int, default=0)
    args = ap.parse_args()
    seqs = [s for s in H.list_sequences() if H.split_of(s.subject) in ("train", "val")]
    assert not any(H.split_of(s.subject) == "test" for s in seqs)
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    t0, rows = time.time(), []
    with get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, [s.name for s in seqs])):
            rows += r
            print(f"{i + 1}/{len(seqs)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    summary = {}
    for v in VARIANTS:
        sel = [r for r in rows if r["variant"] == v]
        cnt = Counter(r["reason"] for r in sel)
        summary[v] = {"lifts": len(sel), "success": round(cnt["success"] / max(len(sel), 1), 3),
                      **{k: cnt[k] for k in ("knocked", "closed_empty", "slipped",
                                             "never_closed", "too_wide")}}  # fmt: skip
        print(f"{v:45s}", summary[v], flush=True)
    out = Path("runs/sim_diagnose.json")
    out.write_text(json.dumps({"summary": summary, "trials": rows}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
