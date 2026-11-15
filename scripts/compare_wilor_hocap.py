"""Our grasp model vs WiLoR on HO-Cap (test-only): hand-mesh accuracy by camera group.

Same frames for both (our MediaPipe hand associated to the GT hand, GT valid, both predictions
finite), same convention as eval_hocap.score (wrist-relative, camera axes, left mirrored).
Unlike DexYCB, WiLoR did not train on HO-Cap, so this is a fair zero-shot comparison for both.

    python scripts/compare_wilor_hocap.py --ours runs/hocap/<run>/preds.npz \\
        [--wilor runs/baselines/wilor_hocap.npz]
writes runs/baselines/compare_wilor_hocap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from eval_hocap import cam_group, gt_cam_mesh, gt_labels, procrustes_err  # noqa: E402

from touch.data import hocap as H  # noqa: E402

STATES = ("none", "touch", "hold")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ours",
        type=Path,
        default=Path("runs/hocap/g1_grasp_repvit_20261109-154049/preds.npz"),
    )
    ap.add_argument("--wilor", type=Path, default=Path("runs/baselines/wilor_hocap.npz"))
    ap.add_argument("--out", type=Path, default=Path("runs/baselines/compare_wilor_hocap.json"))
    args = ap.parse_args()
    Op, W = np.load(args.ours), np.load(args.wilor)
    seqs = {s.name: s for s in H.list_sequences(video_only=True, split="test")}
    keys = sorted(k[: -len("/joints")] for k in W.files if k.endswith("/joints"))
    acc: dict = {}  # (group, model, metric) -> list; metric "mj_<state>" for by-state

    def add(group, model, metric, x):
        acc.setdefault((group, model, metric), []).append(x)

    for key in keys:
        _, name, cam, side = key.split("/")
        L = gt_labels(seqs[name], side)
        gj, gv = gt_cam_mesh(L, cam, side == "left")
        pred = {"ours": (Op[f"{key}/joints"], Op[f"{key}/verts"].astype(np.float32)),
                "wilor": (W[f"{key}/joints"], W[f"{key}/verts"].astype(np.float32))}  # fmt: skip
        ok = Op[f"{key}/present"] & L["valid"]
        for pj, _ in pred.values():
            ok &= np.isfinite(pj).all((1, 2))
        idx = np.flatnonzero(ok)
        st = L["state_any2"][idx]
        for g in ("all", cam_group(cam)):
            for m, (pj, pv) in pred.items():
                mj = np.linalg.norm(pj[idx] - gj[idx], axis=-1).mean(1) * 1000
                mv = np.linalg.norm(pv[idx] - gv[idx], axis=-1).mean(1) * 1000
                pa = [procrustes_err(pj[i], gj[i]) * 1000 for i in idx[::5]]
                add(g, m, "mpjpe", mj)
                add(g, m, "mpvpe", mv)
                add(g, m, "pa_mpjpe", np.array(pa))
                for s, sn in enumerate(STATES):
                    add(g, m, f"mpjpe_{sn}", mj[st == s])
    res: dict = {}
    for (g, m, metric), v in sorted(acc.items()):
        x = np.concatenate(v)
        res.setdefault(g, {}).setdefault(m, {})[metric] = round(float(x.mean()), 2)
        if metric == "mpjpe":
            res[g]["frames"] = len(x)
    speed = {"wilor_mps": round(float(W["sec_per_hand_model"]) * 1000, 1), "ours_cpu": 17.2}
    res["meta"] = {"ours": str(args.ours), "wilor": str(args.wilor), "tracks": len(keys),
                   "speed_ms_per_hand": speed}  # fmt: skip
    args.out.write_text(json.dumps(res, indent=1))
    print(f"{'group':9s} {'frames':>7s} | {'MPJPE ours/WiLoR':>17s} | {'PA-MPJPE':>11s} | "
          f"{'MPVPE':>11s} | MPJPE while holding")  # fmt: skip
    for g in ("all", "top-down", "oblique", "low"):
        if g not in res:
            continue
        o, w = res[g]["ours"], res[g]["wilor"]
        print(f"{g:9s} {res[g]['frames']:7d} | {o['mpjpe']:7.1f} / {w['mpjpe']:6.1f} | "
              f"{o['pa_mpjpe']:4.1f} / {w['pa_mpjpe']:4.1f} | {o['mpvpe']:4.1f} / {w['mpvpe']:4.1f}"
              f" | {o['mpjpe_hold']:4.1f} / {w['mpjpe_hold']:4.1f}")  # fmt: skip
    print("wrote", args.out)


if __name__ == "__main__":
    main()
