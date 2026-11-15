"""Tune the causal hold latch (touch.interact.hold_latch) on VALIDATION predictions only:
HO-Cap val (subject 9, runs/hocap_val/<run>/preds.npz, labels "air": hold = held AND off the table)
and DexYCB val (subject 7, runs/<run>/val_preds.npz, labels "any2"). Hand height = the model's own
3D hand (mesh + translation fitted to the MediaPipe keypoints, unsmoothed; the latch smooths
causally) in each dataset's table frame. Selection = mean of the two validation macro F1s.

    python scripts/tune_hold_latch.py --run runs/g6_grasp_table_20261112-101602
writes <run>/hold_latch.json (grid + best parameters)
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from eval_hocap import gt_labels  # noqa: E402

from touch import robot_sim as S  # noqa: E402
from touch.data import dexycb as D  # noqa: E402
from touch.data import hocap as H  # noqa: E402
from touch.handpath import world_hand  # noqa: E402
from touch.interact.hold_latch import HoldLatch, palm_height  # noqa: E402
from touch.train_grasp import _f1  # noqa: E402

PROC = Path("data/dexycb_proc")
GRID = {
    "rise_m": [0.02, 0.03, 0.05, 1e9],  # 1e9 = rise cue off
    "down_m": [0.01, 0.02, 0.03],
    "k_off": [3, 5, 10],
    "k_down": [3, 8],
}


def hocap_tracks(run: str, split: str = "val") -> list[dict]:
    root = Path("runs/hocap_val") if split == "val" else Path("runs/hocap")
    P = np.load(root / run / "preds.npz")
    seqs = {s.name: s for s in H.list_sequences(video_only=True, split=split)}
    out = []
    for key in sorted({k.rsplit("/", 1)[0] for k in P.files}):
        _, name, cam, side = key.split("/")
        L = gt_labels(seqs[name], side)
        wh = world_hand(P[f"{key}/joints"], P[f"{key}/verts"].astype(np.float32),
                        P[f"{key}/lm"], H.intrinsics(cam), H.cam_to_world(cam), side == "left",
                        smooth_k=1)  # fmt: skip
        pres = P[f"{key}/present"]
        out.append({"state": np.where(pres, P[f"{key}/state"].argmax(1), 0), "present": pres,
                    "z": palm_height(wh["joints"]), "gt": L["state_air"], "cam": cam})  # fmt: skip
    return out


def dexycb_tracks(run_dir: Path) -> list[dict]:
    P = np.load(run_dir / "val_preds.npz")
    out = []
    for seq in [s for s in D.list_sequences() if s.split == "val"]:
        c = np.load(PROC / seq.subject / seq.name / "contact.npz")
        gt = D.labels_from_dist(c["part_dist"], c["obj_disp"], c["valid"], hold_rule="any2")
        ex = S.extrinsics(seq)
        tag = ex["apriltag"]
        for cam in D.CAMS:
            key = f"{seq.name}/{cam}"
            if f"{key}/present" not in P.files:
                continue
            pres = P[f"{key}/present"]
            J = P[f"{key}/joints"].astype(np.float64)
            with np.load(PROC / seq.subject / seq.name / f"{cam}.npz") as z:
                uv, flip = z["lm_px"][..., :2].astype(np.float64), bool(z["flip"])
            if flip:
                J[..., 0] *= -1
            K = D.intrinsics(cam)
            Jw = np.full_like(J, np.nan)
            for t in np.flatnonzero(pres):
                Jc = J[t] + S.fit_translation(J[t], uv[t], K)
                Jm = Jc @ ex[cam][:, :3].T + ex[cam][:, 3]  # camera -> master
                Jw[t] = S.master_to_table(Jm, tag)
            out.append({"state": np.where(pres, P[f"{key}/state"].argmax(1), 0), "present": pres,
                        "z": palm_height(Jw), "gt": gt["state"]})  # fmt: skip
    return out


def score(tracks: list[dict], latch: HoldLatch | None) -> dict:
    ps, gs = [], []
    for tr in tracks:
        p = tr["state"] if latch is None else latch.run(tr["state"], tr["present"], tr["z"])
        m = tr["gt"] >= 0
        ps.append(p[m])
        gs.append(tr["gt"][m])
    ps, gs = np.concatenate(ps), np.concatenate(gs)
    per = {n: _f1(ps == c, gs == c) for c, n in enumerate(("none", "touch", "hold"))}
    return {"macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 4),
            "hold": per["hold"], "touch_called_hold": round(float((ps[gs == 1] == 2).mean()), 4),
            "none_called_hold": round(float((ps[gs == 0] == 2).mean()), 4)}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--test", action="store_true",
                    help="FINAL: score the already-tuned latch (hold_latch.json) on HO-Cap test "
                         "(runs/hocap/<run>/preds.npz); no tuning")  # fmt: skip
    args = ap.parse_args()
    if args.test:
        from eval_hocap import cam_group

        prm = json.loads((args.run / "hold_latch.json").read_text())["best"]["params"]
        with np.errstate(all="ignore"):
            tr = hocap_tracks(args.run.name, "test")
        res = {}
        for g in ("all", "top-down", "oblique", "low"):
            sel = [t for t in tr if g == "all" or cam_group(t["cam"]) == g]
            if sel:
                res[g] = {"model": score(sel, None), "latch": score(sel, HoldLatch(**prm))}
                m, la = res[g]["model"], res[g]["latch"]
                print(f"{g:9s} model macro {m['macro_f1']:.3f} hold P/R "
                      f"{m['hold']['precision']:.2f}/{m['hold']['recall']:.2f} | "
                      f"+latch macro {la['macro_f1']:.3f} hold P/R "
                      f"{la['hold']['precision']:.2f}/{la['hold']['recall']:.2f}")  # fmt: skip
        out = args.run / "hold_latch_test.json"
        out.write_text(json.dumps({"params": prm, **res}, indent=1))
        print("wrote", out)
        return
    with np.errstate(all="ignore"):
        data = {"hocap_val": hocap_tracks(args.run.name), "dexycb_val": dexycb_tracks(args.run)}
    base = {k: score(v, None) for k, v in data.items()}
    print("model alone:", json.dumps(base), flush=True)
    rows = []
    for vals in itertools.product(*GRID.values()):
        prm = dict(zip(GRID, vals, strict=True))
        r = {k: score(v, HoldLatch(**prm)) for k, v in data.items()}
        sel = round((r["hocap_val"]["macro_f1"] + r["dexycb_val"]["macro_f1"]) / 2, 4)
        rows.append({"params": prm, "select": sel, **r})
    rows.sort(key=lambda r: -r["select"])
    for r in rows[:5]:
        h, d = r["hocap_val"], r["dexycb_val"]
        print(f"{r['select']:.4f} {r['params']} | HO-Cap macro {h['macro_f1']:.3f} hold P/R "
              f"{h['hold']['precision']:.2f}/{h['hold']['recall']:.2f} touch->hold "
              f"{h['touch_called_hold']:.2f} | DexYCB macro {d['macro_f1']:.3f} hold P/R "
              f"{d['hold']['precision']:.2f}/{d['hold']['recall']:.2f}")  # fmt: skip
    out = args.run / "hold_latch.json"
    out.write_text(json.dumps({"model_alone": base, "best": rows[0], "grid": rows}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
