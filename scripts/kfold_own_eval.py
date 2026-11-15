"""Score grasp checkpoints on our own sessions with the training harness' evaluate_state (stored
crops, same processing as live; state F1 + hold timing). k-fold: each fold model is scored on
the session it did not train on (out-of-fold), plus the shared val session.

    python scripts/kfold_own_eval.py --base runs/g10_.../best.pt --runs "runs/g11A_fold*" ...
Writes results/own_grasp/kfold_<tag>.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]
from touch.data.grasp_store import load_own_tracks  # noqa: E402
from touch.train_grasp import build_grasp_model, evaluate_state, load_grasp_config  # noqa: E402

VAL = "andy_grasp_cup_20261113-200050"


def load(ck: Path, device):
    d = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = load_grasp_config(d["config"])
    m = build_grasp_model({**cfg, "pretrained": False})
    m.load_state_dict(d["model"])
    return m.to(device).eval(), cfg


def tracks(names):
    allt = {t.seq: t for sp in ("train", "val", "test") for t in load_own_tracks(sp)}
    return [allt[n] for n in names]


def score(m, cfg, trs, device) -> dict:
    with torch.no_grad():
        r = evaluate_state(m, trs, cfg["T"], device)
    return {"none_f1": r["state"]["none"]["f1"], "hold_f1": r["state"]["hold"]["f1"],
            "hold_onset_sr5": r["hold_timing"].get("onset", {}).get("sr@5"),
            "hold_release_sr5": r["hold_timing"].get("release", {}).get("sr@5"),
            "hold_release_mae": r["hold_timing"].get("release", {}).get("mae_frames"),
            "score": r["score"],
            **({"finger_f1": r["finger"]["f1"],
                "tap_onset_sr3": r["finger_timing"].get("onset", {}).get("sr@3"),
                "tap_release_sr3": r["finger_timing"].get("release", {}).get("sr@3")}
               if "finger" in r else {})}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", default=[], help="fold run dirs (globs ok)")
    ap.add_argument("--tag", default="g11")
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    runs = sorted({r for g in args.runs for r in glob.glob(g)})
    out = {"base": str(args.base), "folds": {}}
    bm, bcfg = load(args.base, device)
    for r in runs:
        cfg = json.loads(json.dumps(load_grasp_config(Path(r) / "config.yaml")))
        held = [n for n in cfg["own_eval"].split(",") if n != VAL]
        m, mcfg = load(Path(r) / "best.pt", device)
        out["folds"][r] = {"held_out": held,
                           "fold": score(m, mcfg, tracks(held), device),
                           "base_on_fold": score(bm, bcfg, tracks(held), device),
                           "val": score(m, mcfg, tracks([VAL]), device)}  # fmt: skip
        f = out["folds"][r]
        print(f"{Path(r).name}: held {held[0][-15:]} fold {f['fold']} | g10 {f['base_on_fold']} "
              f"| val {f['val']}", flush=True)  # fmt: skip
    out["base_val"] = score(bm, bcfg, tracks([VAL]), device)
    for k in ("fold", "base_on_fold", "val"):
        vals = [f[k] for f in out["folds"].values()]
        if vals:
            out[f"mean_{k}"] = {
                m: round(float(np.mean([v[m] for v in vals if v[m] is not None])), 4)
                for m in vals[0]
            }
    print("MEAN", {k: out.get(k) for k in ("mean_fold", "mean_base_on_fold", "mean_val")},
          "g10 val", out["base_val"], flush=True)  # fmt: skip
    p = Path("results/own_grasp") / f"kfold_{args.tag}.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
