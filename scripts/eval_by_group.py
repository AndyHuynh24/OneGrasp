"""Validation metrics of a checkpoint split by gesture group (presses vs touch-table gestures).

Post-processing is tuned on the whole validation set (as in touch.eval.run), then the same setting
is evaluated on each group. Never touches the test split and never writes into the checkpoint.

Usage:
    python scripts/eval_by_group.py runs/<run>/best.pt
Writes runs/<run>/eval_val_groups_<stem>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from touch.data.catalog import load_catalog  # noqa: E402
from touch.data.patch_store import load_tracks  # noqa: E402
from touch.eval.run import evaluate, predict, tune_postprocessing  # noqa: E402
from touch.train import build_model, load_config, pick_device  # noqa: E402

TOUCH_TABLE = ("draw_word", "pinch-zoom", "type_ipad")
KEYS = (
    "click_f1_pm2",
    "finger_auroc",
    "hover_fp_per_min",
    "timing_down_mae_ms",
    "timing_up_mae_ms",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = load_config(ck["config"])
    device = pick_device()
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    seqs = load_catalog(source="egopressure", split="val")
    assert all(s.split == "val" for s in seqs)
    tracks = load_tracks(seqs, cfg["views"], lm_feats=cfg["lm_feats"])
    probs = predict(model, tracks, cfg["T"], cfg["input_repr"], device)
    post, _ = tune_postprocessing(tracks, probs, "event")

    groups = {
        "all": lambda g: True,
        "presses": lambda g: g not in TOUCH_TABLE,
        "touch_table": lambda g: g in TOUCH_TABLE,
        **{g: (lambda g0: lambda g: g == g0)(g) for g in TOUCH_TABLE},
    }
    out = {"ckpt": args.ckpt, "epoch": ck.get("epoch"), "post": post}
    for name, keep in groups.items():
        idx = [i for i, tr in enumerate(tracks) if keep(tr.seq.meta["gesture"])]
        res = evaluate([tracks[i] for i in idx], [probs[i] for i in idx], post)
        out[name] = res
        for view, r in res.items():
            vals = "  ".join(f"{k} {r[k]:.3f}" for k in KEYS)
            print(f"{name:12s} {view:9s} n_down {r['n_touchdowns']:5d}  {vals}", flush=True)
    path = Path(args.ckpt).parent / f"eval_val_groups_{Path(args.ckpt).stem}.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
