"""Average the per-frame touch probabilities of several checkpoints and evaluate on VALIDATION.

Post-processing (median k, hysteresis on/off) is tuned on val, exactly as in touch.eval.run.
Per-checkpoint val probabilities are cached next to the checkpoint (probs_val_<stem>.pkl).
Writes results/ensemble_<tag>.json.

Usage:
    python scripts/ensemble_eval.py runs/a/best.pt runs/b/last.pt [--tag c4+c2]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from touch.data.catalog import load_catalog  # noqa: E402
from touch.data.patch_store import load_tracks, with_lm_feats  # noqa: E402
from touch.eval.run import evaluate, predict, tune_postprocessing  # noqa: E402
from touch.train import build_model, load_config, pick_device  # noqa: E402

VIEWS = ["ego", "overhead"]


def val_probs(ckpt: Path, tracks) -> list[np.ndarray]:
    cache = ckpt.parent / f"probs_val_{ckpt.stem}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = load_config(ck["config"])
    assert cfg["views"] == VIEWS, "ensemble members must be evaluated on the same views"
    device = pick_device()
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    tracks = with_lm_feats(tracks, cfg["lm_feats"])
    probs = predict(model, tracks, cfg["T"], cfg["input_repr"], device)
    cache.write_bytes(pickle.dumps(probs))
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", type=Path)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tracks = load_tracks(load_catalog(source="egopressure", split="val"), VIEWS)
    members = [val_probs(c, tracks) for c in args.ckpts]
    probs = [np.mean(np.stack(ps), axis=0) for ps in zip(*members, strict=True)]
    post, _ = tune_postprocessing(tracks, probs, "event")
    post_frame, _ = tune_postprocessing(tracks, probs, "frame")
    rec = {
        "members": [str(c) for c in args.ckpts],
        "split": "val",
        "post": post,
        "post_frame": post_frame,
        "val": evaluate(tracks, probs, post),
        "frame_op": evaluate(tracks, probs, post_frame),
    }
    tag = args.tag or "+".join(c.parent.name.split("_s")[0] for c in args.ckpts)
    out = Path("results") / f"ensemble_{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    for view in ("all", "ego", "overhead"):
        r = rec["val"][view]
        print(
            f"{view:9s} auroc {r['finger_auroc']:.4f} acc {r['finger_acc']:.4f} "
            f"hand {r['hand_acc']:.4f} F1±2 {r['click_f1_pm2']:.3f} "
            f"exact {r['click_f1_exact']:.3f} hoverFP/min {r['hover_fp_per_min']:.1f}"
        )
    print("post", post, "->", out)


if __name__ == "__main__":
    main()
