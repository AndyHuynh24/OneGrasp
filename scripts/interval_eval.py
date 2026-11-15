"""Onset / release timing of contact intervals on validation, scored like the hand-object
literature (EgoLoc / HOCALT: SR@+-3/5/10 frames, MAE), next to OneTouch's own event metrics.

    python scripts/interval_eval.py runs/<run>/best.pt

Per finger (and for any-finger contact), within each valid run of a track. Uses the checkpoint's
validation-tuned post-processing; no tuning here. Writes results/interval_timing_<run>.json.
Note: EgoPressure GT is a pressure sensor on a flat pad, theirs is hand labels on objects, so
the numbers are comparable in format only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch.data.catalog import load_catalog  # noqa: E402
from touch.data.patch_store import load_tracks  # noqa: E402
from touch.eval.metrics import interval_timing  # noqa: E402
from touch.eval.run import _runs, binarize, predict  # noqa: E402
from touch.train import build_model, load_config, pick_device  # noqa: E402

TOLS = (3, 5, 10)


def summarize(results: list[dict]) -> dict:
    n_gt = sum(r["n_gt"] for r in results)
    out = {"n_gt": n_gt, "n_pred": sum(r["n_pred"] for r in results),
           "matched": sum(r["matched"] for r in results)}  # fmt: skip
    for b in ("onset", "release"):
        err = np.concatenate([r[b]["errors_frames"] for r in results]).astype(float)
        out[b] = {f"sr@{k}": round(float((np.abs(err) <= k).sum()) / max(n_gt, 1), 3) for k in TOLS}
        out[b]["mae_frames"] = round(float(np.abs(err).mean()), 2) if len(err) else None
        out[b]["mae_ms"] = round(float(np.abs(err).mean()) * 1000 / 30, 1) if len(err) else None
        out[b]["median_signed_frames"] = float(np.median(err)) if len(err) else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = load_config(ck["config"])
    device = pick_device()
    model = build_model({**cfg, "pretrained": False}).to(device)
    model.load_state_dict(ck["model"])
    seqs = load_catalog(source="egopressure", split="val")
    assert all(s.split == "val" for s in seqs)
    tracks = load_tracks(seqs, cfg["views"], lm_feats=cfg["lm_feats"])
    probs = predict(model, tracks, cfg["T"], cfg["input_repr"], device)
    per: dict[str, list[dict]] = {}
    for tr, p in zip(tracks, probs, strict=True):
        b = binarize(p, ck["post"]) & tr.present[:, None]
        for a, e in _runs(tr.valid):
            g, q = tr.contact[a:e], b[a:e]
            for key in (tr.view, "all"):
                per.setdefault(f"finger/{key}", []).extend(
                    interval_timing(q[:, f], g[:, f], TOLS) for f in range(5)
                )
                per.setdefault(f"any_finger/{key}", []).append(
                    interval_timing(q.any(1), g.any(1), TOLS)
                )
    out = {"ckpt": str(args.ckpt), "post": ck["post"], **{k: summarize(v) for k, v in per.items()}}
    for k, v in out.items():
        if isinstance(v, dict) and "onset" in v:
            o, r = v["onset"], v["release"]
            print(f"{k:22s} n_gt {v['n_gt']:5d} matched {v['matched']:5d} | onset SR@3/5/10 "
                  f"{o['sr@3']:.2f}/{o['sr@5']:.2f}/{o['sr@10']:.2f} MAE {o['mae_frames']} fr | "
                  f"release {r['sr@3']:.2f}/{r['sr@5']:.2f}/{r['sr@10']:.2f} "
                  f"MAE {r['mae_frames']} fr")  # fmt: skip
    path = Path("results") / f"interval_timing_{args.ckpt.parent.name}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
