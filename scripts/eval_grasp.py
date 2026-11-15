"""Score a grasp checkpoint on DexYCB s1 val (default) or, once at the very end, on test.

    python scripts/eval_grasp.py runs/<run>/best.pt                    # val
    python scripts/eval_grasp.py runs/<run>/best.pt --split test --final  # test, once

Writes <run>/eval_<split>.json. No thresholds are tuned here (0.5 everywhere). The test split
(subjects 8-9) needs --final: use it only after every choice has been made on validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch.data.grasp_store import load_grasp_tracks  # noqa: E402
from touch.train import pick_device  # noqa: E402
from touch.train_grasp import (  # noqa: E402
    build_grasp_model,
    evaluate,
    load_grasp_config,
    load_part_dist,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true", help="required for --split test")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("test split: pass --final, and only once every choice was made on validation")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = load_grasp_config(ck["config"])
    model = build_grasp_model({**cfg, "pretrained": False})
    model.load_state_dict(ck["model"])
    device = pick_device()
    model.to(device)
    tracks, mesh, vlab = load_grasp_tracks(args.split)
    assert tracks and all(t.split == args.split for t in tracks)
    ev = evaluate(model, tracks, (mesh, vlab), cfg["T"], device, part_dist=load_part_dist(tracks))
    ev["ckpt"], ev["split"], ev["n_tracks"] = str(args.ckpt), args.split, len(tracks)
    out = args.ckpt.parent / f"eval_{args.split}.json"
    out.write_text(json.dumps(ev, indent=1))
    print(json.dumps(ev, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
