"""Save a grasp checkpoint's per-frame predictions on DexYCB val (subject 7) for offline use
(metric wrist fit, robot-sim replay). Val only; the test split is never loaded.

    python scripts/export_grasp_preds.py runs/<run>/best.pt   # -> <run>/val_preds.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch.data.grasp_store import load_grasp_tracks  # noqa: E402
from touch.train import pick_device  # noqa: E402
from touch.train_grasp import build_grasp_model, load_grasp_config, predict_track  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = load_grasp_config(ck["config"])
    model = build_grasp_model({**cfg, "pretrained": False})
    model.load_state_dict(ck["model"])
    device = pick_device()
    model.to(device).eval()
    tracks, mesh, vlab = load_grasp_tracks("val")
    assert tracks and all(t.split == "val" for t in tracks)
    out = {}
    for i, tr in enumerate(tracks):
        r = predict_track(model, tr, (mesh, vlab), cfg["T"], device)
        key = f"{tr.seq}/{tr.cam}"
        out[f"{key}/finger"] = r["finger"].astype(np.float16)
        out[f"{key}/state"] = r["state"].astype(np.float16)
        out[f"{key}/vert"] = (r["vert"] * 255).astype(np.uint8)
        out[f"{key}/verts"] = r["verts"].astype(np.float16)
        out[f"{key}/joints"] = r["joints"].astype(np.float32)
        out[f"{key}/present"] = tr.present
        if i % 50 == 0:
            print(i, len(tracks), flush=True)
    dst = args.ckpt.parent / "val_preds.npz"
    np.savez_compressed(dst, **out)
    print("wrote", dst)


if __name__ == "__main__":
    main()
