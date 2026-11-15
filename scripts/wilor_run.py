"""Run WiLoR (frozen, official weights) on the crops listed by scripts/wilor_bench.py export.
Runs inside third_party/WiLoR with its own venv (third_party/wilor_venv); see wilor_bench.py.
Writes runs/wilor_bench/wilor.npz: joints [N, 21, 3] wrist-relative, camera axes, left hands
in WiLoR's mirrored (right-hand) frame = the eval convention; verts [N, 778, 3]; ok [N].
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
from wilor.configs import get_config  # noqa: E402
from wilor.datasets.vitdet_dataset import ViTDetDataset  # noqa: E402
from wilor.models.wilor import WiLoR  # noqa: E402


def load(
    ckpt="./pretrained_models/wilor_final.ckpt", cfg_path="./pretrained_models/model_config.yaml"
):
    """As wilor.models.load_wilor, without the renderer (needs EGL, not on macOS)."""
    cfg = get_config(cfg_path, update_cachedir=True)
    cfg.defrost()
    if "vit" in cfg.MODEL.BACKBONE.TYPE and "BBOX_SHAPE" not in cfg.MODEL:
        cfg.MODEL.BBOX_SHAPE = [192, 256]
    if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    cfg.MANO.DATA_DIR = cfg.MANO.MODEL_PATH = "./mano_data/"
    cfg.MANO.MEAN_PARAMS = "./mano_data/mano_mean_params.npz"
    cfg.freeze()
    model = WiLoR.load_from_checkpoint(ckpt, strict=False, cfg=cfg, init_renderer=False)
    return model.eval(), cfg


def main() -> None:
    jobs = np.load(sys.argv[1])
    out = Path(sys.argv[1]).with_name("wilor.npz")
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, cfg = load()
    model = model.to(dev)
    n = len(jobs["img"])
    J = np.full((n, 21, 3), np.nan, np.float32)
    V = np.full((n, 778, 3), np.nan, np.float32)
    ok = np.zeros(n, bool)
    t0 = time.time()
    from torch.utils.data import default_collate

    def to_dev(b):
        return {k: (v.float() if v.dtype == torch.float64 else v).to(dev)
                if torch.is_tensor(v) else v for k, v in b.items()}  # fmt: skip

    B = 16
    for s0 in range(0, n, B):
        items, idx = [], []
        for i in range(s0, min(s0 + B, n)):
            img = cv2.imread(str(jobs["img"][i]))
            if img is None:
                continue
            ds = ViTDetDataset(cfg, img, jobs["box"][i][None], np.array([float(jobs["right"][i])]),
                               rescale_factor=2.0)  # fmt: skip
            items.append(ds[0])
            idx.append(i)
        if not items:
            continue
        with torch.no_grad():
            o = model(to_dev(default_collate(items)))
        j = o["pred_keypoints_3d"].float().cpu().numpy()
        v = o["pred_vertices"].float().cpu().numpy()
        J[idx], V[idx], ok[idx] = j - j[:, :1], v - j[:, :1], True  # left: mirrored frame kept
        if s0 % 320 == 0:
            print(f"{s0}/{n} {(time.time() - t0) / max(s0, 1):.3f} s/crop", flush=True)
    np.savez(out, joints=J, verts=V, ok=ok)
    print("wrote", out, f"{ok.sum()}/{n}", f"{(time.time() - t0) / n:.3f} s/crop")


if __name__ == "__main__":
    main()
