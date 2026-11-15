"""WiLoR baseline (CVPR 2025; weights CC BY-NC-ND, evaluation only) on DexYCB val tracks.

Runs in its own venv (third_party/.venv_wilor, WiLoR-mini). For a fair mesh comparison WiLoR gets
the SAME hand as our pipeline: the box of the MediaPipe landmarks we used (padded), on the same
frames, with the GT handedness (left hands mirrored, exactly like our model's input).

    third_party/.venv_wilor/bin/python scripts/run_wilor.py --out runs/baselines/wilor_val.npz
    third_party/.venv_wilor/bin/python scripts/run_wilor.py --hocap runs/hocap/<run>/preds.npz \
        --out runs/baselines/wilor_hocap.npz       # HO-Cap test (same associated hand boxes)

Writes per track "{seq}/{cam}/joints" [N,21,3] f32 and "/verts" [N,778,3] f16, wrist-relative,
camera axes, mirrored to a right hand when the GT hand is left (= our GT convention); NaN where
our pipeline had no hand. Also "{...}/sec" (mean seconds per hand) for the speed table.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch.data import dexycb as D  # noqa: E402

PROC = Path("data/dexycb_proc")
PAD = 0.1  # box padding per side, fraction of the landmark box size
RESCALE = 2.5  # WiLoR's crop = 2.5 x box (its pipeline default)


def load_pipe(device):
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    pipe = WiLorHandPose3dEstimationPipeline(
        device=device,
        dtype=torch.float32,
        verbose=False,
        wilor_pretrained_dir="third_party/wilor_models",
    )
    for m in pipe.wilor_model.modules():  # MPS conv rejects the permuted (non-contiguous) input
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_pre_hook(lambda mod, inp: (inp[0].contiguous(),))
    return pipe


def crop(img_rgb: np.ndarray, lm: np.ndarray, flip: bool) -> np.ndarray:
    """WiLoR's own patch code (blur when downsampling, square crop, mirror for left hands)."""
    from skimage.filters import gaussian
    from wilor_mini.utils import utils

    lo, hi = lm[:, :2].min(0), lm[:, :2].max(0)
    lo, hi = lo - PAD * (hi - lo), hi + PAD * (hi - lo)
    c = (lo + hi) / 2
    size = RESCALE * (hi - lo).max()
    im = img_rgb
    ds = size / 256 / 2
    if ds > 1.1:
        im = gaussian(im, sigma=(ds - 1) / 2, channel_axis=2, preserve_range=True)
    patch, _ = utils.generate_image_patch_cv2(
        im, c[0], c[1], size, size, 256, 256, flip, 1.0, 0, border_mode=cv2.BORDER_CONSTANT
    )
    return patch


def predict(pipe, patches: list[np.ndarray], device) -> tuple[np.ndarray, np.ndarray, float]:
    """-> wrist-relative joints [B, 21, 3], verts [B, 778, 3], model seconds."""
    x = torch.from_numpy(np.stack(patches)).to(device=device, dtype=torch.float32)
    ts = time.time()
    with torch.no_grad():
        o = pipe.wilor_model(x)
    j = o["pred_keypoints_3d"].float().cpu().numpy()
    v = o["pred_vertices"].float().cpu().numpy()
    return j - j[:, :1], v - j[:, :1], time.time() - ts


def run_hocap(pipe, args, device) -> None:
    """HO-Cap test tracks: the MediaPipe hand eval_hocap.py associated with each GT hand."""
    from touch.data import hocap as H

    P = np.load(args.hocap)
    keys = sorted(k[: -len("/lm")] for k in P.files if k.endswith("/lm"))
    if args.limit:
        keys = keys[: args.limit]
    seqs = {s.name: s for s in H.list_sequences(video_only=True, split="test")}
    out, t_model, n_model, t0 = {}, 0.0, 0, time.time()
    for k, key in enumerate(keys):
        _, name, cam, side = key.split("/")
        lm = P[f"{key}/lm"]
        idx = np.flatnonzero(np.isfinite(lm).all((1, 2)))
        n = len(lm)
        J = np.full((n, 21, 3), np.nan, np.float32)
        V = np.full((n, 778, 3), np.nan, np.float16)
        for b in range(0, len(idx), args.batch):
            ii = idx[b : b + args.batch]
            patches = [crop(cv2.imread(str(seqs[name].image(cam, i)))[..., ::-1].copy(), lm[i],
                            side == "left") for i in ii]  # fmt: skip
            j, v, dt = predict(pipe, patches, device)
            t_model, n_model = t_model + dt, n_model + len(ii)
            J[ii], V[ii] = j, v.astype(np.float16)
        out[f"{key}/joints"], out[f"{key}/verts"] = J, V
        print(f"{k + 1}/{len(keys)} {key} {len(idx)} hands | {(time.time() - t0) / 60:.1f} min",
              flush=True)  # fmt: skip
    out["sec_per_hand_model"] = np.float32(t_model / max(n_model, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"wrote {args.out}; model-only {1000 * t_model / max(n_model, 1):.1f} ms/hand")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/baselines/wilor_val.npz"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N tracks (smoke test)")
    ap.add_argument("--hocap", type=Path, default=None,
                    help="HO-Cap preds.npz from eval_hocap.py (same hand boxes)")  # fmt: skip
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pipe = load_pipe(device)
    if args.hocap is not None:
        return run_hocap(pipe, args, device)
    seqs = {s.name: s for s in D.list_sequences() if s.split == "val"}
    assert seqs and all(s.split == "val" for s in seqs.values())
    files = sorted(p for p in PROC.glob("*/*/*.npz") if p.name != "contact.npz")
    files = [p for p in files if p.parent.name in seqs]
    if args.limit:
        files = files[: args.limit]
    out, t_model, n_model, t0 = {}, 0.0, 0, time.time()
    for k, p in enumerate(files):
        seq, cam = seqs[p.parent.name], p.stem
        with np.load(p) as z:
            present, lm, flip = z["present"], z["lm_px"], bool(z["flip"])
        n = len(present)
        J = np.full((n, 21, 3), np.nan, np.float32)
        V = np.full((n, 778, 3), np.nan, np.float16)
        idx = np.flatnonzero(present)
        for b in range(0, len(idx), args.batch):
            ii = idx[b : b + args.batch]
            patches = [
                crop(cv2.imread(str(seq.image(cam, i)))[..., ::-1].copy(), lm[i], flip) for i in ii
            ]
            x = torch.from_numpy(np.stack(patches)).to(device=device, dtype=torch.float32)
            ts = time.time()
            with torch.no_grad():
                o = pipe.wilor_model(x)
            j = o["pred_keypoints_3d"].float().cpu().numpy()
            v = o["pred_vertices"].float().cpu().numpy()
            t_model += time.time() - ts
            n_model += len(ii)
            J[ii] = j - j[:, :1]
            V[ii] = (v - j[:, :1]).astype(np.float16)
        key = f"{seq.name}/{cam}"
        out[f"{key}/joints"], out[f"{key}/verts"] = J, V
        print(f"{k + 1}/{len(files)} {key} {len(idx)} hands | {(time.time() - t0) / 60:.1f} min",
              flush=True)  # fmt: skip
    out["sec_per_hand_model"] = np.float32(t_model / max(n_model, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"wrote {args.out}; model-only {1000 * t_model / max(n_model, 1):.1f} ms/hand "
          f"(batch {args.batch}, {device})")  # fmt: skip


if __name__ == "__main__":
    main()
