"""How much do trained models drop when a projector shines UI content onto the hand? (estimate)

Simulates a top-down projector on the OVERHEAD validation patches (touch.data.projection: a
random UI "screen" fixed in image coordinates, warped into every patch, applied as light).
Training with proj_aug > 0 uses the same generator, so for those models this is an
in-distribution (optimistic) check. Not modelled: MediaPipe failures under projection.

Post-processing is tuned on the CLEAN validation predictions and then frozen, as in deployment.

Usage:
    python scripts/projection_stress.py --ckpt runs/a/last.pt runs/b/best.pt [--viz]
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from touch.data.catalog import load_catalog  # noqa: E402
from touch.data.patch_store import PatchReader, load_tracks, with_lm_feats  # noqa: E402
from touch.data.projection import (  # noqa: E402
    apply_light,
    hand_geometry,
    make_screen,
    screen_light,
)
from touch.eval.run import evaluate, predict, tune_postprocessing  # noqa: E402
from touch.train import build_model, load_config, pick_device  # noqa: E402

# name -> (ambient share of the light, frames between content changes; 0 = static UI)
CONDITIONS = {
    "static_mild": (0.7, 0),
    "static_strong": (0.4, 0),
    "dynamic_strong": (0.4, 15),
}
KEYS = ["finger_auroc", "finger_acc", "hand_acc", "click_f1_pm2", "hover_fp_per_min"]


class Projection:
    """patch_fn for touch.eval.run.predict: adds simulated projector light to every patch."""

    def __init__(self, ambient: float, change_every: int, seed: int = 0):
        self.ambient, self.change_every, self.seed = ambient, change_every, seed
        self._key, self._screens = None, {}

    def _load(self, tr) -> None:
        key = (tr.seq.name, tr.view)
        if key == self._key:
            return
        self._geo, self._hw = hand_geometry(tr.seq, tr.view)
        self._key, self._screens = key, {}
        self._rng_seed = zlib.crc32(f"{tr.seq.name}/{tr.view}/{self.seed}".encode())

    def _screen(self, i: int) -> np.ndarray:
        k = i // self.change_every if self.change_every else 0
        if k not in self._screens:
            S = make_screen(np.random.default_rng([self._rng_seed, k]), *self._hw)
            self._screens[k] = S / S.mean()
        return self._screens[k]

    def __call__(self, tr, f: int, frames: np.ndarray, patches: np.ndarray) -> np.ndarray:
        self._load(tr)
        x = patches.astype(np.float32) / 255.0
        for j, i in enumerate(frames):
            light = screen_light(self._screen(int(i)), self._geo[i, f])
            x[j] = apply_light(x[j], light, self.ambient)
        return (x * 255 + 0.5).astype(np.uint8)


def load_model(ckpt: Path, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = load_config(ck["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    return model, cfg


def viz(tracks, out: Path, n: int = 6) -> None:
    """Grid: rows = clean + each condition, columns = index-finger patches from a few tracks."""
    reader = PatchReader()
    picks = []
    for tr in tracks[:: max(1, len(tracks) // n)][:n]:
        have = np.flatnonzero(tr.offset[:, 1] >= 0)
        i = int(have[len(have) // 2])
        picks.append((tr, i))
    rows = []
    for cond in [None, *CONDITIONS]:
        row = []
        for tr, i in picks:
            p = reader.get(tr.seq.source, int(tr.offset[i, 1]), int(tr.length[i, 1]))[None]
            if cond:
                p = Projection(*CONDITIONS[cond])(tr, 1, np.array([i]), p)
            row.append(p[0])
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(str(out), np.concatenate(rows, axis=0)[:, :, ::-1])
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", type=Path, required=True)
    ap.add_argument("--viz", action="store_true")
    args = ap.parse_args()
    seqs = load_catalog(source="egopressure", split="val")
    tracks = load_tracks(seqs, ["overhead"])
    Path("results").mkdir(exist_ok=True)
    if args.viz:
        viz(tracks, Path("results/projection_stress_examples.jpg"))
    device = pick_device()
    probs = {}  # (member, condition) -> list of [N, 5]
    for ck in args.ckpt:
        model, cfg = load_model(ck, device)
        tr_m = with_lm_feats(tracks, cfg["lm_feats"])
        for cond in [None, *CONDITIONS]:
            fn = Projection(*CONDITIONS[cond]) if cond else None
            probs[(str(ck), cond or "clean")] = predict(
                model, tr_m, cfg["T"], cfg["input_repr"], device, patch_fn=fn
            )
            print("predicted", ck, cond or "clean", flush=True)
    members = {str(c): [str(c)] for c in args.ckpt}
    if len(args.ckpt) > 1:
        members["ensemble"] = [str(c) for c in args.ckpt]
    res = {}
    for name, mem in members.items():

        def avg(cond, mem=mem):
            return [
                np.mean(np.stack(ps), 0)
                for ps in zip(*[probs[(m, cond)] for m in mem], strict=True)
            ]

        post, _ = tune_postprocessing(tracks, avg("clean"), "event")  # clean val only
        res[name] = {"post": post}
        for cond in ["clean", *CONDITIONS]:
            r = evaluate(tracks, avg(cond), post)["overhead"]
            res[name][cond] = {k: r[k] for k in KEYS}
            print(f"{name[-45:]:45s} {cond:15s} " + " ".join(f"{k} {r[k]:.3f}" for k in KEYS))
    Path("results/projection_stress.json").write_text(json.dumps(res, indent=2))
    print("wrote results/projection_stress.json")


if __name__ == "__main__":
    main()
