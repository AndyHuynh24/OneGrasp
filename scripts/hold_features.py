"""Extra per-frame HOLD features (touch.interact.hand_features) for every track of the fusion
study, in the same track order as scripts/grab_features.py:
  pose_mano   grasp shape from the model's predicted MANO joints (32)
  pose_mp     the same from the MediaPipe landmarks (32)
  img_motion  2D hand velocity in palm lengths / frame (3)
  comotion    optical-flow co-motion of the band around the hand (7)
  tipcolor    fingertip colour (L*, a*) relative to the palm skin: blood pushed out by pressing (13)
Validation sets only (HO-Cap val, DexYCB val, OakInk2); --test for the final HO-Cap test check.

    python scripts/hold_features.py --run runs/<run> [--only hocap_val,...] [--test]
writes runs/grab_fusion/<run>/<dataset>_hand.npz (X [n, F], names, keys); co-motion depends only
on the frames + landmarks and is cached in runs/grab_fusion/comotion/<dataset>.npz.
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]

from touch.interact.hand_features import (  # noqa: E402
    COMOTION_NAMES,
    POSE_NAMES,
    TIPCOLOR_NAMES,
    comotion_track,
    image_motion,
    pose_features,
    tip_color,
)

OUT = Path("runs/grab_fusion")


def _gray(path) -> np.ndarray:
    """Decode colour, then convert: exactly what the live demo does with camera frames."""
    img = cv2.imread(str(path))
    assert img is not None, path
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def frames_and_lm(ds: str, key: str, run: str):
    """-> (frame paths [N], landmarks px [N, 21, 3], predicted MANO joints [N, 21, 3])."""
    if ds.startswith("hocap"):
        from touch.data import hocap as H

        split = "val" if ds == "hocap_val" else "test"
        root = Path("runs/hocap_val" if split == "val" else "runs/hocap") / run
        P = np.load(root / "preds.npz")
        _, name, cam, _ = key.split("/")
        seq = next(s for s in H.list_sequences(video_only=True, split=split) if s.name == name)
        lm = P[f"{key}/lm"]
        return [seq.image(cam, i) for i in range(len(lm))], lm, P[f"{key}/joints"]
    if ds == "dexycb_val":
        from touch.data import dexycb as D

        P = np.load(Path("runs") / run / "val_preds.npz")
        name, cam = key.split("/")
        seq = next(s for s in D.list_sequences() if s.name == name and s.split == "val")
        with np.load(Path("data/dexycb_proc") / seq.subject / name / f"{cam}.npz") as z:
            lm = z["lm_px"].astype(np.float32)
            lm[~z["present"].astype(bool)] = np.nan
        paths = [seq.root / cam / f"color_{i:06d}.jpg" for i in range(len(lm))]
        return paths, lm, P[f"{key}/joints"]
    if ds == "oakink2":
        from eval_oakink2 import Seq

        P = np.load(Path("runs/oakink2") / run / "preds.npz")
        name, cam, _ = key.split("/")
        seq = Seq(name)
        lm = P[f"{key}/lm"]
        return [seq.image(cam, i) for i in range(len(lm))], lm, P[f"{key}/joints"]
    raise ValueError(ds)


def comotion_job(job):
    ds, key, run = job
    paths, lm, _ = frames_and_lm(ds, key, run)
    return key, comotion_track((_gray(p) for p in paths), lm)


def tipcolor_job(job):
    ds, key, run = job
    paths, lm, _ = frames_and_lm(ds, key, run)
    return key, np.stack([tip_color(cv2.imread(str(p)), lm[i] if np.isfinite(lm[i]).all() else None)
                          for i, p in enumerate(paths)])  # fmt: skip


def cached(ds: str, kind: str, fn, keys: list[str], run: str, workers: int) -> dict:
    """Per-track features that depend only on frames + landmarks (not on the model)."""
    f = OUT / kind / f"{ds}.npz"
    C = dict(np.load(f)) if f.exists() else {}
    todo = [k for k in keys if k not in C]  # e.g. a camera added after the cache was built
    if not todo:
        return C
    with get_context("spawn").Pool(workers) as pool:
        for k, r in pool.imap_unordered(fn, [(ds, k, run) for k in todo]):
            C[k] = r
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **C)
    print(f"{kind}/{ds}: computed {len(todo)} new tracks", flush=True)
    return C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--only", default="hocap_val,dexycb_val,oakink2")
    ap.add_argument("--test", action="store_true", help="HO-Cap TEST (final check only)")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    run = args.run.name
    todo = ["hocap_test"] if args.test else args.only.split(",")
    t0 = time.time()
    for ds in todo:
        base = np.load(OUT / run / f"{ds}_start.npz")  # grab_features.py output: track order
        keys = [str(k) for k in base["keys"]]
        C = cached(ds, "comotion", comotion_job, keys, run, args.workers)
        T = cached(ds, "tipcolor", tipcolor_job, keys, run, args.workers)
        rows = []
        for k in keys:
            _, lm, joints = frames_and_lm(ds, k, run)
            pres = np.isfinite(lm).all((1, 2))
            jm = np.where(pres[:, None, None], joints, np.nan)
            lmm = np.where(pres[:, None, None], lm, np.nan)
            im, im_names = image_motion(lmm)
            rows.append(np.concatenate([pose_features(jm), pose_features(lmm), im, C[k], T[k]], 1))
        names = ([f"mano_{n}" for n in POSE_NAMES] + [f"mp_{n}" for n in POSE_NAMES] + im_names
                 + COMOTION_NAMES + TIPCOLOR_NAMES)  # fmt: skip
        X = np.concatenate(rows).astype(np.float32)
        assert len(X) == len(base["X"]), (len(X), len(base["X"]))
        np.savez_compressed(OUT / run / f"{ds}_hand.npz", X=X, names=np.array(names),
                            keys=np.array(keys))  # fmt: skip
        print(f"{ds}: {len(keys)} tracks, {X.shape} | {(time.time() - t0) / 60:.1f} min",
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
