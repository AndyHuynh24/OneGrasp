"""Robustness of grab detection to webcam-like image degradations, on HO-Cap VAL (subject 9).

Same pipeline as scripts/eval_hocap.py (MediaPipe on the frame, association to the GT hand, the
live model one frame at a time), but every frame is degraded first (deterministic per frame).
Scored with the live demo's grab rule (scripts/grab_fusion.gate_rule / touch.live_robot):
grabbed / false release / F1 against the GT any2 hold, plus the MediaPipe hand rate.

    python scripts/degrade_hocap.py --ckpt runs/<run>/best.pt [--kinds none,blur,...]
writes runs/degrade/<run>/<kind>.npz (per-track state / present) and summary.json
"""

from __future__ import annotations

import argparse
import json
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

OUT = Path("runs/degrade")


def degrade(img: np.ndarray, kind: str, i: int) -> np.ndarray:
    """Webcam-like degradations. Seeded by frame index (same noise every run)."""
    if kind == "none":
        return img
    rng = np.random.default_rng(i)
    x = img.astype(np.float32)

    def noise(x, s):
        return x + rng.normal(0, s, x.shape).astype(np.float32)

    def jpeg(x, q):
        ok, b = cv2.imencode(".jpg", np.clip(x, 0, 255).astype(np.uint8),
                             [cv2.IMWRITE_JPEG_QUALITY, q])  # fmt: skip
        return cv2.imdecode(b, cv2.IMREAD_COLOR).astype(np.float32)

    def blur(x, s):
        return cv2.GaussianBlur(x, (0, 0), s)

    def warm(x):  # BGR: auto white balance gone warm
        return x * np.array([0.82, 1.0, 1.18], np.float32)

    def lowres(x, f):
        h, w = x.shape[:2]
        s = cv2.resize(x, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)

    if kind == "blur":
        x = blur(x, 1.5)
    elif kind == "noise":
        x = jpeg(noise(x, 8), 70)
    elif kind == "jpeg":
        x = jpeg(x, 35)
    elif kind == "warm":
        x = warm(x)
    elif kind == "cool":
        x = x * np.array([1.18, 1.0, 0.82], np.float32)
    elif kind == "lowres":
        x = lowres(x, 0.5)
    elif kind == "dark":
        x = jpeg(noise(x * 0.55, 6), 80)
    elif kind == "webcam":  # a bit of everything: soft, noisy, compressed, warm
        x = jpeg(noise(warm(blur(lowres(x, 0.7), 0.8)), 5), 75)
    else:
        raise ValueError(kind)
    return np.clip(x, 0, 255).astype(np.uint8)


def work(job) -> dict:
    ckpt, name, cam, kind = job
    from collections import deque

    from eval_hocap import ASSOC_FRAC, associate, gt_labels, world_to_cam  # noqa: F401
    from live_grasp import LiveGrasp
    from mediapipe import Image, ImageFormat

    from touch.data import dexycb as D
    from touch.data import hocap as H
    from touch.data.landmarks import DEFAULT_MODEL, _make_landmarker

    seq = next(s for s in H.list_sequences(video_only=True, split="val") if s.name == name)
    n, K = seq.n_frames, H.intrinsics(cam)
    lm = np.full((n, 2, 21, 3), np.nan, np.float32)
    with _make_landmarker(DEFAULT_MODEL, 0.3) as det:
        for i in range(n):
            img = cv2.imread(str(seq.image(cam, i)))
            if img is None:
                continue
            img = degrade(img, kind, i)
            h, w = img.shape[:2]
            r = det.detect_for_video(Image(image_format=ImageFormat.SRGB,
                                           data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                                     int(i * 1000 / H.FPS))  # fmt: skip
            for k, hand in enumerate(r.hand_landmarks[:2]):
                lm[i, k] = [[p.x * w, p.y * h, p.z * w] for p in hand]
    gt_uv, gt_ok = {}, {}
    for side in seq.sides:
        L = gt_labels(seq, side)
        jc = world_to_cam(L["joints"], cam)
        gt_uv[side] = D.project(jc.reshape(-1, 3), K).reshape(n, 21, 2)
        gt_ok[side] = L["valid"] & (jc[:, :, 2] > 0.05).all(1)
    tracks = associate(lm, gt_uv, gt_ok)
    live = LiveGrasp(Path(ckpt), threads=1)
    res = {}
    for side, tlm in tracks.items():
        live.hist = deque([(live.pad_tok, live.pad_he, False)] * (live.T - 1), maxlen=live.T - 1)
        live.lm_hist.clear()
        live.last_imgs = None
        pres = np.isfinite(tlm).all((1, 2))
        st = np.zeros((n, 3), np.float32)
        for i in range(n):
            img = degrade(cv2.imread(str(seq.image(cam, i))), kind, i)
            st[i] = live.step(live.prep(img, tlm[i] if pres[i] else None, side == "left"))["state"]
        res[f"{seq.subject}/{name}/{cam}/{side}"] = {
            "state": st,
            "present": pres,
            "y": gt_labels(seq, side)["state_any2"],
        }
    return res


def summarize(res: dict) -> dict:
    import grab_fusion as G

    keys = sorted(res)
    d = {"state": np.concatenate([res[k]["state"] for k in keys]),
         "present": np.concatenate([res[k]["present"] for k in keys]).astype(float),
         "y_any2": np.concatenate([res[k]["y"] for k in keys]),
         "tid": np.concatenate([np.full(len(res[k]["y"]), i)
                                for i, k in enumerate(keys)])}  # fmt: skip
    r = G.score(d, G.run_rule(d))
    y, p = d["y_any2"], d["state"].argmax(1) * (d["present"] > 0)
    m = y >= 0
    hold_p = float(((p == 2) & (y == 2) & m).sum() / max(((p == 2) & m).sum(), 1))
    hold_r = float(((p == 2) & (y == 2) & m).sum() / max(((y == 2) & m).sum(), 1))
    return {**r, "hand_rate": round(float(d["present"].mean()), 3),
            "hold_P": round(hold_p, 3), "hold_R": round(hold_r, 3)}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--kinds", default="none,blur,noise,jpeg,warm,cool,lowres,dark,webcam")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    from touch.data import hocap as H

    out = OUT / args.ckpt.parent.name
    out.mkdir(parents=True, exist_ok=True)
    seqs = H.list_sequences(video_only=True, split="val")
    base = [(s.name, c) for s in seqs for c in s.serials
            if (s.root / c / f"color_{s.n_frames - 1:06d}.jpg").exists()]  # fmt: skip
    summ = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    t0 = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        for kind in args.kinds.split(","):
            if kind in summ:
                continue
            res = {}
            for r in pool.imap_unordered(work, [(str(args.ckpt), nm, c, kind) for nm, c in base]):
                res.update(r)
            np.savez_compressed(out / f"{kind}.npz", **{f"{k}/{f}": v for k, r in res.items()
                                                       for f, v in r.items()})  # fmt: skip
            summ[kind] = summarize(res)
            (out / "summary.json").write_text(json.dumps(summ, indent=1))
            print(f"{kind:7s} {summ[kind]} | {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
