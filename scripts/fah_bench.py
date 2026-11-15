"""Can FAH run on this Mac? Speed (ms/frame), MPS-vs-CPU parity, per-frame outputs to compare.

Runs in FAH's own environment (data/third_party/fah_dl/FAH-main/.venv), not the OneTouch one:

    FAH=data/third_party/fah_dl/FAH-main
    $FAH/.venv/bin/python scripts/fah_bench.py results/annotate/val_pinch.mp4 --frames 150
    $FAH/.venv/bin/python scripts/fah_bench.py results/fah_bench/clips/<seq>__<view>.txt  # JPEGs

For each (model, device, setting) it times frame-by-frame inference after a warm-up and saves the
hands of each frame (2D joints, contact) to results/fah_bench/<clip>_<model>_<device>_<setting>.npz
  track    image mode, no mesh fit, no contact: FAH as a drop-in hand tracker (MediaPipe's job)
  coreml   the same with the Core ML exports (coreml_video: video mode); needs fah-build-coreml
  contact  video mode + per-finger contact heads (PyTorch only: Core ML exports omit contact)
Model loading is excluded from the timing, like FAH's own numbers. Nothing here trains anything.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

SETTINGS = {
    "track": dict(mode="image", mesh="none", contact=False),
    "contact": dict(mode="video", mesh="none", contact=True),
    # Core ML exports (fah-build-coreml): tracking only, Core ML has no contact output
    "coreml": dict(mode="image", mesh="none", contact=False, coreml=True),
    "coreml_video": dict(mode="video", mesh="none", contact=False, coreml=True),
}


def read_frames(path: Path, n: int) -> list[np.ndarray]:
    """A video, or a .txt list of image paths (one per line, in frame order)."""
    if path.suffix == ".txt":
        files = path.read_text().split()[:n]
        return [cv2.imread(f) for f in files]
    cap, out = cv2.VideoCapture(str(path)), []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    assert out, f"no frames in {path}"
    return out


def pack(hands: list[dict], groups: np.ndarray | None = None) -> dict:
    """Hands of one frame -> arrays, slot 0 = left, slot 1 = right (FAH's is_right); NaN if absent.
    The first hand of each side is kept (FAH returns at most one per side in practice)."""
    out = dict(
        n=np.int8(len(hands)),
        j2d=np.full((2, 21, 2), np.nan, np.float32),
        fingers=np.full((2, 6), np.nan, np.float32),  # palm, thumb, index, middle, ring, pinky
        score=np.full(2, np.nan, np.float32),
        flag=np.full(2, np.nan, np.float32),
        # per finger at FAH's 5 / 10 / 20 / 35 mm labels, and the video contact model's
        fingers_k=np.full((2, 4, 6), np.nan, np.float32),
        fingers_video=np.full((2, 6), np.nan, np.float32),
    )
    for side in (0, 1):
        h = next((x for x in hands if bool(x.get("is_right")) == bool(side)), None)
        if h is None:
            continue
        if h.get("joints2d") is not None:
            out["j2d"][side] = np.asarray(h["joints2d"], np.float32)[:, :2]
        if h.get("contact_fingers") is not None:
            out["fingers"][side] = np.asarray(h["contact_fingers"], np.float32)
        sc = h.get("contact_score_video", h.get("contact_score"))
        if sc is not None:
            out["score"][side] = float(sc)
        if h.get("contact_flag_video") is not None:
            out["flag"][side] = float(h["contact_flag_video"])
        if groups is not None:  # vertex -> palm, thumb .. pinky (FAH's MANO skinning groups)
            for key, dst in (("contact_k", "fingers_k"), ("contact_video", "fingers_video")):
                if h.get(key) is not None:
                    v = np.asarray(h[key], np.float32)
                    g = np.stack([v[groups == k].max(0) for k in range(6)], -1)  # [4, 6] or [6]
                    out[dst][side] = g
    return out


def run(
    clip: Path, frames, model: str, device: str, setting: str, warmup: int, out_dir: Path
) -> dict:
    import fah
    import torch

    torch.manual_seed(0)
    t0 = time.perf_counter()
    m = fah.load(model, device=device, **SETTINGS[setting])
    load_s = time.perf_counter() - t0
    m.reset()
    for f in frames[:warmup]:  # warm-up (kernel compilation, caches); not timed, not saved
        m(f)
    m.reset()
    ms, rows = [], []
    for f in frames:
        if device == "mps":
            torch.mps.synchronize()
        t = time.perf_counter()
        hands = m(f)
        if device == "mps":
            torch.mps.synchronize()
        ms.append(1000 * (time.perf_counter() - t))
        rows.append(pack(hands, getattr(m.method.est, "contact_groups", None)))
    ms = np.array(ms)
    stem = f"{clip.stem}_{model}_{device}_{setting}"
    np.savez_compressed(
        out_dir / f"{stem}.npz",
        ms=ms,
        **{k: np.stack([np.asarray(r[k]) for r in rows]) for k in rows[0]},
    )
    det = np.mean([int(r["n"]) > 0 for r in rows])
    res = dict(
        clip=clip.name,
        model=model,
        device=device,
        setting=setting,
        frames=len(frames),
        load_s=round(load_s, 1),
        ms_median=round(float(np.median(ms)), 1),
        ms_p90=round(float(np.percentile(ms, 90)), 1),
        fps_median=round(1000 / float(np.median(ms)), 1),
        hand_found=round(float(det), 3),
    )
    print(json.dumps(res), flush=True)
    return res


def parity(out_dir: Path, clip: Path, model: str, setting: str) -> dict | None:
    """MPS vs CPU on the same frames: max / median |diff| of 2D joints (px), per-finger contact."""
    a, b = (out_dir / f"{clip.stem}_{model}_{d}_{setting}.npz" for d in ("mps", "cpu"))
    if not (a.exists() and b.exists()):
        return None
    za, zb = np.load(a), np.load(b)
    dj = np.abs(za["j2d"] - zb["j2d"])
    df = np.abs(za["fingers"] - zb["fingers"])
    res = dict(
        model=model,
        setting=setting,
        joints_px_max=float(np.nanmax(dj)) if np.isfinite(dj).any() else None,
        joints_px_median=float(np.nanmedian(dj)) if np.isfinite(dj).any() else None,
        contact_max=float(np.nanmax(df)) if np.isfinite(df).any() else None,
        hands_differ=int((za["n"] != zb["n"]).sum()),
    )
    print("parity", json.dumps(res), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", type=Path, nargs="+")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--models", nargs="+", default=["fah-s", "fah-b"])
    ap.add_argument("--devices", nargs="+", default=["mps", "cpu"])
    ap.add_argument("--settings", nargs="+", default=["track", "contact"])
    ap.add_argument("--out", type=Path, default=Path("results/fah_bench"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch

    env = dict(
        machine=platform.machine(),
        macos=platform.mac_ver()[0],
        torch=torch.__version__,
        mps=torch.backends.mps.is_available(),
    )
    print(json.dumps(env), flush=True)
    results, par = [], []
    for clip in args.clips:
        frames = read_frames(clip, args.frames + args.warmup)
        for model in args.models:
            for setting in args.settings:
                for device in args.devices:
                    try:
                        results.append(
                            run(clip, frames, model, device, setting, args.warmup, args.out)
                        )
                    except Exception as e:  # report and keep going
                        print(
                            f"FAILED {clip.name} {model} {device} {setting}: {e!r}",
                            flush=True,
                        )
                        results.append(
                            dict(
                                clip=clip.name,
                                model=model,
                                device=device,
                                setting=setting,
                                error=f"{type(e).__name__}: {e}",
                            )
                        )
                p = parity(args.out, clip, model, setting)
                if p:
                    par.append({"clip": clip.name, **p})
    (args.out / "bench.json").write_text(
        json.dumps(dict(env=env, runs=results, parity=par), indent=1)
    )
    print("wrote", args.out / "bench.json")


if __name__ == "__main__":
    main()
