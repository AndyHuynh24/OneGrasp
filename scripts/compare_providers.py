"""CPU vs CoreML (Mac GPU / Neural Engine) for the live grasp model: same frames, same crops ->
do the outputs match, and how long does one model step take on each? Run on an idle machine.

    python scripts/compare_providers.py --ckpt runs/<run>/best.pt --source <video> [--frames N]
Prints max / mean |difference| of the per-finger touch and hand-state probabilities, how often
the decisions (finger >= 0.5, state argmax) differ, and the median / p95 step time.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from live_grasp import HandTracker, LiveGrasp, default_ckpt, frames_with_hands  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--source", required=True, help="video file")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    ckpt = args.ckpt or default_ckpt()
    nets = {p: LiveGrasp(ckpt, args.threads, p) for p in ("cpu", "coreml")}
    cap = cv2.VideoCapture(args.source)
    tracker = HandTracker(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    times = {p: [] for p in nets}
    diff = {"finger": [], "state": []}
    flips = {"finger": 0, "state": 0}
    n = 0
    for frame, hands, handedness, _, _ in frames_with_hands(cap, args.source, 30.0,
                                                             threading.Event()):  # fmt: skip
        lm, left = tracker.pick(hands, handedness)
        crops = nets["cpu"].crops(frame, lm, left)
        out = {}
        for p, net in nets.items():
            x = net.inputs(crops)
            t0 = time.perf_counter()
            out[p] = net.step(x)
            times[p].append((time.perf_counter() - t0) * 1000)
        if lm is not None:
            for k in diff:
                a, b = np.asarray(out["cpu"][k]).ravel(), np.asarray(out["coreml"][k]).ravel()
                diff[k].append(np.abs(a - b))
            fa, fb = np.asarray(out["cpu"]["finger"]), np.asarray(out["coreml"]["finger"])
            flips["finger"] += int(((fa >= 0.5) != (fb >= 0.5)).sum())
            flips["state"] += int(out["cpu"]["state"].argmax() != out["coreml"]["state"].argmax())
        n += 1
        if n >= args.frames:
            break
    m = len(diff["state"])
    print(f"{n} frames, {m} with a hand, ckpt {ckpt}")
    for k in diff:
        d = np.concatenate(diff[k]) if diff[k] else np.zeros(1)
        print(f"  {k:6s} |cpu - coreml| max {d.max():.4f} mean {d.mean():.5f}; "
              f"decisions differ {flips[k]} times")  # fmt: skip
    for p, t in times.items():
        t = np.array(t[10:])  # skip warm-up (CoreML compiles on the first calls)
        print(
            f"  {p:6s} model step median {np.median(t):.1f} ms, p95 {np.percentile(t, 95):.1f} ms"
        )


if __name__ == "__main__":
    main()
