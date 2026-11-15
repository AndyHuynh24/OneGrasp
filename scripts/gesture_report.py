"""How the rule-based gesture layer (touch.interact) behaves on real touch-table sequences.

Feeds per-finger touch states through TouchTracker + GestureRecognizer for every draw_word /
pinch-zoom / type_ipad sequence of a split and reports, per sequence type: gesture counts and the
measured touch durations / travel (hand sizes), next to the GestureConfig thresholds. Touch states
come from the ground truth (--source gt) or a checkpoint (--ckpt, validation-tuned post-processing).
Tune GestureConfig on train / val only, never on test.

Usage:
    python scripts/gesture_report.py --split val                       # ground-truth touches
    python scripts/gesture_report.py --split val --ckpt runs/<run>/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from touch.data.catalog import load_catalog  # noqa: E402
from touch.data.patch_store import load_tracks  # noqa: E402
from touch.data.patches import primary_hand  # noqa: E402
from touch.interact.gestures import GestureConfig, GestureRecognizer  # noqa: E402
from touch.interact.touches import TouchTracker, hand_points  # noqa: E402

TOUCH_TABLE = ("draw_word", "pinch-zoom", "type_ipad")
KINDS = ("tap", "double_tap", "long_press", "drag_start", "swipe", "pinch_start")


def run_sequence(lms, touch, prob, fps, cfg):
    """-> gestures, closed touches [(duration_s, travel_hand_sizes, max_open_touches)]."""
    tracker, rec = TouchTracker(), GestureRecognizer(cfg)
    gestures, downs, touches = [], {}, []
    for i in range(len(touch)):
        lm = lms[i] if np.isfinite(lms[i]).all() else None
        tips, size = hand_points(lm)
        evs = tracker.update(i / fps, touch[i], tips, size, prob[i])
        gestures += rec.update(evs)
        n_open = len(tracker.active)
        for d in downs.values():
            d["max_open"] = max(d["max_open"], n_open)
        for e in evs:
            if e.phase == "down":
                downs[e.touch_id] = {"ev": e, "max_open": n_open}
            elif e.phase == "up":
                d = downs.pop(e.touch_id)
                size = 0.5 * (d["ev"].hand_size + e.hand_size)
                travel = float(np.hypot(*np.subtract(e.xy, d["ev"].xy))) / max(size, 1e-6)
                touches.append((e.t - d["ev"].t, travel, d["max_open"]))
    return gestures, touches


def _pct(x: np.ndarray) -> list[float]:
    return list(np.percentile(x, [10, 50, 90])) if len(x) else [float("nan")] * 3


def touch_states(tracks, ckpt):
    """-> per track (touch [N, 5] bool, prob [N, 5])."""
    if ckpt is None:
        out = []
        for tr in tracks:
            t = tr.contact & tr.valid[:, None] & tr.present[:, None]
            out.append((t, t.astype(np.float32)))
        return out
    import torch

    from touch.eval.run import binarize, predict, smooth
    from touch.train import build_model, load_config, pick_device

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = load_config(ck["config"])
    device = pick_device()
    model = build_model({**cfg, "pretrained": False}).to(device)
    model.load_state_dict(ck["model"])
    probs = predict(model, tracks, cfg["T"], cfg["input_repr"], device)
    return [(binarize(p, ck["post"]), smooth(p, ck["post"]["k"])) for p in probs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    cfg = GestureConfig()
    seqs = [
        s
        for s in load_catalog(source="egopressure", split=args.split)
        if s.meta["gesture"] in TOUCH_TABLE
    ]
    tracks = load_tracks(seqs, ["overhead"], lm_feats="v1")
    states = touch_states(tracks, args.ckpt)
    counts, touches, n_seq = defaultdict(Counter), defaultdict(list), Counter()
    for tr, (touch, prob) in zip(tracks, states, strict=True):
        g = tr.seq.meta["gesture"]
        lms = primary_hand(tr.seq.landmarks("overhead"))
        gestures, ts = run_sequence(lms, touch, prob, tr.seq.meta["fps"], cfg)
        counts[g].update(x.kind for x in gestures)
        touches[g] += ts
        n_seq[g] += 1

    src = args.ckpt or "ground truth"
    print(f"split {args.split}, touches from {src}; GestureConfig {asdict(cfg)}\n")
    print("gestures per sequence (mean)")
    print(f"{'':12s}" + "".join(f"{k:>13s}" for k in ("touches", *KINDS)))
    for g in TOUCH_TABLE:
        n = n_seq[g]
        row = [len(touches[g]) / n] + [counts[g][k] / n for k in KINDS]
        print(f"{g:12s}" + "".join(f"{v:13.1f}" for v in row))
    print("\ntouch duration s / travel (hand sizes) / max fingers down: p10 p50 p90")
    report = {"split": args.split, "source": src, "config": asdict(cfg), "per_type": {}}
    for g in TOUCH_TABLE:
        a = np.array(touches[g]) if touches[g] else np.zeros((0, 3))
        d, tv, mo = (_pct(a[:, c]) for c in range(3))
        print(
            f"{g:12s} dur {d[0]:.2f} {d[1]:.2f} {d[2]:.2f} | travel {tv[0]:.2f} {tv[1]:.2f} "
            f"{tv[2]:.2f} | fingers {mo[0]:.0f} {mo[1]:.0f} {mo[2]:.0f}"
        )
        report["per_type"][g] = {
            "n_seq": n_seq[g],
            "gestures": dict(counts[g]),
            "n_touches": len(a),
            "duration_s_p10_50_90": [round(float(x), 3) for x in d],
            "travel_hs_p10_50_90": [round(float(x), 3) for x in tv],
            "max_fingers_p10_50_90": [float(x) for x in mo],
        }
    tag = "gt" if args.ckpt is None else Path(args.ckpt).parent.name
    out = Path("results") / f"gesture_report_{args.split}_{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
