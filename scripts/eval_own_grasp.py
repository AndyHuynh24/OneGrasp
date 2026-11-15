"""Score grasp checkpoints on our own key-labelled sessions (data/own_grasp/<split>) by running the
LIVE pipeline over each session video: MediaPipe (frames_with_hands) -> HandTracker ->
LiveGrasp ONNX -> learned grab gate (runs/grab_fusion/<run>/fusion_hold.json, release profile
as live_sim --release). Labels: the cleaned state of scripts/prepare_own_grasp.py.

    python scripts/eval_own_grasp.py --split test --ckpt runs/g9_.../best.pt runs/g10_.../best.pt
Run --split test only for the final comparison (never to choose anything).
Writes results/own_grasp/<split>_<run>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
import grab_fusion as G  # noqa: E402
from live_grasp import HandTracker, LiveGrasp, frames_with_hands  # noqa: E402

from touch.data.grasp_store import OWN_PROC  # noqa: E402
from touch.interact.grab_gate import FusionGate  # noqa: E402

OWN = Path("data/own/grasp")


def run_session(ckpt: Path, name: str, split: str, release: str) -> dict:
    tr = np.load(OWN_PROC / split / name / "track.npz")
    y, present_prep = tr["state"], tr["present"]
    live = LiveGrasp(ckpt, threads=4)
    fdir = Path("runs/grab_fusion") / ckpt.parent.name
    fusion = FusionGate.from_dir(fdir, "hold")
    prof = json.loads((fdir / "fusion_hold.json").read_text()).get("release_profiles", {})
    for k, v in prof.get(release, {}).get("release", {}).items():
        setattr(fusion.hyst, k, v)
    cap = cv2.VideoCapture(str(OWN / name / "video.mp4"))
    tracker = HandTracker(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    closed, state, pres, score, pnone = [], [], [], [], []
    for frame, hands, handedness, _, _ in frames_with_hands(cap, name, 30.0, threading.Event()):
        lm, left = tracker.pick(hands, handedness)
        out = live.step(live.prep(frame, lm, left))
        extra = {}
        if fusion.needs_frames:
            extra = {"joints_pred": out.get("joints"), "lm": lm, "bgr": frame,
                     "gray": cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)}  # fmt: skip
        closed.append(fusion.step(out["state"], out["finger"], lm is not None, **extra))
        state.append(int(out["state"].argmax()) if lm is not None else 0)
        pres.append(lm is not None)
        score.append(fusion.score)
        pnone.append(float(out["state"][0]))
    n = min(len(closed), len(y))
    return {"closed": np.array(closed[:n]), "state": np.array(state[:n]), "y": y[:n],
            "present": np.array(pres[:n]), "present_prep": present_prep[:n],
            "score": np.array(score[:n]), "p_none": np.array(pnone[:n])}  # fmt: skip


def metrics(rs: list[dict]) -> dict:
    y = np.concatenate([r["y"] for r in rs])
    d = {"y_any2": y, "tid": np.concatenate([np.full(len(r["y"]), k) for k, r in enumerate(rs)]),
         "present": np.concatenate([r["present"] for r in rs])}  # fmt: skip
    closed = np.concatenate([r["closed"] for r in rs])
    st = np.concatenate([r["state"] for r in rs])
    m = y >= 0
    per = {}
    for c, name in enumerate(["none", "touch", "hold"]):
        tp = int(((st == c) & (y == c) & m).sum())
        fp, fn = int(((st == c) & (y != c) & m).sum()), int(((st != c) & (y == c) & m).sum())
        per[name] = round(2 * tp / max(2 * tp + fp + fn, 1), 4)
    return {"gate": G.score(d, closed), "model_state_f1": per,
            "hand_tracked": round(float(d["present"].mean()), 3)}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--release", default="fast",
                    choices=["normal", "fast", "fastest", "own_fast"])  # fmt: skip
    ap.add_argument("--dump", action="store_true",
                    help="save per-frame gate scores (tune_release_profiles --own)")  # fmt: skip
    args = ap.parse_args()
    names = sorted(p.name for p in (OWN_PROC / args.split).iterdir())
    out = Path("results/own_grasp")
    out.mkdir(parents=True, exist_ok=True)
    for ck in args.ckpt:
        rs = [run_session(ck, n, args.split, args.release) for n in names]
        res = {"ckpt": str(ck), "split": args.split, "sessions": names,
               "release": args.release, **metrics(rs)}  # fmt: skip
        (out / f"{args.split}_{ck.parent.name}.json").write_text(json.dumps(res, indent=1))
        if args.dump:  # per-frame gate inputs (release-agnostic: the score is the MLP's)
            np.savez(out / f"{args.split}_{ck.parent.name}_frames.npz",
                     **{k: np.concatenate([r[k] for r in rs]) for k in
                        ("y", "present", "score", "p_none", "closed")},
                     tid=np.concatenate([np.full(len(r["y"]), i)
                                         for i, r in enumerate(rs)]))  # fmt: skip
        g = res["gate"]
        print(
            f"{ck.parent.name} [{args.split}]: gate F1 {g['f1']}, grabbed {g['grabbed']}, "
            f"false release {g['false_release']}, close delay {g['close_delay']}, release "
            f"delay {g['release_delay']} frames, false grabs/min {g['false_grab_per_min']}; "
            f"model state F1 {res['model_state_f1']}; hand {res['hand_tracked']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
