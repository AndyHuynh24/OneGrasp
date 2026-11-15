"""Lab showcase on BioVL2 (Nishimura et al., Zenodo 10.5281/zenodo.21495188, CC BY 4.0):
egocentric biochemistry videos (gloved hands, pipettes, tubes) + protocol steps with start / end
times. No hand or contact labels exist, so this is qualitative, plus one weak check: how often
the model says HOLD inside protocol steps vs between them (scientists hold a pipette / tube
during a step; between steps hands are more often free or resting).

Runs the live causal pipeline (scripts/live_grasp.py) on both hands: MediaPipe 2 hands, one
LiveGrasp state per handedness label.

    python scripts/showcase_biovl.py --video data/biovl2/BioVL2/videos/pcr/pcr_2.mp4 \\
        --protocol data/biovl2/BioVL2/protocols/pcr/pcr_2.csv --ckpt runs/<run>/best.pt
writes results/biovl/<video>_<run>.mp4 and .json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from live_grasp import LiveGrasp  # noqa: E402

from touch.data.landmarks import DEFAULT_MODEL, _make_landmarker  # noqa: E402
from touch.interact.grab_gate import FusionGate  # noqa: E402

STATES = ["none", "touch", "HOLD"]
COL = [(200, 200, 200), (0, 200, 255), (0, 0, 255)]


def label(img, text, org, scale=0.8, col=(255, 255, 255), th=2):
    cv2.putText(img, text, org, 0, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, 0, scale, col, th, cv2.LINE_AA)


def reset(live: LiveGrasp) -> None:
    live.hist = deque([(live.pad_tok, live.pad_he, False)] * (live.T - 1), maxlen=live.T - 1)
    live.lm_hist.clear()
    live.last_imgs = None
    live.sticky.prev = 0


def main() -> None:
    from mediapipe import Image, ImageFormat

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, default=None, help="BioVL2 step CSV (optional)")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--width", type=int, default=960, help="processing / output width")
    ap.add_argument("--fps", type=float, default=30.0, help="model rate (the model is 30 fps)")
    ap.add_argument("--landmarks", type=Path, default=None,
                    help="wilor_video_landmarks.py output, replaces MediaPipe")  # fmt: skip
    ap.add_argument("--tag", default="", help="suffix for the output name")
    ap.add_argument("--out-dir", type=Path, default=Path("results/biovl"))
    ap.add_argument("--gate", default="start",
                    help="grab gate variant in runs/grab_fusion/<run>/ ('start' = "
                         "probabilities only: no table calibration here, so no 3D-motion "
                         "features); '' = off")  # fmt: skip
    args = ap.parse_args()
    steps = []
    if args.protocol:
        with open(args.protocol) as fh:
            rd = csv.DictReader(fh)
            steps = [(r["sentence"], float(r["start"]), float(r["end"])) for r in rd]
    cap = cv2.VideoCapture(str(args.video))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    every = max(1, round(src_fps / args.fps))
    hands = {"Left": LiveGrasp(args.ckpt, threads=4), "Right": LiveGrasp(args.ckpt, threads=4)}
    for h in hands.values():
        reset(h)
    gates = {}
    if args.gate:
        fdir = Path("runs/grab_fusion") / args.ckpt.parent.name
        gates = {n: FusionGate.from_dir(fdir, args.gate) for n in hands}
    out = args.out_dir / f"{args.video.stem}_{args.ckpt.parent.name}{args.tag}.mp4"
    pre = dict(np.load(args.landmarks)) if args.landmarks else None
    if pre is not None:
        assert int(pre["width"]) == args.width, "landmarks were made at another width"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".raw.mp4")
    vw = None
    rows, i = [], -1
    with _make_landmarker(DEFAULT_MODEL, 0.3) as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % every:
                continue
            t = i / src_fps
            h0, w0 = frame.shape[:2]
            frame = cv2.resize(frame, (args.width, round(h0 * args.width / w0)))
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            seen = {}
            if pre is not None:
                j = np.flatnonzero(np.abs(pre["t"] - t) < 0.5 / args.fps)
                if len(j) == 0:
                    if t > pre["t"][-1]:
                        break
                    continue
                for k, name in enumerate(("Left", "Right")):
                    if np.isfinite(pre["lm"][j[0], k]).all():
                        seen[name] = pre["lm"][j[0], k]
            else:
                img = Image(image_format=ImageFormat.SRGB, data=rgb)
                r = det.detect_for_video(img, int(t * 1000))
                for lms, hd in zip(r.hand_landmarks, r.handedness, strict=True):
                    name = hd[0].category_name
                    if name in hands and name not in seen:
                        seen[name] = np.array(
                            [[p.x * w, p.y * h, p.z * w] for p in lms], np.float32
                        )
            step = next((s for s, a, b in steps if a <= t < b), None)
            row = {"t": round(t, 3), "step": step}
            for name, live in hands.items():
                lm = seen.get(name)
                o = live.step(live.prep(frame, lm, name == "Left"))
                k = int(o["state_sticky"]) if lm is not None else 0
                row[name], row[f"{name}_tracked"] = k, lm is not None
                if gates:
                    hold = gates[name].step(o["state"], o["finger"], lm is not None)
                    row[f"{name}_gate"] = bool(hold)
                    row[f"{name}_p"] = [round(float(v), 3) for v in o["state"]]
                    if lm is not None:  # the gate decides HOLD; touch / none from the model
                        k = 2 if hold else min(k, 1)
                if lm is not None:
                    x0, y0 = lm[:, :2].min(0).astype(int)
                    x1, y1 = lm[:, :2].max(0).astype(int)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), COL[k], 3)
                    label(frame, f"{name.lower()}: {STATES[k]}", (x0, max(24, y0 - 8)), 0.8, COL[k])
            rows.append(row)
            if steps:
                label(frame, f"protocol: {step or '(between steps)'}", (14, h - 18), 0.75)
            label(frame, f"{t:6.1f} s", (w - 120, 30), 0.7)
            if vw is None:
                vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            vw.write(frame)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "26", str(out)], check=True)  # fmt: skip
    raw.unlink()
    held = np.array([r.get("Left_gate", r["Left"] == 2) or r.get("Right_gate", r["Right"] == 2)
                     for r in rows])  # fmt: skip
    in_step = np.array([r["step"] is not None for r in rows])
    per_step = {
        s: round(float(held[[r["step"] == s for r in rows]].mean()), 3) for s, _, _ in steps
    }
    rate = lambda m: round(float(held[m].mean()), 3) if m.any() else None  # noqa: E731
    summary = {"frames": len(rows), "hold_rate": rate(np.ones_like(held)),
               "hold_rate_in_steps": rate(in_step), "hold_rate_between_steps": rate(~in_step),
               "hold_rate_per_step": per_step,
               "hand_tracked_rate": {n: round(float(np.mean([r[f"{n}_tracked"] for r in rows])), 3)
                                     for n in hands}}  # fmt: skip
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "frames": rows}))
    print(json.dumps(summary, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
