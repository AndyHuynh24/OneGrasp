"""Before / after video on the HO-Cap TEST set (top-down camera): two models' hand states side by
side, with the ground truth and a timeline strip per hand (grey none / orange touch / red hold).

    python scripts/render_compare_hocap.py \\
        --a runs/hocap/<old>/preds.npz --b runs/hocap/<new>/preds.npz --b-gate gate_<cand>.npz
writes results/hocap/compare_<a>_vs_<b>.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from eval_hocap import gt_labels  # noqa: E402

from touch.data import hocap as H  # noqa: E402
from touch.interact.object_gate import latch  # noqa: E402

NAMES = ["none", "touch", "HOLD"]
COL = [(170, 170, 170), (0, 165, 255), (0, 0, 230)]


def label(img, text, org, scale=0.6, col=(255, 255, 255), th=2):
    cv2.putText(img, text, org, 0, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, 0, scale, col, th, cv2.LINE_AA)


def states(P, G, key: str) -> np.ndarray:
    pres = P[f"{key}/present"]
    s = np.where(pres, P[f"{key}/state"].argmax(1), 0)
    return latch(s, G[f"{key}/near"]) if G is not None else s.astype(np.int8)


def strip(s: np.ndarray, w: int, t: int, h: int = 10) -> np.ndarray:
    x = np.zeros((h, w, 3), np.uint8)
    idx = (np.arange(w) * len(s) / w).astype(int)
    x[:] = np.array(COL, np.uint8)[np.clip(s[idx], 0, 2)][None]
    x[:, int(t * w / len(s))] = 255
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--a-gate", default="")
    ap.add_argument("--b-gate", default="")
    ap.add_argument("--a-name", default="")
    ap.add_argument("--b-name", default="")
    ap.add_argument("--cam", default="105322251564")
    ap.add_argument("--seconds", type=float, default=20.0, help="per sequence")
    ap.add_argument("--seed", type=int, default=None,
                    help="random segment per sequence (default: the middle)")  # fmt: skip
    args = ap.parse_args()
    PA, PB = np.load(args.a), np.load(args.b)
    GA = np.load(args.a.parent / args.a_gate) if args.a_gate else None
    GB = np.load(args.b.parent / args.b_gate) if args.b_gate else None
    na = args.a_name or args.a.parent.name.split("_")[0] + (" + gate" if GA is not None else "")
    nb = args.b_name or args.b.parent.name.split("_")[0] + (" + gate" if GB is not None else "")
    tag = f"_seed{args.seed}" if args.seed is not None else ""
    fname = f"compare_{na}_vs_{nb}{tag}.mp4".replace(" ", "").replace("+", "_")
    out = Path("results/hocap") / fname
    rng = np.random.default_rng(args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 560))
    for seq in H.list_sequences(video_only=True, split="test"):
        sides = [s for s in seq.sides
                 if f"{seq.subject}/{seq.name}/{args.cam}/{s}/present" in PA.files]  # fmt: skip
        if not sides:
            continue
        keys = {s: f"{seq.subject}/{seq.name}/{args.cam}/{s}" for s in sides}
        SA = {s: states(PA, GA, k) for s, k in keys.items()}
        SB = {s: states(PB, GB, k) for s, k in keys.items()}
        GT = {s: gt_labels(seq, s)["state_any2"] for s in sides}
        LM = {s: PA[f"{k}/lm"] for s, k in keys.items()}
        n = seq.n_frames
        m = int(args.seconds * 30)
        a0 = max(0, n // 2 - m // 2) if args.seed is None else int(rng.integers(0, max(1, n - m)))
        for i in range(a0, min(n, a0 + m)):
            img = cv2.imread(str(seq.image(args.cam, i)))
            tiles = []
            for name, S in ((na, SA), (nb, SB)):
                t = img.copy()
                for s in sides:
                    lm = LM[s][i]
                    if not np.isfinite(lm).all():
                        continue
                    k, g = int(S[s][i]), int(GT[s][i])
                    x0, y0 = lm[:, :2].min(0).astype(int)
                    x1, y1 = lm[:, :2].max(0).astype(int)
                    cv2.rectangle(t, (x0, y0), (x1, y1), COL[k], 3)
                    ok = "  (GT ?)" if g < 0 else ("  = GT" if k == g else f"  (GT {NAMES[g]})")
                    label(t, f"{NAMES[k]}{ok}", (x0, max(20, y0 - 8)), 0.6, COL[k])
                label(t, name, (12, 30), 0.9)
                bar = np.vstack([strip(S[s], 640, i) for s in sides]
                                + [strip(np.maximum(GT[s], 0), 640, i) for s in sides])  # fmt: skip
                pad = np.zeros((80 - bar.shape[0], 640, 3), np.uint8)
                label(pad, f"strips: model per hand, then GT per hand ({' / '.join(sides)})",
                      (8, pad.shape[0] - 12), 0.45)  # fmt: skip
                tiles.append(np.vstack([t, bar, pad]))
            vw.write(np.hstack(tiles))
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "26", str(out)], check=True)  # fmt: skip
    raw.unlink()
    print("wrote", out)


if __name__ == "__main__":
    main()
