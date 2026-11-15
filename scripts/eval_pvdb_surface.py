"""Top-down "hand on a table, no object" check on the PressureVisionDB TEST fold (held-out
participants; never trained / tuned on): every grasp-state call other than "none" is wrong (there
is no object), fingertip contact is scored against the pressure pad (>= 1 kPa). Runs the live
causal path (LiveGrasp, MediaPipe landmarks from the processed catalog); PVDB is 15 fps, each
frame is fed twice (30 fps model). Optionally renders the first clips side by side.

    python scripts/eval_pvdb_surface.py --ckpts runs/<a>/best.pt runs/<b>/best.pt [--render 4]
writes runs/pvdb_surface/<a>__<b>.json (+ results/pvdb/surface_<a>_vs_<b>.mp4)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from live_grasp import LiveGrasp  # noqa: E402
from showcase_biovl import reset  # noqa: E402

from touch.data.catalog import load_catalog  # noqa: E402
from touch.data.patches import primary_hand  # noqa: E402

NAMES = ["none", "touch", "HOLD"]
COL = [(170, 170, 170), (0, 165, 255), (0, 0, 230)]


def run(live: LiveGrasp, seq, frames: int, render: bool):
    with np.load(seq.root / "landmarks_overhead.npz") as z:
        hand = primary_hand({k: z[k] for k in z.files})[:frames]
    with np.load(seq.root / "fingers.npz") as z:
        fing = np.where(z["valid"][:frames, None], z["contact"][:frames], -1)
    left = seq.meta["hand"] == "left"
    reset(live)
    st, fp, imgs = [], [], []
    for i in range(len(hand)):
        img = cv2.imread(str(seq.image_path("overhead", i)))
        lm = hand[i] if np.isfinite(hand[i]).all() else None
        for _ in range(2):  # 15 -> 30 fps
            o = live.step(live.prep(img, lm, left))
        st.append(int(o["state"].argmax()) if lm is not None else -1)
        fp.append(o["finger"] >= 0.5 if lm is not None else np.zeros(5, bool))
        if render:
            imgs.append((img, lm, st[-1], fp[-1]))
    return np.array(st), np.array(fp), fing, imgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=Path, nargs="+", required=True)
    ap.add_argument("--tracks", type=int, default=24)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--render", type=int, default=0)
    args = ap.parse_args()
    seqs = load_catalog(source="pvdb", split="test")
    assert seqs and all(s.split == "test" for s in seqs)
    rng = np.random.default_rng(0)
    seqs = [seqs[i] for i in sorted(rng.permutation(len(seqs))[: args.tracks])]
    names = [c.parent.name.split("_")[0] for c in args.ckpts]
    res, clips = {}, {n: [] for n in names}
    for c, name in zip(args.ckpts, names, strict=True):
        live = LiveGrasp(c, threads=4)
        S, FP, FG = [], [], []
        for k, s in enumerate(seqs):
            st, fp, fg, imgs = run(live, s, args.frames, k < args.render)
            S.append(st), FP.append(fp), FG.append(fg)
            clips[name].append(imgs)
        st, fp, fg = np.concatenate(S), np.concatenate(FP), np.concatenate(FG)
        m = st >= 0
        mf = (fg >= 0) & m[:, None]
        tp = (fp & (fg == 1))[mf].sum()
        prec, rec = tp / max(fp[mf].sum(), 1), tp / max((fg == 1)[mf].sum(), 1)
        res[name] = {"frames": int(m.sum()),
                     "state_says_touch": round(float((st[m] == 1).mean()), 4),
                     "state_says_hold": round(float((st[m] == 2).mean()), 4),
                     "finger_f1": round(float(2 * prec * rec / max(prec + rec, 1e-9)), 4),
                     "finger_precision": round(float(prec), 4),
                     "finger_recall": round(float(rec), 4)}  # fmt: skip
        print(name, res[name], flush=True)
    out = Path("runs/pvdb_surface") / ("__".join(names) + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tracks": [s.root.name for s in seqs], **res}, indent=1))
    print("wrote", out)
    if args.render:
        vid = Path("results/pvdb") / f"surface_{'_vs_'.join(names)}.mp4"
        vid.parent.mkdir(parents=True, exist_ok=True)
        raw = vid.with_suffix(".raw.mp4")
        vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 15,
                             (480 * len(names), 270))  # fmt: skip
        for k in range(args.render):
            for i in range(len(clips[names[0]][k])):
                tiles = []
                for name in names:
                    img, lm, s, fp = clips[name][k][i]
                    t = img.copy()
                    if lm is not None:
                        for f, tip in enumerate((4, 8, 12, 16, 20)):
                            col = (0, 0, 255) if fp[f] else (220, 220, 220)
                            cv2.circle(t, tuple(lm[tip, :2].astype(int)), 14, col, 4)
                    t = cv2.resize(t, (480, 270))
                    txt = f"{name}: {NAMES[s] if s >= 0 else 'no hand'}"
                    cv2.putText(t, txt, (10, 30), 0, 0.8, (0, 0, 0), 5)
                    cv2.putText(t, txt, (10, 30), 0, 0.8, COL[max(s, 0)], 2)
                    cv2.putText(t, "no object: state should be none", (10, 258), 0, 0.5,
                                (255, 255, 255), 1)  # fmt: skip
                    tiles.append(t)
                vw.write(np.hstack(tiles))
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "26", str(vid)], check=True)  # fmt: skip
        raw.unlink()
        print("wrote", vid)


if __name__ == "__main__":
    main()
