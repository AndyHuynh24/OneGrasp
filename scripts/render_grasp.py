"""Render a grasp-model visualisation on DexYCB val sequences (subject 7; test is never used).

Per frame: camera view with fingertip contact + state, the predicted MANO mesh (camera view and a
side view, contact vertices red), per-finger bars, and a predicted vs ground-truth state timeline.
Runs the same causal live pipeline as scripts/live_grasp.py (MediaPipe -> GraspStep ONNX), with the
model history reset at the start of each sequence.

    python scripts/render_grasp.py --ckpt runs/<run>/best.pt --out results/dexycb/grasp_render.mp4
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_grasp import STATES, LiveGrasp, draw  # noqa: E402

from touch.data import dexycb as D  # noqa: E402
from touch.data.landmarks import DEFAULT_MODEL, _make_landmarker  # noqa: E402

PROC = Path("data/dexycb_proc")
STATE_COL = [(170, 170, 170), (0, 200, 255), (0, 0, 255)]  # BGR: none, touch, hold
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def pick_sequences(n: int, cam: str) -> list:
    """Val sequences with a clear none -> touch -> hold arc, varied objects, both hands."""
    out, seen = [], set()
    for s in D.list_sequences():
        if s.split != "val":
            continue
        st = np.load(PROC / s.subject / s.name / "contact.npz")["state"]
        if (st == 2).sum() < 15 or (st == 0).sum() < 10:
            continue
        key = (s.grasp_object, s.side)
        if key in seen:
            continue
        seen.add(key)
        out.append((s, st))
        if len(out) >= n:
            break
    return out


def render_mesh(verts, faces, hot, size, yaw_deg=0.0):
    """Painter's-algorithm render of a root-relative mesh (metres, camera axes) -> BGR image."""
    img = np.full((size, size, 3), 28, np.uint8)
    a = np.deg2rad(yaw_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    v = (verts - verts.mean(0)) @ R.T
    s = 0.8 * size / max(np.ptp(v[:, 0]), np.ptp(v[:, 1]), 1e-6)
    p = v[:, :2] * s + size / 2
    tri = v[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    shade = 0.35 + 0.65 * np.abs(n @ np.array([0.0, -0.3, -1.0]) / np.linalg.norm([0, -0.3, -1]))
    hot_f = hot[faces].mean(1)
    order = np.argsort(-tri[:, :, 2].mean(1))  # far (large z) first
    base = np.array([150, 185, 225.0])  # skin, BGR
    red = np.array([40, 40, 235.0])
    for f in order:
        c = (base * (1 - hot_f[f]) + red * hot_f[f]) * shade[f]
        cv2.fillConvexPoly(img, p[faces[f]].astype(np.int32), tuple(int(x) for x in c), cv2.LINE_AA)
    return img


def timeline(pred, gt, t, w, h):
    img = np.full((h, w, 3), 18, np.uint8)
    n = len(pred)
    xs = np.linspace(0, w, n + 1).astype(int)
    rows = [("model", pred, 26), ("truth", gt, 66)]
    for name, seq, y in rows:
        cv2.putText(img, name, (6, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        for i, k in enumerate(seq):
            col = (60, 60, 60) if k < 0 else STATE_COL[k]
            cv2.rectangle(
                img,
                (70 + xs[i] * (w - 80) // w, y),
                (70 + xs[i + 1] * (w - 80) // w, y + 30),
                col,
                -1,
            )
    x = 70 + xs[t] * (w - 80) // w
    cv2.line(img, (x, 18), (x, 104), (255, 255, 255), 2)
    lx = 70
    for k, name in enumerate(STATES):
        cv2.rectangle(img, (lx, 112), (lx + 14, 126), STATE_COL[k], -1)
        cv2.putText(
            img, name.lower(), (lx + 20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1
        )
        lx += 100
    cv2.rectangle(img, (lx, 112), (lx + 14, 126), (60, 60, 60), -1)
    cv2.putText(
        img, "unlabelled", (lx + 20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1
    )
    return img


def finger_bars(p, w, h):
    img = np.full((h, w, 3), 18, np.uint8)
    for f in range(5):
        y = 14 + f * 22
        cv2.putText(
            img, FINGERS[f], (8, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1
        )
        bw = int((w - 80) * float(p[f]))
        col = (0, 0, 255) if p[f] >= 0.5 else (150, 150, 150)
        cv2.rectangle(img, (70, y), (70 + bw, y + 14), col, -1)
        cv2.rectangle(img, (70, y), (w - 10, y + 14), (90, 90, 90), 1)
    return img


def main() -> None:
    from mediapipe import Image, ImageFormat

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/dexycb/grasp_render.mp4"))
    ap.add_argument("--n-seq", type=int, default=8)
    ap.add_argument("--cam", type=int, default=1, help="index into dexycb.CAMS")
    args = ap.parse_args()
    cam = D.CAMS[args.cam]
    live = LiveGrasp(args.ckpt, threads=4)
    faces = {s: np.asarray(D.mano_model(s)["f"]).astype(np.int64) for s in ("right", "left")}
    seqs = pick_sequences(args.n_seq, cam)
    print("sequences:", [(s.name, s.side, s.grasp_object) for s, _ in seqs])
    W, H, M = 640, 480, 240
    writer = cv2.VideoWriter(
        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W + 2 * M, H + 132)
    )
    ms = 0
    with _make_landmarker(DEFAULT_MODEL, 0.3) as det:
        for s, gt in seqs:
            live.hist = deque(
                [(live.pad_tok, live.pad_he, False)] * (live.T - 1), maxlen=live.T - 1
            )
            live.lm_hist.clear()
            live.last_imgs = None
            live.sticky.prev = 0
            frames, outs = [], []
            for i in range(s.n_frames):
                frame = cv2.imread(str(s.image(cam, i)))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                r = det.detect_for_video(Image(image_format=ImageFormat.SRGB, data=rgb), ms)
                ms += 33
                lm, left = None, False
                if r.hand_landmarks:
                    k = int(np.argmax([hd[0].score for hd in r.handedness]))
                    lm = np.array(
                        [[p.x * W, p.y * H, p.z * W] for p in r.hand_landmarks[k]], np.float32
                    )
                    left = r.handedness[k][0].category_name == "Left"
                out = live.step(live.prep(frame, lm, left))
                frames.append((frame, lm, left))
                outs.append(out)
            pred = [int(o["state"].argmax()) for o in outs]
            for t, ((frame, lm, left), out) in enumerate(zip(frames, outs, strict=True)):
                cv_img = frame.copy()
                draw(cv_img, lm, out, left, 30.0)
                cv2.rectangle(cv_img, (W - 160, 8), (W, 40), (0, 0, 0), -1)  # hide fps (offline)
                cv2.putText(cv_img, f"{s.grasp_object[4:]}  {s.side}", (16, H - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # fmt: skip
                v = out["verts"].copy()
                if left:  # undo training-time mirroring
                    v[:, 0] *= -1
                fc = faces["left" if left else "right"]
                hot = (out["vert"] >= 0.5).astype(np.float32)
                if lm is None:
                    m1 = m2 = np.full((M, M, 3), 28, np.uint8)
                    cv2.putText(
                        m1,
                        "no hand",
                        (70, M // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (200, 200, 200),
                        2,
                    )
                else:
                    m1 = render_mesh(v, fc, hot, M, 0)
                    m2 = render_mesh(v, fc, hot, M, 90)
                cv2.putText(
                    m1,
                    "mesh: camera view",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (230, 230, 230),
                    1,
                )
                cv2.putText(
                    m2,
                    "mesh: side view",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (230, 230, 230),
                    1,
                )
                bars = finger_bars(out["finger"], 2 * M, H - M)
                right = np.vstack([np.hstack([m1, m2]), bars])
                top = np.hstack([cv_img, right])
                tl = timeline(pred, gt.tolist(), t, W + 2 * M, 132)
                writer.write(np.vstack([top, tl]))
    writer.release()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
