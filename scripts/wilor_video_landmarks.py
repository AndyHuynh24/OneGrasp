"""OFFLINE hand detection + keypoints with the official WiLoR (CVPR 2025,
github.com/rolpotamias/WiLoR; weights CC BY-NC-ND: evaluation / research only) and its own YOLO
hand detector, for videos where
MediaPipe misses hands (gloves, hands at the frame edge). On sampled frames MediaPipe found a hand
in 61 % of Transfyr FPV frames and 3 % of a white-latex-glove video; this detector in 99 % / 37 %.
Same 21-keypoint order as MediaPipe (WiLoR uses the OpenPose hand order = MediaPipe's).

Runs inside third_party/WiLoR with its venv:
    cd third_party/WiLoR && PYOPENGL_PLATFORM=darwin TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \\
        ../wilor_venv/bin/python ../../scripts/wilor_video_landmarks.py \\
        --video V.mp4 --out ../../data/wilor_lm/V.npz [--start 0 --end 100 --width 960 --fps 30]
(several --video files + --out FOLDER: one model load, writes FOLDER/<video stem>.npz, skips done)
(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD: the official detector.pt is an old full-pickle checkpoint.)
--video may also be a folder of images (sorted by name, --fps sets their rate).

Writes, per processed frame (index 0 = left hand, 1 = right; one per side = largest box):
  lm    [N, 2, 21, 3] f32  x, y px in the resized frame; z = MediaPipe-style relative depth
                           (joint depth - wrist depth) * focal / wrist depth, in px; NaN = no hand
  j3d   [N, 2, 21, 3] f32  camera-space joints (m, WiLoR's assumed focal length, un-mirrored)
  conf  [N, 2] f32         detector confidence (0 = none)
  t     [N] seconds, width (resized frame width)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")


GLOVE_HSV = {"blue": ((95, 120, 60), (115, 255, 255))}  # OpenCV HSV (H 0-180); nitrile blue
MIN_BLOB = 2500 / (960 * 540)  # min glove area, fraction of the frame
MERGE_PX = 30 / 960  # blobs whose boxes come this close (fraction of width) are one hand


class GloveBoxes:
    """Hand boxes from glove colour, for footage where the learned detector misses gloved hands
    (BioVL2 blue nitrile: YOLO missed most pipette-holding hands even at conf 0.05). Blobs of the
    glove colour, merged when a held object splits one glove, at most 2 kept (largest). Side:
    with 2 hands the left box is the left hand; with 1, the side whose last box centre is nearer
    (egocentric view: hands rarely cross). Only for videos where the gloves have this colour."""

    def __init__(self, colour: str = "blue"):
        self.lo, self.hi = GLOVE_HSV[colour]
        self.last = {0: None, 1: None}

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        m = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), self.lo, self.hi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        boxes = [[x, y, x + bw, y + bh, a] for x, y, bw, bh, a in st[1:]
                 if a >= MIN_BLOB * h * w]  # fmt: skip
        gap = MERGE_PX * w
        merged = True
        while merged:  # union boxes that touch after growing by `gap`
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    if (a[0] - gap < b[2] and b[0] - gap < a[2]
                            and a[1] - gap < b[3] and b[1] - gap < a[3]):  # fmt: skip
                        boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]),
                                    max(a[3], b[3]), a[4] + b[4]]  # fmt: skip
                        del boxes[j]
                        merged = True
                        break
                if merged:
                    break
        boxes = sorted(boxes, key=lambda b: -b[4])[:2]
        if not boxes:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
        b = np.array([x[:4] for x in boxes], np.float64)
        cx = (b[:, 0] + b[:, 2]) / 2
        if len(b) == 2:
            side = (cx > cx.min()).astype(np.float64) if cx[0] != cx[1] else np.array([0.0, 1.0])
        else:
            d = {s: abs(cx[0] - c) for s, c in self.last.items() if c is not None}
            side = np.array([float(min(d, key=d.get) if d else cx[0] > w / 2)])
        for s, c in zip(side, cx, strict=True):
            self.last[int(s)] = c
        return b, side, np.ones(len(b))


def frames(src: Path, fps: float):
    """(index, seconds, BGR frame) from a video or a folder of images."""
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for i, p in enumerate(files):
            yield i, i / fps, cv2.imread(str(p))
        return
    cap = cv2.VideoCapture(str(src))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    every = max(1, round(src_fps / fps))
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        i += 1
        if i % every == 0:
            yield i, i / src_fps, frame


def main() -> None:
    from torch.utils.data import default_collate
    from ultralytics import YOLO
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils.renderer import cam_crop_to_full

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from wilor_run import load

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, nargs="+", required=True, help="videos / image folders")
    ap.add_argument("--out", type=Path, required=True, help="X.npz (one video) or a folder")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--conf", type=float, default=0.3, help="detector threshold (official demo)")
    ap.add_argument("--hands", default="yolo", choices=["yolo", "glove-blue"],
                    help="hand boxes: WiLoR's YOLO, or blue-glove colour (BioVL2)")  # fmt: skip
    ap.add_argument("--batch", type=int, default=16, help="hand crops per WiLoR forward pass")
    ap.add_argument("--cpu", action="store_true", help="leave the GPU to other jobs")
    args = ap.parse_args()
    mps = torch.backends.mps.is_available() and not args.cpu
    dev = torch.device("mps" if mps else "cpu")
    model, cfg = load()
    model = model.to(dev)
    det = YOLO("./pretrained_models/detector.pt")

    jobs = [(v, args.out if len(args.video) == 1 and args.out.suffix == ".npz" else
             args.out / f"{v.stem}.npz") for v in args.video]  # fmt: skip
    for video, out in jobs:
        if out.exists() and not args.overwrite:
            continue
        glove = GloveBoxes(args.hands.split("-")[1]) if args.hands.startswith("glove") else None
        lms, j3ds, confs, ts, t0 = [], [], [], [], time.time()
        pend: list = []  # (frame row, side, detector conf, dataset item) waiting for a batch

        def flush(pend: list, lms: list, j3ds: list, confs: list) -> None:
            if not pend:
                return
            batch = default_collate([p[3] for p in pend])
            batch = {k: (v.float() if v.dtype == torch.float64 else v).to(dev)
                     if torch.is_tensor(v) else v for k, v in batch.items()}  # fmt: skip
            with torch.no_grad():
                res = model(batch)
            cam = res["pred_cam"].clone()
            cam[:, 1] = (2 * batch["right"] - 1) * cam[:, 1]
            img_size = batch["img_size"].float()
            focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * img_size.max()
            cam_t = cam_crop_to_full(cam, batch["box_center"].float(), batch["box_size"].float(),
                                     img_size, focal).cpu().numpy()  # fmt: skip
            f = float(focal)
            joints = res["pred_keypoints_3d"].float().cpu().numpy()
            for n, (row, s, score, _) in enumerate(pend):
                J = joints[n].copy()
                J[:, 0] *= 2 * s - 1  # un-mirror left hands (official demo)
                P = J + cam_t[n]
                x = f * P[:, 0] / P[:, 2] + img_size[n, 0].item() / 2
                y = f * P[:, 1] / P[:, 2] + img_size[n, 1].item() / 2
                z = (P[:, 2] - P[0, 2]) * f / P[0, 2]
                lms[row][s] = np.stack([x, y, z], 1)
                j3ds[row][s], confs[row][s] = P, score
            pend.clear()

        for _, t, frame in frames(video, args.fps):
            if t < args.start:
                continue
            if t >= args.end or frame is None:
                break
            h0, w0 = frame.shape[:2]
            frame = cv2.resize(frame, (args.width, round(h0 * args.width / w0)))
            row = len(ts)
            lms.append(np.full((2, 21, 3), np.nan, np.float32))
            j3ds.append(np.full((2, 21, 3), np.nan, np.float32))
            confs.append(np.zeros(2, np.float32))
            ts.append(t)
            if glove is not None:
                boxes, right, score = glove(frame)
            else:
                d = det(frame, conf=args.conf, verbose=False)[0]
                boxes = d.boxes.xyxy.cpu().numpy()
                right = d.boxes.cls.cpu().numpy().astype(np.float64)
                score = d.boxes.conf.cpu().numpy()
            if len(boxes):
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                keep = [max(np.flatnonzero(right == s), key=lambda k: area[k])
                        for s in (0, 1) if (right == s).any()]  # fmt: skip
                ds = ViTDetDataset(cfg, frame, boxes[keep], right[keep], rescale_factor=2.0)
                for n, k in enumerate(keep):
                    pend.append((row, int(right[k]), float(score[k]), ds[n]))
            if len(pend) >= args.batch:
                flush(pend, lms, j3ds, confs)
            if len(ts) % 300 == 0:
                rate = (time.time() - t0) / len(ts)
                print(f"{len(ts)} frames, t={t:.1f} s | {rate:.3f} s/frame", flush=True)
        flush(pend, lms, j3ds, confs)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, lm=np.stack(lms), j3d=np.stack(j3ds), conf=np.stack(confs),
                 t=np.array(ts), width=args.width)  # fmt: skip
        found = np.isfinite(np.stack(lms)[..., 0, 0]).any(1).mean()
        print(
            f"wrote {out} {len(ts)} frames, hand found in {found:.0%}, "
            f"{(time.time() - t0) / max(len(ts), 1):.3f} s/frame"
        )


if __name__ == "__main__":
    main()
