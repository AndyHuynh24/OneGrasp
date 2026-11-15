"""Touch-down times from the tap sounds in a recording (video + microphone): a tap on a desk is a
short broadband click. Weak labels: onset times only (no finger, no release); check them with the
contact sheet before using them.

    python scripts/label_taps_audio.py --video data/own_touch/tap_A.mp4 [--k 8] [--min-gap 0.15]
writes <video>.taps.json ({"taps_s": [...], "fps": ..., "params": ...}) and <video>.taps.jpg
(frame at each detected tap, for a visual check).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

SR = 16000


def audio(video: Path) -> np.ndarray:
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(SR),
           "-f", "s16le", "-"]  # fmt: skip
    raw = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(raw.stdout, np.int16).astype(np.float32) / 32768.0


def onsets(x: np.ndarray, k: float, min_gap: float, hop_s: float = 0.005) -> np.ndarray:
    """Rises of high-frequency (> 1.5 kHz) log energy above its 1 s running median by more than
    k robust standard deviations (MAD), one onset per min_gap."""
    from scipy.ndimage import median_filter
    from scipy.signal import butter, sosfiltfilt

    hp = sosfiltfilt(butter(4, 1500, "highpass", fs=SR, output="sos"), x)
    hop = int(hop_s * SR)
    n = len(hp) // hop
    e = np.log(1e-10 + (hp[: n * hop].reshape(n, hop) ** 2).mean(1))
    d = e - median_filter(e, size=int(1.0 / hop_s), mode="nearest")
    thr = np.median(d) + k * 1.4826 * np.median(np.abs(d - np.median(d)))
    out, last = [], -1e9
    for i in np.flatnonzero((d > thr) & (np.diff(d, prepend=d[0]) > 0)):
        t = i * hop_s
        if t - last >= min_gap:
            out.append(t)
            last = t
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--k", type=float, default=8.0, help="threshold: median + k * MAD")
    ap.add_argument("--min-gap", type=float, default=0.15, help="s between taps")
    args = ap.parse_args()
    taps = onsets(audio(args.video), args.k, args.min_gap)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = args.video.with_suffix(".taps.json")
    res = {"taps_s": [round(float(t), 3) for t in taps], "fps": fps,
           "params": {"k": args.k, "min_gap": args.min_gap}}  # fmt: skip
    out.write_text(json.dumps(res, indent=1))
    tiles = []
    for t in taps[:48]:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, im = cap.read()
        if ok:
            im = cv2.resize(im, (320, 180))
            cv2.putText(im, f"{t:.2f}s", (6, 20), 0, 0.6, (0, 255, 255), 2)
            tiles.append(im)
    if tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
        sheet = np.vstack([np.hstack(tiles[i : i + 6]) for i in range(0, len(tiles), 6)])
        cv2.imwrite(str(args.video.with_suffix(".taps.jpg")), sheet)
    print(f"{len(taps)} taps -> {out}")


if __name__ == "__main__":
    main()
