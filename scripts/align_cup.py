"""Turn a recorded cup-rig session into per-frame contact labels (scripts/record_cup.py output).

    python scripts/align_cup.py data/own/cup/<session>                 # select the LED box once
    python scripts/align_cup.py data/own/cup/<session> --led 600 40 20 20

Steps: LED brightness per frame -> LED edges -> clock fit (t_video = a * t_mcu + b) against the
logged LED edges -> electrode touch (hysteresis) -> per-frame contact [N, 12].
Writes <session>/labels.npz (contact [N, 12] bool, frame_t [N], led [N]) and <session>/align.json
(fit, residual, matched edges). Check `rms_ms` (should be under ~1 frame) before using labels.
Touch thresholds: choose on validation sessions only, never on test sessions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch.data.cup_rig import (  # noqa: E402
    contact_per_frame,
    edges,
    electrode_touch,
    fit_clock,
    led_states,
    parse_log,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--led", type=int, nargs=4, default=None, help="LED box x y w h (px)")
    ap.add_argument("--delta", type=float, default=12.0, help="touch threshold (counts)")
    ap.add_argument("--release", type=float, default=6.0, help="release threshold (counts)")
    args = ap.parse_args()
    s = args.session
    frame_t = np.loadtxt(s / "frames.csv", delimiter=",", skiprows=1, ndmin=2)[:, 1]
    frame_t = frame_t - frame_t[0]
    # the sensor log has host receive times in front; drop them (the LED fit does the sync)
    lines = [ln.split(" ", 1)[1] for ln in (s / "sensor.log").read_text().splitlines() if " " in ln]
    (s / "sensor_mcu.log").write_text("\n".join(lines))
    log = parse_log(s / "sensor_mcu.log")
    cap = cv2.VideoCapture(str(s / "video.mp4"))
    box = args.led
    bright = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if box is None:
            box = cv2.selectROI("draw a box around the LED, then Enter", f)
            cv2.destroyAllWindows()
        x, y, w, h = box
        bright.append(f[y : y + h, x : x + w].mean())
    cap.release()
    n = min(len(bright), len(frame_t))
    led = led_states(np.array(bright[:n]))
    idx, vs = edges(led)
    # an LED change happens somewhere between the previous frame and the one that shows it:
    # use the midpoint (removes a half-frame bias; the camera's own exposure latency is
    # calibrated separately, docs/CUP_RIG.md)
    dt = float(np.median(np.diff(frame_t[:n])))
    fit = fit_clock(frame_t[idx] - dt / 2, vs, log.led_t, log.led_state)
    touch = electrode_touch(log, args.delta, args.release)
    contact = contact_per_frame(log, touch, frame_t[:n], fit)
    np.savez_compressed(s / "labels.npz", contact=contact, frame_t=frame_t[:n], led=led)
    info = dict(a=fit.a, b=fit.b, n_matched=fit.n_matched, video_led_edges=len(idx),
                logged_led_edges=len(log.led_t), rms_ms=round(fit.rms_s * 1000, 2),
                frames=n, led_box=list(map(int, box)), delta=args.delta, release=args.release,
                contact_frames_per_electrode=contact.sum(0).tolist())  # fmt: skip
    (s / "align.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
