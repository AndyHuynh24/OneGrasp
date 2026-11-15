"""Record a labelled grab session on the demo rig (your hand, your webcam, your cylinder).

Grab with one hand; with the OTHER hand press SPACE once when your grabbing hand closes on the
object (holding starts) and once when it lets go (holding ends). Press u to toggle an "unsure"
span (ignored when scoring), q to stop. Frames are saved as JPEG (quality 95) exactly as captured
(after the calibration's flip), so every model can be re-run on them later.

    python scripts/record_grabs.py --cam 1 [--name session_a]
writes data/own/grabs/<name>/{frames/000000.jpg ..., labels.npz, meta.json}
labels.npz: hold [N] (0/1), unsure [N] (bool), t [N] (s since start)
Use: one session to tune on (validation), another, recorded separately, as the test set.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/own/grabs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--calib", type=Path, default=Path("results/rig/calib.json"))
    ap.add_argument("--name", default=time.strftime("session_%Y%m%d_%H%M%S"))
    args = ap.parse_args()
    cal = json.loads(args.calib.read_text()) if args.calib.exists() else {}
    W, H = cal.get("image_size", [1280, 720])
    flip = cal.get("flip", "")
    out = ROOT / args.name
    assert not out.exists(), f"{out} exists: pick another --name"
    (out / "frames").mkdir(parents=True)
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    hold, unsure = False, False
    labels, uns, ts = [], [], []
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter() - t0
        if flip:
            frame = cv2.flip(frame, {"h": 1, "v": 0, "hv": -1}[flip])
        i = len(labels)
        cv2.imwrite(str(out / "frames" / f"{i:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        labels.append(int(hold))
        uns.append(unsure)
        ts.append(t)
        view = frame.copy()
        txt = "HOLDING" if hold else "not holding"
        cv2.putText(
            view,
            f"{txt}{'  (unsure)' if unsure else ''}   {i / 30:.0f} s",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 255) if hold else (255, 255, 255),
            3,
        )
        cv2.putText(view, "SPACE = hold starts / ends   u = unsure   q = stop", (20, H - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # fmt: skip
        cv2.imshow("record grabs", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            hold = not hold
        elif k == ord("u"):
            unsure = not unsure
        elif k in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    n = len(labels)
    np.savez(out / "labels.npz", hold=np.array(labels, np.int8), unsure=np.array(uns),
             t=np.array(ts))  # fmt: skip
    fps = (n - 1) / max(ts[-1] - ts[0], 1e-6) if n > 1 else 0.0
    meta = {"cam": args.cam, "calib": cal, "frames": n, "fps_measured": round(fps, 2),
            "holds": int(np.sum(np.diff(np.r_[0, labels]) == 1))}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}: {n} frames, {fps:.1f} fps, {meta['holds']} holds labelled")


if __name__ == "__main__":
    main()
