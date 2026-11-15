"""Live check of the cylinder pose estimate (touch.cup_pose: SAM 2.1 Hiera-L + silhouette fit
through the rig calibration): click the cylinder in the camera window -> its outline (green =
trusted fit), standing / lying, heading, position in the sim's world frame.

    python scripts/cup_pose_check.py --cam 1
    keys: click = estimate at that point   SPACE = again at the last point   S = save   Q = quit
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]

from touch import cup_pose as CP  # noqa: E402
from touch.live_robot import CYL_D, CYL_H  # noqa: E402

TITLE = "cup pose check (click the cylinder, Q quits)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="1", help="webcam index or an image / video file")
    ap.add_argument("--calib", type=Path, default=Path("results/rig/calib.json"))
    ap.add_argument("--obj-h", type=float, default=CYL_H)
    ap.add_argument("--obj-d", type=float, default=CYL_D)
    args = ap.parse_args()
    cal = json.loads(args.calib.read_text())
    K, c2w, size = np.asarray(cal["K"]), np.asarray(cal["cam_to_world"])[:3], cal["image_size"]
    flip = cal.get("flip", "")
    print("loading SAM 2.1 Hiera-L ...", flush=True)
    seg = CP.Segmenter()
    src = int(args.cam) if args.cam.isdigit() else args.cam
    still = isinstance(src, str) and Path(src).suffix.lower() in (".jpg", ".jpeg", ".png")
    cap = None if still else cv2.VideoCapture(src)
    if cap is not None and isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    view_w = 1280
    click = {"pt": None, "new": False}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            click["pt"], click["new"] = (x, y), True

    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(TITLE, on_mouse)
    shown, frame = None, None
    while True:
        if still:
            frame = cv2.imread(src) if frame is None else frame
        else:
            ok, f = cap.read()
            if ok:
                if "h" in flip and "v" in flip:
                    f = cv2.flip(f, -1)
                elif "h" in flip:
                    f = cv2.flip(f, 1)
                elif "v" in flip:
                    f = cv2.flip(f, 0)
                frame = f
        if frame is None:
            continue
        sc = view_w / frame.shape[1]
        if click["new"]:
            click["new"] = False
            uv = np.array(click["pt"], float) / sc  # px of this frame
            Ki = CP.scale_K(K, size, frame.shape)
            hint = CP.ray_to_plane(Ki, c2w, uv, args.obj_d / 2)[:2]
            t0 = time.time()
            est = CP.estimate(seg, frame, K, size, c2w, args.obj_h, args.obj_d, hint)
            kind = (
                f"lying, heading {math.degrees(est['yaw']):.0f} deg" if est["lying"] else "standing"
            )
            xy = "-" if est["xy"] is None else np.round(est["xy"], 3).tolist()
            print(f"{kind}  xy {xy} m  IoU {est['iou']:.2f} (other pose {est['iou_other']:.2f})  "
                  f"{'trusted' if est['ok'] else 'NOT trusted'}  {time.time() - t0:.1f} s",
                  flush=True)  # fmt: skip
            shown = CP.draw(frame, K, size, c2w, est, args.obj_h, args.obj_d)
            still_until = time.time() + 4.0
        img = shown if shown is not None and time.time() < still_until else frame
        cv2.imshow(TITLE, cv2.resize(img, (view_w, int(img.shape[0] * sc))))
        k = cv2.waitKey(1 if not still else 30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" ") and click["pt"] is not None:
            click["new"] = True
        if k == ord("s") and shown is not None:
            out = Path("results/rig") / f"cup_pose_{time.strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(out), shown)
            print("saved", out)
    if cap is not None:
        cap.release()


if __name__ == "__main__":
    main()
