"""Demo rig calibration with live guidance: webcam + printed ArUco markers (scripts/make_markers.py)
flat on the table. Marker 0 defines the table frame: origin at its centre, x to the right along
its printed top edge, y toward that top edge, z up. Use the same --flip as the live demo.

Two parts (Andy 2026-11-14: one-marker calibrations gave f 1141 in the morning and 1313 in the
afternoon for the same camera -> the hand's depth was off):
  1. the focal length, ONCE per camera + resolution (+ Camo lens / zoom): both markers in view,
     the camera tilted >= 20 deg from straight down. Saved to results/rig/intrinsics.json.
  2. the camera's pose, every time the camera moves: marker 0 alone, the focal length fixed
     (well conditioned). Saved to results/rig/calib.json (the old one kept in results/rig/
     history/).
The view is re-checked continuously over the last --frames frames; a list of checks (marker
size, edge margin, steadiness, fit error, tilt; for the focal length also marker 1 and how
steady the estimate is) says what to adjust. S saves only when every check passes.

    python scripts/calib_rig.py                    # pose (focal length from intrinsics.json)
    python scripts/calib_rig.py --measure-focal    # the focal length (both markers)
keys: s = save, f = switch focal length / pose mode, h / v = flip, q = quit
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch.rig import calibrate, detect, view_quality  # noqa: E402

OUT = Path("results/rig/calib.json")
INTR = Path("results/rig/intrinsics.json")
HIST = Path("results/rig/history")
GOOD, BAD, WHITE = (120, 220, 120), (90, 90, 255), (255, 255, 255)


def draw_axes(img: np.ndarray, cal: dict, length: float = 0.1) -> None:
    c2w = np.asarray(cal["cam_to_world"])
    R, t = c2w[:, :3].T, -c2w[:, :3].T @ c2w[:, 3]  # table -> camera
    pts = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]], np.float64)
    uv = cv2.projectPoints(pts, cv2.Rodrigues(R)[0], t, np.asarray(cal["K"]), None)[0]
    o = tuple(uv[0, 0].astype(int))
    for k, col in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0)), strict=True):
        cv2.line(img, o, tuple(uv[k, 0].astype(int)), col, 4, cv2.LINE_AA)


def load_focal(w: int, h: int) -> float | None:
    """The saved focal length for this image size, else None."""
    if not INTR.exists():
        return None
    d = json.loads(INTR.read_text())
    return float(d["f"]) if d.get("image_size") == [w, h] else None


def backup(p: Path) -> None:
    if p.exists():
        HIST.mkdir(parents=True, exist_ok=True)
        (HIST / f"{p.stem}_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(p.read_text())


def text(img, s, x, y, col=WHITE, scale=0.9, th=2) -> None:
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--size-mm", type=float, default=100.0, help="printed marker side (black)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--frames", type=int, default=30, help="frames per check (rolling)")
    ap.add_argument("--hfov", type=float, default=70.0, help="fallback horizontal FOV (deg)")
    ap.add_argument("--measure-focal", action="store_true",
                    help="measure the focal length (both markers, tilted view)")  # fmt: skip
    ap.add_argument(
        "--flip",
        default="hv",
        choices=["", "h", "v", "hv"],
        help="flip the image exactly as the live demo will (h left-right, v up-down)",
    )
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    size = args.size_mm / 1000
    buf: collections.deque = collections.deque(maxlen=args.frames)
    flip = {"h": "h" in args.flip, "v": "v" in args.flip}
    measure_f = args.measure_focal
    cal, checks, n_seen, note = None, [], 0, ""
    win = "calib_rig (red x, green y, blue up)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    while True:
        ok, img = cap.read()
        if not ok:
            sys.exit("camera read failed")
        fl = ("h" if flip["h"] else "") + ("v" if flip["v"] else "")
        if fl:
            img = cv2.flip(img, {"h": 1, "v": 0, "hv": -1}[fl])
        h, w = img.shape[:2]
        f_saved = load_focal(w, h)
        if f_saved is None and not measure_f:
            measure_f = True
            note = f"no focal length saved for {w}x{h}: measuring it first (both markers)"
        corners = detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        for i, c in corners.items():
            cv2.polylines(img, [c.astype(np.int32)], True, (0, 255, 255), 3, cv2.LINE_AA)
            text(img, str(i), *c[0].astype(int))
        if 0 in corners:
            buf.append(corners)
        n_seen += 1
        if len(buf) >= args.frames and n_seen % 5 == 0:  # re-check the view
            mean = {0: np.mean([b[0] for b in buf], 0)}
            ones = [b[1] for b in buf if 1 in b]
            if len(ones) >= 0.8 * len(buf):
                mean[1] = np.mean(ones, 0)
            cal = calibrate(mean, size, w, h, args.hfov, None if measure_f else f_saved)
            checks = view_quality(list(buf), cal, w, h, size, measure_f)
        if 0 not in corners:
            buf.clear()
            cal, checks = None, []
        # ---- overlay
        mode = "FOCAL LENGTH (both markers)" if measure_f else f"POSE (focal {f_saved:.0f} px)"
        text(img, f"mode: {mode}    flip: {fl or 'none'}", 24, 44)
        y = 96
        if cal is None:
            text(img, "marker 0 not in view: lay it flat, fully visible, no hand over it"
                 if 0 not in corners else f"collecting {len(buf)}/{args.frames} frames ...",
                 24, y, BAD if 0 not in corners else WHITE)  # fmt: skip
        else:
            draw_axes(img, cal)
            for c in checks:
                text(img, f"{'OK ' if c['ok'] else 'FIX'}  {c['name']}: {c['value']}", 24, y,
                     GOOD if c["ok"] else BAD)  # fmt: skip
                y += 40
            text(img, f"f {cal['f']:.0f} px   camera {cal['camera_height_m'] * 100:.0f} cm "
                 f"above the table", 24, y + 10)  # fmt: skip
            bad = [c for c in checks if not c["ok"]]
            msg, col = ((f"ADJUST: {bad[0]['tip']}", BAD) if bad
                        else ("GOOD - press S to save", GOOD))  # fmt: skip
            text(img, msg, 24, h - 40, col, 1.0, 2)
        if note:
            text(img, note, 24, h - 90, WHITE, 0.8)
        cv2.imshow(win, img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k in (ord("h"), ord("v")):
            flip[chr(k)] = not flip[chr(k)]
            buf.clear()
        if k == ord("f"):
            measure_f = not measure_f if f_saved is not None else True
            buf.clear()
            note = ""
        if k == ord("s"):
            if cal is None or any(not c["ok"] for c in checks):
                note = "not saved: fix the red checks first"
                continue
            if measure_f:
                backup(INTR)
                INTR.write_text(json.dumps({"f": cal["f"], "image_size": [w, h],
                                            "f_source": cal["f_source"],
                                            "measured_at": time.strftime("%Y-%m-%d %H:%M"),
                                            "markers": cal["markers_seen"]}, indent=1))  # fmt: skip
                print(f"wrote {INTR}: f {cal['f']:.1f} px", flush=True)
            backup(OUT)
            js = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cal.items()}
            js["flip"] = fl
            js["calibrated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(js, indent=1))
            print(f"wrote {OUT}: f {cal['f']:.1f} px ({cal['f_source']}), camera "
                  f"{cal['camera_height_m'] * 100:.0f} cm up, fit {cal['reproj_px']:.2f} px",
                  flush=True)  # fmt: skip
            note = f"saved {OUT}" + (f" + {INTR}" if measure_f else "")
            if measure_f:
                measure_f = False
                buf.clear()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
