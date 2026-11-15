"""Printable table markers for the live robot-sim demo: ArUco DICT_4X4_50, ids 0 and 1, one per
US Letter page, black square exactly --size-mm wide, with a ruler line to check the print scale.
Print at 100% / "Actual size" (no "fit to page"). Marker id 0 = table origin.

    python scripts/make_markers.py [--size-mm 100] [--out results/print/table_markers.pdf]
    python scripts/make_markers.py --clean   # only the marker + its ID (for the demo table)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DICT = cv2.aruco.DICT_4X4_50
DPI = 300
PAGE_IN = (8.5, 11.0)  # US Letter


def mm_px(mm: float) -> int:
    return round(mm / 25.4 * DPI)


def page(marker_id: int, size_mm: float, clean: bool = False) -> Image.Image:
    W, H = round(PAGE_IN[0] * DPI), round(PAGE_IN[1] * DPI)
    img = np.full((H, W), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(DICT)
    s = mm_px(size_mm)
    m = cv2.aruco.generateImageMarker(d, marker_id, s, borderBits=1)
    x0, y0 = (W - s) // 2, mm_px(45)
    img[y0 : y0 + s, x0 : x0 + s] = m
    if clean:  # the marker and its ID only (Andy 2026-11-14)
        t = f"ID {marker_id}"
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
        cv2.putText(img, t, ((W - tw) // 2, y0 + s + mm_px(15)), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                    0, 4, cv2.LINE_AA)  # fmt: skip
        return Image.fromarray(img)
    # ruler: a line exactly size_mm long with end ticks, to check the print scale
    ry = y0 + s + mm_px(25)
    cv2.line(img, (x0, ry), (x0 + s, ry), 0, 4)
    for x in (x0, x0 + s):
        cv2.line(img, (x, ry - mm_px(4)), (x, ry + mm_px(4)), 0, 4)

    def text(t: str, y: int, scale: float = 1.6, th: int = 3) -> None:
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        cv2.putText(img, t, ((W - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, th,
                    cv2.LINE_AA)  # fmt: skip

    text(f"this line must measure exactly {size_mm:g} mm", ry + mm_px(12))
    text(f"OneTouch table marker  ID {marker_id}  (ArUco 4x4_50)", y0 - mm_px(12), 2.0, 4)
    text("Print at 100% / Actual size - NOT fit to page.", ry + mm_px(30))
    text("Tape flat on the table. Keep the white margin around the square.", ry + mm_px(42))
    if marker_id == 0:
        text("ID 0 = table origin. Place ~30-40 cm from ID 1.", ry + mm_px(54))
    return Image.fromarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-mm", type=float, default=100.0)
    ap.add_argument("--out", type=Path, default=Path("results/print/table_markers.pdf"))
    ap.add_argument("--clean", action="store_true", help="only the marker and its ID")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pages = [page(i, args.size_mm, args.clean) for i in (0, 1)]
    pages[0].save(args.out, save_all=True, append_images=pages[1:], resolution=DPI)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
