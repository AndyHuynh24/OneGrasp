"""Demo UI for scripts/panda_demo.py, modelled on a launch webcast: the picture fills the
window, a few thin rings with light numerals, a large clock and one wide timeline arc along the
bottom, clean light type (SF Pro, the macOS system font), white with colour only for state.

The canvas is drawn at the screen's native (Retina) resolution so text stays sharp. Text is
rendered once per (string, weight, size, colour) with PIL and cached as an alpha patch.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SF = "/System/Library/Fonts/SFNS.ttf"  # variable: Light / Regular / Medium ...
WEIGHTS = {"light": b"Light", "regular": b"Regular", "medium": b"Medium", "semibold": b"Semibold"}

# colours (BGR)
BG = (0, 0, 0)
TEXT = (250, 250, 250)
DIM = (165, 165, 165)
FAINT = (90, 90, 90)
LINE = (60, 60, 60)
GOOD = (130, 210, 120)
BAD = (80, 85, 240)
ACCENT = TEXT
PANEL = (16, 16, 16)

PHASES = ("Capture", "Detect", "Plan", "Learn", "Execute", "Scale")
TITLE_BAR = 28  # macOS window title bar (points)


def screen_size() -> tuple[int, int]:
    """Main display size in points (e.g. 1512 x 982 on a 14" MacBook Pro), via CoreGraphics
    (not tkinter: a Tk root next to OpenCV's Cocoa window crashes at exit)."""
    try:
        import ctypes
        import ctypes.util

        class _R(ctypes.Structure):
            _fields_ = [(n, ctypes.c_double) for n in ("x", "y", "w", "h")]

        cg = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreGraphics"))
        cg.CGMainDisplayID.restype = ctypes.c_uint32
        cg.CGDisplayBounds.restype = _R
        cg.CGDisplayBounds.argtypes = [ctypes.c_uint32]
        r = cg.CGDisplayBounds(cg.CGMainDisplayID())
        if r.w > 0 and r.h > 0:
            return int(r.w), int(r.h)
    except Exception:
        pass
    return 1920, 1080


def visible_frame() -> tuple[int, int, int, int]:
    """The main screen's usable area in points, without the menu bar and the Dock:
    (x, y from the top, width, height). Falls back to the whole screen."""
    sw, sh = screen_size()
    try:
        import ctypes

        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AppKit.framework/AppKit")
        objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p

        class _R(ctypes.Structure):
            _fields_ = [(n, ctypes.c_double) for n in ("x", "y", "w", "h")]

        addr = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        get_id = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(addr)
        get_r = ctypes.CFUNCTYPE(_R, ctypes.c_void_p, ctypes.c_void_p)(addr)
        scr = get_id(objc.objc_getClass(b"NSScreen"), objc.sel_registerName(b"mainScreen"))
        if scr:
            r = get_r(scr, objc.sel_registerName(b"visibleFrame"))  # origin bottom-left
            if r.w > 0 and r.h > 0:
                return int(r.x), int(sh - r.y - r.h), int(r.w), int(r.h)
    except Exception:
        pass
    return 0, 0, sw, sh


def window_size() -> tuple[int, int, int, int]:
    """A window filling the usable screen area: (x, y, content width, content height)."""
    x, y, w, h = visible_frame()
    return x, y, w, h - TITLE_BAR


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


class UI:
    def __init__(self, width: int | None = None, aspect: float | None = None):
        _, _, ww, wh = window_size()
        aspect = aspect or ww / wh
        self.W = int(width or 2 * ww)  # Retina: 2 pixels per point
        self.H = int(round(self.W / aspect))
        self.s = self.W / 1920  # sizes below are for a 1920-wide canvas
        self.m = self.px(40)
        self._fonts: dict = {}
        self._cache: dict = {}
        self._grad: dict = {}
        self._blank = np.zeros((self.H, self.W, 3), np.uint8)

    # ------------------------------------------------------------------ primitives
    def px(self, v: float) -> int:
        return int(round(v * self.s))

    def font(self, weight: str, size: int):
        k = (weight, size)
        if k not in self._fonts:
            f = ImageFont.truetype(SF, size)
            try:
                f.set_variation_by_name(WEIGHTS[weight])
            except Exception:
                pass
            self._fonts[k] = f
        return self._fonts[k]

    def _patch(self, s: str, weight: str, size: int, col, track: float):
        k = (s, weight, size, col, track)
        p = self._cache.get(k)
        if p is not None:
            return p
        f = self.font(weight, size)
        sp = track * size
        if track:
            widths = [f.getlength(c) for c in s]
            w = int(np.ceil(sum(widths) + sp * max(0, len(s) - 1))) + 4
        else:
            w = int(np.ceil(f.getlength(s))) + 4
        asc, desc = f.getmetrics()
        im = Image.new("L", (max(w, 1), asc + desc + 4), 0)
        d = ImageDraw.Draw(im)
        if track:
            x = 2.0
            for c, cw in zip(s, widths, strict=True):
                d.text((x, 2), c, font=f, fill=255)
                x += cw + sp
        else:
            d.text((2, 2), s, font=f, fill=255)
        a = np.asarray(im, np.float32)[..., None] / 255.0
        p = (a, np.array(col, np.float32)[None, None])
        if len(self._cache) > 4000:
            self._cache.clear()
        self._cache[k] = p
        return p

    def measure(self, s: str, size: float, weight: str = "regular", track: float = 0.0) -> int:
        return self._patch(s, weight, max(6, self.px(size)), TEXT, track)[0].shape[1] if s else 0

    def text(self, img, s: str, x: float, y: float, size: float = 20, col=TEXT,
             weight: str = "regular", anchor: str = "l", track: float = 0.0) -> int:  # fmt: skip
        """Draw s with its top-left (anchor l), top-centre (c) or top-right (r) at x, y
        (canvas px; size in 1920-canvas px) -> its width."""
        if not s:
            return 0
        a, c = self._patch(s, weight, max(6, self.px(size)), tuple(int(v) for v in col), track)
        h, w = a.shape[:2]
        x0 = int(x - (w if anchor == "r" else w / 2 if anchor == "c" else 0))
        y0 = int(y)
        H, W = img.shape[:2]
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if xa >= xb or ya >= yb:
            return w
        aa = a[ya - y0 : yb - y0, xa - x0 : xb - x0]
        roi = img[ya:yb, xa:xb].astype(np.float32)
        img[ya:yb, xa:xb] = (roi * (1 - aa) + c * aa).astype(np.uint8)
        return w

    def caps(self, img, s: str, x, y, col=DIM, size: float = 14, anchor: str = "l",
             weight: str = "medium") -> int:  # fmt: skip
        """A small uppercase label (lightly tracked)."""
        return self.text(img, s.upper(), x, y, size, col, weight, anchor, 0.1)

    def line(self, img, x1, y1, x2, y2, col=LINE, th: int | None = None) -> None:
        th = th or max(1, self.px(1.5))
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), col, th, cv2.LINE_AA)

    def shade(self, img, y0: int, y1: int, top: float, bottom: float) -> None:
        """Darken rows y0..y1 by a vertical gradient (keep factor top -> bottom)."""
        y0, y1 = max(0, int(y0)), min(img.shape[0], int(y1))
        if y1 <= y0:
            return
        k = (y1 - y0, round(top, 3), round(bottom, 3), img.shape[1])
        g = self._grad.get(k)
        if g is None:
            col = np.linspace(top, bottom, y1 - y0, dtype=np.float32)
            g = np.repeat((col * 255).astype(np.uint8)[:, None, None], img.shape[1], 1)
            g = np.repeat(g, 3, 2)
            self._grad[k] = g
        img[y0:y1] = cv2.multiply(img[y0:y1], g, scale=1 / 255)

    def cover(self, src: np.ndarray, r: Rect) -> tuple[np.ndarray, Rect]:
        """src scaled to cover r (cropped centrally) -> (the r-sized image, the rect of the
        whole scaled src in canvas px: it may extend past r)."""
        h, w = src.shape[:2]
        k = max(r.w / w, r.h / h)
        nw, nh = int(round(w * k)), int(round(h * k))
        interp = cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR
        big = cv2.resize(src, (nw, nh), interpolation=interp)
        ox, oy = (nw - r.w) // 2, (nh - r.h) // 2
        return big[oy : oy + r.h, ox : ox + r.w], Rect(r.x - ox, r.y - oy, nw, nh)

    def fit(self, src: np.ndarray, r: Rect) -> tuple[np.ndarray, Rect]:
        """src scaled to fit inside r (aspect kept, centred) -> (resized, its rect)."""
        h, w = src.shape[:2]
        k = min(r.w / w, r.h / h)
        nw, nh = max(1, int(w * k)), max(1, int(h * k))
        if abs(nw - w) <= 1 and abs(nh - h) <= 1:
            return src, Rect(r.x + (r.w - w) // 2, r.y + (r.h - h) // 2, w, h)
        interp = cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR
        out = cv2.resize(src, (nw, nh), interpolation=interp)
        return out, Rect(r.x + (r.w - nw) // 2, r.y + (r.h - nh) // 2, nw, nh)

    def put(self, img, src, r: Rect) -> None:
        img[r.y : r.y2, r.x : r.x2] = src

    def bar(self, img, x, y, w, frac: float, col=TEXT) -> None:
        th = max(2, self.px(2))
        self.line(img, x, y, x + w, y, LINE, th)
        f = float(np.clip(frac, 0, 1))
        if f > 0:
            self.line(img, x, y, x + w * f, y, col, th)

    # ------------------------------------------------------------------ webcast widgets
    def ring(self, img, cx, cy, r, frac: float, label: str, value: str, unit: str,
             col=TEXT) -> None:  # fmt: skip
        """A gauge: thin full circle, a brighter arc for frac (from the bottom-left, clockwise),
        label on top inside, value in the middle, unit under it."""
        c, ax = (int(cx), int(cy)), (int(r), int(r))
        th = max(2, self.px(2.2))
        cv2.circle(img, c, int(r), (70, 70, 70), th, cv2.LINE_AA)
        f = float(np.clip(frac, 0, 1))
        if f > 0.003:
            cv2.ellipse(img, c, ax, 0, 125, 125 + 290 * f, col, max(3, self.px(3.4)), cv2.LINE_AA)
        self.caps(img, label, cx, cy - r * 0.58, DIM, 13, "c")
        self.text(img, value, cx, cy - r * 0.36, 40, TEXT, "light", "c")
        if unit:
            self.caps(img, unit, cx, cy + r * 0.34, DIM, 12, "c")

    def attitude(self, img, cx, cy, r, tilt_deg: float, label: str = "Attitude") -> None:
        """The object's attitude in a ring: an upright outline = 0 deg, rotated by its tilt;
        the heavy edge is its top, so upside down reads at a glance."""
        th = max(2, self.px(2.2))
        cv2.circle(img, (int(cx), int(cy)), int(r), (70, 70, 70), th, cv2.LINE_AA)
        self.line(img, cx, cy - r * 0.92, cx, cy + r * 0.92, (55, 55, 55))
        h, w = r * 0.95, r * 0.6
        a = np.deg2rad(tilt_deg)
        ca, sa = np.cos(a), np.sin(a)
        oy = r * 0.06
        pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        p = np.array([(cx + x * ca - y * sa, cy - oy + x * sa + y * ca) for x, y in pts])
        p16 = (p * 16).round().astype(np.int32)
        cv2.fillPoly(img, [p16], (38, 38, 38), cv2.LINE_AA, shift=4)
        cv2.polylines(img, [p16], True, TEXT, th, cv2.LINE_AA, shift=4)
        cv2.line(img, tuple(p16[0]), tuple(p16[1]), TEXT, max(3, self.px(5)), cv2.LINE_AA, 4)
        self.caps(img, label, cx, cy - r * 0.86, DIM, 12, "c")
        self.text(img, f"{abs(tilt_deg):.0f}°", cx, cy + r * 0.5, 22, TEXT, "light", "c")

    def clock_block(self, img, cx, y, clock_s: str, sub: str = "", sub_col=DIM,
                    dot=None) -> None:  # fmt: skip
        """The big centred clock with a small caption under it (optionally a status dot)."""
        self.text(img, clock_s, cx, y, 50, TEXT, "light", "c")
        if sub:
            sy = y + self.px(66)
            w = self.caps(img, sub, cx, sy, sub_col, 14, "c")
            if dot is not None:
                cv2.circle(img, (int(cx - w / 2 - self.px(14)), int(sy + self.px(10))),
                           max(3, self.px(5)), dot, -1, cv2.LINE_AA)  # fmt: skip

    def arc(self, img, cx, ytop, half_w, items, now: float, span: float,
            sag: float | None = None) -> None:  # fmt: skip
        """The timeline arc: its top is "now"; items (name, t) sit at (t - now) / span of the
        half arc, sliding right -> left as now advances. Passed = filled, ahead = ring;
        items fade towards the ends."""
        sag = sag or self.px(60)
        R = (half_w**2 + sag**2) / (2 * sag)
        tmax = float(np.arcsin(min(1.0, half_w / R)))
        th = max(2, self.px(1.6))

        def pt(a):
            return cx + R * np.sin(a), ytop + R * (1 - np.cos(a))

        for lo, hi, col in ((-tmax, 0.0, (200, 200, 200)), (0.0, tmax, (75, 75, 75))):
            a = np.linspace(lo, hi, 120)
            p = (np.stack(pt(a), 1) * 16).round().astype(np.int32)
            cv2.polylines(img, [p], False, col, th, cv2.LINE_AA, shift=4)
        self.line(img, cx, ytop - self.px(10), cx, ytop + self.px(10), TEXT, th)
        placed: list = []
        for name, t in items:
            a = (t - now) / span * tmax
            if abs(a) > tmax:
                continue
            fade = float(np.clip((tmax - abs(a)) / (0.3 * tmax), 0, 1))
            fade = round(fade * 6) / 6  # few colour steps: the text cache stays small
            x, y = pt(a)
            c16 = (int(round(x * 16)), int(round(y * 16)))
            past = t <= now + 1e-6
            near = abs(a) < 0.06 * tmax
            col = tuple(int(v * fade) for v in (TEXT if past or near else DIM))
            r = self.px(7 if near else 5.5)
            if past:
                cv2.circle(img, c16, r * 16, col, -1, cv2.LINE_AA, 4)
            else:
                cv2.circle(img, c16, r * 16, BG, -1, cv2.LINE_AA, 4)
                cv2.circle(img, c16, r * 16, col, th, cv2.LINE_AA, 4)
            size = 14
            w = self.measure(name.upper(), size, "medium", 0.1)
            ly = y - self.px(34)
            while any(x - w / 2 < b1 + self.px(14) and x + w / 2 > b0 - self.px(14)
                      and abs(ly - by) < self.px(22) for b0, b1, by in placed):  # fmt: skip
                ly -= self.px(24)
            placed.append((x - w / 2, x + w / 2, ly))
            self.caps(img, name, x, ly, col, size, "c")

    def phases(self, img, phase: int, detail: float | None = None, y: float | None = None):
        """The pipeline phases on the arc; the current one eases to the top (animated across
        frames)."""
        target = phase + 0.5 * float(np.clip(detail or 0.0, 0, 1))
        now = getattr(self, "_phase_now", target)
        now += (target - now) * 0.2 if abs(target - now) > 0.01 else target - now
        self._phase_now = now
        y = self.H - self.px(150) if y is None else y
        self.arc(img, self.W / 2, y, self.W * 0.24, list(zip(PHASES, range(6), strict=True)),
                 now, 2.6)  # fmt: skip

    def keys(self, img, items, x=None, y=None, anchor: str = "r") -> None:
        """Key hints, one line (default: top right): key in white, action in grey."""
        sz = 15
        parts = []
        for k, label in items:
            parts.append((k, TEXT, "semibold"))
            parts.append((label, DIM, "regular"))
        gap_k, gap = self.px(8), self.px(26)
        widths = [self.measure(t, sz, w_) for t, _, w_ in parts]
        total = sum(widths) + gap_k * len(items) + gap * (len(items) - 1)
        x = (self.W - self.m if x is None else x) - (total if anchor == "r" else 0)
        y = self.px(30) if y is None else y
        for i, ((t, col, w_), wd) in enumerate(zip(parts, widths, strict=True)):
            self.text(img, t, x, y, sz, col, w_)
            x += wd + (gap_k if i % 2 == 0 else gap)

    def brand(self, img) -> None:
        self.text(img, "ONETOUCH", self.m, self.px(28), 16, TEXT, "semibold", track=0.32)

    def banner(self, img, r: Rect, s: str, col) -> None:
        """A centred outcome line on a dark band across the middle of r."""
        bh = self.px(84)
        y = r.y + (r.h - bh) // 2
        roi = img[y : y + bh, r.x : r.x2]
        roi[:] = roi // 4
        self.text(img, s, r.x + r.w // 2, y + self.px(20), 34, col, "light", "c")


def clock(secs: float) -> str:
    secs = max(0.0, secs)
    return f"T+ {int(secs // 3600):02d}:{int(secs % 3600 // 60):02d}:{int(secs % 60):02d}"


def wrap(s: str, n: int) -> list[str]:
    out, line = [], ""
    for w in (s or "").split():
        if len(line) + len(w) + 1 > n and line:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out
