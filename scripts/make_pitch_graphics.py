"""Pitch slide graphics (16:9 PNGs) for the 3 steps Detect / Map / Learn, made from one real take
(its video, hand landmarks, 3D joints, grab timeline, the robot's joint angles and render).

    python scripts/make_pitch_graphics.py                       # take 20261114-235738
    python scripts/make_pitch_graphics.py --take results/robot_sim/panda_demo/<take> --demo demo_390

-> results/pitch/slide_detect.png, slide_map.png, slide_learn.png (1920x1080).
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

W, H = 1920, 1080
BG = (11, 11, 13)
TEXT, DIM, FAINT = (245, 245, 245), (154, 154, 160), (70, 70, 76)
AMBER, BLUE, GREEN, RED, CYAN = (
    (245, 166, 35),
    (106, 167, 255),
    (127, 210, 122),
    (235, 87, 87),
    (80, 220, 230),
)
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
         (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19),
         (19, 20)]  # fmt: skip


SYM_FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"  # → ✓ ✕ (not in Helvetica)
SYMS = "→✓✕"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def text(d, xy, s: str, size: int, fill, bold: bool = False) -> float:
    """Text with the symbols in SYMS drawn from a font that has them -> width."""
    x, y = xy
    f, fs = font(size, bold), ImageFont.truetype(SYM_FONT, size)
    for ch in s:
        if ch == "→":  # drawn: the font's arrow is small and sits low
            yy = y + size * 0.62
            arrow(d, (x + size * 0.1, yy), (x + size * 1.0, yy), fill, max(2, size // 9), size // 3)
            x += size * 1.15
            continue
        g = fs if ch in SYMS else f
        d.text((x, y + (size * 0.06 if ch in SYMS else 0)), ch, font=g, fill=fill)
        x += d.textlength(ch, font=g)
    return x - xy[0]


def text_w(d, s: str, size: int, bold: bool = False) -> float:
    f, fs = font(size, bold), ImageFont.truetype(SYM_FONT, size)
    return sum(size * 1.15 if ch == "→" else d.textlength(ch, font=fs if ch in SYMS else f)
               for ch in s)  # fmt: skip


def canvas(step: str, title: str, sub: str):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((80, 64), step, font=font(26, True), fill=AMBER)
    d.text((80, 104), title, font=font(54, True), fill=TEXT)
    d.text((80, 178), sub, font=font(28), fill=DIM)
    return im, d


def paste(im, img_bgr, box, radius: int = 18, border=None):
    """BGR image -> fit (cover) into box (x0, y0, x1, y1) with rounded corners."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    h, w = img_bgr.shape[:2]
    s = max(bw / w, bh / h)
    r = cv2.resize(img_bgr, (int(w * s + 0.5), int(h * s + 0.5)), interpolation=cv2.INTER_AREA)
    oy, ox = (r.shape[0] - bh) // 2, (r.shape[1] - bw) // 2
    r = r[oy : oy + bh, ox : ox + bw]
    tile = Image.fromarray(cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
    mask = Image.new("L", (bw, bh), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, bw - 1, bh - 1), radius, fill=255)
    im.paste(tile, (x0, y0), mask)
    if border:
        ImageDraw.Draw(im).rounded_rectangle(box, radius, outline=border, width=3)


def arrow(d, a, b, color=FAINT, width: int = 5, head: int = 22):
    a, b = np.array(a, float), np.array(b, float)
    d.line([tuple(a), tuple(b)], fill=color, width=width)
    u = (b - a) / np.linalg.norm(b - a)
    n = np.array([-u[1], u[0]])
    d.polygon([tuple(b + u * 4), tuple(b - u * head + n * head * 0.55),
               tuple(b - u * head - n * head * 0.55)], fill=color)  # fmt: skip


def tag(d, xy, s, color, size: int = 22):
    f = font(size, True)
    x, y = xy
    tw = d.textlength(s, font=f)
    d.rounded_rectangle((x, y, x + tw + 24, y + size + 16), 6, fill=color)
    d.text((x + 12, y + 7), s, font=f, fill=(0, 0, 0))


def fig_image(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    a = np.frombuffer(buf.getvalue(), np.uint8)
    return cv2.imdecode(a, cv2.IMREAD_COLOR)


def frame(path: Path, k: int) -> np.ndarray:
    c = cv2.VideoCapture(str(path))
    c.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, f = c.read()
    assert ok, (path, k)
    return f


def scale_up(img, s: int = 2):
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------- slide 1: detect
def held_can(J: np.ndarray, close: np.ndarray, k: int, c0: np.ndarray):
    """The can's centre and axis at frame k (cm): standing at c0 until the grab, then carried
    rigidly with the palm (rotation included), like panda_from_video.your_cup_path."""
    on = np.flatnonzero(close)
    up = np.array([0.0, 0.0, 1.0])
    if not len(on) or k < on[0] or k > on[-1]:
        return c0, up
    ka = int(on[0])
    P = J[:, [0, 5, 9, 13, 17]].mean(1)  # palm centre
    x = J[:, 9] - J[:, 0]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = J[:, 5] - J[:, 17]
    y -= (y * x).sum(1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    R = np.stack([x, y, np.cross(x, y)], -1)  # palm frames
    rc, ra = R[ka].T @ (c0 - P[ka]), R[ka].T @ up
    return P[k] + R[k] @ rc, R[k] @ ra


def draw_can(ax, c: np.ndarray, a: np.ndarray, r: float, h: float) -> np.ndarray:
    """A wireframe can (centre c, axis a); returns its rim points (for the axis limits)."""
    a = a / np.linalg.norm(a)
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    th = np.linspace(0, 2 * np.pi, 40)
    circ = r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
    lo, hi = c - a * h / 2 + circ, c + a * h / 2 + circ
    for q in (lo, hi):
        ax.plot(*q.T, color="#9a9aa0", lw=1.5)
    for i in range(0, 40, 5):
        ax.plot(*np.stack([lo[i], hi[i]]).T, color="#55555a", lw=1)
    return np.vstack([lo, hi])


def slide_detect(take: Path, out: Path, k: int) -> None:
    z = np.load(take / "take.npz")
    cal = json.loads((take / "calib.json").read_text())
    img = scale_up(frame(take / "take.mp4", k))  # 1600 x 900
    s = img.shape[1] / cal["image_size"][0]
    lm = z["lm"][k][:, :2] * s
    for a, b in EDGES:
        cv2.line(img, tuple(lm[a].astype(int)), tuple(lm[b].astype(int)), (255, 255, 255), 4,
                 cv2.LINE_AA)  # fmt: skip
    for p in lm.astype(int):
        cv2.circle(img, tuple(p), 7, (35, 166, 245), -1, cv2.LINE_AA)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = det.detectMarkers(img)
    mk = None
    if ids is not None:
        c = corners[0][0]
        cv2.polylines(img, [c.astype(int)], True, (230, 220, 80), 6, cv2.LINE_AA)
        o = c[0]  # the marker's corner: draw its x / y edges as the table axes
        for e, col in ((c[1], (87, 87, 235)), (c[3], (122, 210, 127))):
            cv2.arrowedLine(img, tuple(o.astype(int)), tuple((o + 1.6 * (e - o)).astype(int)), col,
                            7, cv2.LINE_AA, tipLength=0.15)  # fmt: skip
        mk = c.mean(0)
    im, d = canvas("01  DETECT", "One camera rebuilds my hand in 3D",
                   "Every frame, on real-world coordinates from the printed markers")  # fmt: skip
    box = (80, 250, 1000, 768)
    paste(im, img, box)
    sx, sy = (box[2] - box[0]) / img.shape[1], (box[3] - box[1]) / img.shape[0]
    sc = max(sx, sy)
    ox = (img.shape[1] * sc - (box[2] - box[0])) / 2
    oy = (img.shape[0] * sc - (box[3] - box[1])) / 2

    def to_c(p):
        return box[0] + p[0] * sc - ox, box[1] + p[1] * sc - oy

    tag(d, (box[0] + 20, box[1] + 20), "WEBCAM", (230, 230, 230))
    hx, hy = to_c(lm[9])
    tag(d, (hx + 70, hy - 110), "HOLD", AMBER)
    if mk is not None:
        mx, my = to_c(mk)
        tag(d, (mx - 120, my + 120), "MARKER 0 = (0, 0, 0)", CYAN)

    # the 3D hand in the table frame (cm), with the can and marker 0
    j = z["joints"][k] * 100
    fig = plt.figure(figsize=(7.4, 5.2), facecolor="#0b0b0d")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0b0b0d")
    ms = cal["marker_size_m"] * 100
    ax.plot([0, ms, ms, 0, 0], [0, 0, ms, ms, 0], [0] * 5, color="#e6dc50", lw=2.5)
    ax.quiver(0, 0, 0, 12, 0, 0, color="#eb5757", lw=2.5, arrow_length_ratio=0.2)
    ax.quiver(0, 0, 0, 0, 12, 0, color="#7fd27a", lw=2.5, arrow_length_ratio=0.2)
    ax.quiver(0, 0, 0, 0, 0, 12, color="#6aa7ff", lw=2.5, arrow_length_ratio=0.2)
    summ = json.loads((take / "summary.json").read_text())["take"][0]
    cx, cy = (np.array(summ["cup_pose"]["xy"]) * 100).tolist()
    r_, h_ = 2.6, 13.4  # the can: 5.2 cm x 13.4 cm
    c, a = held_can(z["joints"] * 100, z["close"].astype(bool), k, np.array([cx, cy, h_ / 2]))
    ring = draw_can(ax, c, a, r_, h_)
    for a, b in EDGES:
        ax.plot(*zip(j[a], j[b], strict=True), color="white", lw=2.5)
    ax.scatter(*j.T, color="#f5a623", s=22, depthshade=False)
    pts = np.vstack([j, ring])
    lo = np.minimum(pts.min(0), [0, 0, 0]) - 4
    hi = np.maximum(pts.max(0), [ms, ms, h_]) + 4
    span = (hi - lo).max()
    mid = (hi + lo) / 2
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax.set_zlim(0, span * 0.7)
    ax.view_init(elev=28, azim=-60)
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.set_pane_color((0.08, 0.08, 0.09, 1))
        a_.label.set_color("#9a9aa0")
        a_._axinfo["grid"]["color"] = (0.25, 0.25, 0.27, 1)
    ax.tick_params(colors="#55555a", labelsize=8)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    fig.subplots_adjust(0, 0, 1, 1)
    box3 = (1100, 250, 1840, 768)
    paste(im, fig_image(fig), box3, border=(38, 38, 42))
    tag(d, (box3[0] + 20, box3[1] + 20), "3D HAND + CAN, TABLE COORDINATES", (230, 230, 230))
    arrow(d, (1018, 509), (1082, 509), AMBER)

    # grab / let go timeline (30 fps), from the double-click on
    cl = z["close"].astype(bool)
    k0 = int(json.loads((take / "click.json").read_text()).get("frame", 0))
    cl = cl[k0:]
    x0, x1, y0 = 80, 1840, 850
    n = len(cl)
    d.text((x0, y0 - 46), "GRAB / LET GO, FOUND AUTOMATICALLY EVERY FRAME", font=font(22, True),
           fill=DIM)  # fmt: skip
    d.rounded_rectangle((x0, y0, x1, y0 + 34), 8, fill=(38, 38, 42))
    on = np.flatnonzero(cl)
    if len(on):
        a, b = on[0], on[-1] + 1
        xa, xb = x0 + (x1 - x0) * a / n, x0 + (x1 - x0) * b / n
        d.rounded_rectangle((xa, y0, xb, y0 + 34), 8, fill=AMBER)
        d.text((xa + 14, y0 + 4), "HOLDING", font=font(22, True), fill=(0, 0, 0))
        for x, s_ in ((xa, "grab"), (xb, "let go")):
            d.line([(x, y0 - 8), (x, y0 + 50)], fill=TEXT, width=3)
            d.text((x - d.textlength(s_, font=font(24)) / 2, y0 + 56), s_, font=font(24), fill=TEXT)
    for t in range(0, int(n / 30) + 1, 2):
        x = x0 + (x1 - x0) * t * 30 / n
        d.text((x - 8, y0 + 100), f"{t}s", font=font(20), fill=FAINT)
    im.save(out)


# ---------------------------------------------------------------------------- slide 2: map
def slide_map(take: Path, out: Path, k: int, demo: str, dataset: Path, robot_k: int) -> None:
    z = np.load(take / "take.npz")
    cal = json.loads((take / "calib.json").read_text())
    img = scale_up(frame(take / "take.mp4", k))
    s = img.shape[1] / cal["image_size"][0]
    lm = z["lm"][k][:, :2] * s
    for a, b in EDGES:
        cv2.line(img, tuple(lm[a].astype(int)), tuple(lm[b].astype(int)), (200, 200, 200), 3,
                 cv2.LINE_AA)  # fmt: skip
    palm = lm[[0, 5, 9, 13, 17]].mean(0)
    fwd = lm[9] - lm[0]
    fwd /= np.linalg.norm(fwd)
    side = lm[5] - lm[17]
    side /= np.linalg.norm(side)
    L = 150
    cv2.arrowedLine(img, tuple(palm.astype(int)), tuple((palm + L * fwd).astype(int)),
                    (87, 87, 235), 9, cv2.LINE_AA, tipLength=0.2)  # fmt: skip
    cv2.arrowedLine(img, tuple(palm.astype(int)), tuple((palm + L * side).astype(int)),
                    (122, 210, 127), 9, cv2.LINE_AA, tipLength=0.2)  # fmt: skip
    cv2.circle(img, tuple(palm.astype(int)), 22, (255, 167, 106), 6, cv2.LINE_AA)
    cv2.circle(img, tuple(palm.astype(int)), 6, (255, 167, 106), -1, cv2.LINE_AA)
    c = lm.mean(0)
    half = 330
    x0_, y0_ = (
        int(np.clip(c[0] - half * 1.2, 0, img.shape[1] - 2 * half * 1.2)),
        int(np.clip(c[1] - half, 0, img.shape[0] - 2 * half)),
    )
    crop = img[y0_ : y0_ + 2 * half, x0_ : x0_ + int(2 * half * 1.2)]

    im, d = canvas("02  MAP", "Copy what the hand does, not its joints",
                   "Every robot arm has different joints, so we record the hand's pose and let "
                   "each robot solve for its own")  # fmt: skip
    b1 = (80, 270, 640, 760)
    paste(im, crop, b1)
    tag(d, (b1[0] + 20, b1[1] + 20), "MY HAND", (230, 230, 230))
    d.text((b1[0], b1[3] + 24), "WHERE IT IS", font=font(22, True), fill=AMBER)
    d.text((b1[0], b1[3] + 54), "x · y · z", font=font(30), fill=TEXT)
    d.text((b1[0] + 290, b1[3] + 24), "HOW IT'S TURNED", font=font(22, True), fill=AMBER)
    d.text((b1[0] + 290, b1[3] + 54), "roll · pitch · yaw", font=font(30), fill=TEXT)

    # inverse kinematics: the 7 joint angles over the take (the robot's real commands)
    with h5py.File(dataset, "r") as f:
        q = f["data"][demo]["actions"][:, :7]
    fig, ax = plt.subplots(figsize=(5.4, 3.6), facecolor="#0b0b0d")
    ax.set_facecolor("#0b0b0d")
    cols = plt.cm.cool(np.linspace(0, 1, 7))
    t = np.arange(len(q)) / 30
    ends = q[-1].copy()
    order = np.argsort(ends)
    ly = ends[order].copy()
    gap = 0.22 * (q.max() - q.min()) / 3
    for i in range(1, 7):
        ly[i] = max(ly[i], ly[i - 1] + gap)
    lab = np.empty(7)
    lab[order] = ly
    for i in range(7):
        ax.plot(t, q[:, i], color=cols[i], lw=2.2)
        ax.text(t[-1] + 0.15, lab[i], f"θ{i + 1}", color=cols[i], fontsize=11, va="center")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors="#55555a", labelsize=9)
    ax.set_xlabel("seconds", color="#9a9aa0")
    ax.set_ylabel("joint angle (rad)", color="#9a9aa0")
    ax.grid(color="#26262a")
    fig.subplots_adjust(0.12, 0.16, 0.92, 0.97)
    b2 = (730, 330, 1190, 640)
    d.rounded_rectangle((710, 270, 1210, 760), 18, fill=(18, 18, 21), outline=(38, 38, 42), width=3)
    d.text((740, 290), "INVERSE KINEMATICS", font=font(26, True), fill=TEXT)
    paste(im, fig_image(fig), b2, radius=8)
    text(d, (740, 660), "hand pose  →  7 joint angles", 28, TEXT)
    d.text((740, 702), "30 times a second", font=font(24), fill=DIM)
    arrow(d, (648, 515), (700, 515), AMBER)
    arrow(d, (1218, 515), (1270, 515), BLUE)

    r = frame(take / "teleop.mp4", robot_k)
    r = r[:, r.shape[1] // 2 :]
    rc = r[40:, int(r.shape[1] * 0.33) :]
    b3 = (1280, 270, 1840, 760)
    paste(im, scale_up(rc), b3)
    tag(d, (b3[0] + 20, b3[1] + 20), "ROBOT ARM (7 JOINTS)", (230, 230, 230))

    # the same hand data, any arm
    y = 800
    text(d, (1280, y), "SAME HAND RECORDING  →  ANY ARM", 22, DIM, bold=True)
    y += 20
    for i, (name, segs) in enumerate((("7-joint arm", 4), ("6-joint arm", 3), ("small arm", 2))):
        bx = 1290 + i * 190
        pts = [(bx + 20, y + 150)]
        ang = [-80, -35, 20, 60][:segs]
        ln = [60, 55, 45, 35][:segs]
        for a_, l_ in zip(ang, ln, strict=True):
            px, py = pts[-1]
            pts.append((px + l_ * np.cos(np.radians(a_)), py + l_ * np.sin(np.radians(a_))))
        d.rectangle((bx, y + 150, bx + 40, y + 162), fill=FAINT)
        d.line(pts, fill=BLUE, width=7, joint="curve")
        for p in pts[:-1]:
            d.ellipse((p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7), fill=TEXT)
        d.text((bx, y + 172), name, font=font(20), fill=DIM)
    im.save(out)


# ---------------------------------------------------------------------------- slide 3: learn
def slide_learn(take: Path, out: Path, robot_k: int) -> None:
    summ = json.loads((take / "summary.json").read_text())["take"][0]
    sub = "Camera tracking is never perfect, and a centimetre off means a miss"
    im, d = canvas("03  LEARN", "Practise in physics, keep the grip that lifts", sub)
    # left: how the search works (illustration): candidate grips on the can, side view
    d.rounded_rectangle((80, 270, 900, 900), 18, fill=(18, 18, 21), outline=(38, 38, 42), width=3)
    d.text((110, 292), "SMALL REINFORCEMENT-LEARNING SEARCH", font=font(24, True), fill=TEXT)
    d.text((110, 326), "in a physics simulator (illustration)", font=font(22), fill=DIM)
    gy = 830  # table
    d.line([(110, gy), (870, gy)], fill=FAINT, width=4)
    can_h, can_w = 320, 124  # 13.4 cm x 5.2 cm, drawn
    px_cm = can_h / 13.4

    def can(x, color=(150, 150, 156), tilt=0.0, lift=0.0):
        c = np.array([[-can_w / 2, 0], [can_w / 2, 0], [can_w / 2, -can_h], [-can_w / 2, -can_h]])
        R = np.array([[np.cos(tilt), -np.sin(tilt)], [np.sin(tilt), np.cos(tilt)]])
        p = c @ R.T + [x, gy - lift]
        d.polygon([tuple(q) for q in p], outline=color, width=4)
        return p

    def gripper(x, h_cm, color, width=6, lift=0.0):
        y = gy - h_cm * px_cm - lift
        g = can_w / 2 + 6
        d.line([(x - g, y - 34), (x - g, y + 34)], fill=color, width=width + 4)
        d.line([(x + g, y - 34), (x + g, y + 34)], fill=color, width=width + 4)
        d.line([(x - g, y - 34), (x - g, y - 64), (x + g, y - 64), (x + g, y - 34)], fill=color,
               width=width)  # fmt: skip
        d.line([(x, y - 64), (x, y - 130)], fill=color, width=width)
        return y

    # three tries: slips (too high), tips the can (off-centre), lifts (kept)
    xs = (250, 490, 730)
    can(xs[0])
    gripper(xs[0], 12.2, RED)
    can(xs[1], tilt=0.18)
    gripper(xs[1] + 30, 7.0, RED)
    lift = 70
    can(xs[2], color=(220, 220, 225), lift=lift)
    gripper(xs[2], summ["grasp_height_cm"], GREEN, lift=lift)
    arrow(d, (xs[2] + 110, gy - 10), (xs[2] + 110, gy - lift - 40), GREEN, width=4, head=16)
    for x, s_, col in ((xs[0], "✕  slips off", RED), (xs[1], "✕  tips over", RED),
                       (xs[2], "✓  lifts, kept", GREEN)):  # fmt: skip
        text(d, (x - text_w(d, s_, 26, True) / 2, gy + 22), s_, 26, col, bold=True)

    # middle: the real numbers from this take
    d.text((960, 300), "THIS TAKE", font=font(22, True), fill=AMBER)
    rows = [(f"{summ['your_grip_height_cm']:.1f} cm", "grip height the camera saw"),
            (f"{summ['grasp_height_cm']:.1f} cm", "grip height practice picked"),
            (f"{len(summ['rl_rounds'])} rounds",
             f"of search, {summ['seconds']:.0f} s in total")]  # fmt: skip
    for i, (a, b) in enumerate(rows):
        y = 350 + i * 150
        d.text((960, y), a, font=font(64, True), fill=TEXT)
        d.text((960, y + 76), b, font=font(24), fill=DIM)

    r = frame(take / "teleop.mp4", robot_k)
    r = r[:, r.shape[1] // 2 :]
    b3 = (1340, 270, 1840, 900)
    rc = r[48:, int(r.shape[1] * 0.22) : int(r.shape[1] * 0.72)]  # gripper + can, no caption
    paste(im, scale_up(rc), b3)
    tag(d, (b3[0] + 20, b3[1] + 20), "THE GRIP THAT LIFTS", GREEN)
    im.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--take", type=Path, default=Path("results/robot_sim/panda_demo/20261114-235738")
    )
    ap.add_argument("--frame", type=int, default=260, help="take frame (hand holding the can)")
    ap.add_argument("--robot-frame", type=int, default=144, help="teleop.mp4 frame (robot lifting)")
    ap.add_argument(
        "--dataset", type=Path, default=Path("results/robot_sim/orient_diag/test_demos.hdf5")
    )
    ap.add_argument("--demo", default="demo_390", help="the take's teleop demo in --dataset")
    ap.add_argument("--out", type=Path, default=Path("results/pitch"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    slide_detect(a.take, a.out / "slide_detect.png", a.frame)
    slide_map(a.take, a.out / "slide_map.png", a.frame, a.demo, a.dataset, a.robot_frame)
    slide_learn(a.take, a.out / "slide_learn.png", a.robot_frame)
    print("wrote", *sorted(str(p) for p in a.out.glob("slide_*.png")))


if __name__ == "__main__":
    main()
