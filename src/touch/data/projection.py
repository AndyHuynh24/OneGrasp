"""Simulated top-down projector light on fingertip patches (augmentation + stress test).

A random projected "screen" (background, coloured windows/buttons, text lines) lives in IMAGE
coordinates and is warped into each patch with the patch's own transform, so the projection
slides across the finger as the finger moves. It is applied as extra illumination in linear
light:  out = patch * (ambient + (1 - ambient) * screen / mean(screen)).
Not modelled: hand shadows, projector/camera parallax, DLP flicker.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from touch.data.patches import PATCH_SIZE, fingertip_matrix
from touch.data.pressure import FINGER_DIPS, FINGER_TIPS

GAMMA = 2.2


def make_screen(rng: np.random.Generator, h: int, w: int, px: float = 1.0) -> np.ndarray:
    """Random projected UI frame [h, w, 3] float32 in (0, 1] (projector light colour).

    px scales the text-line geometry (1.0 = full-resolution image pixels).
    """
    bg = [rng.uniform(0.8, 1.0, 3), rng.uniform(0.02, 0.15, 3), rng.random(3)][rng.integers(3)]
    S = np.empty((h, w, 3), np.float32)
    S[:] = bg
    m, step, thick = max(1, round(6 * px)), max(2, round(14 * px)), max(1, round(5 * px))
    for _ in range(rng.integers(5, 16)):
        ww, hh = int(w * rng.uniform(0.05, 0.4)), int(h * rng.uniform(0.05, 0.4))
        x0, y0 = int(rng.integers(0, w - ww)), int(rng.integers(0, h - hh))
        S[y0 : y0 + hh, x0 : x0 + ww] = rng.random(3)
        if rng.random() < 0.5:  # text lines
            ink = rng.uniform(0.0, 0.2, 3) if S[y0, x0].mean() > 0.5 else rng.uniform(0.8, 1, 3)
            for y in range(y0 + m, y0 + hh - m, step):
                S[y : y + thick, x0 + m : x0 + m + int((ww - 2 * m) * rng.uniform(0.3, 1.0))] = ink
    return np.clip(S, 0.02, 1.0)  # projector black level


def hand_geometry(seq, view: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Primary-hand fingertip/DIP pixels [N, 5, 2, 2] (tip, dip; NaN = no hand) and image (h, w)."""
    lm = seq.landmarks(view)
    prim = lm["primary"]
    hand = np.full((len(prim), 21, 2), np.nan, np.float32)
    rows = np.flatnonzero(prim >= 0)
    hand[rows] = lm["lm_px"][rows, prim[rows], :, :2]
    geo = np.stack([hand[:, list(FINGER_TIPS)], hand[:, list(FINGER_DIPS)]], axis=2)
    first = int(rows[0]) if len(rows) else 0
    with Image.open(seq.image_path(view, first)) as im:
        w, h = im.size
    return geo, (h, w)


def screen_light(screen: np.ndarray, tip_dip: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Relative projector light on one patch [S, S, 3]; screen is normalized to mean 1.

    scale = screen resolution / image resolution (screen rendered at lower resolution).
    """
    M = fingertip_matrix(tip_dip[0], tip_dip[1], PATCH_SIZE) * scale
    return cv2.warpAffine(
        screen,
        M,
        (PATCH_SIZE, PATCH_SIZE),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def apply_light(x: np.ndarray, light: np.ndarray, ambient: float) -> np.ndarray:
    """x [..., S, S, 3] in [0, 1] (sRGB), light same shape (mean ~1) -> lit patch in [0, 1]."""
    lin = x**GAMMA * (ambient + (1.0 - ambient) * light)
    return np.clip(lin, 0.0, 1.0) ** (1.0 / GAMMA)


def random_window_projection(
    rng: np.random.Generator,
    x: np.ndarray,
    src: np.ndarray,
    geo: np.ndarray,
    hw: tuple[int, int],
    finger: int,
    scale: float = 0.5,
) -> np.ndarray:
    """Training augmentation for one causal window.

    x [W, S, S, 3] float in [0, 1]; src [W] frame each row's patch came from (-1 = no patch).
    Ambient share ~ U(0.3, 0.85); half the time the content changes once inside the window.
    """
    h, w = int(hw[0] * scale), int(hw[1] * scale)
    ambient = rng.uniform(0.3, 0.85)
    screens = [make_screen(rng, h, w, px=scale)]
    switch = len(x)
    if rng.random() < 0.5:
        screens.append(make_screen(rng, h, w, px=scale))
        switch = int(rng.integers(1, len(x)))
    screens = [s / s.mean() for s in screens]
    out = x.copy()
    for r, i in enumerate(src):
        if i < 0 or not np.isfinite(geo[i, finger]).all():
            continue
        light = screen_light(screens[int(r >= switch)], geo[i, finger], scale)
        out[r] = apply_light(x[r], light, ambient)
    return out
