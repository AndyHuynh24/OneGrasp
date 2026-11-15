"""Pressure-map units, cleaning, contact labels and per-finger assignment.

Both EgoPressure and PressureVisionDB use a Sensel Morph pad: a (105, 185) grid with 1.25 mm pitch.
All values here are in "PressureVision-convention kPa" (see docs/DECISIONS.md, 2026-10-03):
comparable across both datasets and with PV++'s 1 kPa contact threshold, but not a trustworthy
absolute physical scale. Never convert these to newtons for reporting.
"""

from __future__ import annotations

import numpy as np

SENSEL_SHAPE = (105, 185)  # rows, cols
SENSEL_PITCH_MM = 1.25
CELL_AREA_MM2 = SENSEL_PITCH_MM**2

NOISE_FLOOR_KPA = 0.5  # EgoPressure's floor: values below are diffuse noise (DECISIONS option C)
CONTACT_KPA = 1.0  # PV++ eval threshold

# PressureVision recording/util.py: counts -> "newtons" -> kPa.
_PV_COUNTS_PER_NEWTON = 1736
_PV_PITCH_M = 0.00125

FINGER_TIPS = (4, 8, 12, 16, 20)  # MediaPipe landmark order: thumb, index, middle, ring, pinky
FINGER_DIPS = (3, 7, 11, 15, 19)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
# A finger can only take pressure if its tip joint is at most this high above the pad (EgoPressure,
# where 3D joints exist). Chosen on TRAIN participants: firm index contacts (>= 5 kPa) have tip
# height p99 8.1 mm, p99.9 11.1 mm; at 12 mm 0.07% of them are dropped while 72% of the spurious
# thumb "contacts" in index presses (thumb hovering over the index blob) disappear.
MAX_TIP_HEIGHT_MM = 12.0


def pv_counts_to_kpa(counts: np.ndarray) -> np.ndarray:
    """PressureVisionDB raw Sensel counts -> PV-convention kPa (same formula as PV's code)."""
    return np.asarray(counts, dtype=np.float32) / _PV_COUNTS_PER_NEWTON / _PV_PITCH_M**2 / 1000


def clean(kpa: np.ndarray, floor: float = NOISE_FLOOR_KPA) -> np.ndarray:
    """Zero out values below the noise floor. Works on one map or a stack."""
    out = np.array(kpa, dtype=np.float32, copy=True)
    out[out < floor] = 0
    return out


def frame_stats(
    kpa: np.ndarray, floor: float = NOISE_FLOOR_KPA, contact: float = CONTACT_KPA
) -> dict[str, np.ndarray]:
    """Per-frame summary of a pressure stack [N, 105, 185] (kPa, uncleaned).

    Returns max_kpa [N], area_mm2 [N] (cells >= contact), total_kpa [N] (sum after cleaning),
    centroid_rc [N, 2] (pressure-weighted over cells >= contact; NaN when no contact) and
    contact [N] (bool, max >= contact after cleaning).
    """
    stack = clean(kpa, floor)
    assert stack.ndim == 3 and stack.shape[1:] == SENSEL_SHAPE, stack.shape
    n = stack.shape[0]
    max_kpa = stack.reshape(n, -1).max(axis=1)
    above = np.where(stack >= contact, stack, 0)
    weight = above.reshape(n, -1).sum(axis=1)
    rows, cols = np.indices(SENSEL_SHAPE)
    with np.errstate(invalid="ignore", divide="ignore"):
        cr = (above * rows).reshape(n, -1).sum(axis=1) / weight
        cc = (above * cols).reshape(n, -1).sum(axis=1) / weight
    return {
        "max_kpa": max_kpa.astype(np.float32),
        "area_mm2": ((stack >= contact).reshape(n, -1).sum(axis=1) * CELL_AREA_MM2).astype(
            np.float32
        ),
        "total_kpa": stack.reshape(n, -1).sum(axis=1).astype(np.float32),
        "centroid_rc": np.stack([cr, cc], axis=1).astype(np.float32),
        "contact": max_kpa >= contact,
    }


def count_transitions(contact: np.ndarray) -> tuple[int, int]:
    """Number of touch-downs (0->1) and touch-ups (1->0) in a boolean sequence."""
    c = np.asarray(contact, dtype=np.int8)
    d = np.diff(c)
    return int((d == 1).sum()), int((d == -1).sum())


def contact_segments(contact: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end_exclusive), ...] of runs where contact is True."""
    c = np.concatenate([[0], np.asarray(contact, dtype=np.int8), [0]])
    d = np.diff(c)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


def _point_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points p [M, 2] to segments a->b [F, 2] -> [M, F]."""
    ab = b - a  # [F, 2]
    ap = p[:, None, :] - a[None]  # [M, F, 2]
    denom = np.maximum((ab**2).sum(-1), 1e-9)  # [F]
    t = np.clip((ap * ab[None]).sum(-1) / denom, 0, 1)  # [M, F]
    closest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(p[:, None, :] - closest, axis=-1)


def finger_pressure(
    kpa: np.ndarray,
    tips_rc: np.ndarray,
    dips_rc: np.ndarray,
    radius_mm: float = 20.0,
    floor: float = NOISE_FLOOR_KPA,
    eligible: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Split one pressure map between the five fingers.

    tips_rc, dips_rc: [5, 2] fingertip and DIP positions in pad-grid units (row, col), i.e. the
    hand projected onto the pad. Each cell above the floor goes to the nearest distal phalanx
    (DIP->tip segment) if it lies within radius_mm (20 mm: pressure spreads ~15 mm through the
    overlay, see docs/DATA_FORMAT.md); otherwise it counts as "other" (palm, other hand, ...).
    The contact centroid of a press sits ~70% of the way from tip to DIP (fitted on EgoPressure).
    eligible: optional [5] bool; fingers marked False (e.g. tip hovering > 12 mm above the pad)
    cannot receive pressure, so their cells go to the nearest eligible finger or to "other".

    Returns max_kpa [5], area_mm2 [5] (cells >= CONTACT_KPA) and other_max_kpa (scalar).
    """
    assert kpa.shape == SENSEL_SHAPE, kpa.shape
    assert tips_rc.shape == (5, 2) and dips_rc.shape == (5, 2)
    rr, cc = np.nonzero(kpa >= floor)
    out_max = np.zeros(5, np.float32)
    out_area = np.zeros(5, np.float32)
    if len(rr) == 0:
        return {"max_kpa": out_max, "area_mm2": out_area, "other_max_kpa": np.float32(0)}
    vals = kpa[rr, cc]
    cells = np.stack([rr, cc], axis=1).astype(np.float32)
    dist_mm = _point_segment_dist(cells, tips_rc.astype(np.float32), dips_rc.astype(np.float32))
    dist_mm *= SENSEL_PITCH_MM
    if eligible is not None:
        dist_mm[:, ~np.asarray(eligible, bool)] = np.inf
    nearest = dist_mm.argmin(axis=1)
    near_ok = dist_mm[np.arange(len(rr)), nearest] <= radius_mm
    for f in range(5):
        sel = near_ok & (nearest == f)
        if sel.any():
            out_max[f] = vals[sel].max()
            out_area[f] = (vals[sel] >= CONTACT_KPA).sum() * CELL_AREA_MM2
    other = vals[~near_ok]
    return {
        "max_kpa": out_max,
        "area_mm2": out_area,
        "other_max_kpa": np.float32(other.max() if len(other) else 0),
    }
