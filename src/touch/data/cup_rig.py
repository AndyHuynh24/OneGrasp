"""Instrumented-object ground truth: microcontroller sensor log -> per-video-frame contact labels.

Rig (docs/CUP_RIG.md): copper-foil strips on a cup -> MPR121 capacitive board -> microcontroller,
which also blinks an LED in the camera's view in a pseudo-random pattern and logs every LED edge
on its own clock. Offline:
  1. led_states(): LED on / off per video frame from the brightness of a small image region.
  2. fit_clock(): match LED edges seen in the video to the logged edges and fit
     t_video = a * t_mcu + b (least squares over all matched edges; absorbs clock drift).
  3. contact_per_frame(): electrode readings -> touched / not per electrode, sampled at each
     video frame's time on the microcontroller clock.

Log format (firmware/cup_rig/cup_rig.ino), one record per line, times in microseconds:
  L,<t_us>,<0|1>                    LED edge
  C,<t_us>,<filt_0>,...,<filt_11>   MPR121 filtered data (lower = more capacitance = touch)
  B,<t_us>,<base_0>,...,<base_11>   MPR121 baselines (logged at a lower rate)
  W,<t_us>,<raw>                    HX711 load cell (optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_ELECTRODES = 12


@dataclass
class SensorLog:
    led_t: np.ndarray  # [E] s, microcontroller clock, LED edge times
    led_state: np.ndarray  # [E] int8, state after the edge
    cap_t: np.ndarray  # [M] s
    cap: np.ndarray  # [M, 12] filtered counts
    base_t: np.ndarray  # [B] s
    base: np.ndarray  # [B, 12] baseline counts
    load_t: np.ndarray  # [W] s
    load: np.ndarray  # [W] raw load-cell counts


def parse_log(path: Path | str) -> SensorLog:
    rows: dict[str, list] = {"L": [], "C": [], "B": [], "W": []}
    for line in Path(path).read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 3 or parts[0] not in rows:
            continue  # comments, partial lines at start-up
        try:
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        rows[parts[0]].append(vals)

    def arr(k: str, width: int) -> tuple[np.ndarray, np.ndarray]:
        r = [v for v in rows[k] if len(v) == width + 1]
        a = np.array(r, np.float64).reshape(-1, width + 1)
        return a[:, 0] * 1e-6, a[:, 1:]

    lt, ls = arr("L", 1)
    ct, c = arr("C", N_ELECTRODES)
    bt, b = arr("B", N_ELECTRODES)
    wt, w = arr("W", 1)
    return SensorLog(lt, ls[:, 0].astype(np.int8), ct, c, bt, b, wt, w[:, 0])


def led_states(brightness: np.ndarray) -> np.ndarray:
    """Per-frame mean brightness of the LED region [N] -> on / off [N] bool (threshold halfway
    between the two brightness clusters, Otsu style)."""
    x = np.asarray(brightness, np.float64)
    assert x.ndim == 1 and len(x) > 1
    xs = np.sort(x)
    best, thr = -1.0, float(xs[len(xs) // 2])
    for i in range(1, len(xs)):  # maximise between-class variance over split points
        a, b = xs[:i], xs[i:]
        v = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if v > best:
            best, thr = v, (xs[i - 1] + xs[i]) / 2
    return x > thr


def edges(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frame indices where a boolean sequence changes [E] and the new state [E]."""
    s = np.asarray(states, np.int8)
    idx = np.flatnonzero(np.diff(s) != 0) + 1
    return idx, s[idx]


@dataclass
class ClockFit:
    a: float  # t_video = a * t_mcu + b (seconds)
    b: float
    n_matched: int
    rms_s: float  # residual of matched edges (includes the camera's frame quantization)


def fit_clock(
    video_t: np.ndarray,
    video_state: np.ndarray,
    mcu_t: np.ndarray,
    mcu_state: np.ndarray,
    tol_s: float = 0.05,
) -> ClockFit:
    """Match LED edges seen in the video (times in s, e.g. frame index / fps) to the logged edges.

    Every (video edge, logged edge) pair of the same direction proposes an offset b (with a = 1);
    the offset that explains the most edges within tol_s wins (the random blink pattern makes it
    unique). Then a and b are refit by least squares on those matches, and matched again once.
    """
    vt, vs = np.asarray(video_t, np.float64), np.asarray(video_state)
    mt, ms = np.asarray(mcu_t, np.float64), np.asarray(mcu_state)
    assert len(vt) >= 3 and len(mt) >= 3, "need at least 3 LED edges in both streams"

    def match(a: float, b: float) -> list[tuple[int, int]]:
        pred = a * mt + b
        pairs = []
        for i, t in enumerate(vt):
            j = int(np.argmin(np.abs(pred - t)))
            if abs(pred[j] - t) <= tol_s and ms[j] == vs[i]:
                pairs.append((i, j))
        return pairs

    best: list[tuple[int, int]] = []
    for i in range(len(vt)):
        for j in np.flatnonzero(ms == vs[i]):
            p = match(1.0, vt[i] - mt[j])
            if len(p) > len(best):
                best = p
    assert len(best) >= 3, "could not match the LED patterns (check the LED region / log)"
    a, b = 1.0, 0.0
    for _ in range(2):
        i, j = np.array(best).T
        a, b = np.polyfit(mt[j], vt[i], 1)
        best = match(a, b)
    i, j = np.array(best).T
    res = vt[i] - (a * mt[j] + b)
    return ClockFit(float(a), float(b), len(best), float(np.sqrt(np.mean(res**2))))


def electrode_touch(log: SensorLog, delta: float = 12.0, release: float = 6.0) -> np.ndarray:
    """Touched / not per electrode at each C record [M, 12] with hysteresis on
    (baseline - filtered): touch when the drop exceeds `delta`, release below `release`.
    Defaults are the Adafruit MPR121 library's touch / release thresholds (12 / 6); choose your
    own on validation sessions only."""
    if len(log.base_t):
        k = np.clip(np.searchsorted(log.base_t, log.cap_t, side="right") - 1, 0, None)
        base = log.base[k]
    else:  # no baselines logged: the first second's median per electrode
        base = np.median(log.cap[log.cap_t < log.cap_t[0] + 1.0], axis=0, keepdims=True)
    drop = base - log.cap
    out = np.zeros(drop.shape, bool)
    on = np.zeros(drop.shape[1], bool)
    for i, d in enumerate(drop):
        on = np.where(on, d > release, d > delta)
        out[i] = on
    return out


def contact_per_frame(
    log: SensorLog, touch: np.ndarray, frame_t: np.ndarray, fit: ClockFit
) -> np.ndarray:
    """Electrode state [M, 12] -> per video frame [N, 12]: the latest reading at or before each
    frame's time on the microcontroller clock (NaN-free; frames before the first reading = 0)."""
    t_mcu = (np.asarray(frame_t, np.float64) - fit.b) / fit.a
    k = np.searchsorted(log.cap_t, t_mcu, side="right") - 1
    out = np.zeros((len(t_mcu), touch.shape[1]), bool)
    ok = k >= 0
    out[ok] = touch[k[ok]]
    return out
