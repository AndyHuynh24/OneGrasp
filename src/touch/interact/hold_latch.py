"""Causal hold latch on top of the grasp model's per-frame state (docs/DECISIONS.md 2026-11-12).

The model misses holds mostly as "touch" mid-carry and lets go early. Physical cue: while the hand
is in contact, if it rises above where the contact began, the object went up with it (held).
Per frame (live: one call per frame; offline: `run`):

  contact episode  starts at the first touch/hold frame, ends after `k_off` frames of none /
                   no hand. z0 = the hand's height at the start (median of its first frames).
  enter HOLD       the model says hold, or the hand rose >= rise_m above z0 during contact.
  stay HOLD        until the contact episode ends (release), or the hand is back within down_m
                   of z0 and the model has not said hold for k_down frames (set down -> touch).
  otherwise        the model's own state.

Hand height = causal median (last `med` frames) of the palm height in the table frame (m),
NaN when unknown (no hand / no 3D fit): rises are then not detected, the model state still is.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

PALM = (0, 5, 9, 13, 17)  # wrist + finger bases (our 21-joint order)


@dataclass
class HoldLatch:
    rise_m: float = 0.03
    down_m: float = 0.015
    k_off: int = 5
    k_down: int = 5
    med: int = 5
    z0_frames: int = 3
    _hist: deque = field(default_factory=deque, repr=False)
    _in_contact: bool = False
    _off: int = 0
    _z0_buf: list = field(default_factory=list, repr=False)
    _z0: float = float("nan")
    _hold: bool = False
    _no_hold: int = 0

    def reset(self) -> None:
        self._hist.clear()
        self._in_contact, self._off, self._z0_buf = False, 0, []
        self._z0, self._hold, self._no_hold = float("nan"), False, 0

    def _height(self, z: float) -> float:
        self._hist.append(z)
        while len(self._hist) > self.med:
            self._hist.popleft()
        v = [h for h in self._hist if np.isfinite(h)]
        return float(np.median(v)) if v else float("nan")

    def step(self, state: int, present: bool, palm_z: float = float("nan")) -> int:
        """state: the model's argmax (0 none, 1 touch, 2 hold); returns the latched state."""
        z = self._height(palm_z if present else float("nan"))
        s = state if present else 0
        if s >= 1:
            if not self._in_contact:
                self._in_contact, self._z0_buf, self._z0 = True, [], float("nan")
            self._off = 0
        elif self._in_contact:
            self._off += 1
            if self._off >= self.k_off:  # contact over: release
                self._in_contact, self._hold = False, False
        if self._in_contact and not np.isfinite(self._z0) and np.isfinite(z):
            self._z0_buf.append(z)
            if len(self._z0_buf) >= self.z0_frames:
                self._z0 = float(np.median(self._z0_buf))
        rise = z - self._z0 if np.isfinite(z) and np.isfinite(self._z0) else float("nan")
        self._no_hold = 0 if s == 2 else self._no_hold + 1
        if self._in_contact and (s == 2 or (np.isfinite(rise) and rise >= self.rise_m)):
            self._hold = True
        elif (
            self._hold
            and np.isfinite(rise)
            and rise <= self.down_m
            and self._no_hold >= self.k_down
        ):
            self._hold = False  # back down where it started, model agrees it is not holding
        if self._hold and self._in_contact:
            return 2
        return s if s != 2 else 1  # model said hold but latch is off: cannot happen (s==2 latches)

    def run(self, states: np.ndarray, present: np.ndarray, palm_z: np.ndarray) -> np.ndarray:
        self.reset()
        rows = zip(states, present, palm_z, strict=True)
        return np.array([self.step(int(s), bool(p), float(z)) for s, p, z in rows], np.int8)


def palm_height(joints_world: np.ndarray) -> np.ndarray:
    """[N, 21, 3] table-frame joints (NaN = none) -> [N] palm height (m)."""
    z = joints_world[:, PALM, 2]
    ok = np.isfinite(z).all(1)
    out = np.full(len(z), np.nan)
    out[ok] = z[ok].mean(1)
    return out
