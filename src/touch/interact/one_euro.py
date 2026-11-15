"""1 Euro filter (Casiez, Roussel, Vogel, CHI 2012, "1 Euro Filter: A Simple Speed-based Low-pass
Filter for Noisy Input in Interactive Systems"): an exponential low-pass whose cutoff grows with
the signal's speed, so slow motion is smoothed hard (no jitter) and fast motion lightly (little
lag).

    cutoff = min_cutoff + beta * |filtered derivative|
    alpha  = 1 / (1 + tau / Te),  tau = 1 / (2 pi cutoff),  Te = sampling period

OneEuro filters vectors (one cutoff from the vector's speed norm, so all components lag
equally); OneEuroRotation filters rotations through sign-aligned quaternions."""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0,
                 rate: float = 30.0):  # fmt: skip
        """min_cutoff (Hz): smoothing at rest; beta (1 / signal unit): how fast the cutoff grows
        with speed; d_cutoff (Hz): smoothing of the speed estimate; rate (Hz): when no time
        stamps are given."""
        self.min_cutoff, self.beta, self.d_cutoff, self.rate = min_cutoff, beta, d_cutoff, rate
        self.reset()

    def reset(self) -> None:
        self.x = self.dx = self.t = None

    def __call__(self, x, t: float | None = None) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x.copy()
        dt = 1.0 / self.rate if t is None or self.t is None else max(t - self.t, 1e-6)
        self.t = t
        dx = (x - self.x) / dt
        self.dx = self.dx + _alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + _alpha(cutoff, dt) * (x - self.x)
        return self.x.copy()


class OneEuroRotation:
    """Rotation matrices [3, 3] through a OneEuro on quaternions (sign-aligned to the previous
    output, renormalised). beta is per unit of quaternion speed (~ rad/s / 2)."""

    def __init__(self, **kw):
        self.f = OneEuro(**kw)
        self.q = None

    def reset(self) -> None:
        self.f.reset()
        self.q = None

    def __call__(self, R: np.ndarray, t: float | None = None) -> np.ndarray:
        from scipy.spatial.transform import Rotation as Rot

        q = Rot.from_matrix(R).as_quat()
        if self.q is not None and float(q @ self.q) < 0:
            q = -q
        qf = self.f(q, t)
        qf = qf / max(np.linalg.norm(qf), 1e-12)
        self.q = qf
        return Rot.from_quat(qf).as_matrix()
