"""Learned causal grab gate: a small MLP ensemble over the grasp model's per-frame probabilities
and their short causal history (scripts/grab_fusion.py fits it, on DexYCB val + OakInk2; the
hysteresis gate is selected on HO-Cap val), then a hysteresis on its grab score.

`prob_features` is the single definition of the features, used offline (batch) and live
(`FusionGate`, which recomputes it on the last WINDOW frames, so both match exactly).
"""

from __future__ import annotations

import json
import warnings
from collections import deque
from pathlib import Path

import numpy as np

WINDOW = 30  # the longest causal window in prob_features


def rolling(x: np.ndarray, w: int, fn) -> np.ndarray:
    """Causal rolling statistic over the last w frames (NaN-aware) of x [N, k] -> [N, k]."""
    from numpy.lib.stride_tricks import sliding_window_view

    pad = np.full((w - 1, *x.shape[1:]), np.nan)
    v = sliding_window_view(np.concatenate([pad, x]), w, axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(v, axis=-1)


def prob_features(state: np.ndarray, finger: np.ndarray, present: np.ndarray):
    """state [N, 3] (none, touch, hold), finger [N, 5], present [N] -> X [N, 20], names."""
    pres = np.asarray(present).astype(bool)
    S = np.where(pres[:, None], state, np.nan)
    Fp = np.where(pres[:, None], finger, np.nan)
    cols = [S, Fp, pres[:, None].astype(float)]
    names = ["p_none", "p_touch", "p_hold"] + [f"p_f{f}" for f in range(5)] + ["present"]
    for w in (5, 15, 30):
        cols.append(rolling(S, w, np.nanmean))
        names += [f"{n}_m{w}" for n in ("p_none", "p_touch", "p_hold")]
    cols.append(rolling(S[:, 2:3], 15, np.nanmax))
    names.append("p_hold_max15")
    cols.append(rolling(S[:, 0:1], 10, np.nanmin))
    names.append("p_none_min10")
    return np.concatenate(cols, 1).astype(np.float32), names


class Fusion:
    """MLP (ReLU) over standardised features; NaN -> 0 plus a NaN flag per feature. numpy only."""

    def __init__(self, mean, std, layers):
        self.mean, self.std, self.layers = mean, std, layers

    def prep(self, X: np.ndarray) -> np.ndarray:
        nan = ~np.isfinite(X)
        Z = np.where(nan, 0.0, (np.nan_to_num(X) - self.mean) / self.std)
        return np.concatenate([Z, nan.astype(np.float32)], 1).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        h = self.prep(X)
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        p = 1.0 / (1.0 + np.exp(-np.clip(h[:, 0], -50, 50)))
        return np.stack([1 - p, p], 1)

    def save(self, path: Path) -> None:
        arrs = {"mean": self.mean, "std": self.std}
        for i, (W, b) in enumerate(self.layers):
            arrs[f"W{i}"], arrs[f"b{i}"] = W, b
        np.savez(path, **arrs)

    @classmethod
    def load(cls, path: Path) -> Fusion:
        z = np.load(path)
        n = sum(k.startswith("W") for k in z.files)
        return cls(z["mean"], z["std"], [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)])


class Ensemble:
    def __init__(self, members: list[Fusion]):
        self.members = members

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self.members], 0)


class Hysteresis:
    """Close after k_on frames with score >= on, open after k_off frames with score < off;
    frames without a hand keep the decision.

    Fast release (k_none > 0): also open after k_none frames in a row where the grasp model's
    own P(NONE) >= p_none. The learned score lingers for 1-2 s after an object is put down
    (it averages recent history); the model's NONE reacts at once."""

    def __init__(self, on: float, off: float, k_on: int, k_off: int,
                 p_none: float = 1.0, k_none: int = 0):  # fmt: skip
        self.on, self.off, self.k_on, self.k_off = on, off, k_on, k_off
        self.p_none, self.k_none = p_none, k_none
        self.reset()

    def reset(self) -> None:
        self.close, self.run, self.run_none = False, 0, 0

    def step(self, s: float, present: bool, p_none: float = float("nan")) -> bool:
        if present and np.isfinite(s):
            if not self.close:
                self.run = self.run + 1 if s >= self.on else 0
                if self.run >= self.k_on:
                    self.close, self.run, self.run_none = True, 0, 0
            else:
                self.run = self.run + 1 if s < self.off else 0
                self.run_none = self.run_none + 1 if p_none >= self.p_none else 0
                if self.run >= self.k_off or (self.k_none and self.run_none >= self.k_none):
                    self.close, self.run, self.run_none = False, 0, 0
        return self.close


class FusionGate:
    """Live: feed the grasp model's outputs (and, for the richer feature sets, the frame, the
    landmarks and the hand joints) one frame at a time -> gripper closed?, grab score."""

    def __init__(self, model: Ensemble, gate: dict, features: list[str] | None = None):
        self.model, self.hyst = model, Hysteresis(**gate)
        _, pnames = prob_features(np.zeros((1, 3)), np.zeros((1, 5)), np.ones(1))
        self.features = list(features) if features is not None else pnames
        self.stream = None
        if self.features != pnames:  # needs more than the probabilities
            from touch.interact.hold_stream import HoldStream

            self.stream = HoldStream(self.features)
        self.reset()

    @classmethod
    def from_dir(cls, run_dir: Path, variant: str = "start") -> FusionGate:
        """runs/grab_fusion/<run>/: fusion_<variant>.json (features, gate) and
        fusion_<variant>_m<k>.npz (the MLP ensemble). variant "start": probabilities-only gate;
        "hold": the gate chosen by scripts/hold_study.py --final."""
        sel = json.loads((run_dir / f"fusion_{variant}.json").read_text())
        mem = sorted(run_dir.glob(f"fusion_{variant}_m*.npz"))
        assert mem, f"no fusion weights in {run_dir}"
        return cls(Ensemble([Fusion.load(p) for p in mem]), sel["gate"], sel["features"])

    @property
    def needs_frames(self) -> bool:
        return self.stream is not None

    def frame_features(self, present: bool, lm=None, gray=None, bgr=None) -> dict:
        """Image-dependent features of one frame (HoldStream.frame_features; {} for the
        probabilities-only gate): pass them to step(**) instead of the frame."""
        return {} if self.stream is None else self.stream.frame_features(present, lm, gray, bgr)

    def reset(self) -> None:
        self.buf: deque = deque(maxlen=WINDOW)
        if self.stream is not None:
            self.stream.reset()
        self.hyst.reset()
        self.score = float("nan")

    def step(self, state: np.ndarray, finger: np.ndarray, present: bool, **frame) -> bool:
        """frame (only for the richer feature sets): joints_pred, lm, gray, joints_world, bgr
        (see touch.interact.hold_stream.HoldStream.step)."""
        if self.stream is not None:
            x = self.stream.step(state, finger, present, **frame)[None]
        else:
            self.buf.append((np.asarray(state, float), np.asarray(finger, float), bool(present)))
            st, fi, pr = (np.array(v) for v in zip(*self.buf, strict=True))
            x = prob_features(st, fi, pr)[0][-1:]
        self.score = float(self.model.predict_proba(x)[0, 1]) if present else float("nan")
        return self.hyst.step(self.score, present, float(np.asarray(state)[0]))
