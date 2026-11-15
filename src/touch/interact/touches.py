"""Per-finger touch tracks from the per-frame touch state: down / move / up events.

Input per frame: the post-processed touch state of the 5 fingers (thumb..pinky), the MediaPipe
fingertip pixels and the hand size (wrist -> middle MCP, px). Positions stay in image pixels; the
gesture layer divides distances by the hand size so its thresholds don't depend on camera
distance. No surface calibration yet. Causal: each frame only updates state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from touch.data.pressure import FINGER_NAMES, FINGER_TIPS


@dataclass
class TouchEvent:
    t: float  # seconds
    phase: str  # down | move | up
    touch_id: int
    finger: int  # 0 thumb .. 4 pinky
    xy: tuple[float, float]  # fingertip, image px
    hand_size: float  # px
    prob: float  # smoothed touch probability this frame
    lost: bool = False  # "up" because the hand was lost, not because the finger lifted

    def to_dict(self) -> dict:
        return {
            "type": "touch",
            "t": round(self.t, 4),
            "phase": self.phase,
            "id": self.touch_id,
            "finger": FINGER_NAMES[self.finger],
            "xy": [round(float(v), 1) for v in self.xy],
            "hand_size": round(float(self.hand_size), 1),
            "prob": round(float(self.prob), 3),
            **({"lost": True} if self.lost else {}),
        }


def hand_points(lm: np.ndarray | None) -> tuple[np.ndarray, float]:
    """MediaPipe landmarks [21, >=2] px (or None) -> fingertips [5, 2] (NaN if none), hand size."""
    if lm is None:
        return np.full((5, 2), np.nan), 0.0
    return lm[list(FINGER_TIPS), :2].astype(np.float64), float(np.hypot(*(lm[9, :2] - lm[0, :2])))


class TouchTracker:
    """One hand. A touch starts when a finger's state turns on and ends when it turns off or the
    hand is lost."""

    def __init__(self):
        self.active: dict[int, TouchEvent] = {}  # finger -> last event of its open touch
        self.next_id = 0

    def update(
        self, t: float, touch: np.ndarray, tips: np.ndarray, hand_size: float, prob: np.ndarray
    ) -> list[TouchEvent]:
        """touch [5] bool, tips [5, 2] px (NaN when no hand), prob [5] -> events of this frame."""
        hand = bool(np.isfinite(tips).all()) and hand_size > 0
        out = []
        for f in range(5):
            on = hand and bool(touch[f])
            p = float(prob[f])
            if f in self.active:
                last = self.active[f]
                if on:
                    ev = TouchEvent(t, "move", last.touch_id, f, tuple(tips[f]), hand_size, p)
                    self.active[f] = ev
                else:  # position = last contact point (the finger has already left the surface)
                    ev = TouchEvent(
                        t, "up", last.touch_id, f, last.xy, last.hand_size, p, lost=not hand
                    )
                    del self.active[f]
            elif on:
                ev = TouchEvent(t, "down", self.next_id, f, tuple(tips[f]), hand_size, p)
                self.next_id += 1
                self.active[f] = ev
            else:
                continue
            out.append(ev)
        return out
