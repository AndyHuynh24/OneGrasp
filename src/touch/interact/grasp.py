"""Per-finger contact [N, 5] -> hand-object interaction states for labelling demonstrations.

Derived states, not a separate classifier (reports/Hand object contact from RGB.md):
  NONE   no finger in contact
  TOUCH  some finger in contact, but no grasp
  GRASP  thumb + at least one opposing finger in contact (EgoContact's thumb + "virtual finger")
  HOLD   a grasp that has lasted at least `hold_frames` (default 15 = 0.5 s at 30 fps, ARCTIC's
         "stable contact" duration); frames before that point stay GRASP (causal: a frame is HOLD
         only once the grasp has lasted long enough, so live use needs no future frames)
Telling "holding it in the air" from "gripping it on the table" needs an object signal (e.g. a
load cell): not modelled here.
"""

from __future__ import annotations

import numpy as np

NONE, TOUCH, GRASP, HOLD = 0, 1, 2, 3
STATE_NAMES = ("none", "touch", "grasp", "hold")
THUMB = 0


def grasp_states(contact: np.ndarray, hold_frames: int = 15) -> np.ndarray:
    """contact [N, 5] bool (thumb..pinky) -> states [N] int8 (NONE / TOUCH / GRASP / HOLD)."""
    c = np.asarray(contact, dtype=bool)
    assert c.ndim == 2 and c.shape[1] == 5, c.shape
    grasp = c[:, THUMB] & c[:, 1:].any(1)
    state = np.where(grasp, GRASP, np.where(c.any(1), TOUCH, NONE)).astype(np.int8)
    run = 0
    for i, g in enumerate(grasp):
        run = run + 1 if g else 0
        if run >= hold_frames:
            state[i] = HOLD
    return state


def state_mask(states: np.ndarray, which: str) -> np.ndarray:
    """Boolean [N] for interval scoring: "contact" (any finger), "grasp" (GRASP or HOLD), "hold"."""
    s = np.asarray(states)
    return {"contact": s >= TOUCH, "grasp": s >= GRASP, "hold": s == HOLD}[which]


class StickyHold:
    """Causal post-processing of the grasp model's 3-way state (0 none, 1 touch, 2 hold): once
    HOLD, a TOUCH prediction stays HOLD until contact ends (NONE or no hand). A held object cannot
    become "just touched" without being let go; DexYCB's hold label needs the thumb within 3 mm,
    so the model calls many real holds "touch" (val vs a thumb-free hold label: hold recall
    0.61 -> 0.71 at precision 0.97; docs/DECISIONS.md)."""

    def __init__(self) -> None:
        self.prev = 0

    def step(self, state: int) -> int:
        s = 2 if (self.prev == 2 and state == 1) else int(state)
        self.prev = s
        return s


def sticky_hold(states: np.ndarray) -> np.ndarray:
    """[N] model states (0 none / 1 touch / 2 hold; no hand = 0) -> [N] with StickyHold applied."""
    sh = StickyHold()
    return np.array([sh.step(int(s)) for s in states], np.int8)
