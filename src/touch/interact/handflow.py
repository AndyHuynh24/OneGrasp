"""TouchSession results -> HandFlow's touch gesture names, so OneTouch can replace HandFlow's
landmark-only touch classes (touch_hover / touch / touch_hold).

Per frame, for one finger (index by default):
    None          no hand
    "touch_hover" hand present, finger not touching (cursor follows the fingertip)
    "touch_hold"  finger touching for longer than a tap, or moved like a drag (press-and-drag)
    "touch"       emitted once, on the frame a tap ends (short, still contact) -> click
Thresholds are GestureConfig's (tap_max_s, tap_max_move in hand sizes), tuned on train data.
`xy` is the fingertip in surface units (or px without a surface); for "touch" it is where the
finger went down, as on a real touchscreen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from touch.interact.gestures import GestureConfig
from touch.interact.session import FrameResult

INDEX = 1


@dataclass
class HandFlowTouch:
    gesture: str | None
    xy: tuple[float, float] | None
    prob: float


class HandFlowAdapter:
    def __init__(self, finger: int = INDEX, cfg: GestureConfig | None = None):
        self.finger, self.cfg = finger, cfg or GestureConfig()
        self.down = None  # (t, image xy, surface-or-image xy, hand size) of the open touch
        self.holding = False

    def update(self, res: FrameResult) -> HandFlowTouch:
        f, c = self.finger, self.cfg
        uv = res.tips_surface if res.tips_surface is not None else res.tips
        xy = tuple(float(v) for v in uv[f]) if res.present else None
        prob = float(res.prob[f])
        for e in res.events:
            if e.finger != f:
                continue
            if e.phase == "down":
                d_uv = xy if xy is not None else e.xy
                self.down, self.holding = (e.t, np.array(e.xy), d_uv, e.hand_size), False
            elif e.phase == "up" and self.down is not None:
                t0, xy0, uv0, size = self.down
                was_hold = self.holding or self._is_hold(e.t, e.xy, xy0, t0, size)
                self.down, self.holding = None, False
                if not was_hold and not e.lost and e.t - t0 >= c.tap_min_s:
                    return HandFlowTouch("touch", uv0, prob)
        if not res.present:
            return HandFlowTouch(None, None, prob)
        if self.down is not None:
            t0, xy0, _, size = self.down
            self.holding = self.holding or self._is_hold(res.t, res.tips[f], xy0, t0, size)
            if self.holding:
                return HandFlowTouch("touch_hold", xy, prob)
        return HandFlowTouch("touch_hover", xy, prob)

    def _is_hold(self, t, xy, xy0, t0, size) -> bool:
        moved = float(np.hypot(*(np.asarray(xy) - xy0))) / max(size, 1e-6)
        return t - t0 > self.cfg.tap_max_s or moved > self.cfg.tap_max_move
