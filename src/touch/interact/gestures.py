"""Rule-based touch gestures from touch events (touch.interact.touches).

Gestures (SurfaceXR's set of tap / double tap / swipes / pinch, plus long press and drag):
    tap, double_tap        short touch, little movement (double: a second tap soon after, nearby)
    long_press             touch held still for long_press_s (fires while still touching)
    drag_start, drag_end   touch moved more than drag_min_move (drag_end on lift)
    swipe                  fast, short moving touch; direction left / right / up / down (image axes)
    pinch_start, pinch_end exactly two touches whose distance changes by pinch_min_log_scale
                           (in / out); with 3+ fingers down (e.g. a whole-hand press) no pinch
Distances are in hand sizes (wrist -> middle MCP), speeds in hand sizes per second. Tap / long-press
thresholds were set from ground-truth touches of the TRAIN touch-table sequences
(scripts/gesture_report.py, docs/DECISIONS.md 2026-11-08); the rest are still starting values.
Tune on train / validation or our own recordings, never on the test set.
Causal: a gesture is emitted on the frame it is decided (a tap is reported at lift-off, and a
double tap is reported after the tap it completes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from touch.data.pressure import FINGER_NAMES
from touch.interact.touches import TouchEvent


@dataclass
class GestureConfig:
    tap_min_s: float = 0.05  # shorter touches (1 frame at 30 FPS) are flicker, not taps
    tap_max_s: float = 0.80  # = long_press_s: a still touch released before a long press is a tap
    tap_max_move: float = 0.20
    double_tap_gap_s: float = 0.35  # lift of the first tap -> touch-down of the second
    double_tap_max_dist: float = 0.30
    long_press_s: float = 0.80  # type_ipad key presses: p50 0.37 s, p98 0.73 s (train)
    drag_min_move: float = 0.25
    swipe_max_s: float = 0.40
    swipe_min_speed: float = 2.0
    pinch_min_log_scale: float = 0.15


@dataclass
class Gesture:
    t: float
    kind: str
    fingers: tuple[int, ...]
    touch_ids: tuple[int, ...]
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": "gesture",
            "t": round(self.t, 4),
            "kind": self.kind,
            "fingers": [FINGER_NAMES[f] for f in self.fingers],
            "ids": list(self.touch_ids),
            **self.info,
        }


@dataclass
class _Track:
    down: TouchEvent
    last: TouchEvent
    max_move: float = 0.0
    long_fired: bool = False
    dragging: bool = False
    in_pinch: bool = False


def _disp(a: TouchEvent, b: TouchEvent) -> np.ndarray:
    """Displacement a -> b in hand sizes."""
    size = 0.5 * (a.hand_size + b.hand_size)
    return (np.asarray(b.xy) - np.asarray(a.xy)) / max(size, 1e-6)


def _direction(d: np.ndarray) -> str:
    if abs(d[0]) >= abs(d[1]):
        return "right" if d[0] > 0 else "left"
    return "down" if d[1] > 0 else "up"  # image y points down


class GestureRecognizer:
    def __init__(self, cfg: GestureConfig | None = None):
        self.cfg = cfg or GestureConfig()
        self.tracks: dict[int, _Track] = {}  # open touches by id
        self.last_tap: dict[int, TouchEvent] = {}  # finger -> lift event of its last single tap
        self.pinch: tuple[int, int] | None = None  # candidate / active pinch touch ids
        self.pinch_d0 = self.pinch_d1 = 0.0  # pair distance at candidate start / now
        self.pinch_on = False

    def update(self, events: list[TouchEvent]) -> list[Gesture]:
        out: list[Gesture] = []
        for e in events:
            if e.phase == "down":
                self.tracks[e.touch_id] = _Track(e, e)
            elif e.phase == "move":
                self._move(self.tracks[e.touch_id], e, out)
            else:
                self._up(self.tracks.pop(e.touch_id), e, out)
        self._pinch(out)
        return out

    def _move(self, tr: _Track, e: TouchEvent, out: list[Gesture]) -> None:
        c = self.cfg
        tr.last = e
        tr.max_move = max(tr.max_move, float(np.linalg.norm(_disp(tr.down, e))))
        if tr.in_pinch:
            return
        if (
            not tr.long_fired
            and not tr.dragging
            and tr.max_move < c.tap_max_move
            and e.t - tr.down.t >= c.long_press_s
        ):
            tr.long_fired = True
            out.append(Gesture(e.t, "long_press", (e.finger,), (e.touch_id,)))
        if not tr.dragging and tr.max_move >= c.drag_min_move:
            tr.dragging = True
            out.append(Gesture(e.t, "drag_start", (e.finger,), (e.touch_id,)))

    def _up(self, tr: _Track, e: TouchEvent, out: list[Gesture]) -> None:
        c = self.cfg
        tr.last = e
        if self.pinch and e.touch_id in self.pinch:
            if self.pinch_on:
                out.append(self._pinch_gesture(e.t, "pinch_end", e))
            self.pinch, self.pinch_on = None, False
        if tr.in_pinch:
            return
        dur = e.t - tr.down.t
        d = _disp(tr.down, e)
        dist = float(np.linalg.norm(d))
        ids, fingers = (e.touch_id,), (e.finger,)
        info = {
            "duration_s": round(dur, 3),
            "dx": round(float(d[0]), 3),
            "dy": round(float(d[1]), 3),
        }
        if (
            not tr.long_fired
            and dur <= c.swipe_max_s
            and dist >= c.drag_min_move
            and dist / max(dur, 1e-3) >= c.swipe_min_speed
        ):
            out.append(Gesture(e.t, "swipe", fingers, ids, {"direction": _direction(d), **info}))
        elif tr.dragging:
            out.append(Gesture(e.t, "drag_end", fingers, ids, info))
        elif (
            not tr.long_fired and c.tap_min_s <= dur <= c.tap_max_s and tr.max_move < c.tap_max_move
        ):
            prev = self.last_tap.get(e.finger)
            if (
                prev is not None
                and tr.down.t - prev.t <= c.double_tap_gap_s
                and np.linalg.norm(_disp(prev, tr.down)) <= c.double_tap_max_dist
            ):
                out.append(Gesture(e.t, "double_tap", fingers, (prev.touch_id, e.touch_id), info))
                del self.last_tap[e.finger]
            else:
                out.append(Gesture(e.t, "tap", fingers, ids, info))
                self.last_tap[e.finger] = e

    def _pair_dist(self) -> float:
        a, b = (self.tracks[i].last for i in self.pinch)
        return float(np.linalg.norm(_disp(a, b)))

    def _pinch_gesture(self, t: float, kind: str, lifted: TouchEvent | None = None) -> Gesture:
        """lifted: the pinch touch that just ended (already removed from self.tracks)."""
        fingers = tuple(
            self.tracks[i].last.finger if i in self.tracks else lifted.finger for i in self.pinch
        )
        info = {"scale": round(self.pinch_d1 / max(self.pinch_d0, 1e-6), 3)}
        if kind == "pinch_start":
            info["direction"] = "out" if self.pinch_d1 > self.pinch_d0 else "in"
        return Gesture(t, kind, fingers, self.pinch, info)

    def _pinch(self, out: list[Gesture]) -> None:
        free = [i for i, tr in self.tracks.items() if not tr.in_pinch]
        if self.pinch is None:
            if len(free) == 2:
                self.pinch = tuple(sorted(free))
                self.pinch_d0 = self.pinch_d1 = self._pair_dist()
            return
        if not self.pinch_on and len(self.tracks) > 2:  # a third finger joined: not a pinch
            self.pinch = None
            return
        if not all(i in self.tracks for i in self.pinch):
            return
        self.pinch_d1 = self._pair_dist()
        t = max(self.tracks[i].last.t for i in self.pinch)
        if (
            not self.pinch_on
            and abs(np.log(max(self.pinch_d1, 1e-6) / max(self.pinch_d0, 1e-6)))
            >= self.cfg.pinch_min_log_scale
        ):
            self.pinch_on = True
            for i in self.pinch:
                self.tracks[i].in_pinch = True
            out.append(self._pinch_gesture(t, "pinch_start"))
