import numpy as np
import pytest

from touch.interact.gestures import GestureConfig, GestureRecognizer
from touch.interact.touches import TouchTracker, hand_points

FPS = 30.0
SIZE = 100.0  # hand size, px


def run(frames, size=SIZE, scale=1.0):
    """frames: list of {finger: (x, y)} for fingers touching that frame (positions in hand sizes,
    None = hand lost) -> (touch events, gesture kinds)."""
    tracker, rec = TouchTracker(), GestureRecognizer()
    evs, gestures = [], []
    for i, fr in enumerate(frames):
        tips = np.full((5, 2), np.nan) if fr is None else np.full((5, 2), 300.0)
        touch = np.zeros(5, bool)
        for f, xy in (fr or {}).items():
            tips[f] = 300.0 + np.asarray(xy) * size * scale
            touch[f] = True
        e = tracker.update(i / FPS, touch, tips, size * scale if fr is not None else 0.0, touch)
        evs += e
        gestures += rec.update(e)
    return evs, [g.kind for g in gestures], gestures


def hold(f, n, xy=(0.0, 0.0)):
    return [{f: xy}] * n


def idle(n):
    return [{}] * n


def move(f, n, a, b):
    return [{f: tuple(np.asarray(a) + (np.asarray(b) - a) * k / (n - 1))} for k in range(n)]


def test_tracker_phases_ids_and_lost_hand():
    evs, _, _ = run(idle(2) + hold(1, 3) + idle(1) + hold(1, 2) + [None])
    assert [(e.phase, e.touch_id) for e in evs] == [
        ("down", 0), ("move", 0), ("move", 0), ("up", 0),
        ("down", 1), ("move", 1), ("up", 1),
    ]  # fmt: skip
    assert evs[-1].lost and not evs[3].lost
    assert evs[0].to_dict()["finger"] == "index"


def test_hand_points_shape():
    lm = np.random.default_rng(0).random((21, 3)) * 100
    tips, size = hand_points(lm)
    assert tips.shape == (5, 2) and size > 0
    tips, size = hand_points(None)
    assert tips.shape == (5, 2) and np.isnan(tips).all() and size == 0


def test_tap_and_double_tap():
    _, kinds, _ = run(idle(2) + hold(1, 4) + idle(20))
    assert kinds == ["tap"]
    _, kinds, g = run(idle(2) + hold(1, 4) + idle(5) + hold(1, 4, (0.05, 0.0)) + idle(5))
    assert kinds == ["tap", "double_tap"] and g[1].touch_ids == (0, 1)


def test_two_taps_far_apart_in_time_are_two_taps():
    _, kinds, _ = run(hold(1, 4) + idle(30) + hold(1, 4) + idle(2))
    assert kinds == ["tap", "tap"]


def test_long_press_fires_while_held_and_no_tap():
    _, kinds, g = run(hold(1, 40) + idle(2))
    assert kinds == ["long_press"]
    first = int(np.ceil(GestureConfig().long_press_s * FPS - 1e-9))  # first frame held long enough
    assert g[0].t == pytest.approx(first / FPS)


def test_slow_drag():
    _, kinds, g = run(move(1, 30, (0, 0), (1.0, 0.0)) + idle(2))
    assert kinds == ["drag_start", "drag_end"]
    assert g[1].info["dx"] == pytest.approx(1.0) and g[1].info["dy"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "end, direction",
    [((0.8, 0), "right"), ((-0.8, 0), "left"), ((0, -0.8), "up"), ((0, 0.8), "down")],
)
def test_fast_swipe_directions(end, direction):
    _, kinds, g = run(move(2, 5, (0, 0), end) + idle(2))
    assert kinds[-1] == "swipe" and g[-1].info["direction"] == direction
    assert "drag_end" not in kinds


def test_pinch_out_and_in():
    n = 15
    out = [{0: (-0.2 - 0.3 * k / n, 0.0), 1: (0.2 + 0.3 * k / n, 0.0)} for k in range(n)]
    _, kinds, g = run(out + idle(2))
    assert kinds == ["pinch_start", "pinch_end"]
    assert g[0].info["direction"] == "out" and g[1].info["scale"] > 2.0
    assert set(g[0].fingers) == {0, 1}
    inn = list(reversed(out))
    _, kinds, g = run(inn + idle(2))
    assert kinds == ["pinch_start", "pinch_end"] and g[0].info["direction"] == "in"


def test_three_fingers_spreading_is_not_a_pinch():
    n = 15
    spread = [
        {0: (-0.2 - 0.3 * k / n, 0), 1: (0.2 + 0.3 * k / n, 0), 2: (0.5, 0)} for k in range(n)
    ]
    _, kinds, _ = run(spread + idle(2))
    assert "pinch_start" not in kinds


def test_two_fingers_still_is_not_a_pinch():
    _, kinds, _ = run([{0: (-0.2, 0.0), 1: (0.2, 0.0)}] * 5 + idle(2))
    assert "pinch_start" not in kinds


def test_gestures_invariant_to_hand_size():
    frames = move(1, 30, (0, 0), (1.0, 0.0)) + idle(2) + hold(3, 4) + idle(2)
    assert run(frames)[1] == run(frames, scale=2.5)[1] == ["drag_start", "drag_end", "tap"]


def test_one_frame_flicker_is_not_a_tap():
    _, kinds, _ = run(idle(2) + hold(1, 1) + idle(20))
    assert kinds == []


def test_slow_key_press_is_still_a_tap():  # typing: p50 0.37 s on real data
    _, kinds, _ = run(idle(2) + hold(1, 15) + idle(20))
    assert kinds == ["tap"]
