import numpy as np

from touch.interact.object_gate import ObjectGate, latch


def _hand(cx, cy, s=40.0):
    """A fake MediaPipe hand [21, 3]: wrist at (cx, cy + s), fingers pointing up."""
    lm = np.zeros((21, 3), np.float32)
    lm[0] = [cx, cy + s, 0]
    for f in range(5):
        x = cx + (f - 2) * 0.3 * s
        for j in range(4):
            lm[1 + 4 * f + j] = [x, cy - j * 0.3 * s, 0]
    lm[9] = [cx, cy, 0]
    return lm


def test_gate_sees_object_not_bare_table():
    table = np.full((240, 320, 3), (170, 200, 210), np.uint8)
    frame = table.copy()
    frame[20:60, 140:180] = (30, 30, 200)  # red block on the table
    g = ObjectGate()
    near_obj = _hand(160, 90)  # fingertips reach y ~ 54 -> touching the block
    bare = _hand(60, 170)
    for _ in range(10):  # the hand approaches from elsewhere: the block is seen first
        g.update(frame, [_hand(260, 170), bare])
    out = [g.update(frame, [near_obj, bare]) for _ in range(12)][-1]
    assert out == [True, False]


def test_latch_starts_only_near_object_then_persists():
    states = np.array([0, 2, 2, 2, 0, 1, 1, 0])
    near = np.array([0, 0, 1, 0, 0, 0, 0, 0], bool)
    out = latch(states, near)
    assert out.shape == (8,) and out.dtype == np.int8
    assert out.tolist() == [0, 0, 2, 2, 0, 0, 0, 0]
