import numpy as np

from touch.interact.grasp import StickyHold, sticky_hold


def test_sticky_hold_keeps_hold_until_contact_ends():
    s = np.array([0, 1, 2, 1, 1, 2, 1, 0, 1, 2])
    out = sticky_hold(s)
    assert out.shape == s.shape
    assert out.tolist() == [0, 1, 2, 2, 2, 2, 2, 0, 1, 2]
    sh = StickyHold()
    assert [sh.step(x) for x in (1, 2, 1, 0, 1)] == [1, 2, 2, 0, 1]


def test_labels_from_dist_hold_rules():
    from touch.data.dexycb import labels_from_dist

    d = np.full((3, 6), 0.05)  # metres: palm, thumb..pinky
    d[0, [0, 2]] = 0.001  # palm + index touching, thumb far (a wrap grip)
    d[1, [1, 2]] = 0.001  # thumb + index
    d[2, [2]] = 0.001  # one finger only
    disp, ok = np.full(3, 0.05), np.ones(3, bool)
    thumb = labels_from_dist(d, disp, ok)["state"]
    any2 = labels_from_dist(d, disp, ok, hold_rule="any2")["state"]
    assert thumb.shape == any2.shape == (3,)
    assert thumb.tolist() == [1, 2, 1]
    assert any2.tolist() == [2, 2, 1]
