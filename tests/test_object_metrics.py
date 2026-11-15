import numpy as np
import pytest

from touch.eval.metrics import interval_timing, intervals, match_intervals
from touch.interact.grasp import GRASP, HOLD, NONE, TOUCH, grasp_states, state_mask


def _seq(n, *ivs):
    x = np.zeros(n, bool)
    for a, b in ivs:
        x[a:b] = True
    return x


def test_intervals_shape_and_edges():
    iv = intervals(_seq(20, (0, 3), (10, 20)))
    assert iv.shape == (2, 2) and iv.tolist() == [[0, 3], [10, 20]]
    assert intervals(np.zeros(5, bool)).shape == (0, 2)


def test_perfect_prediction():
    g = _seq(100, (10, 30), (50, 80))
    r = interval_timing(g, g)
    assert r["matched"] == 2
    for k in ("onset", "release"):
        assert r[k]["sr@3"] == 1.0 and r[k]["mae_frames"] == 0.0


def test_shifted_prediction():
    g = _seq(100, (10, 30), (50, 80))
    p = _seq(100, (12, 34), (46, 80))  # onset +2 / -4, release +4 / 0
    r = interval_timing(p, g)
    assert r["onset"]["errors_frames"].tolist() == [2, -4]
    assert r["onset"]["sr@3"] == 0.5 and r["onset"]["sr@5"] == 1.0
    assert r["release"]["errors_frames"].tolist() == [4, 0]
    assert r["onset"]["mae_frames"] == 3.0 and r["onset"]["mae_ms"] == pytest.approx(100.0)


def test_missed_interval_counts_as_failure():
    g = _seq(100, (10, 30), (50, 80))
    p = _seq(100, (10, 30))
    r = interval_timing(p, g)
    assert r["matched"] == 1 and r["onset"]["sr@10"] == 0.5 and r["onset"]["mae_frames"] == 0.0


def test_merged_prediction_matches_one_gt_only():
    g = _seq(100, (10, 30), (35, 60))
    p = _seq(100, (10, 60))
    pairs = match_intervals(intervals(p), intervals(g))
    assert pairs == [(1, 0)]  # higher IoU with the longer GT interval
    r = interval_timing(p, g)
    assert r["matched"] == 1 and r["release"]["sr@3"] == 0.5


def test_no_gt_gives_nan_sr():
    r = interval_timing(_seq(50, (5, 10)), np.zeros(50, bool))
    assert np.isnan(r["onset"]["sr@3"]) and r["n_pred"] == 1 and r["matched"] == 0


def test_grasp_states():
    c = np.zeros((40, 5), bool)
    c[5:10, 1] = True  # index only: touch
    c[20:40, 0] = True  # thumb + middle: grasp, hold after 15 frames
    c[20:40, 2] = True
    s = grasp_states(c, hold_frames=15)
    assert s.shape == (40,) and s.dtype == np.int8
    assert (s[5:10] == TOUCH).all() and (s[10:20] == NONE).all()
    assert (s[20:34] == GRASP).all() and (s[34:] == HOLD).all()
    assert state_mask(s, "grasp")[20:].all() and not state_mask(s, "grasp")[5:10].any()


def test_thumb_alone_is_touch_not_grasp():
    c = np.zeros((10, 5), bool)
    c[:, 0] = True
    assert (grasp_states(c) == TOUCH).all() and not (grasp_states(c) == GRASP).any()
    assert HOLD not in grasp_states(c)
    assert GRASP == 2
