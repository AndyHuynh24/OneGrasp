import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_own_grasp import EDGE_S, clean_state, touch_labels  # noqa: E402


def test_clean_state_ignores_invalid_and_none_near_hold_edges():
    n, k = 300, int(EDGE_S * 30)
    label = np.zeros(n, np.int8)
    label[100:200] = 2
    label[60:70] = 1  # an F press before the grab stays touch
    valid = np.ones(n, bool)
    valid[200:213] = False  # release pad
    s = clean_state(label, valid)
    assert s.shape == (n,) and s.dtype == np.int8
    assert (s[100:200] == 2).all() and (s[60:70] == 1).all()
    assert (s[100 - k : 100] == -1).all()  # none within EDGE_S before the grab
    assert (s[200 : 200 + k] == -1).all() and (s[200 + k :] == 0).all()
    assert (s[:60] == 0).all()


def test_touch_labels_selected_finger_only_and_no_object_state():
    n = 10
    label = np.array([0, 1, 1, 0, 2, 2, 0, 0, 1, 0], np.int8)  # tap, long press, tap
    valid = np.ones(n, bool)
    valid[9] = False
    finger = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2], np.int8)  # index, then middle
    state, fc = touch_labels(label, valid, finger)
    assert state.shape == (n,) and fc.shape == (n, 5) and fc.dtype == np.int8
    assert (state[:9] == 0).all() and state[9] == -1  # no object: none wherever labelled
    assert fc[1, 1] == 1 and fc[4, 1] == 1 and fc[8, 2] == 1  # tap + long press = contact
    assert fc[0, 1] == 0 and fc[6, 2] == 0  # selected finger up
    assert fc[1, 0] == 0 and fc[8, 1] == 0  # the other fingers: not touching
    assert (fc[9] == -1).all()  # invalid frame
