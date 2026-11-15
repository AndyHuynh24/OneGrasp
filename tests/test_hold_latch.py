import numpy as np

from touch.interact.hold_latch import HoldLatch, palm_height


def run(states, z=None, present=None, **kw):
    n = len(states)
    z = np.zeros(n) if z is None else np.asarray(z, float)
    present = np.ones(n, bool) if present is None else np.asarray(present)
    out = HoldLatch(**kw).run(np.asarray(states), present, z)
    assert out.shape == (n,) and out.dtype == np.int8
    return out


def test_hold_survives_brief_touch_flicker():
    s = [0, 1, 2, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    out = run(s, k_off=5, rise_m=1e9)
    assert (out[2:9] == 2).all()  # touch flicker inside the hold stays hold
    assert out[-1] == 0  # released after k_off none frames


def test_released_when_set_down():
    s = [1, 2, 2, 2] + [1] * 12
    z = [0.0, 0.0, 0.0, 0.10, 0.10, 0.05] + [0.0] * 10
    out = run(s, z, k_down=3, down_m=0.02, rise_m=1e9, med=1)
    assert out[3] == 2 and out[-1] == 1  # back at the start height, model says touch -> touch


def test_rise_cue_enters_hold_without_model_hold():
    s = [1] * 10
    z = [0.0] * 4 + [0.06] * 6
    assert run(s, z, rise_m=0.03, med=1)[-1] == 2
    assert run(s, z, rise_m=1e9, med=1)[-1] == 1


def test_lost_hand_releases_after_k_off():
    s = [2, 2, 2, 2, 2, 2]
    out = run(s, present=[1, 1, 0, 0, 0, 0], k_off=3)
    assert out[1] == 2 and out[-1] == 0


def test_palm_height_nan_rows():
    j = np.zeros((2, 21, 3))
    j[0, :, 2] = 0.1
    j[1, 0, 2] = np.nan
    h = palm_height(j)
    assert h.shape == (2,) and np.isclose(h[0], 0.1) and np.isnan(h[1])
