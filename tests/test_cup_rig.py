import numpy as np

from touch.data.cup_rig import (
    ClockFit,
    contact_per_frame,
    edges,
    electrode_touch,
    fit_clock,
    led_states,
    parse_log,
)


def _blinks(rng, n=40, t0=0.5):
    """Pseudo-random LED pattern: edge times (s, microcontroller clock) and states."""
    t = t0 + np.cumsum(rng.uniform(0.2, 0.8, n))
    return t, (np.arange(n) % 2 == 0).astype(np.int8)  # on, off, on, ...


def test_led_states_threshold():
    b = np.array([10, 11, 200, 210, 12, 205, 9], float)
    assert led_states(b).tolist() == [False, False, True, True, False, True, False]


def test_fit_clock_recovers_offset_and_drift():
    rng = np.random.default_rng(0)
    mt, ms = _blinks(rng)
    a_true, b_true, fps = 1.0002, 3.7, 30.0
    # what the camera sees: LED state per frame, quantized to frames
    n = int((a_true * mt[-1] + b_true + 1) * fps)
    ft = np.arange(n) / fps
    t_mcu = (ft - b_true) / a_true
    k = np.searchsorted(mt, t_mcu, side="right") - 1
    state = np.where(k >= 0, ms[np.clip(k, 0, None)], 0).astype(bool)
    idx, vs = edges(state)
    fit = fit_clock(idx / fps, vs, mt, ms)
    assert fit.n_matched >= len(mt) - 2
    assert abs(fit.a - a_true) < 2e-3
    # an edge is seen at the first frame after it: up to one frame late
    assert abs(fit.b - b_true) < 1.0 / fps
    assert fit.rms_s < 1.0 / fps


def test_fit_clock_ignores_extra_video_edges():
    rng = np.random.default_rng(1)
    mt, ms = _blinks(rng, 30)
    vt, vs = mt + 2.0, ms.copy()
    vt = np.concatenate([vt, [vt[5] + 0.11]])  # a spurious flicker
    vs = np.concatenate([vs, [1 - vs[5]]])
    o = np.argsort(vt)
    fit = fit_clock(vt[o], vs[o], mt, ms)
    assert fit.n_matched == 30 and abs(fit.b - 2.0) < 1e-6


def _write_log(path, t_us, filt, base):
    lines = ["# header line from the sketch", "L,1000,1", "L,600000,0"]
    lines += [f"B,{t_us[0]}," + ",".join(str(int(v)) for v in base)]
    lines += [f"C,{t}," + ",".join(str(int(v)) for v in f) for t, f in zip(t_us, filt, strict=True)]
    lines += ["C,99", "W,700000,12345"]  # partial line + one load-cell record
    path.write_text("\n".join(lines))


def test_parse_and_touch_with_hysteresis(tmp_path):
    t_us = np.arange(20) * 5000
    base = np.full(12, 600)
    filt = np.tile(base, (20, 1))
    filt[5:12, 3] = 580  # drop 20 > 12: touch
    filt[12:15, 3] = 592  # drop 8: stays on (release below 6)
    filt[15:, 3] = 598  # drop 2: released
    _write_log(tmp_path / "log.csv", t_us, filt, base)
    log = parse_log(tmp_path / "log.csv")
    assert log.cap.shape == (20, 12) and log.led_t.shape == (2,) and log.load.shape == (1,)
    touch = electrode_touch(log)
    assert touch.shape == (20, 12)
    assert touch[5:15, 3].all() and not touch[:5, 3].any() and not touch[15:, 3].any()
    assert touch[:, [0, 1, 2, 4]].sum() == 0
    fit = ClockFit(1.0, 10.0, 3, 0.0)  # video time = mcu time + 10 s
    frames = contact_per_frame(log, touch, 10.0 + np.array([0.0, 0.03, 0.06, 0.09]), fit)
    assert frames.shape == (4, 12)
    assert frames[:, 3].tolist() == [False, True, True, False]  # 60 ms: in hysteresis
