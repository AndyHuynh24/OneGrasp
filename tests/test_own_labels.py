import numpy as np

from touch.data.own_labels import frame_labels, intervals

FPS = 30.0
T = 100.0 + np.arange(90) / FPS  # 3 s of frame read times


def ev(t, key, down):
    return {"t": t, "key": key, "down": down}


def test_touch_press_labels_frames_inside_it():
    e = [ev(T[30], "h", True), ev(T[45], "h", False)]
    lab = frame_labels(T, e, "touch")
    assert lab["label"].shape == lab["valid"].shape == lab["edge_dist"].shape == (90,)
    assert lab["label"].dtype == np.int8
    assert np.flatnonzero(lab["label"]).tolist() == list(range(30, 45))
    assert lab["valid"].all() and lab["edge_dist"][30] == 0 and lab["edge_dist"][37] == 7


def test_latency_moves_labels_later_and_edge_shift_moves_one_edge():
    e = [ev(T[30], "h", True), ev(T[45], "h", False)]
    lab = frame_labels(T, e, "touch", latency_s=3 / FPS)
    assert np.flatnonzero(lab["label"]).tolist() == list(range(33, 48))
    lab = frame_labels(T, e, "touch", shifts={"1": -2 / FPS})  # release 2 frames earlier
    assert np.flatnonzero(lab["label"]).tolist() == list(range(30, 43))


def test_grasp_hold_wins_over_touch_and_unreleased_key_runs_to_the_end():
    e = [ev(T[10], "f", True), ev(T[20], "h", True), ev(T[30], "f", False),
         ev(T[60], "h", False), ev(T[80], "f", True)]  # fmt: skip
    lab = frame_labels(T, e, "grasp")["label"]
    assert (lab[10:20] == 1).all() and (lab[20:60] == 2).all() and (lab[60:80] == 0).all()
    assert (lab[80:] == 1).all()
    assert [r["i1"] for r in intervals(e, T[-1])] == [2, 3, None]


def test_bad_mark_and_dropped_press_make_frames_invalid():
    e = [ev(T[10], "h", True), ev(T[20], "h", False), ev(T[70], "bad", True)]
    lab = frame_labels(T, e, "touch", dropped=[0])
    assert not lab["valid"][10:20].any() and lab["label"][10:20].sum() == 0
    assert not lab["valid"][70] and lab["valid"][50]  # +- 0.5 s around the mark
    assert lab["valid"][:10].all()


def test_unmapped_keys_and_repeated_downs_are_ignored():
    e = [ev(T[10], "f", True), ev(T[20], "f", False), ev(T[30], "h", True),
         ev(T[35], "h", True), ev(T[40], "h", False)]  # fmt: skip
    lab = frame_labels(T, e, "touch")["label"]  # F = touch in the touch task
    assert np.flatnonzero(lab).tolist() == list(range(10, 20)) + list(range(30, 40))
    e2 = [ev(T[10], "x", True), ev(T[20], "x", False)]  # unknown key
    assert frame_labels(T, e2, "touch")["label"].sum() == 0


def test_old_space_key_sessions_still_label():
    e = [ev(T[30], "space", True), ev(T[45], "space", False)]
    assert np.flatnonzero(frame_labels(T, e, "grasp")["label"] == 2).tolist() == list(range(30, 45))


def test_release_pad_and_invalid_ranges():
    e = [ev(T[30], "h", True), ev(T[45], "h", False)]
    lab = frame_labels(T, e, "grasp", release_pad_s={"h": 6 / FPS}, invalid=[[T[70], T[75]]])
    assert (lab["label"][30:45] == 2).all()
    assert not lab["valid"][45:51].any() and lab["valid"][51] and lab["valid"][44]
    assert not lab["valid"][70:75].any() and lab["valid"][75]


def test_finger_switches_give_per_frame_finger():
    sw = {"t": T[40], "key": "finger", "down": True, "finger": "middle"}
    e = [
        ev(T[10], "h", True),
        ev(T[20], "h", False),
        sw,
        ev(T[50], "h", True),
        ev(T[60], "h", False),
    ]
    lab = frame_labels(T, e, "touch", finger="index")
    assert lab["finger"].shape == (90,) and lab["finger"].dtype == np.int8
    assert (lab["finger"][:40] == 1).all() and (lab["finger"][40:] == 2).all()
    assert np.flatnonzero(lab["label"]).tolist() == list(range(10, 20)) + list(range(50, 60))
    assert (frame_labels(T, e, "grasp")["finger"] == -1).all()


def test_touch_task_h_is_hold():
    e = [ev(T[10], "f", True), ev(T[20], "f", False), ev(T[30], "h", True), ev(T[50], "h", False)]
    lab = frame_labels(T, e, "touch")["label"]
    assert (lab[10:20] == 1).all() and (lab[30:50] == 2).all() and lab[20:30].sum() == 0
