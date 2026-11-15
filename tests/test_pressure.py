import numpy as np
import pytest

from touch.data.pressure import (
    CELL_AREA_MM2,
    clean,
    contact_segments,
    count_transitions,
    finger_pressure,
    frame_stats,
    pv_counts_to_kpa,
)


def test_pv_counts_to_kpa_matches_pressurevision_formula():
    # 1736 counts = 1 "newton" over one 1.25 mm cell = 640 kPa (PressureVision recording/util.py)
    assert pv_counts_to_kpa(np.array([1736.0]))[0] == pytest.approx(640.0)
    assert pv_counts_to_kpa(np.zeros((105, 185))).shape == (105, 185)


def test_clean_zeroes_below_floor_without_mutating():
    x = np.array([0.2, 0.5, 3.0], np.float32)
    assert clean(x).tolist() == [0.0, 0.5, 3.0]
    assert x[0] == pytest.approx(0.2)


def test_frame_stats_shapes_and_contact():
    stack = np.zeros((3, 105, 185), np.float32)
    stack[1, 10:12, 20:23] = 2.0  # 6 cells above 1 kPa
    stack[2, 0, 0] = 0.9  # above floor, below contact
    s = frame_stats(stack)
    assert s["max_kpa"].shape == (3,) and s["centroid_rc"].shape == (3, 2)
    assert s["contact"].tolist() == [False, True, False]
    assert s["area_mm2"][1] == pytest.approx(6 * CELL_AREA_MM2)
    assert s["centroid_rc"][1] == pytest.approx([10.5, 21.0])
    assert np.isnan(s["centroid_rc"][0]).all()


def test_frame_stats_rejects_wrong_grid():
    with pytest.raises(AssertionError):
        frame_stats(np.zeros((2, 185, 105)))


def test_transitions_and_segments():
    c = np.array([0, 1, 1, 0, 0, 1, 0, 1], bool)
    assert count_transitions(c) == (3, 2)
    assert contact_segments(c) == [(1, 3), (5, 6), (7, 8)]
    assert contact_segments(np.zeros(4, bool)) == []


def _hand(tip_rc, dip_rc):
    tips = np.array([[90, 10], tip_rc, [10, 150], [10, 165], [10, 180]], np.float32)
    dips = tips + np.array([10, 0], np.float32)
    dips[1] = dip_rc
    return tips, dips


def test_finger_pressure_assigns_blob_to_nearest_phalanx():
    kpa = np.zeros((105, 185), np.float32)
    kpa[52:57, 88:93] = 7.0  # under the index distal phalanx
    kpa[100, 100] = 4.0  # far from every finger -> "other"
    tips, dips = _hand([50, 90], [62, 90])
    r = finger_pressure(kpa, tips, dips)
    assert r["max_kpa"].shape == (5,)
    assert r["max_kpa"].tolist() == [0, 7, 0, 0, 0]
    assert r["area_mm2"][1] == pytest.approx(25 * CELL_AREA_MM2)
    assert r["other_max_kpa"] == pytest.approx(4.0)


def test_finger_pressure_empty_map():
    tips, dips = _hand([50, 90], [62, 90])
    r = finger_pressure(np.zeros((105, 185), np.float32), tips, dips)
    assert r["max_kpa"].sum() == 0 and r["other_max_kpa"] == 0


def test_finger_pressure_ineligible_finger_gets_nothing():
    kpa = np.zeros((105, 185), np.float32)
    kpa[52:57, 88:93] = 7.0
    tips, dips = _hand([50, 90], [62, 90])
    tips[0], dips[0] = [54, 95], [64, 100]  # thumb projected right next to the index blob
    gated = finger_pressure(kpa, tips, dips, eligible=np.array([False, True, True, True, True]))
    assert gated["max_kpa"][0] == 0 and gated["max_kpa"][1] == 7
