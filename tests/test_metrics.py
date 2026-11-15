import math

import numpy as np
import pytest

from touch.eval.metrics import (
    ContactIoU,
    contact_iou,
    event_metrics,
    events,
    frame_metrics,
    match_events,
)

# Worked example from the P2 walkthrough
GT = np.array([0, 0, 1, 1, 1, 0], bool)
PRED = np.array([0, 1, 1, 1, 0, 0], bool)


def test_frame_metrics_worked_example():
    m = frame_metrics(PRED, GT)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["tpr"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 3)


def test_frame_metrics_perfect_and_all_zero():
    assert frame_metrics(GT, GT) == {"accuracy": 1.0, "tpr": 1.0, "fpr": 0.0}
    m = frame_metrics(np.zeros_like(GT), GT)
    assert m["accuracy"] == pytest.approx(3 / 6)
    assert m["tpr"] == 0.0 and m["fpr"] == 0.0


def test_frame_metrics_always_contact_on_20pct_sequence():
    gt = np.zeros(10, bool)
    gt[:2] = True
    m = frame_metrics(np.ones(10, bool), gt)
    assert m["accuracy"] == pytest.approx(0.2)
    assert m["tpr"] == 1.0 and m["fpr"] == 1.0


def test_frame_metrics_undefined_rates_are_nan():
    m = frame_metrics(np.zeros(4, bool), np.zeros(4, bool))
    assert m["accuracy"] == 1.0 and math.isnan(m["tpr"]) and m["fpr"] == 0.0


def test_contact_iou_three_vs_three_cells():
    gt = np.zeros((1, 2, 3), np.float32)
    pred = np.zeros((1, 2, 3), np.float32)
    gt[0, 0, :3] = 2.0  # A, B, C
    pred[0, 0, 1:3] = 2.0  # B, C
    pred[0, 1, 0] = 2.0  # D
    assert contact_iou(pred, gt) == pytest.approx(0.5)
    assert contact_iou(gt, gt) == 1.0
    assert contact_iou(np.zeros_like(gt), gt) == 0.0


def test_contact_iou_threshold_is_strict_like_pv2():
    x = np.full((1, 1, 1), 1.0, np.float32)
    assert math.isnan(contact_iou(x, x))  # exactly 1.0 kPa is not contact
    y = np.full((1, 1, 1), 1.01, np.float32)
    assert contact_iou(y, y) == 1.0


def test_contact_iou_sums_over_dataset_not_mean_of_frames():
    m = ContactIoU()
    # frame 1: IoU 1/1; frame 2: IoU 1/3. Mean of frames = 0.667, dataset sum = 2/4 = 0.5
    a = np.array([[[2.0, 0, 0]]], np.float32)
    m.update(a, a)
    m.update(np.array([[[2.0, 2.0, 0]]], np.float32), np.array([[[2.0, 0, 2.0]]], np.float32))
    assert m.compute() == pytest.approx(0.5)


def test_events_down_and_up():
    x = np.array([1, 1, 0, 1, 1, 0, 0, 1], bool)
    assert events(x, "down").tolist() == [3, 7]  # starting in contact is not a touch-down
    assert events(x, "up").tolist() == [2, 5]
    assert events(np.zeros(5, bool)).tolist() == []


def test_event_metrics_worked_example_tolerance_and_timing():
    exact = event_metrics(PRED, GT, tol=0)
    assert (exact["n_gt"], exact["n_pred"], exact["n_hit"], exact["n_false"]) == (1, 1, 0, 1)
    assert exact["hit_rate"] == 0.0
    one = event_metrics(PRED, GT, tol=1)
    assert one["hit_rate"] == 1.0 and one["n_false"] == 0
    assert one["timing_ms"].tolist() == pytest.approx([-1000 / 30])
    up = event_metrics(PRED, GT, tol=1, kind="up")
    assert up["timing_ms"].tolist() == pytest.approx([-1000 / 30])  # pred lifts at 4, GT at 5


def test_event_metrics_perfect_and_all_zero():
    p = event_metrics(GT, GT, tol=0)
    assert p["hit_rate"] == 1.0 and p["n_false"] == 0 and p["timing_ms"].tolist() == [0.0]
    z = event_metrics(np.zeros_like(GT), GT, tol=2)
    assert z["n_hit"] == 0 and z["hit_rate"] == 0.0 and z["timing_ms"].shape == (0,)
    none = event_metrics(np.zeros(4, bool), np.zeros(4, bool))
    assert none["n_gt"] == 0 and math.isnan(none["hit_rate"])


def test_one_prediction_cannot_match_two_gt_events():
    assert match_events(np.array([5]), np.array([4, 6]), tol=1) == [(4, 5)]
    # nearest wins, not first in list
    assert match_events(np.array([2, 9]), np.array([8]), tol=10) == [(8, 9)]


def test_auroc_and_ap():
    from touch.eval.metrics import auroc, average_precision

    y = np.array([0, 0, 1, 1], bool)
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert auroc(np.full(4, 0.5), y) == 0.5  # all ties -> chance
    assert auroc(np.array([0.1, 0.6, 0.4, 0.9]), y) == pytest.approx(0.75)
    assert average_precision(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    # ranking 1,0,1,0 -> precision at positives 1/1 and 2/3
    assert average_precision(np.array([0.9, 0.8, 0.7, 0.1]), np.array([1, 0, 1, 0], bool)) == (
        pytest.approx((1 + 2 / 3) / 2)
    )


def test_causal_median_uses_only_past():
    from touch.eval.metrics import causal_median

    x = np.array([0, 0, 1, 0, 0, 1, 1, 1], float)
    assert causal_median(x, 3).tolist() == [0, 0, 0, 0, 0, 0, 1, 1]
    y = x.copy()
    y[-1] = 0  # changing the last frame must not change earlier outputs
    assert causal_median(y, 3)[:-1].tolist() == causal_median(x, 3)[:-1].tolist()


def test_hysteresis():
    from touch.eval.metrics import hysteresis

    p = np.array([0.1, 0.6, 0.8, 0.6, 0.4, 0.2, 0.6, 0.9])
    # on at >= 0.7, off below 0.5: 0.6 alone never switches on; dips to 0.6 do not switch off
    assert hysteresis(p, 0.7, 0.5).astype(int).tolist() == [0, 0, 1, 1, 0, 0, 0, 1]
    assert hysteresis(p, 0.5, 0.5).astype(int).tolist() == [0, 1, 1, 1, 0, 0, 1, 1]
    two = hysteresis(np.stack([p, p], 1), 0.7, 0.5)
    assert two.shape == (8, 2) and (two[:, 0] == two[:, 1]).all()
