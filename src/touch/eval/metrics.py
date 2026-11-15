"""Contact metrics: frame accuracy/TPR/FPR, contact IoU, click events and timing error.

Conventions (docs/DECISIONS.md, P2 metrics entry):
- Contact = pressure strictly > threshold, as in PV++ prediction/evaluator.py (CONTACT_THRESH 1.0).
- Contact IoU is summed over the whole dataset, then divided once (PV++ ContactIOU), so frames with
  no GT and no predicted contact contribute nothing.
- An event is a 0->1 step (touch-down) or 1->0 step (touch-up). A sequence that starts in contact
  has no touch-down at frame 0 (same as touch.data.pressure.count_transitions).
- Each GT event matches at most one predicted event: the nearest unused one within +-tol frames.
  tol=0 is EgoTouch's "exact frame" click-event accuracy.
"""

from __future__ import annotations

import numpy as np

from touch.data.pressure import CONTACT_KPA

FPS = 30.0


def _safe_div(a: float, b: float) -> float:
    return float(a) / b if b > 0 else float("nan")


def frame_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Per-frame contact metrics for boolean arrays [N]. TPR/FPR are NaN when undefined."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    assert pred.shape == gt.shape and pred.ndim == 1, (pred.shape, gt.shape)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    return {
        "accuracy": _safe_div((pred == gt).sum(), gt.size),
        "tpr": _safe_div(tp, gt.sum()),
        "fpr": _safe_div(fp, (~gt).sum()),
    }


class ContactIoU:
    """Pixel contact IoU accumulated over many batches of pressure maps [N, H, W] (kPa)."""

    def __init__(self, thresh: float = CONTACT_KPA):
        self.thresh = thresh
        self.inter = 0
        self.union = 0

    def update(self, pred_kpa: np.ndarray, gt_kpa: np.ndarray) -> None:
        assert pred_kpa.shape == gt_kpa.shape, (pred_kpa.shape, gt_kpa.shape)
        p = np.asarray(pred_kpa) > self.thresh
        g = np.asarray(gt_kpa) > self.thresh
        self.inter += int((p & g).sum())
        self.union += int((p | g).sum())

    def compute(self) -> float:
        return _safe_div(self.inter, self.union)


def contact_iou(pred_kpa: np.ndarray, gt_kpa: np.ndarray, thresh: float = CONTACT_KPA) -> float:
    """One-shot ContactIoU for a single stack [N, H, W]."""
    m = ContactIoU(thresh)
    m.update(pred_kpa, gt_kpa)
    return m.compute()


def events(contact: np.ndarray, kind: str = "down") -> np.ndarray:
    """Frame indices of touch-downs (0->1) or touch-ups (1->0) in a boolean sequence [N]."""
    d = np.diff(np.asarray(contact, dtype=np.int8))
    step = {"down": 1, "up": -1}[kind]
    return np.flatnonzero(d == step) + 1


def match_events(pred_ev: np.ndarray, gt_ev: np.ndarray, tol: int) -> list[tuple[int, int]]:
    """Pairs (gt_frame, pred_frame): each GT event takes the nearest unused prediction within tol.

    GT events are visited in time order; ties go to the earlier prediction.
    """
    pairs = []
    if len(pred_ev) == 0:
        return pairs
    used = np.zeros(len(pred_ev), dtype=bool)
    for g in gt_ev:
        dist = np.abs(np.asarray(pred_ev) - g).astype(float)
        dist[used] = np.inf
        j = int(np.argmin(dist))
        if dist[j] <= tol:
            used[j] = True
            pairs.append((int(g), int(pred_ev[j])))
    return pairs


def event_metrics(
    pred: np.ndarray, gt: np.ndarray, tol: int = 0, kind: str = "down", fps: float = FPS
) -> dict:
    """Click-event metrics for one sequence of boolean contact [N].

    Returns n_gt, n_pred, n_hit, hit_rate (n_hit / n_gt, NaN if no GT events),
    n_false (unmatched predicted events) and timing_ms (pred - gt per hit in ms; negative = early).
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    assert pred.shape == gt.shape and pred.ndim == 1, (pred.shape, gt.shape)
    pe, ge = events(pred, kind), events(gt, kind)
    pairs = match_events(pe, ge, tol)
    timing = np.array([(p - g) * 1000.0 / fps for g, p in pairs], dtype=np.float64)
    return {
        "n_gt": len(ge),
        "n_pred": len(pe),
        "n_hit": len(pairs),
        "hit_rate": _safe_div(len(pairs), len(ge)),
        "n_false": len(pe) - len(pairs),
        "timing_ms": timing,
    }


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney U) form; ties get average ranks."""
    s = np.asarray(scores, np.float64)
    y = np.asarray(labels, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), np.float64)
    sorted_s = s[order]
    # average ranks over ties
    _, first, counts = np.unique(sorted_s, return_index=True, return_counts=True)
    avg = first + (counts - 1) / 2.0 + 1.0
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """AP = sum over positives of precision at that positive's rank (scores sorted descending)."""
    y = np.asarray(labels, bool)[np.argsort(-np.asarray(scores, np.float64), kind="mergesort")]
    if not y.any():
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float(precision[y].mean())


def causal_median(x: np.ndarray, k: int) -> np.ndarray:
    """Median of the last k values (only past frames; the first frames use what exists)."""
    x = np.asarray(x, np.float64)
    if k <= 1:
        return x.copy()
    out = np.empty_like(x)
    for i in range(min(k - 1, len(x))):  # first frames: median of what exists so far
        out[i] = np.median(x[: i + 1])
    if len(x) >= k:
        out[k - 1 :] = np.median(np.lib.stride_tricks.sliding_window_view(x, k), axis=1)
    return out


def hysteresis(p: np.ndarray, on: float, off: float) -> np.ndarray:
    """Causal two-threshold switch, per column of p [N] or [N, F]: becomes True when p >= on,
    stays True until p < off. Starts False. Requires off <= on."""
    assert off <= on, (on, off)
    p = np.asarray(p, np.float64)
    n = len(p)
    idx = np.arange(n).reshape((n,) + (1,) * (p.ndim - 1))
    last_on = np.maximum.accumulate(np.where(p >= on, idx, -1), axis=0)
    last_off = np.maximum.accumulate(np.where(p < off, idx, -1), axis=0)
    return last_on > last_off


# ---------------------------------------------------------------------------------------------
# Interval timing (hand-object contact): onset and release of whole contact intervals.
# Scoring as described by EgoLoc / HOCALT (arXiv 2606.10743): predicted and GT intervals are
# matched one-to-one by interval IoU; per GT boundary, success within +-k frames (SR@k) and the
# mean absolute error. An unmatched GT interval is a failure for SR. Their exact matching
# details (IoU threshold, tie-breaking, MAE over unmatched) were not verified from their code:
# here any IoU > 0 can match, greedily by highest IoU, and MAE is over matched boundaries only
# (`matched` reports how many). Report OneTouch's own event metrics alongside this.
# ---------------------------------------------------------------------------------------------


def intervals(contact: np.ndarray) -> np.ndarray:
    """Contact intervals [K, 2] (start, end) of a boolean sequence [N]; end is exclusive, i.e. the
    release frame (first frame without contact). An interval touching the clip start / end keeps
    the clip boundary as its start / end."""
    c = np.concatenate([[0], np.asarray(contact, dtype=np.int8), [0]])
    d = np.diff(c)
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], 1).astype(np.int64)


def match_intervals(pred_iv: np.ndarray, gt_iv: np.ndarray) -> list[tuple[int, int]]:
    """One-to-one (gt_index, pred_index) pairs with IoU > 0, greedily by highest IoU."""
    pred_iv, gt_iv = np.asarray(pred_iv).reshape(-1, 2), np.asarray(gt_iv).reshape(-1, 2)
    if len(pred_iv) == 0 or len(gt_iv) == 0:
        return []
    inter = np.clip(
        np.minimum(gt_iv[:, None, 1], pred_iv[None, :, 1])
        - np.maximum(gt_iv[:, None, 0], pred_iv[None, :, 0]),
        0,
        None,
    )
    union = (gt_iv[:, None, 1] - gt_iv[:, None, 0]) + (pred_iv[None, :, 1] - pred_iv[None, :, 0])
    iou = inter / np.maximum(union - inter, 1)
    pairs, used_g, used_p = [], set(), set()
    for flat in np.argsort(-iou, axis=None, kind="stable"):
        g, p = np.unravel_index(flat, iou.shape)
        if iou[g, p] <= 0:
            break
        if g in used_g or p in used_p:
            continue
        used_g.add(int(g))
        used_p.add(int(p))
        pairs.append((int(g), int(p)))
    return sorted(pairs)


def interval_timing(
    pred: np.ndarray, gt: np.ndarray, tols: tuple[int, ...] = (3, 5, 10), fps: float = FPS
) -> dict:
    """Onset / release timing of contact intervals for one boolean sequence [N] (one finger, or
    any-finger contact, or a grasp state). Returns n_gt, n_pred, matched and, for "onset" and
    "release": sr@k (NaN without GT intervals), mae_frames and mae_ms over matched boundaries
    (NaN if none), errors_frames (pred - gt; negative = early)."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    assert pred.shape == gt.shape and pred.ndim == 1, (pred.shape, gt.shape)
    pi, gi = intervals(pred), intervals(gt)
    pairs = match_intervals(pi, gi)
    out: dict = {"n_gt": len(gi), "n_pred": len(pi), "matched": len(pairs)}
    for name, col in (("onset", 0), ("release", 1)):
        err = np.array([pi[p, col] - gi[g, col] for g, p in pairs], dtype=np.int64)
        res = {f"sr@{k}": _safe_div(int((np.abs(err) <= k).sum()), len(gi)) for k in tols}
        mae = float(np.abs(err).mean()) if len(err) else float("nan")
        res.update(mae_frames=mae, mae_ms=mae * 1000.0 / fps, errors_frames=err)
        out[name] = res
    return out
