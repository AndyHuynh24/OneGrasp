"""Streaming prediction + per-view evaluation of a TouchNet checkpoint.

Prediction is causal: the score at frame t comes from the window t-T+1..t only (except for an
offline temporal="bigru" checkpoint, which also uses later frames; see window_logits). Per-frame
encoder features are computed once per (track, finger) and then windowed, which gives exactly
the training-time windows (missing frames hold the last patch with present=0).

Usage:
    python -m touch.eval.run --ckpt runs/<run>/best.pt --split val
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from touch.data.catalog import load_catalog
from touch.data.patch_store import PatchReader, Track, load_tracks, represent
from touch.eval.metrics import (
    auroc,
    average_precision,
    causal_median,
    event_metrics,
    events,
    frame_metrics,
    hysteresis,
)

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _frame_inputs(
    tr: Track,
    f: int,
    reader: PatchReader,
    input_repr: str,
    size: int = 112,
    load_images=True,
    patch_fn=None,
):
    """Per-frame network inputs for one (track, finger) [N, C, S, S] plus present [N].

    patch_fn(tr, f, frames, patches uint8 [K, S, S, 3]) -> patches: optional perturbation
    (e.g. scripts/projection_stress.py) applied to the decoded patches before forward fill.
    """
    n = len(tr)
    raw = np.zeros((n, size, size, 3), np.uint8)
    present = np.zeros(n, bool)
    have = np.flatnonzero(tr.offset[:, f] >= 0)
    present[have] = True
    if load_images and len(have):
        src = tr.seq.source
        with ThreadPoolExecutor(8) as ex:  # cv2.imdecode releases the GIL
            imgs = list(
                ex.map(lambda i: reader.get(src, int(tr.offset[i, f]), int(tr.length[i, f])), have)
            )
        raw[have] = np.stack(imgs)
        if patch_fn is not None:
            raw[have] = patch_fn(tr, f, have, raw[have])
        # hold the last seen patch on frames without one (causal forward fill)
        last = np.maximum.accumulate(np.where(present, np.arange(n), -1))
        ok = last >= 0
        raw[ok] = raw[last[ok]]
    x = raw.astype(np.float32) / 255.0
    prev = np.concatenate([np.zeros_like(x[:1]), x[:-1]])
    return represent(x, prev, input_repr), present


@torch.no_grad()
def window_logits(model, feats: torch.Tensor, pad: torch.Tensor, T: int, batch: int = 512):
    """Per-frame logits [N] from per-frame features [N, d] of one finger.

    Frame i is scored by the window that puts it at c = model.window_target(T): frames
    i-T+1..i for causal models (exactly the training / live windows), i-c..i+T-1-c for the
    offline bigru. Frames outside the sequence use the `pad` feature [d].
    """
    if model.temporal == "none":
        return model.temporal_head(feats[None])[0]
    n, c = len(feats), model.window_target(T)
    padded = torch.cat([pad[None].expand(c, -1), feats, pad[None].expand(T - 1 - c, -1)])
    idx = torch.arange(n, device=feats.device)[:, None] + torch.arange(T, device=feats.device)
    return torch.cat(
        [model.temporal_head(padded[idx[a : a + batch]])[:, c] for a in range(0, n, batch)]
    )


@torch.no_grad()
def predict_track(
    model, tr: Track, T: int, input_repr: str, device, reader, batch: int = 512, patch_fn=None
):
    """Contact probabilities [N, 5] for one track (NaN where the frame has no patch)."""
    model.eval()
    n = len(tr)
    probs = np.full((n, 5), np.nan, np.float32)
    lm = torch.from_numpy(tr.lm_feat).to(device)
    for f in range(5):
        x, present = _frame_inputs(
            tr, f, reader, input_repr, load_images=model.use_image, patch_fn=patch_fn
        )
        pres = torch.from_numpy(present).to(device)
        fing = torch.full((n,), f, dtype=torch.long, device=device)
        feats = []
        for a in range(0, n, batch):
            b = min(n, a + batch)
            xi = torch.from_numpy(x[a:b]).to(device)[:, None]
            feats.append(model.frame_features(xi, lm[a:b, None], pres[a:b, None], fing[a:b])[:, 0])
        feats = torch.cat(feats)  # [N, d]
        # frames before the start of the sequence: zero image, no landmarks, present=0
        zero = np.zeros((1, 112, 112, 3), np.float32)
        z_img = torch.from_numpy(represent(zero, zero, input_repr))
        pad = model.frame_features(
            z_img.float().to(device)[:, None],
            torch.zeros((1, 1, lm.shape[1]), device=device),
            torch.zeros((1, 1), dtype=torch.bool, device=device),
            torch.tensor([f], device=device),
        )[0, 0]
        logits = window_logits(model, feats, pad, T, batch)
        p = torch.sigmoid(logits).float().cpu().numpy()
        probs[present, f] = p[present]
    return probs


def predict(model, tracks, T, input_repr, device, patch_fn=None) -> list[np.ndarray]:
    reader = PatchReader()
    return [
        predict_track(model, tr, T, input_repr, device, reader, patch_fn=patch_fn) for tr in tracks
    ]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    c = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(c)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def smooth(probs: np.ndarray, k: int) -> np.ndarray:
    """Causal median per finger; frames without a patch count as 0."""
    p = np.nan_to_num(probs, nan=0.0)
    return np.stack([causal_median(p[:, f], k) for f in range(p.shape[1])], 1)


def binarize(probs: np.ndarray, post: dict) -> np.ndarray:
    """Causal median (k) then hysteresis: switch on at p >= on, off at p < off (off <= on)."""
    return hysteresis(smooth(probs, post["k"]), post["on"], post["off"])


def evaluate(tracks, probs, post: dict, fps: float = 30.0) -> dict:
    """Metrics per view and overall at one post-processing setting {k, on, off}.

    Per-finger frame metrics use frames whose label is known (valid) and that have a patch.
    Frames where MediaPipe found no hand are counted (frames_no_hand) and scored as "no contact"
    in the event metrics, never silently skipped. AUROC/AP use the smoothed probabilities.
    """
    out = {}
    sm = [smooth(p, post["k"]) for p in probs]
    bins = [hysteresis(ps, post["on"], post["off"]) for ps in sm]
    for view in sorted({tr.view for tr in tracks}) + ["all"]:
        sel = [i for i, tr in enumerate(tracks) if view in ("all", tr.view)]
        s_all, y_all, b_all, s_idx, y_idx, b_idx = [], [], [], [], [], []
        hand_pred, hand_gt = [], []
        ev = {tol: [0, 0, 0] for tol in (0, 1, 2)}  # n_gt, n_hit, n_pred
        timing, timing_up = [], []
        hover_events, hover_minutes = 0, 0.0
        dropped, labelled = 0, 0
        for i in sel:
            tr, ps = tracks[i], sm[i]
            pred_bin = bins[i] & tr.present[:, None]
            ok = tr.valid & tr.present
            dropped += int((tr.valid & ~tr.present).sum())
            labelled += int(tr.valid.sum())
            s_all.append(ps[ok].reshape(-1))
            b_all.append(pred_bin[ok].reshape(-1))
            y_all.append(tr.contact[ok].reshape(-1))
            s_idx.append(ps[ok, 1])
            b_idx.append(pred_bin[ok, 1])
            y_idx.append(tr.contact[ok, 1])
            hand_pred.append(pred_bin[ok].any(1))
            hand_gt.append(tr.contact[ok].any(1))
            for a, b in _runs(tr.valid):
                for f in range(5):
                    g, q = tr.contact[a:b, f], pred_bin[a:b, f]
                    for tol in ev:
                        m = event_metrics(q, g, tol=tol, fps=fps)
                        ev[tol][0] += m["n_gt"]
                        ev[tol][1] += m["n_hit"]
                        ev[tol][2] += m["n_pred"]
                    timing.extend(event_metrics(q, g, tol=2, fps=fps)["timing_ms"].tolist())
                    timing_up.extend(
                        event_metrics(q, g, tol=2, kind="up", fps=fps)["timing_ms"].tolist()
                    )
            if tr.seq.meta.get("variant") == "no-contact":
                for a, b in _runs(tr.valid):
                    if tr.contact[a:b].any():
                        continue
                    hover_minutes += (b - a) / fps / 60.0
                    for f in range(5):
                        hover_events += len(events(pred_bin[a:b, f]))
        s, y, bb = np.concatenate(s_all), np.concatenate(y_all), np.concatenate(b_all)
        si, yi, bi = np.concatenate(s_idx), np.concatenate(y_idx), np.concatenate(b_idx)
        fm = frame_metrics(bb, y)
        fi = frame_metrics(bi, yi)
        hm = frame_metrics(np.concatenate(hand_pred), np.concatenate(hand_gt))
        r = {
            "frames_labelled": labelled,
            "frames_no_hand": dropped,
            "finger_acc": fm["accuracy"],
            "finger_bal_acc": (fm["tpr"] + 1 - fm["fpr"]) / 2,
            "finger_tpr": fm["tpr"],
            "finger_fpr": fm["fpr"],
            "finger_auroc": auroc(s, y),
            "finger_ap": average_precision(s, y),
            "index_acc": fi["accuracy"],
            "index_bal_acc": (fi["tpr"] + 1 - fi["fpr"]) / 2,
            "index_auroc": auroc(si, yi),
            "hand_acc": hm["accuracy"],
            "hand_tpr": hm["tpr"],
            "hand_fpr": hm["fpr"],
            "hover_fp_per_min": hover_events / hover_minutes if hover_minutes else float("nan"),
            "timing_down_mae_ms": float(np.mean(np.abs(timing))) if timing else float("nan"),
            "timing_down_mean_ms": float(np.mean(timing)) if timing else float("nan"),
            "timing_up_mae_ms": float(np.mean(np.abs(timing_up))) if timing_up else float("nan"),
            "timing_up_mean_ms": float(np.mean(timing_up)) if timing_up else float("nan"),
        }
        for tol, (n_gt, n_hit, n_pred) in ev.items():
            name = "exact" if tol == 0 else f"pm{tol}"
            r[f"click_recall_{name}"] = n_hit / n_gt if n_gt else float("nan")
            r[f"click_precision_{name}"] = n_hit / n_pred if n_pred else float("nan")
            r[f"click_f1_{name}"] = 2 * n_hit / (n_gt + n_pred) if (n_gt + n_pred) else float("nan")
        r["n_touchdowns"] = ev[0][0]
        out[view] = r
    return out


def _click_f1(tracks, bins, tol: int = 2) -> float:
    n_gt = n_hit = n_pred = 0
    for tr, b in zip(tracks, bins, strict=True):
        pb = b & tr.present[:, None]
        for a, e in _runs(tr.valid):
            for f in range(5):
                m = event_metrics(pb[a:e, f], tr.contact[a:e, f], tol=tol)
                n_gt, n_hit, n_pred = n_gt + m["n_gt"], n_hit + m["n_hit"], n_pred + m["n_pred"]
    return 2 * n_hit / (n_gt + n_pred) if (n_gt + n_pred) else 0.0


def tune_postprocessing(
    tracks, probs, objective: str = "event", ks=(1, 3, 5)
) -> tuple[dict, float]:
    """Pick post-processing {k, on, off} on THIS (validation) set. Returns (post, score).

    objective="frame": single threshold (on == off), maximizes per-finger balanced accuracy.
    objective="event": causal median + hysteresis, maximizes touch-down F1 at +-2 frames.
    """
    best: tuple[dict, float] = ({"k": 1, "on": 0.5, "off": 0.5}, -1.0)
    for k in ks:
        sm = [smooth(p, k) for p in probs]
        if objective == "frame":
            s = np.concatenate(
                [ps[tr.valid & tr.present].reshape(-1) for tr, ps in zip(tracks, sm, strict=True)]
            )
            y = np.concatenate([tr.contact[tr.valid & tr.present].reshape(-1) for tr in tracks])
            for tau in np.linspace(0.05, 0.95, 37):
                m = frame_metrics(s > tau, y)
                bal = (m["tpr"] + 1 - m["fpr"]) / 2
                if bal > best[1]:
                    best = ({"k": k, "on": float(tau), "off": float(tau)}, float(bal))
        else:
            for on in np.linspace(0.2, 0.95, 16):
                for gap in (0.0, 0.1, 0.2, 0.3, 0.45):
                    off = on - gap
                    if off < 0.05:
                        continue
                    f1 = _click_f1(tracks, [hysteresis(ps, on, off) for ps in sm])
                    if f1 > best[1]:
                        best = ({"k": k, "on": float(on), "off": float(off)}, float(f1))
    return best


def main() -> None:
    from touch.train import build_model, load_config, pick_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = load_config(ck["config"])
    device = pick_device()
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    seqs = load_catalog(source="egopressure", split=args.split)
    tracks = load_tracks(seqs, cfg["views"], lm_feats=cfg["lm_feats"])
    probs = predict(model, tracks, cfg["T"], cfg["input_repr"], device)
    if args.split == "val":
        post, _ = tune_postprocessing(tracks, probs, "event")
        post_frame, _ = tune_postprocessing(tracks, probs, "frame")
    else:
        # test: only settings frozen on validation, never re-tuned
        assert "post" in ck and "post_frame" in ck, "evaluate on val first"
        post, post_frame = ck["post"], ck["post_frame"]
    rec = {
        "epoch": ck.get("epoch"),
        "split": args.split,
        "post": post,
        "post_frame": post_frame,
        "val" if args.split == "val" else "test": evaluate(tracks, probs, post),
        "frame_op": evaluate(tracks, probs, post_frame),
    }
    if args.split == "val":
        ck["post"], ck["post_frame"] = post, post_frame
        torch.save(ck, args.ckpt)
    stem = Path(args.ckpt).stem
    suffix = "" if stem == "best" else f"_{stem}"  # leaderboard reads the best.pt result only
    out = Path(args.ckpt).parent / f"eval_{args.split}{suffix}.json"
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
