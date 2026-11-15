"""Grab decision (gripper closed / open) for the live demo: rule baselines vs a learned causal
fusion of the grasp model's probabilities + 3D-mesh geometry (scripts/grab_features.py).

Fit: DexYCB val + OakInk2. Select (fusion variant, gate thresholds): HO-Cap val (a subject and
cameras the fusion never saw). Final: HO-Cap test, once, with --test (no re-tuning).
Target: GT "any2" hold (thumb + another part on an object that moved), i.e. the gripper should be
closed while the person holds the object, open otherwise. Scored per frame (F1) and per hold
episode (>= 15 frames): grabbed, close delay, false release (opened > 5 frames before the end),
release delay, and false grabs per minute (closed runs touching no GT hold).

    python scripts/grab_fusion.py --run runs/g6_grasp_table_20261112-101602 [--variant cur]
writes runs/grab_fusion/<run>/fusion_<variant>.json (+ fusion_<variant>_m<k>.npz: the chosen
ensemble's weights, numpy-only)
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch.interact.grab_gate import Ensemble, Fusion, Hysteresis  # noqa: E402

OUT = Path("runs/grab_fusion")
FPS = 30.0


def load(run: str, ds: str, variant: str) -> dict:
    z = np.load(OUT / run / f"{ds}_{variant}.npz")
    return {k: z[k] for k in z.files}


def tracks(d: dict):
    for t in np.unique(d["tid"]):
        yield np.flatnonzero(d["tid"] == t)


# ------------------------------------------------------------------ gates (causal)
def gate_rule(state: np.ndarray, present: np.ndarray, rel_k=20, fast_p=0.99, fast_k=10, on=2):
    """The live demo's current rule (touch.live_robot): close after `on` HOLD frames, open after
    rel_k NONE frames or P(NONE) >= fast_p for fast_k frames; no-hand frames hold."""
    out = np.zeros(len(state), bool)
    close, run, sure = False, 0, 0
    for t in range(len(state)):
        if present[t] > 0:
            st = int(np.argmax(state[t]))
            sure = sure + 1 if state[t, 0] >= fast_p else 0
            if not close:
                run = run + 1 if st == 2 else 0
                if run >= on:
                    close, run = True, 0
            else:
                run = run + 1 if st == 0 else 0
                if run >= rel_k or sure >= fast_k:
                    close, run = False, 0
        out[t] = close
    return out


def gate_score(s: np.ndarray, present: np.ndarray, p_none_t=None, **kw) -> np.ndarray:
    """touch.interact.grab_gate.Hysteresis over a per-frame grab score (the live gate).
    p_none_t: the grasp model's per-frame P(NONE) (for the fast release)."""
    h = Hysteresis(**kw)
    pn = np.full(len(s), np.nan) if p_none_t is None else p_none_t
    return np.array([h.step(float(v), bool(p > 0), float(q))
                     for v, p, q in zip(s, present, pn, strict=True)])  # fmt: skip


# ------------------------------------------------------------------ scoring
def score(d: dict, closed: np.ndarray, label: str = "y_any2") -> dict:
    y = d[label]
    m = y >= 0
    tp = (closed & (y == 2) & m).sum()
    fp = (closed & (y != 2) & m).sum()
    fn = (~closed & (y == 2) & m).sum()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    n_ep = grabbed = false_rel = 0
    on_d, off_d = [], []
    false_grab, minutes = 0, 0.0
    for idx in tracks(d):
        c, g = closed[idx], y[idx] == 2
        minutes += len(idx) / FPS / 60
        e = np.flatnonzero(np.diff(np.r_[0, g.astype(int), 0]))
        for a, b in zip(e[::2], e[1::2], strict=True):
            if b - a < 15:
                continue
            n_ep += 1
            ins = c[a:b]
            if ins.any():
                grabbed += 1
                on_d.append(int(np.argmax(ins)))
                fr = np.flatnonzero(ins[:-1] & ~ins[1:])
                if len(fr) and fr[0] < (b - a) - 5:
                    false_rel += 1
                if c[b - 1]:
                    after = np.flatnonzero(~c[b:])
                    off_d.append(int(after[0]) if len(after) else len(c) - b)
        gd = np.convolve(g, np.ones(21), "same") > 0  # GT hold +- 10 frames
        ce = np.flatnonzero(np.diff(np.r_[0, c.astype(int), 0]))
        for a, b in zip(ce[::2], ce[1::2], strict=True):
            false_grab += int(not gd[a:b].any())
    return {"f1": round(float(f1), 4), "holds": n_ep, "grabbed": round(grabbed / max(n_ep, 1), 3),
            "false_release": round(false_rel / max(grabbed, 1), 3),
            "close_delay": float(np.median(on_d)) if on_d else None,
            "release_delay": float(np.median(off_d)) if off_d else None,
            "false_grab_per_min": round(false_grab / max(minutes, 1e-9), 2)}  # fmt: skip


def run_gate(d, s, **kw):
    out = np.zeros(len(s), bool)
    for idx in tracks(d):
        out[idx] = gate_score(s[idx], d["present"][idx], d["state"][idx, 0], **kw)
    return out


def run_rule(d, **kw):
    out = np.zeros(len(d["state"]), bool)
    for idx in tracks(d):
        out[idx] = gate_rule(d["state"][idx], d["present"][idx], **kw)
    return out


GATE_GRID = {"on": [0.5, 0.6, 0.7, 0.8], "off": [0.1, 0.2, 0.3], "k_on": [2, 4],
             "k_off": [5, 10, 20, 30]}  # fmt: skip
FR_CAP, FG_CAP = 0.01, 0.1  # selection: best F1 with <= 1 % false releases, <= 0.1 false grabs/min


def tune_gate(d, s) -> tuple[dict, dict]:
    rows = []
    for vals in itertools.product(*GATE_GRID.values()):
        kw = dict(zip(GATE_GRID, vals, strict=True))
        rows.append((kw, score(d, run_gate(d, s, **kw))))
    ok = [
        x for x in rows if x[1]["false_release"] <= FR_CAP and x[1]["false_grab_per_min"] <= FG_CAP
    ]
    return max(ok or rows, key=lambda x: x[1]["f1"])


def fit(Xtr, ytr, seed=0, hidden=64, epochs=12):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        mean = np.nan_to_num(np.nanmean(Xtr, 0))
        std = np.nan_to_num(np.nanstd(Xtr, 0)) + 1e-6
    f = Fusion(mean.astype(np.float32), std.astype(np.float32), [])
    Z = torch.from_numpy(f.prep(Xtr))
    y = torch.from_numpy(ytr.astype(np.float32))
    net = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], hidden), torch.nn.ReLU(),
                              torch.nn.Dropout(0.1), torch.nn.Linear(hidden, hidden),
                              torch.nn.ReLU(), torch.nn.Linear(hidden, 1))  # fmt: skip
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    pw = torch.tensor((1 - ytr.mean()) / max(ytr.mean(), 1e-6))
    for _ in range(epochs):
        net.train()
        for i in np.array_split(rng.permutation(len(Z)), max(1, len(Z) // 512)):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                net(Z[i])[:, 0], y[i], pos_weight=pw
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    f.layers = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()) for m in lin]
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--variant", default="cur", choices=["cur", "start"])
    ap.add_argument("--seeds", type=int, default=5, help="ensemble size")
    ap.add_argument("--test", action="store_true", help="FINAL: score the saved choice on test")
    args = ap.parse_args()
    run = args.run.name
    out_json = OUT / run / f"fusion_{args.variant}.json"
    if args.test:
        sel = json.loads(out_json.read_text())
        clf = Ensemble(
            [Fusion.load(p) for p in sorted((OUT / run).glob(f"fusion_{args.variant}_m*.npz"))]
        )
        te = load(run, "hocap_test", args.variant)
        cols = [list(te["names"]).index(n) for n in sel["features"]]
        s = clf.predict_proba(te["X"][:, cols])[:, 1]
        res = {"rule (live now)": score(te, run_rule(te)),
               "fusion": score(te, run_gate(te, s, **sel["gate"]))}  # fmt: skip
        for k, v in res.items():
            print(f"TEST {k:16s} {v}")
        (OUT / run / f"fusion_{args.variant}_TEST.json").write_text(json.dumps(res, indent=1))
        return
    tr = [load(run, ds, args.variant) for ds in ("dexycb_val", "oakink2")]
    va = load(run, "hocap_val", args.variant)
    names = list(va["names"])
    res = {"rule (live now)": score(va, run_rule(va))}
    print(f"HO-Cap val  rule (live now)       {res['rule (live now)']}", flush=True)
    groups = {
        "model probs only": [n for n in names if n.startswith("p_") or n == "present"],
        "model + geometry": names,
    }
    best = None
    for gname, feats in groups.items():
        cols = [names.index(n) for n in feats]
        Xtr = np.concatenate([d["X"][:, cols] for d in tr])
        ytr = np.concatenate([d["y_any2"] for d in tr])
        ptr = np.concatenate([d["present"] for d in tr]) > 0
        m = (ytr >= 0) & ptr
        clf = Ensemble([fit(Xtr[m], (ytr[m] == 2).astype(int), seed=k) for k in range(args.seeds)])
        s = clf.predict_proba(va["X"][:, cols])[:, 1]
        kw, r = tune_gate(va, s)
        res[gname] = {**r, "gate": kw}
        print(f"HO-Cap val  fusion: {gname:18s} {r} gate {kw}", flush=True)
        if best is None or r["f1"] > best[0]:
            best = (r["f1"], gname, feats, clf, kw)
    _, gname, feats, clf, kw = best
    for k, mem in enumerate(clf.members):
        mem.save(OUT / run / f"fusion_{args.variant}_m{k}.npz")
    out_json.write_text(json.dumps({"chosen": gname, "features": feats, "gate": kw,
                                    "hocap_val": res}, indent=1))  # fmt: skip
    print("chosen:", gname, "->", out_json)


if __name__ == "__main__":
    main()
