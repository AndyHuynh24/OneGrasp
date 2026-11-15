"""Which features make the HOLD (gripper closed) decision more reliable? Ablation on validation.

Feature groups (all causal, all available live):
  P    the grasp model's probabilities + their short history (touch.interact.grab_gate)
  M    3D hand motion from the calibrated rig: rise over 1 s, vertical speed, speed
  H    grasp shape from the predicted MANO joints (touch.interact.hand_features.pose_features)
  Hmp  the same from the MediaPipe landmarks
  I    2D hand velocity (palm lengths / frame)
  C    optical-flow co-motion of the band around the hand (CoMotion) + 15-frame mean
  N    fingertip colour vs palm skin (blood pushed out of a pressed fingertip: less red) + its
       15-frame mean and its drop below the redness of the last 3 s
Two fitting protocols, both scored on HO-Cap val (subject 9) with the live hysteresis gate tuned
on the same scores (best F1 with <= 1 % false releases, <= 0.1 false grabs / min):
  xds  fit on DexYCB val + OakInk2 only (HO-Cap val unseen)
  cv   + HO-Cap val sequences in 4 folds; every val sequence scored by a model that did not see it
Per frame, also: share of GT-hold frames with the gripper open (hold missed), GT-touch / GT-none
frames with it closed, and the AUROC of the grab score.

    python scripts/hold_study.py --run runs/g6_grasp_table_20261112-101602
writes runs/grab_fusion/<run>/hold_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]

import grab_fusion as G  # noqa: E402

from touch.interact.grab_gate import Ensemble, rolling  # noqa: E402

OUT = Path("runs/grab_fusion")
M_NAMES = ["rise30", "vz", "speed"]


def load(run: str, ds: str) -> dict:
    a = G.load(run, ds, "start")
    h = np.load(OUT / run / f"{ds}_hand.npz")
    assert list(h["keys"]) == list(a["keys"])
    names = list(a["names"]) + list(h["names"])
    X = np.concatenate([a["X"], h["X"]], 1)
    extra, en = [], []
    for col in ("co_frac", "mano_aperture_min", "mano_curl_index"):  # 15-frame causal means
        x = X[:, names.index(col)]
        r = np.full(len(x), np.nan, np.float32)
        for idx in G.tracks(a):
            r[idx] = rolling(x[idx, None], 15, np.nanmean)[:, 0]
        extra.append(r[:, None])
        en.append(f"{col}_m15")
    if "tip_da_mean" in names:  # less red than this hand's fingertips were in the last 3 s
        x = X[:, names.index("tip_da_mean")]
        m15, rel = (np.full(len(x), np.nan, np.float32) for _ in range(2))
        for idx in G.tracks(a):
            m15[idx] = rolling(x[idx, None], 15, np.nanmean)[:, 0]
            rel[idx] = m15[idx] - rolling(m15[idx, None], 90, np.nanmax)[:, 0]
        extra += [m15[:, None], rel[:, None]]
        en += ["tip_da_mean_m15", "tip_da_rel90"]
    a["X"] = np.concatenate([X, *extra], 1)
    a["names"] = names + en
    a["seq"] = np.array([str(a["keys"][t]).split("/")[-3] if ds.startswith("hocap") else ds
                         for t in a["tid"]])  # fmt: skip
    return a


def groups(names: list[str]) -> dict[str, list[str]]:
    P = [n for n in names if n.startswith("p_") or n == "present"]
    M = M_NAMES
    H = [n for n in names if n.startswith("mano_")]
    Hmp = [n for n in names if n.startswith("mp_")]
    Im = [n for n in names if n.startswith("img_")]
    C = [n for n in names if n.startswith("co_")] + ["still_frac", "hand_flow"]
    N = [n for n in names if n.startswith("tip_")]
    return {"P": P, "P+M": P + M, "P+H": P + H, "P+Hmp": P + Hmp, "P+I": P + Im, "P+C": P + C,
            "P+N": P + N, "P+M+C": P + M + C, "P+H+C": P + H + C, "P+M+C+N": P + M + C + N,
            "P+M+H+C": P + M + H + C, "all": P + M + H + Hmp + Im + C,
            "all+N": P + M + H + Hmp + Im + C + N}  # fmt: skip


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(s)
    s, y = s[m], y[m]
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def frame_confusion(d: dict, closed: np.ndarray) -> dict:
    y = d["y_any2"]
    out = {}
    for k, name in ((2, "hold_open"), (1, "touch_closed"), (0, "none_closed")):
        m = y == k
        v = (~closed[m]).mean() if k == 2 else closed[m].mean()
        out[name] = round(float(v), 3)
    return out


def fit_ens(datasets: list[dict], cols: list[int], seeds: int) -> Ensemble:
    X = np.concatenate([d["X"][:, cols] for d in datasets])
    y = np.concatenate([d["y_any2"] for d in datasets])
    p = np.concatenate([d["present"] for d in datasets]) > 0
    m = (y >= 0) & p
    return Ensemble([G.fit(X[m], (y[m] == 2).astype(int), seed=k) for k in range(seeds)])


class GRUFusion:
    """Causal GRU over the per-frame features of a track -> per-frame hold logit (frame t only
    sees frames <= t). Inputs standardised, NaN -> 0 plus a NaN flag, as in grab_gate.Fusion."""

    def __init__(self, mean, std, net):
        self.mean, self.std, self.net = mean, std, net

    def prep(self, X):
        nan = ~np.isfinite(X)
        Z = np.where(nan, 0.0, (np.nan_to_num(X) - self.mean) / self.std)
        return np.concatenate([Z, nan], 1).astype(np.float32)

    def predict_tracks(self, d: dict, cols: list[int]) -> np.ndarray:
        import torch

        out = np.full(len(d["tid"]), np.nan)
        with torch.no_grad():
            for idx in G.tracks(d):
                z = torch.from_numpy(self.prep(d["X"][idx][:, cols]))[None]
                out[idx] = torch.sigmoid(self.net(z)[0, :, 0]).numpy()
        return out


def fit_gru(datasets: list[dict], cols: list[int], seed: int, steps: int = 1500,
            win: int = 64, hidden: int = 32, bidir: bool = False) -> GRUFusion:  # fmt: skip
    """bidir: OFFLINE (frame t also sees later frames; whole track at prediction)."""
    import warnings

    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = np.concatenate([d["X"][:, cols] for d in datasets])
        mean = np.nan_to_num(np.nanmean(X, 0)).astype(np.float32)
        std = (np.nan_to_num(np.nanstd(X, 0)) + 1e-6).astype(np.float32)

    class Net(torch.nn.Module):
        def __init__(self, f):
            super().__init__()
            self.inp = torch.nn.Sequential(torch.nn.Linear(f, 64), torch.nn.ReLU())
            self.gru = torch.nn.GRU(64, hidden, batch_first=True, bidirectional=bidir)
            self.out = torch.nn.Linear(hidden * (2 if bidir else 1) + 64, 1)

        def forward(self, z):
            h = self.inp(z)
            g, _ = self.gru(h)
            return self.out(torch.cat([g, h], -1))

    gf = GRUFusion(mean, std, Net(2 * len(cols)))
    seqs = []
    for d in datasets:
        for idx in G.tracks(d):
            y = d["y_any2"][idx]
            m = (y >= 0) & (d["present"][idx] > 0)
            seqs.append((gf.prep(d["X"][idx][:, cols]), (y == 2).astype(np.float32), m))
    pos = np.mean(np.concatenate([s[1][s[2]] for s in seqs]))
    pw = torch.tensor((1 - pos) / max(pos, 1e-6))
    opt = torch.optim.AdamW(gf.net.parameters(), 2e-3, weight_decay=1e-4)
    lens = np.array([len(s[0]) for s in seqs], float)
    p = lens / lens.sum()
    for _ in range(steps):
        zb, yb, mb = [], [], []
        for k in rng.choice(len(seqs), 32, p=p):
            z, y, m = seqs[k]
            a = int(rng.integers(0, max(1, len(z) - win + 1)))
            pad = win - len(z[a : a + win])
            zb.append(np.pad(z[a : a + win], ((0, pad), (0, 0))))
            yb.append(np.pad(y[a : a + win], (0, pad)))
            mb.append(np.pad(m[a : a + win], (0, pad)))
        z, y, m = (torch.from_numpy(np.stack(v)) for v in (zb, yb, mb))
        logit = gf.net(z)[..., 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit[m], y[m], pos_weight=pw)
        opt.zero_grad()
        loss.backward()
        opt.step()
    gf.net.eval()
    return gf


class GRUEnsemble:
    def __init__(self, members):
        self.members = members

    def predict_tracks(self, d, cols):
        return np.mean([m.predict_tracks(d, cols) for m in self.members], 0)


def fit_any(kind: str, datasets: list[dict], cols: list[int], seeds: int):
    if kind == "gru":
        return GRUEnsemble([fit_gru(datasets, cols, seed=k) for k in range(seeds)])
    if kind == "bigru":  # OFFLINE: ~4 s training windows, sees the future
        return GRUEnsemble([fit_gru(datasets, cols, seed=k, win=128, bidir=True)
                            for k in range(seeds)])  # fmt: skip
    return fit_ens(datasets, cols, seeds)


def predict(model, d: dict, cols: list[int]) -> np.ndarray:
    if isinstance(model, GRUEnsemble):
        return model.predict_tracks(d, cols)
    return model.predict_proba(d["X"][:, cols])[:, 1]


def subset(d: dict, mask: np.ndarray) -> dict:
    return {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v)
            for k, v in d.items()}  # fmt: skip


def final(run: str, gname: str, tr: list[dict], va: dict, folds, seeds: int) -> None:
    """Live gate: the hysteresis is tuned on out-of-fold HO-Cap val scores (each sequence scored
    by a model that did not see it); the deployed MLP ensemble is then fit on all validation
    data (DexYCB val + OakInk2 + HO-Cap val). Writes fusion_hold.json + fusion_hold_m<k>.npz."""
    feats = groups(va["names"])[gname]
    cols = [va["names"].index(n) for n in feats]
    s = np.full(len(va["y_any2"]), np.nan)
    for f in folds:
        held = np.isin(va["seq"], f)
        s[held] = predict(fit_ens([*tr, subset(va, ~held)], cols, seeds), va, cols)[held]
    kw, r = G.tune_gate(va, s)
    closed = G.run_gate(va, s, **kw)
    r = {**r, **frame_confusion(va, closed)}
    np.savez_compressed(OUT / run / "fusion_hold_oof.npz", score=s, closed=closed,
                        rule=G.run_rule(va), y=va["y_any2"], tid=va["tid"],
                        state=va["state"], present=va["present"], keys=va["keys"])  # fmt: skip
    ens = fit_ens([*tr, va], cols, seeds)
    for p in (OUT / run).glob("fusion_hold_m*.npz"):
        p.unlink()
    for k, mem in enumerate(ens.members):
        mem.save(OUT / run / f"fusion_hold_m{k}.npz")
    (OUT / run / "fusion_hold.json").write_text(json.dumps(
        {"group": gname, "features": feats, "gate": kw, "hocap_val_out_of_fold": r,
         "fit": "dexycb_val + oakink2 + hocap_val", "seeds": seeds}, indent=1))  # fmt: skip
    print(f"live gate {gname}: out-of-fold HO-Cap val {r} gate {kw}")


def test(run: str) -> None:
    """FINAL, once: score the saved live gate (fusion_hold.*, no refit, no retuning) and the old
    rule on HO-Cap test. Needs eval_hocap --split test + grab_features / hold_features --test."""
    from touch.interact.grab_gate import Ensemble, Fusion

    sel = json.loads((OUT / run / "fusion_hold.json").read_text())
    te = load(run, "hocap_test")
    cols = [te["names"].index(n) for n in sel["features"]]
    clf = Ensemble([Fusion.load(p) for p in sorted((OUT / run).glob("fusion_hold_m*.npz"))])
    s = clf.predict_proba(te["X"][:, cols])[:, 1]
    res = {}
    for k, closed in (("rule", G.run_rule(te)), ("gate", G.run_gate(te, s, **sel["gate"]))):
        res[k] = {**G.score(te, closed), **frame_confusion(te, closed)}
        print(f"TEST {k:5s} {res[k]}")
    res["n_tracks"], res["n_frames"] = int(len(np.unique(te["tid"]))), int(len(te["tid"]))
    (OUT / run / "fusion_hold_TEST.json").write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--only", default="", help="comma list of group names")
    ap.add_argument("--protocols", default="xds,cv")
    ap.add_argument("--model", default="mlp", choices=["mlp", "gru", "bigru"])
    ap.add_argument("--out", default="hold_study.json")
    ap.add_argument("--final", default="", help="group name: build the live gate fusion_hold.*")
    ap.add_argument("--test", action="store_true", help="FINAL, once: saved gate on HO-Cap test")
    args = ap.parse_args()
    run = args.run.name
    if args.test:
        test(run)
        return
    tr = [load(run, "dexycb_val"), load(run, "oakink2")]
    va = load(run, "hocap_val")
    names = va["names"]
    for d in tr:
        assert d["names"] == names
    res_path = OUT / run / args.out
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    closed = G.run_rule(va)
    res["rule (live before tonight)"] = {**G.score(va, closed), **frame_confusion(va, closed)}
    seqs = np.unique(va["seq"])
    folds = [seqs[i::4] for i in range(4)]
    if args.final:
        final(run, args.final, tr, va, folds, args.seeds)
        return
    print(f"HO-Cap val: {len(seqs)} sequences, {len(va['y_any2'])} frames", flush=True)
    print(f"{'rule (live before tonight)':28s} {res['rule (live before tonight)']}", flush=True)
    for gname, feats in groups(names).items():
        if args.only and gname not in args.only.split(","):
            continue
        cols = [names.index(n) for n in feats]
        for proto in args.protocols.split(","):
            if proto == "xds":
                s = predict(fit_any(args.model, tr, cols, args.seeds), va, cols)
            else:
                s = np.full(len(va["y_any2"]), np.nan)
                for f in folds:
                    held = np.isin(va["seq"], f)
                    model = fit_any(args.model, [*tr, subset(va, ~held)], cols, args.seeds)
                    s[held] = predict(model, va, cols)[held]
            kw, r = G.tune_gate(va, s)
            closed = G.run_gate(va, s, **kw)
            pres = va["present"] > 0
            r = {**r, **frame_confusion(va, closed),
                 "auroc": round(auroc(s[pres & (va["y_any2"] >= 0)],
                                      va["y_any2"][pres & (va["y_any2"] >= 0)] == 2), 4),
                 "gate": kw, "n_feat": len(feats)}  # fmt: skip
            res[f"{gname} [{proto}]"] = r
            res_path.write_text(json.dumps(res, indent=1))
            print(f"{gname + ' [' + proto + ']':28s} F1 {r['f1']:.4f} AUROC {r['auroc']:.4f} "
                  f"FR {r['false_release']:.3f} grabbed {r['grabbed']:.3f} "
                  f"hold_open {r['hold_open']:.3f} touch_closed {r['touch_closed']:.3f} "
                  f"none_closed {r['none_closed']:.3f} close {r['close_delay']} "
                  f"release {r['release_delay']} FG/min {r['false_grab_per_min']}",
                  flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
