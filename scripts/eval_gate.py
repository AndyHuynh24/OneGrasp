"""Object-proximity gate (touch.interact.object_gate): pick on DexYCB val, check on HO-Cap val
(subject 9), score once on the HO-Cap test set (subjects 5 + 8).

--mode val    DexYCB val: per gate candidate (CANDIDATES), the share of trackable GT contact
              episodes (state > 0 runs with a MediaPipe hand in their first 5 frames) the latch
              would let start (gate true within the first 5 frames) = what the gate costs.
--mode hocap  HO-Cap preds (eval_hocap.py; test -> runs/hocap/, val -> runs/hocap_val/): gate
              every MediaPipe hand, latch the model's state (argmax and sticky), rescore with
              eval_hocap.score, save gate_<cand>.npz (for sim_hocap.py) + metrics_gate_<cand>.json.

  python scripts/eval_gate.py --mode val
  python scripts/eval_gate.py --mode hocap --preds runs/hocap_val/<run>/preds.npz --gate NAME
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from touch.interact.object_gate import ObjectGate, latch  # noqa: E402

CANDIDATES = {  # name -> ObjectGate kwargs (picked on DexYCB val, then checked on HO-Cap val)
    "tips_r0.4": {"region": "tips", "radius": 0.4},
    "tips_r0.6": {"region": "tips", "radius": 0.6},
    "hull10": {"region": "hull", "min_px": 10},
    "hull10_arm": {"region": "hull", "min_px": 10, "arm": 2.5},
    "hull10_arm_skin": {"region": "hull", "min_px": 10, "arm": 2.5, "skin_tau": 20.0},
    "hull10_arm_skin_fast": {"region": "hull", "min_px": 10, "arm": 2.5, "skin_tau": 20.0,
                             "keep": 12, "every": 4},
}  # fmt: skip


def val_job(path: str) -> dict:
    from touch.data import dexycb as D

    p = Path(path)
    seq = next(s for s in D.list_sequences(split="val") if s.name == p.parent.name)
    z, c = np.load(p), np.load(p.parent / "contact.npz")
    lm, pres, st = z["lm_px"], z["present"], c["state"]
    gates = {k: ObjectGate(**kw) for k, kw in CANDIDATES.items()}
    near = {k: np.zeros(len(lm), bool) for k in gates}
    for i in range(len(lm)):
        img = cv2.imread(str(seq.image(p.stem, i)))
        hands = [lm[i]] if pres[i] else []
        for k, g in gates.items():
            f = g.update(img, hands)
            if pres[i]:
                near[k][i] = f[0]
    out = {}
    runs = np.flatnonzero(np.diff(np.r_[0, (st > 0).astype(int), 0]))
    ab = [
        (a, b) for a, b in zip(runs[::2], runs[1::2], strict=True) if pres[a : min(b, a + 5)].any()
    ]  # trackable episodes only (hand seen at start)
    for key, nr in near.items():
        out[key] = [bool(nr[a : min(b, a + 5)].any()) for a, b in ab]
    return out


def hocap_job(job) -> dict:
    from eval_hocap import mediapipe_cache

    from touch.data import hocap as H

    preds_path, name, cam, cand = job
    seq = next(s for s in H.list_sequences(video_only=True) if s.name == name)
    P = np.load(preds_path)
    keys = [f"{seq.subject}/{name}/{cam}/{s}" for s in seq.sides]
    keys = [k for k in keys if f"{k}/lm" in P.files]
    alm = {k: P[f"{k}/lm"] for k in keys}
    n = len(next(iter(alm.values())))
    mp = mediapipe_cache(seq, cam, n)["lm"]
    g = ObjectGate(**CANDIDATES[cand])
    near = {k: np.zeros(n, bool) for k in keys}
    for i in range(n):
        idx = [h for h in range(2) if np.isfinite(mp[i, h]).all()]
        img = cv2.imread(str(seq.image(cam, i)))
        flags = g.update(img, [mp[i, h] for h in idx])
        for k in keys:
            if np.isfinite(alm[k][i]).all():
                for h, f in zip(idx, flags, strict=True):
                    if np.allclose(mp[i, h], alm[k][i]):
                        near[k][i] = f
    return near


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "hocap"], required=True)
    ap.add_argument("--preds", type=Path)
    ap.add_argument("--gate", choices=sorted(CANDIDATES), default="hull10_arm_skin_fast")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    if args.mode == "val":
        files = sorted(p for p in Path("data/dexycb_proc").glob("*/*/*.npz")
                       if p.name != "contact.npz")  # fmt: skip
        from touch.data import dexycb as D

        val = {s.name for s in D.list_sequences(split="val")}
        files = [str(p) for p in files if p.parent.name in val]
        kept: dict = {}
        with get_context("spawn").Pool(args.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(val_job, files)):
                for key, v in r.items():
                    kept.setdefault(key, []).extend(v)
                if k % 50 == 0:
                    print(f"{k}/{len(files)} | {(time.time() - t0) / 60:.1f} min", flush=True)
        res = {k: {"episodes": len(v), "kept": round(float(np.mean(v)), 4)}
               for k, v in kept.items()}  # fmt: skip
        out = Path("runs/hocap/gate_val.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
        for k, v in sorted(res.items()):
            print(k, v)
        return

    from eval_hocap import cam_group, gt_cam_mesh, gt_labels, score

    from touch.data import hocap as H

    split = "test" if args.preds.parent.parent.name == "hocap" else "val"
    seqs = H.list_sequences(video_only=True, split=split)
    P0 = np.load(args.preds)
    jobs = [(str(args.preds), s.name, c, args.gate)
            for s in seqs for c in s.serials
            if any(f"{s.subject}/{s.name}/{c}/{sd}/lm" in P0.files for sd in s.sides)]  # fmt: skip
    near: dict = {}
    with get_context("spawn").Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(hocap_job, jobs)):
            near.update(r)
            print(f"{k + 1}/{len(jobs)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    out_dir = args.preds.parent
    np.savez(out_dir / f"gate_{args.gate}.npz", **{f"{k}/near": v for k, v in near.items()})
    P = np.load(args.preds)
    seq_by = {s.name: s for s in seqs}
    groups: dict = {"all": []}
    for key in sorted(near):
        _, name, cam, side = key.split("/")
        pr = {f: P[f"{key}/{f}"] for f in ("state", "finger", "vert", "joints", "verts", "sticky",
                                            "present")}  # fmt: skip
        nr = near[key]
        ps = np.where(pr["present"], pr["state"].argmax(1), 0)
        gated = latch(ps, nr)
        pr["state"] = np.eye(3, dtype=np.float32)[gated]
        pr["sticky"] = latch(pr["sticky"].astype(int), nr)
        L = gt_labels(seq_by[name], side)
        gj, gv = gt_cam_mesh(L, cam, side == "left")
        it = {"P": pr, "L": L, "gj": gj, "gv": gv}
        groups["all"].append(it)
        groups.setdefault(cam_group(cam), []).append(it)
    metrics = {g: score(v) for g, v in groups.items()}
    metrics["meta"] = {"preds": str(args.preds), "gate": args.gate,
                       "params": CANDIDATES[args.gate],
                       "minutes": round((time.time() - t0) / 60, 1)}  # fmt: skip
    (out_dir / f"metrics_gate_{args.gate}.json").write_text(json.dumps(metrics, indent=1))
    for g in groups:
        m = metrics[g]
        tr = m["table_rest"]
        print(f"{g:9s} state macro any2 {m['state_any2_raw']['macro_f1']:.3f} | table-rest "
              f"touch {tr['says_touch']:.2f} hold {tr['says_hold']:.2f}")  # fmt: skip
    print("wrote", out_dir / f"metrics_gate_{args.gate}.json")


if __name__ == "__main__":
    main()
