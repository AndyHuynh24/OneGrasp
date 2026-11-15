"""Score (and optionally adapt) the grab decision on YOUR labelled recordings
(scripts/record_grabs.py): the live pipeline (MediaPipe -> same-hand tracker -> grasp model ->
3D hand in the rig's table frame, one frame at a time) on the saved frames, then every gate against
your SPACE-key hold labels:
  rule        the old rule gate
  P           probabilities-only learned gate (fusion_start)
  hold        the live gate chosen tonight (fusion_hold: probabilities + 3D motion)
  hold+own    (--adapt TRAIN_SESSION) the same gate re-fit with your training session added to its
              fit data (DexYCB val + OakInk2 + HO-Cap val + yours); scored on the OTHER session.

Your key presses lag your hand by a reaction time (~0.2-0.3 s): trust per-hold numbers (grabbed,
false release, false grabs / min) more than per-frame F1 and the delays. --label-shift N moves the
labels N frames earlier for fitting and scoring.

    python scripts/eval_own_grabs.py data/own/grabs/<test_session>
    python scripts/eval_own_grabs.py data/own/grabs/<test_session> --adapt data/own/grabs/<train>
writes <session>/score_<run>.json; --adapt also writes runs/grab_fusion/<run>/fusion_hold_own.*
(live_sim.py --fusion-variant hold_own uses it).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]

import grab_fusion as G  # noqa: E402

from touch.interact.grab_gate import Ensemble, Fusion, FusionGate, Hysteresis  # noqa: E402
from touch.interact.hold_stream import HoldStream  # noqa: E402

CKPT = None  # default: live_grasp.default_ckpt() (g10, else g9, g8, g6)
FDIR = Path("runs/grab_fusion")


def run_model(session: Path, ckpt: Path) -> dict:
    """Per frame: state [3], finger [5], present, predicted joints [21, 3], landmarks [21, 3] px,
    world joints [21, 3] (needs the rig calibration saved by record_grabs.py)."""
    from live_grasp import HandTracker, LiveGrasp, frames_with_hands

    from touch.handpath import world_hand

    f = session / f"preds_{ckpt.parent.name}.npz"
    if f.exists():
        return dict(np.load(f))
    cal = json.loads((session / "meta.json").read_text()).get("calib") or {}
    K = np.asarray(cal["K"]) if cal else None
    c2w = np.asarray(cal["cam_to_world"]) if cal else None
    cap = cv2.VideoCapture(str(session / "frames" / "%06d.jpg"), cv2.CAP_IMAGES)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    live, tracker = LiveGrasp(ckpt, 4), HandTracker(W)
    rows = {k: [] for k in ("state", "finger", "present", "joints", "lm", "jw")}
    nan21 = np.full((21, 3), np.nan, np.float32)
    for frame, hands, handed, _, _ in frames_with_hands(cap, "file", 30.0, threading.Event()):
        lm, left = tracker.pick(hands, handed)
        out = live.step(live.prep(frame, lm, left))
        jw = nan21
        if lm is not None and K is not None:
            wh = world_hand(out["joints"][None], out["verts"][None].astype(np.float32),
                            lm[None], K, c2w, left, smooth_k=1)  # fmt: skip
            jw = wh["joints"][0] if wh["ok"][0] else nan21
        rows["state"].append(out["state"])
        rows["finger"].append(out["finger"])
        rows["present"].append(lm is not None)
        rows["joints"].append(out["joints"] if lm is not None else nan21)
        rows["lm"].append(lm if lm is not None else nan21)
        rows["jw"].append(jw)
    res = {k: np.asarray(v, np.float32 if k != "present" else bool) for k, v in rows.items()}
    np.savez(f, **res)
    return res


def labels(session: Path, n: int, shift: int) -> np.ndarray:
    L = np.load(session / "labels.npz")
    hold = np.roll(L["hold"].astype(int), -shift)[:n]
    if shift:
        hold[-shift:] = hold[-shift - 1]
    return np.where(L["unsure"][:n], -1, hold * 2)


def features(session: Path, P: dict, names: list[str]) -> np.ndarray:
    """The live gate's features, computed exactly as live (HoldStream, frame by frame)."""
    hs = HoldStream(names)
    rows = []
    for i in range(len(P["state"])):
        bgr = gray = None
        if hs.need_flow or hs.need_tip:
            bgr = cv2.imread(str(session / "frames" / f"{i:06d}.jpg"))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        pres = bool(P["present"][i])
        rows.append(hs.step(P["state"][i], P["finger"][i], pres, P["joints"][i], P["lm"][i],
                            gray, P["jw"][i], bgr))  # fmt: skip
    return np.stack(rows)


def own_dataset(session: Path, ckpt: Path, names: list[str], shift: int) -> dict:
    P = run_model(session, ckpt)
    n = len(P["state"])
    return {"X": features(session, P, names), "y_any2": labels(session, n, shift),
            "present": P["present"].astype(float), "tid": np.zeros(n, int),
            "state": P["state"], "finger": P["finger"]}  # fmt: skip


def gate_closed(model, gate: dict, X: np.ndarray, present: np.ndarray,
                p_none: np.ndarray) -> np.ndarray:  # fmt: skip
    s = model.predict_proba(X)[:, 1]
    h = Hysteresis(**gate)
    return np.array([h.step(float(v), bool(p), float(q))
                     for v, p, q in zip(s, present > 0, p_none, strict=True)])  # fmt: skip


def adapt(run: str, sel: dict, own: dict, seeds: int) -> Ensemble:
    """Re-fit the hold gate's MLP with the validation sets + your training session."""
    import hold_study as HS

    feats = sel["features"]
    data = []
    for ds in ("dexycb_val", "oakink2", "hocap_val"):
        d = HS.load(run, ds)
        data.append({**d, "X": d["X"][:, [d["names"].index(n) for n in feats]]})
    data.append(own)
    return HS.fit_ens(data, list(range(len(feats))), seeds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path, help="session to SCORE")
    from live_grasp import default_ckpt

    ap.add_argument("--ckpt", type=Path, default=CKPT or default_ckpt())
    ap.add_argument("--adapt", type=Path, default=None, help="another session to fit on")
    ap.add_argument("--label-shift", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    run = args.ckpt.parent.name
    assert args.adapt is None or args.adapt.resolve() != args.session.resolve(), (
        "fit and score on different sessions")  # fmt: skip
    P = run_model(args.session, args.ckpt)
    n = len(P["state"])
    y = labels(args.session, n, args.label_shift)
    d = {"state": P["state"], "present": P["present"].astype(float), "y_any2": y,
         "tid": np.zeros(n, int)}  # fmt: skip
    res = {"frames": n, "hand_rate": round(float(d["present"].mean()), 3),
           "rule": G.score(d, G.run_rule(d))}  # fmt: skip
    fdir = FDIR / run
    if (fdir / "fusion_start.json").exists():
        g = FusionGate.from_dir(fdir, "start")
        closed = [g.step(P["state"][t], P["finger"][t], bool(P["present"][t])) for t in range(n)]
        res["P"] = G.score(d, np.array(closed))
    if (fdir / "fusion_hold.json").exists():
        sel = json.loads((fdir / "fusion_hold.json").read_text())
        X = features(args.session, P, sel["features"])
        mem = sorted(fdir.glob("fusion_hold_m*.npz"))
        model = Ensemble([Fusion.load(p) for p in mem])
        res["hold"] = G.score(d, gate_closed(model, sel["gate"], X, d["present"], P["state"][:, 0]))
        if args.adapt is not None:
            own = own_dataset(args.adapt, args.ckpt, sel["features"], args.label_shift)
            ens = adapt(run, sel, own, args.seeds)
            res["hold+own"] = G.score(
                d, gate_closed(ens, sel["gate"], X, d["present"], P["state"][:, 0])
            )
            for p in fdir.glob("fusion_hold_own_m*.npz"):
                p.unlink()
            for k, mem_k in enumerate(ens.members):
                mem_k.save(fdir / f"fusion_hold_own_m{k}.npz")
            (fdir / "fusion_hold_own.json").write_text(json.dumps(
                {**sel, "fit": sel["fit"] + f" + own:{args.adapt}",
                 "scored_on": str(args.session), "score": res["hold+own"]}, indent=1))  # fmt: skip
    for k in ("rule", "P", "hold", "hold+own"):
        if k in res:
            print(f"{k:9s} {res[k]}")
    print(f"hand found in {res['hand_rate']:.0%} of {n} frames")
    (args.session / f"score_{run}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
