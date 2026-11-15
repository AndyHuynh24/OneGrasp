"""Release-speed profiles of the live grab gate (live_sim / panda_demo --release
normal|fast|fastest), tuned on HO-Cap val out-of-fold grab scores (hold_study.py --final;
each sequence scored by a model that did not see it). Validation only.

Sweep of the release side (off, k_off, p_none, k_none; on / k_on as tuned): normal = the tuned
gate; fast / fastest = the shortest median release delay with <= 2 % / 3.5 % of holds released
early, <= 0.1 false grabs / min and grabs found as well as the tuned gate (-0.5 pt), ties
by F1 (the recipe used for g9's profiles, DECISIONS 2026-11-13).

    python scripts/tune_release_profiles.py --run runs/<run> [--own <frames.npz>] [--write]
--write stores them as "release_profiles" in runs/grab_fusion/<run>/fusion_hold.json.

--own results/own_grasp/val_<run>_frames.npz (eval_own_grasp.py --split val --dump): also
"own_fast": among the settings reliable on BOTH HO-Cap val OOF (as "fast": <= 2 % early
releases, false grabs, grabs found) and Andy's own val session (no early release, every
hold grabbed), the fastest median release on his own data (ties: HO-Cap delay, F1). Own val
has few holds (6 in 200050), hence the HO-Cap constraint.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
import grab_fusion as G  # noqa: E402

GRID = {"off": [0.1, 0.2, 0.3, 0.4, 0.5], "k_off": [1, 2, 3, 5],
        "p_none": [0.5, 0.7, 0.9, 0.97], "k_none": [1, 2, 3]}  # fmt: skip
OWN_GRID = {"off": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "k_off": [1, 2, 3, 5],
            "p_none": [0.3, 0.5, 0.7, 0.8, 0.9, 0.97], "k_none": [0, 1, 2, 3]}  # fmt: skip
CAPS = {"fast": 0.02, "fastest": 0.035}
KEYS = ("off", "k_off", "p_none", "k_none")
SHOW = ("f1", "holds", "grabbed", "false_release", "close_delay", "release_delay",
        "false_grab_per_min")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--own", type=Path, default=None, help="own val frames npz (--dump)")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    fdir = Path("runs/grab_fusion") / args.run.name
    sel = json.loads((fdir / "fusion_hold.json").read_text())
    z = np.load(fdir / "fusion_hold_oof.npz")
    d = {"y_any2": z["y"], "tid": z["tid"], "state": z["state"], "present": z["present"]}
    gate = sel["gate"]
    base = {"on": gate["on"], "k_on": gate["k_on"]}
    cur_kw = {**base, **{k: gate.get(k, 1.0 if k == "p_none" else 0) for k in KEYS}}
    cur = G.score(d, G.run_gate(d, z["score"], **cur_kw))
    rows = []
    for vals in itertools.product(*GRID.values()):
        kw = {**base, **dict(zip(GRID, vals, strict=True))}
        rows.append((kw, G.score(d, G.run_gate(d, z["score"], **kw))))
    prof = {"normal": {"release": {k: cur_kw[k] for k in KEYS},
                       "hocap_val_oof": {k: cur[k] for k in SHOW}}}  # fmt: skip
    print("normal ", prof["normal"])
    for name, cap in CAPS.items():
        ok = [x for x in rows if x[1]["false_release"] <= cap
              and x[1]["false_grab_per_min"] <= G.FG_CAP
              and x[1]["grabbed"] >= cur["grabbed"] - 0.005]  # fmt: skip
        kw, r = min(ok, key=lambda x: (x[1]["release_delay"], -x[1]["f1"])) if ok else (cur_kw, cur)
        prof[name] = {"release": {k: kw[k] for k in KEYS}, "hocap_val_oof": {k: r[k] for k in SHOW}}
        print(f"{name:7s}", prof[name])
    if args.own is not None:
        o = np.load(args.own)
        do = {"y_any2": o["y"], "tid": o["tid"], "state": o["p_none"][:, None],
              "present": o["present"]}  # fmt: skip
        cands = []
        for vals in itertools.product(*OWN_GRID.values()):
            kw = {**base, **dict(zip(OWN_GRID, vals, strict=True))}
            if kw["k_none"] == 0:
                kw["p_none"] = 1.0
            rh = G.score(d, G.run_gate(d, z["score"], **kw))
            ro = G.score(do, G.run_gate(do, o["score"], **kw))
            if (rh["false_release"] <= CAPS["fast"] and rh["false_grab_per_min"] <= G.FG_CAP
                    and rh["grabbed"] >= cur["grabbed"] - 0.005
                    and ro["false_release"] <= 0.0 and ro["grabbed"] >= 1.0):  # fmt: skip
                cands.append((kw, rh, ro))
        if cands:
            kw, rh, ro = min(cands, key=lambda x: (x[2]["release_delay"], x[1]["release_delay"],
                                                   -x[1]["f1"]))  # fmt: skip
            prof["own_fast"] = {"release": {k: kw[k] for k in KEYS},
                                "hocap_val_oof": {k: rh[k] for k in SHOW},
                                "own_val": {k: ro[k] for k in SHOW}}  # fmt: skip
            print("own_fast", prof["own_fast"])
    if args.write:
        sel["release_profiles"] = prof
        (fdir / "fusion_hold.json").write_text(json.dumps(sel, indent=1))
        print("wrote", fdir / "fusion_hold.json")


if __name__ == "__main__":
    main()
