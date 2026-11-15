"""Tune the live gate's fast release (grab_gate.Hysteresis p_none / k_none) on HO-Cap val, using
the out-of-fold grab scores saved by hold_study.py --final (each sequence scored by a model that
did not see it). The base gate (on / off / k_on / k_off) stays as tuned. Selection: same rule as
the gate (best F1 with <= 1 % false releases and <= 0.1 false grabs / min). Validation only.

    python scripts/tune_fast_release.py --run runs/<run> [--write]
--write stores the choice in runs/grab_fusion/<run>/fusion_hold.json ("gate"; the previous gate is
kept as "gate_no_fast_release").
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
from hold_study import frame_confusion  # noqa: E402

GRID = {"p_none": [0.5, 0.7, 0.9, 0.97], "k_none": [2, 3, 5, 8, 12]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    fdir = Path("runs/grab_fusion") / args.run.name
    sel = json.loads((fdir / "fusion_hold.json").read_text())
    base = sel.get("gate_no_fast_release", sel["gate"])
    z = np.load(fdir / "fusion_hold_oof.npz")
    d = {"y_any2": z["y"], "tid": z["tid"], "state": z["state"], "present": z["present"]}
    rows = []
    for vals in [(1.0, 0), *itertools.product(*GRID.values())]:
        kw = {**base, "p_none": vals[0], "k_none": vals[1]}
        closed = G.run_gate(d, z["score"], **kw)
        rows.append((kw, {**G.score(d, closed), **frame_confusion(d, closed)}))
    print(f"HO-Cap val out-of-fold, base gate {base}")
    cols = ["p_none", "k_none", "F1", "FR", "grab", "FG/min", "hold_open", "none_closed",
            "touch_closed", "release"]  # fmt: skip
    print(" ".join(f"{c:>6s}" for c in cols))
    for kw, r in rows:
        print(f"{kw['p_none']:6.2f} {kw['k_none']:6d} {r['f1']:6.4f} {r['false_release']:6.3f} "
              f"{r['grabbed']:6.3f} {r['false_grab_per_min']:6.2f} {r['hold_open']:9.3f} "
              f"{r['none_closed']:11.3f} {r['touch_closed']:12.3f} "
              f"{r['release_delay']:7.1f}")  # fmt: skip
    ok = [x for x in rows if x[1]["false_release"] <= G.FR_CAP
          and x[1]["false_grab_per_min"] <= G.FG_CAP]  # fmt: skip
    kw, r = max(ok or rows, key=lambda x: x[1]["f1"])
    print("chosen", kw, r)
    if args.write:
        sel["gate_no_fast_release"] = base
        sel["gate"] = kw
        sel["fast_release_val_oof"] = r
        (fdir / "fusion_hold.json").write_text(json.dumps(sel, indent=1))
        print("wrote", fdir / "fusion_hold.json")


if __name__ == "__main__":
    main()
