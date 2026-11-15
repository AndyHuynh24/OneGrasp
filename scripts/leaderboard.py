"""Collect the best-epoch validation results of every run into results/leaderboard.csv.

All numbers are on the EgoPressure VALIDATION split (tau / median length tuned on val too).

Usage:
    python scripts/leaderboard.py            # prints a table, writes results/leaderboard.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

COLS = [
    "finger_auroc",
    "finger_bal_acc",
    "finger_acc",
    "index_acc",
    "hand_acc",
    "click_f1_exact",
    "click_f1_pm1",
    "click_f1_pm2",
    "click_recall_pm2",
    "click_precision_pm2",
    "timing_down_mae_ms",
    "timing_up_mae_ms",
    "hover_fp_per_min",
]


def main() -> None:
    rows = []
    for ev in sorted(Path("runs").glob("*/eval_val.json")):
        run = ev.parent
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        rec = json.loads(ev.read_text())
        if "val" not in rec:  # pre-harness format (smoke run)
            print(f"skipping {ev} (old format)")
            continue
        lat = run / "latency.json"
        cpu_ms = json.loads(lat.read_text())["cpu_ms_per_finger_frame"] if lat.exists() else ""
        for view in ("all", "ego", "overhead"):
            r = rec["val"].get(view)
            if r is None:
                continue
            row = {
                "run": run.name,
                "name": cfg["name"],
                "seed": cfg["seed"],
                "view": view,
                "epoch": rec["epoch"],
                "post": json.dumps(rec.get("post", {})),
                "cpu_ms": cpu_ms,
            }
            row.update({c: round(r[c], 4) for c in COLS})
            rows.append(row)
    if not rows:
        sys.exit("no runs with eval_val.json yet")
    out = Path("results/leaderboard.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = [
        "name",
        "seed",
        "view",
        "epoch",
        "finger_auroc",
        "finger_bal_acc",
        "finger_acc",
        "hand_acc",
        "click_f1_exact",
        "click_f1_pm2",
        "timing_down_mae_ms",
        "hover_fp_per_min",
        "cpu_ms",
    ]
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
