"""Timelines on HO-Cap val tracks: GT state, the model's raw per-frame label, the old rule gate
and the new learned hold gate (out-of-fold scores: each sequence scored by a model that did not see
it). Shows where the gripper would be closed.

    python scripts/plot_gate_timeline.py --run runs/g6_grasp_table_20261112-101602 [--n 6]
writes results/grab_gate/timeline_<run>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = {0: "#d9d9d9", 1: "#f0a35e", 2: "#3b7dd8", -1: "#ffffff"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6, help="tracks with the most GT hold frames")
    args = ap.parse_args()
    z = np.load(Path("runs/grab_fusion") / args.run.name / "fusion_hold_oof.npz")
    tids = np.unique(z["tid"])
    hold = [(z["y"][z["tid"] == t] == 2).sum() for t in tids]
    pick = tids[np.argsort(hold)[::-1][: args.n]]
    fig, axes = plt.subplots(len(pick), 1, figsize=(14, 1.9 * len(pick)), squeeze=False)
    for ax, t in zip(axes[:, 0], pick, strict=True):
        m = z["tid"] == t
        y, st, pr = z["y"][m], z["state"][m], z["present"][m] > 0
        raw = np.where(pr, st.argmax(1), -1)
        rows = [("GT", y), ("model (frame)", raw),
                ("rule: closed", np.where(z["rule"][m], 2, 0)),
                ("new gate: closed", np.where(z["closed"][m], 2, 0))]  # fmt: skip
        for k, (_, v) in enumerate(rows):
            for c in (0, 1, 2):
                ax.fill_between(np.arange(len(v)), k, k + 0.9, where=v == c, step="post",
                                color=COL[c], linewidth=0)  # fmt: skip
        ax.plot(np.arange(m.sum()), 4.0 + 0.9 * np.nan_to_num(z["score"][m]), color="k", lw=0.8)
        ax.set_yticks([0.45, 1.45, 2.45, 3.45, 4.45])
        ax.set_yticklabels([r[0] for r in rows] + ["grab score"], fontsize=8)
        ax.set_ylim(0, 5)
        ax.set_xlim(0, m.sum())
        ax.set_title(str(z["keys"][t]), fontsize=8, loc="left")
        ax.tick_params(axis="x", labelsize=7)
    axes[-1, 0].set_xlabel("frame (30 fps)    grey none · orange touch · blue hold / closed")
    fig.tight_layout()
    out = Path("results/grab_gate") / f"timeline_{args.run.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
