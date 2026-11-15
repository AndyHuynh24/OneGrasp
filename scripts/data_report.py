"""Summarize the canonical dataset into docs/DATA_REPORT.md (counts, transitions, quality flags).

Usage:
    python scripts/data_report.py    # needs data/processed (run scripts/prepare_data.py first)
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from touch.data.catalog import load_catalog  # noqa: E402
from touch.data.pressure import (  # noqa: E402
    CELL_AREA_MM2,
    MAX_TIP_HEIGHT_MM,
    contact_segments,
    count_transitions,
)

OUT = Path("docs/DATA_REPORT.md")
SPLITS = ("train", "val", "test")


def table(header: list[str], rows: list[list]) -> str:
    def fmt(x):
        if isinstance(x, float):
            return f"{x:,.1f}" if abs(x) >= 10 else f"{x:.2f}"
        return f"{x:,}" if isinstance(x, int) else str(x)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def du(path: str) -> str:
    r = subprocess.run(["du", "-sh", path], capture_output=True, text=True)
    return r.stdout.split()[0] if r.returncode == 0 else "?"


def main() -> None:
    seqs = load_catalog()
    if not seqs:
        sys.exit("no processed sequences; run scripts/prepare_data.py first")

    per = []  # one dict per sequence
    for s in seqs:
        L, F = s.labels, s.fingers
        c = L["contact"]
        down, _ = count_transitions(c)
        d = {
            "s": s,
            "source": s.source,
            "split": s.split or "?",
            "p": s.participant,
            "family": s.meta["family"],
            "variant": s.meta["variant"],
            "hand": s.meta["hand"],
            "frames": len(s),
            "secs": len(s) / s.meta["fps"],
            "contact": int(c.sum()),
            "down": down,
            "press3": sum(1 for a, b in contact_segments(c) if b - a >= 3),
        }
        if F is not None:
            v = F["valid"]
            fc = F["contact"] & v[:, None]
            d["finger_valid"] = int(v.sum())
            d["index_down"] = count_transitions(fc[:, 1])[0]
            d["finger_down"] = sum(count_transitions(fc[:, k])[0] for k in range(5))
            both = c & v
            d["index_share"] = float(fc[both, 1].mean()) if both.any() else float("nan")
            d["other_ge1"] = int((np.nan_to_num(F["other_max_kpa"]) >= 1.0)[both].sum())
        for view in s.views:
            if s.has_landmarks(view):
                z = s.landmarks(view)
                d[f"det_{view}"] = float((z["primary"] >= 0).mean())
                g = s.labels.get(f"gt2d_{view}")
                if g is not None:
                    vw, vh = s.meta["views"][view]["width"], s.meta["views"][view]["height"]
                    inside = (
                        np.isfinite(g).all((1, 2))
                        & (g[..., 0] >= 0).all(1)
                        & (g[..., 0] < vw).all(1)
                        & (g[..., 1] >= 0).all(1)
                        & (g[..., 1] < vh).all(1)
                    )
                    d[f"inside_{view}"] = int(inside.sum())
                    d[f"inside_det_{view}"] = int((inside & (z["primary"] >= 0)).sum())
                ok = z["primary_gt"] >= 0
                if ok.any():
                    d[f"gtmatch_{view}"] = float(ok.mean())
                    d[f"agree_{view}"] = float((z["primary"][ok] == z["primary_gt"][ok]).mean())
                    d[f"gterr_{view}"] = float(np.nanmedian(z["gt_err_px"]))
        per.append(d)

    md = [
        f"# Data report\n\nGenerated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/data_report.py` "
        f"from `data/processed` ({len(per)} sequences). Splits from `configs/splits.yaml` "
        "(chosen 2026-10-04, frozen). Contact = frame max ≥ 1 kPa after zeroing < 0.5 kPa "
        "(DECISIONS 2026-10-03, option C). Counts are descriptive only — no thresholds were "
        "tuned on any split.\n"
    ]

    # 1. Totals per source x split
    rows = []
    for src in sorted({d["source"] for d in per}):
        for sp in SPLITS + ("?",):
            g = [d for d in per if d["source"] == src and d["split"] == sp]
            if not g:
                continue
            rows.append(
                [
                    src,
                    sp,
                    len({d["p"] for d in g}),
                    len(g),
                    sum(d["frames"] for d in g),
                    sum(d["secs"] for d in g) / 60,
                    sum(d["contact"] for d in g),
                    sum(d["down"] for d in g),
                    sum(d["press3"] for d in g),
                    sum(d.get("index_down", 0) for d in g),
                ]
            )
    md += [
        "## 1. Totals by source and split\n",
        table(
            [
                "source",
                "split",
                "participants",
                "sequences",
                "frames",
                "minutes",
                "contact frames",
                "touch-downs (hand)",
                "presses ≥3 frames",
                "index-finger touch-downs",
            ],
            rows,
        ),
        "",
    ]

    # 2. Per gesture
    rows = []
    groups = defaultdict(list)
    for d in per:
        groups[(d["source"], d["family"], d["variant"], d["hand"])].append(d)
    for (src, fam, var, hand), g in sorted(groups.items()):
        rows.append(
            [
                src,
                fam,
                var,
                hand,
                len(g),
                sum(d["frames"] for d in g),
                sum(d["contact"] for d in g),
                sum(d["down"] for d in g),
                sum(d.get("index_down", 0) for d in g),
                sum(d.get("finger_down", 0) - d.get("index_down", 0) for d in g),
            ]
        )
    md += [
        "## 2. By gesture\n",
        table(
            [
                "source",
                "family",
                "variant",
                "hand",
                "seqs",
                "frames",
                "contact frames",
                "touch-downs (hand)",
                "index touch-downs",
                "other-finger touch-downs",
            ],
            rows,
        ),
        "\nHand-level touch-downs count every 0→1 change of frame contact (1-frame blips "
        'included); "presses ≥3 frames" in §1 ignores runs shorter than 3 frames.\n',
    ]

    # 3. Transitions per participant (P1 step 7)
    md += ["## 3. Touch-downs per participant (P1 step 7)\n"]
    for src in sorted({d["source"] for d in per}):
        rows = []
        for p in sorted({d["p"] for d in per if d["source"] == src}):
            g = [d for d in per if d["p"] == p]
            rows.append(
                [
                    p,
                    g[0]["split"],
                    len(g),
                    sum(d["frames"] for d in g),
                    sum(d["down"] for d in g),
                    sum(d["press3"] for d in g),
                    sum(d.get("index_down", 0) for d in g),
                ]
            )
        md += [
            f"### {src}\n",
            table(
                [
                    "participant",
                    "split",
                    "seqs",
                    "frames",
                    "touch-downs (hand)",
                    "presses ≥3 frames",
                    "index touch-downs",
                ],
                rows,
            ),
            "",
        ]

    # 4. Landmarks
    rows = []
    for src in sorted({d["source"] for d in per}):
        for view in ("ego", "overhead"):
            g = [d for d in per if d["source"] == src and f"det_{view}" in d]
            if not g:
                continue
            frames = np.array([d["frames"] for d in g])
            det = np.array([d[f"det_{view}"] for d in g])
            r = [src, view, len(g), float((det * frames).sum() / frames.sum())]
            ins = sum(d.get(f"inside_{view}", 0) for d in g)
            hit = sum(d.get(f"inside_det_{view}", 0) for d in g)
            r.append(float(hit / ins) if ins else "–")
            gg = [d for d in g if f"gtmatch_{view}" in d]
            if gg:
                r += [
                    float(np.mean([d[f"gtmatch_{view}"] for d in gg])),
                    float(np.mean([d[f"agree_{view}"] for d in gg])),
                    float(np.median([d[f"gterr_{view}"] for d in gg])),
                ]
            else:
                r += ["–", "–", "–"]
            rows.append(r)
    md += [
        "## 4. MediaPipe hand landmarks (0.10.35, VIDEO mode, min confidence 0.3)\n",
        table(
            [
                "source",
                "view",
                "seqs",
                "frames with a hand",
                "…when the GT hand is fully in the image",
                "frames matched to GT hand",
                "label rule picks GT hand",
                "median landmark–GT dist (px)",
            ],
            rows,
        ),
        "\nGT columns exist only for EgoPressure (MANO joints projected with the released "
        "calibration). MANO joints sit inside the finger and MediaPipe's on the surface, so "
        "some px distance is expected. Image widths: ego 1920, EgoPressure overhead 2560, "
        "PVDB overhead 1920.\n",
    ]

    # 5. Per-finger label sanity
    rows = []
    for src in sorted({d["source"] for d in per}):
        g = [
            d
            for d in per
            if d["source"] == src
            and d["family"] == "index_press"
            and d["variant"] != "no-contact"
            and "index_share" in d
            and np.isfinite(d["index_share"])
        ]
        if g:
            rows.append(
                [
                    src,
                    len(g),
                    float(np.mean([d["index_share"] for d in g])),
                    sum(d["other_ge1"] for d in g),
                ]
            )
    md += [
        "## 5. Per-finger labels: index_press sanity check\n",
        table(
            [
                "source",
                "index_press seqs (contact variants)",
                "mean share of contact frames given to the index finger",
                "contact frames with ≥1 kPa not near any finger",
            ],
            rows,
        ),
        "\nFingers are located with GT joints (EgoPressure) or MediaPipe on the overhead view "
        "(PVDB), projected onto the pad; each pressure cell goes to the nearest distal "
        "phalanx within 20 mm.\n",
    ]

    # 5b. Speckles: tiny contact areas (isolated sensor spots) that still pass 1 kPa
    rows = []
    for src in sorted({d["source"] for d in per}):
        areas = np.concatenate(
            [d["s"].labels["area_mm2"][d["s"].labels["contact"]] for d in per if d["source"] == src]
            or [np.zeros(0)]
        )
        cells = areas / CELL_AREA_MM2
        if len(cells):
            rows.append(
                [
                    src,
                    len(cells),
                    float(np.mean(cells <= 1) * 100),
                    float(np.mean(cells <= 4) * 100),
                    float(np.mean(cells <= 10) * 100),
                    float(np.median(cells)),
                ]
            )
    md += [
        "### Small contact areas\n",
        table(
            [
                "source",
                "contact frames",
                "% area ≤1 cell",
                "% ≤4 cells",
                "% ≤10 cells",
                "median cells",
            ],
            rows,
        ),
        "\nA real fingertip press covers hundreds of cells (pressure spreads through the "
        "texture overlay). Frames whose contact is only a few cells are mostly isolated "
        "sensor speckles. `area_mm2` is stored per frame (and per finger) so a minimum-area "
        "rule can be chosen in P2 **on validation only**; the stored `contact` label does "
        "not apply one.\n",
    ]

    # 5c. Release lag: pressure vs GT fingertip height around each press (EgoPressure)
    end_lag, start_lead = [], []
    for d in per:
        s, F = d["s"], d["s"].fingers
        if s.source != "egopressure" or d["family"] != "index_press" or F is None:
            continue
        h = F["tip_height_mm"][:, 1]
        for a, b in contact_segments(s.labels["contact"]):
            lo, hi = max(a - 3, 0), min(b + 3, len(h))
            if b - a < 5 or not np.isfinite(h[lo:hi]).all():
                continue
            k = 0
            while k < b - a and h[b - 1 - k] > MAX_TIP_HEIGHT_MM:
                k += 1
            end_lag.append(k)
            k = 0
            while k < b - a and h[a + k] > MAX_TIP_HEIGHT_MM:
                k += 1
            start_lead.append(k)
    if end_lag:
        e, st = np.array(end_lag), np.array(start_lead)
        md += [
            "### Pressure vs. fingertip height at touch-down / up (EgoPressure index_press)\n",
            table(
                [
                    "event",
                    "presses",
                    "% with ≥1 frame mismatch",
                    "mean frames",
                    "p90 frames",
                    "max frames",
                ],
                [
                    [
                        "touch-up: pressure ≥1 kPa but index tip already >12 mm up",
                        len(e),
                        float(np.mean(e > 0) * 100),
                        float(e.mean()),
                        int(np.percentile(e, 90)),
                        int(e.max()),
                    ],
                    [
                        "touch-down: pressure ≥1 kPa before tip is within 12 mm",
                        len(st),
                        float(np.mean(st > 0) * 100),
                        float(st.mean()),
                        int(np.percentile(st, 90)),
                        int(st.max()),
                    ],
                ],
            ),
            "\nAt 30 FPS one frame is 33 ms. Pressure-based touch-**up** labels tend to lag the "
            "visible lift-off (sensor/overlay release, or MANO fit lag); touch-down agrees well. "
            "Keep this in mind for timing metrics (P2) — decide any correction on validation "
            "only.\n",
        ]

    # 6. Flags
    flags = []
    for d in per:
        name = f"{d['p']}/{d['s'].name}"
        if d["variant"] == "no-contact" and d["press3"] > 0:
            flags.append(f"- `{name}`: no-contact gesture but {d['press3']} press(es) ≥3 frames")
        if (
            d["variant"] in ("high", "low")
            and d["family"] in ("index_press", "press_fingers")
            and d["press3"] == 0
        ):
            flags.append(f"- `{name}`: contact gesture but no press ≥3 frames")
        for view in ("ego", "overhead"):
            if d.get(f"det_{view}", 1) < 0.5:
                flags.append(
                    f"- `{name}`: hand detected in only {d[f'det_{view}']:.0%} of {view} frames"
                )
        if (
            "index_share" in d
            and d["family"] == "index_press"
            and d["variant"] != "no-contact"
            and np.isfinite(d["index_share"])
            and d["index_share"] < 0.8
        ):
            flags.append(
                f"- `{name}`: only {d['index_share']:.0%} of contact frames assigned "
                "to the index finger"
            )
    md += [
        "## 6. Flags (check these by eye before training)\n",
        "\n".join(flags) if flags else "None.",
        "",
    ]

    md += [
        "## 7. Disk\n",
        table(
            ["path", "size"],
            [[p, du(p)] for p in ("data/egopressure/raw", "data/pvdb/raw", "data/processed")],
        ),
        "",
    ]
    OUT.write_text("\n".join(md) + "\n")
    print(f"wrote {OUT} ({len(per)} sequences, {len(flags)} flags)")


if __name__ == "__main__":
    main()
