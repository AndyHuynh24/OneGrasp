"""Turn downloaded raw data into the canonical format (data/processed) and index it.

Stages (each skips work that is already done, so re-running is cheap):
    build      raw sequence (with a .done download marker) -> meta.json, pressure/labels/fingers.npz
    landmarks  MediaPipe on every view -> landmarks_<view>.npz
    fingers    PVDB per-finger labels from the overhead landmarks
    index      data/processed/index.csv (one row per sequence, with split and counts)

Usage:
    python scripts/prepare_data.py                    # all stages once
    python scripts/prepare_data.py --watch            # repeat until downloads finish, then once
    python scripts/prepare_data.py --stages build index --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from touch.data import canonical  # noqa: E402
from touch.data.catalog import PROCESSED_ROOT, load_catalog  # noqa: E402
from touch.data.names import parse_sequence_name  # noqa: E402
from touch.data.pressure import contact_segments, count_transitions  # noqa: E402

EP_RAW = Path("data/egopressure/raw")
PV_RAW = Path("data/pvdb/raw")
LOG = PROCESSED_ROOT / "prepare_errors.log"


def _log_error(what: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%F %T')}  {what}\n{traceback.format_exc()}\n")


def _ep_out(seq_dir: Path) -> Path:
    pid = parse_sequence_name(seq_dir.name).participant
    return PROCESSED_ROOT / "egopressure" / f"ep_{pid:03d}" / seq_dir.name


def _pv_out(seq_dir: Path) -> Path:
    return PROCESSED_ROOT / "pvdb" / f"pv_{seq_dir.parent.name.split('_')[0]}" / seq_dir.name


def _stale(raw_done: Path, out_meta: Path) -> bool:
    if not out_meta.exists() or out_meta.stat().st_mtime < raw_done.stat().st_mtime:
        return True
    return json.loads(out_meta.read_text()).get("format_version") != canonical.FORMAT_VERSION


def _build_one(kind: str, seq_dir: str) -> str:
    try:
        if kind == "ep":
            canonical.build_egopressure(Path(seq_dir), PROCESSED_ROOT)
        else:
            canonical.build_pvdb(Path(seq_dir), PROCESSED_ROOT)
        return f"built {seq_dir}"
    except Exception:
        _log_error(f"build {seq_dir}")
        return f"ERROR build {seq_dir}"


def _landmarks_one(seq_root: str, view: str) -> str:
    from touch.data import landmarks

    try:
        landmarks.run_sequence(Path(seq_root), view)
        return f"landmarks {seq_root} {view}"
    except Exception:
        _log_error(f"landmarks {seq_root} {view}")
        return f"ERROR landmarks {seq_root} {view}"


def _run(tasks: list[tuple], fn, workers: int) -> None:
    if not tasks:
        return
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(fn, *t) for t in tasks]
        for k, fut in enumerate(as_completed(futs), 1):
            msg = fut.result()
            if msg.startswith("ERROR") or k % 25 == 0 or k == len(tasks):
                print(f"  [{k}/{len(tasks)} {time.time() - t0:.0f}s] {msg}", flush=True)


def stage_build(workers: int) -> None:
    tasks = []
    for done in sorted(EP_RAW.glob("*/.done")):
        if _stale(done, _ep_out(done.parent) / "meta.json"):
            tasks.append(("ep", str(done.parent)))
    for done in sorted(PV_RAW.glob("*/*/*/.done")):
        if _stale(done, _pv_out(done.parent) / "meta.json"):
            tasks.append(("pv", str(done.parent)))
    print(f"build: {len(tasks)} sequences", flush=True)
    _run(tasks, _build_one, workers)


def stage_landmarks(workers: int) -> None:
    tasks = []
    for s in load_catalog():
        for view in s.views:
            out = s.root / f"landmarks_{view}.npz"
            # Rebuilding labels does not change the frames, so only redo MediaPipe when the
            # file is missing or its frame count no longer matches.
            if not out.exists():
                tasks.append((str(s.root), view))
                continue
            with np.load(out) as z:
                if len(z["primary"]) != len(s):
                    tasks.append((str(s.root), view))
    print(f"landmarks: {len(tasks)} sequence-views", flush=True)
    _run(tasks, _landmarks_one, workers)


def stage_fingers() -> None:
    n = 0
    for s in load_catalog(source="pvdb"):
        lm = s.root / "landmarks_overhead.npz"
        fp = s.root / "fingers.npz"
        pending = s.meta.get("finger_source", "").startswith("pending")
        if lm.exists() and (pending or not fp.exists() or fp.stat().st_mtime < lm.stat().st_mtime):
            try:
                canonical.build_pvdb_fingers(s.root)
                n += 1
            except Exception:
                _log_error(f"fingers {s.root}")
    print(f"fingers: {n} PVDB sequences", flush=True)


def stage_index() -> Path:
    rows = []
    for s in load_catalog():
        L, F, m = s.labels, s.fingers, s.meta
        down, up = count_transitions(L["contact"])
        segs = contact_segments(L["contact"])
        row = {
            "source": s.source,
            "participant": s.participant,
            "split": s.split or "",
            "sequence": s.name,
            "family": m["family"],
            "variant": m["variant"],
            "hand": m["hand"],
            "take": m["take"],
            "fps": m["fps"],
            "n_frames": len(s),
            "contact_frames": int(L["contact"].sum()),
            "touch_downs": down,
            "touch_ups": up,
            "presses_ge_3_frames": sum(1 for a, b in segs if b - a >= 3),
            "max_kpa": round(float(L["max_kpa"].max()), 2),
            "views": "+".join(s.views),
        }
        if "has_anno" in L:
            row["anno_frames"] = int(L["has_anno"].sum())
        if F is not None:
            fd = [count_transitions(F["contact"][:, k] & F["valid"])[0] for k in range(5)]
            row.update(
                {
                    "finger_valid_frames": int(F["valid"].sum()),
                    "index_contact_frames": int((F["contact"][:, 1] & F["valid"]).sum()),
                    "index_touch_downs": fd[1],
                    "other_finger_touch_downs": sum(fd) - fd[1],
                }
            )
        for view in s.views:
            if s.has_landmarks(view):
                z = s.landmarks(view)
                row[f"lm_{view}_detect"] = round(float((z["primary"] >= 0).mean()), 3)
        rows.append(row)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    out = PROCESSED_ROOT / "index.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"index: {len(rows)} sequences -> {out}", flush=True)
    return out


def downloads_running() -> bool:
    r = subprocess.run(["pgrep", "-f", "download_(egopressure|pvdb)"], capture_output=True)
    return r.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stages", nargs="+", default=["build", "landmarks", "fingers", "index"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--watch", action="store_true", help="loop until downloads finish")
    ap.add_argument("--interval", type=int, default=600, help="seconds between --watch rounds")
    args = ap.parse_args()

    while True:
        running = downloads_running()
        t0 = time.time()
        if "build" in args.stages:
            stage_build(args.workers)
        if "landmarks" in args.stages:
            stage_landmarks(args.workers)
        if "fingers" in args.stages:
            stage_fingers()
        if "index" in args.stages:
            stage_index()
        print(f"round done in {time.time() - t0:.0f}s; downloads running: {running}", flush=True)
        if not args.watch or not running:
            break
        time.sleep(max(0, args.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
