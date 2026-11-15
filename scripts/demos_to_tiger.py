"""Upload the robot demonstrations (teleop + generated) from the robomimic HDF5 into Tiger Data
(touch.store.demo_store; docs/TIGER_DATA.md). Safe to re-run: demos stored before are skipped.

    python scripts/demos_to_tiger.py                       # data/panda_demos/cup_pick_place.hdf5
    python scripts/demos_to_tiger.py --compress            # + compress into the columnstore now
    python scripts/demos_to_tiger.py --skip demo_148 demo_149
Database: $TIGER_DATABASE_URL (else a local sqlite file; --url to choose).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch.store.demo_store import DemoStore, demo_from_group  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", type=Path, default=Path("data/panda_demos/cup_pick_place.hdf5"))
    ap.add_argument("--dataset", default=None, help="name in the database (default: file stem)")
    ap.add_argument("--url", default=None, help="database URL (default $TIGER_DATABASE_URL)")
    ap.add_argument("--skip", nargs="*", default=[], help="demo keys not to upload")
    ap.add_argument("--compress", action="store_true", help="compress demo_steps afterwards")
    args = ap.parse_args()
    name = args.dataset or args.hdf5.stem
    st = DemoStore(args.url)
    print(f"database: {'Tiger Data (Postgres)' if st.pg else st.url}", flush=True)
    st.create_schema()
    t0, n_new, n_old, steps = time.time(), 0, 0, 0
    with h5py.File(args.hdf5, "r") as f:
        keys = sorted((k for k in f["data"] if k.startswith("demo_")),
                      key=lambda k: int(k.split("_")[1]))  # fmt: skip
        for i, k in enumerate(keys):
            if k in args.skip:
                continue
            t = time.time()
            d = demo_from_group(f["data"][k], name, k)
            did = st.add_demo(d)
            if did is None:
                n_old += 1
                continue
            n_new += 1
            steps += len(d["arrays"]["action"])
            print(f"[{i + 1}/{len(keys)}] {did} ({d['attrs'].get('source')}, "
                  f"{len(d['arrays']['action'])} steps) {1e3 * (time.time() - t):.0f} ms",
                  flush=True)  # fmt: skip
    dt = time.time() - t0
    print(f"uploaded {n_new} demos ({steps} steps) in {dt:.1f} s; {n_old} were stored before",
          flush=True)  # fmt: skip
    if args.compress:
        print(f"compressed {st.compress_now()} chunks", flush=True)
    print("stats:", st.stats(), flush=True)
    st.close()


if __name__ == "__main__":
    main()
