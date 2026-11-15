"""Read robot demonstrations from Tiger Data and reproduce them in MuJoCo in real time (30 Hz).

    python scripts/replay_from_tiger.py --list                     # what is stored
    python scripts/replay_from_tiger.py                            # newest 5 demos, physics re-run
    python scripts/replay_from_tiger.py --source teleop --n 3
    python scripts/replay_from_tiger.py --take 20261114-160812     # one take: teleop + generated
    python scripts/replay_from_tiger.py --id cup_pick_place:demo_0 --mode states
    python scripts/replay_from_tiger.py --check --n 50             # no window: re-run + compare

--mode actions (default): re-run the physics from the stored commands only, from the recorded
first state; prints where the cup ended vs the recording. --mode states: the recorded MuJoCo
states, exactly. Keys: space = pause, n = next demo, r = restart, q = quit.
Database: $TIGER_DATABASE_URL (--url to choose).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch.store import demo_replay as R  # noqa: E402
from touch.store.demo_store import DemoStore  # noqa: E402

HZ = 30.0


def label(img, s: str, y: int, scale: float = 0.7) -> None:
    cv2.putText(img, s, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, s, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def play(demo: dict, mode: str, size: tuple[int, int], speed: float) -> str:
    """Show one demo in real time -> "next" | "quit"."""
    import mujoco

    m, d = R.build(demo["xml"])
    rnd = mujoco.Renderer(m, size[1], size[0])
    a = demo["arrays"]
    T = len(a["action"])
    src = demo["attrs"].get("source")
    while True:
        t0, paused, restart = time.perf_counter(), False, False
        for t in R.steps(m, d, demo, mode):
            rnd.update_scene(d, camera="view")
            img = np.ascontiguousarray(rnd.render()[..., ::-1])
            label(img, f"{demo['demo_id']}  ({src})  from Tiger Data", 30)
            what = "physics re-run from stored commands" if mode == "actions" else "recorded states"
            grip = "closed" if a["action"][t, -1] > 0 else "open"
            label(img, f"{mode}: {what}   step {t + 1}/{T}  {t / HZ:4.1f} s   gripper {grip}   "
                       f"cup z {100 * a['cup_pos'][t, 2]:.1f} cm", size[1] - 16, 0.55)  # fmt: skip
            cv2.imshow("Tiger Data replay", img)
            while True:  # real time: wait for this step's slot (or a key)
                wait = t0 + (t + 1) / (HZ * speed) - time.perf_counter()
                k = cv2.waitKey(30 if paused else max(1, int(wait * 1000))) & 0xFF
                if k == ord(" "):
                    paused = not paused
                    t0 = time.perf_counter() - (t + 1) / (HZ * speed)
                elif k in (ord("q"), 27, ord("n")):
                    rnd.close()
                    return "next" if k == ord("n") else "quit"
                elif k == ord("r"):
                    restart = True
                if restart or (not paused and time.perf_counter() >= t0 + (t + 1) / (HZ * speed)):
                    break
            if restart:
                break
        if restart:
            continue
        if mode == "actions":
            print(f"  {demo['demo_id']}: {R.compare(m, d, demo)}", flush=True)
        k = cv2.waitKey(700) & 0xFF  # a short hold on the last frame
        rnd.close()
        return "quit" if k in (ord("q"), 27) else "next"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--id", nargs="*", default=None, help="demo ids (dataset:demo_k)")
    ap.add_argument("--source", choices=["teleop", "mimicgen"], default=None)
    ap.add_argument("--take", default=None, help="a take folder name (e.g. 20261114-160812)")
    ap.add_argument("--n", type=int, default=5, help="how many (newest first)")
    ap.add_argument("--mode", choices=["actions", "states"], default="actions")
    ap.add_argument("--size", type=int, nargs=2, default=[960, 540])
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time")
    ap.add_argument("--list", action="store_true", help="print what is stored and quit")
    ap.add_argument("--check", action="store_true", help="no window: re-run each and compare")
    args = ap.parse_args()
    st = DemoStore(args.url)
    print(f"database: {'Tiger Data (Postgres)' if st.pg else st.url}", flush=True)
    if args.id:
        ids = args.id
    else:
        rows = st.list_demos(source=args.source, take=args.take, success=True, limit=args.n)
        ids = [r["demo_id"] for r in rows]
    if args.list:
        print(st.stats())
        for r in st.list_demos(source=args.source, take=args.take, limit=args.n):
            print(f"  {r['demo_id']:28s} {r['source']:9s} take {r['take']}  "
                  f"{r['num_steps']} steps  from {r['src_demo'] or '-'}")  # fmt: skip
        return
    if not ids:
        sys.exit("no demos match")
    if args.check:
        errs = []
        for did in ids:
            t = time.perf_counter()
            demo = st.load_demo(did)
            t_load = time.perf_counter() - t
            m, d = R.build(demo["xml"])
            for _ in R.steps(m, d, demo, "actions"):
                pass
            c = R.compare(m, d, demo)
            errs.append(c["cup_end_vs_recorded_cm"])
            print(f"{did:28s} load {1e3 * t_load:4.0f} ms   cup end vs recorded "
                  f"{c['cup_end_vs_recorded_cm']:.1f} cm", flush=True)  # fmt: skip
        e = np.array(errs)
        print(f"{len(e)} demos: cup end vs recorded median {np.median(e):.1f} cm, max "
              f"{e.max():.1f} cm, within 1 cm {np.mean(e <= 1.0):.0%}")  # fmt: skip
        return
    for did in ids:
        t = time.perf_counter()
        demo = st.load_demo(did)
        print(f"{did}: loaded {len(demo['arrays']['action'])} steps in "
              f"{1e3 * (time.perf_counter() - t):.0f} ms", flush=True)  # fmt: skip
        if play(demo, args.mode, tuple(args.size), args.speed) == "quit":
            break
    cv2.destroyAllWindows()
    st.close()


if __name__ == "__main__":
    main()
