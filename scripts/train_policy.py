"""Train a robot policy straight from the demonstrations in Tiger Data, test it in MuJoCo on
scenes it never saw, and write the result back to Tiger Data (policy_runs, policy_episodes) for
the live dashboard (scripts/tiger_dashboard.py).

    python scripts/train_policy.py                 # train + val + test, results to Tiger
    python scripts/train_policy.py --epochs 5 --no-db --max-test 3    # quick check

Data: every successful demo (teleop + generated), one SELECT. Split by take (a take = one
teleop demo + the scenes generated from it): test takes and val takes are never trained on.
Validation picks the checkpoint (val loss) and is the only split looked at while developing;
test is rolled out once at the end (project rule). Model: touch.policy (MLP, action chunks +
temporal ensembling). Run folder: runs/policy_<time>/ (config + git commit, model, eval).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch import policy as PO  # noqa: E402
from touch.store.demo_store import DemoStore  # noqa: E402

COLS = ["action", "joint_pos", "joint_vel", "gripper_qpos", "eef_pos", "eef_quat", "cup_pos",
        "cup_quat"]  # fmt: skip
DDL = """
CREATE TABLE IF NOT EXISTS policy_runs (
  run_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT now(), name TEXT,
  n_train_demos INTEGER, n_train_steps INTEGER, n_test INTEGER, n_success INTEGER,
  metrics JSONB, config JSONB, git TEXT);
CREATE TABLE IF NOT EXISTS policy_episodes (
  run_id TEXT, demo_id TEXT, split TEXT, success BOOLEAN, why TEXT, metrics JSONB,
  created_at TIMESTAMPTZ DEFAULT now())
"""


def git_commit() -> str:
    """HEAD's commit from .git (read directly, no git command)."""
    g = Path(__file__).resolve().parents[1] / ".git"
    try:
        head = (g / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            p = g / ref
            if p.exists():
                return p.read_text().strip()
            for ln in (g / "packed-refs").read_text().splitlines():
                if ln.endswith(ref):
                    return ln.split()[0]
        return head
    except OSError:
        return "unknown"


def load_all(st: DemoStore) -> dict:
    """Every successful demo's steps, one SELECT -> {demo_id: {column: array}} + meta."""
    t = time.time()
    meta = {r[0]: {"take": r[1], "source": r[2]} for r in st._q(
        "SELECT demo_id, take, source FROM demos WHERE success ORDER BY demo_id")}  # fmt: skip
    rows = st._q(f"SELECT s.demo_id, {', '.join('s.' + c for c in COLS)} FROM demo_steps s "
                 "JOIN demos d USING (demo_id) WHERE d.success "
                 "ORDER BY s.demo_id, s.step")  # fmt: skip
    demos: dict = {}
    for r in rows:
        demos.setdefault(r[0], []).append(r[1:])
    out = {}
    for did, rs in demos.items():
        out[did] = {c: np.array([st._unarr(x[j]) for x in rs], np.float32)
                    for j, c in enumerate(COLS)}  # fmt: skip
    print(f"read {len(out)} demos, {len(rows)} steps from Tiger Data in {time.time() - t:.1f} s",
          flush=True)  # fmt: skip
    return {"demos": out, "meta": meta, "n_rows": len(rows)}


def split_takes(meta: dict, seed: int, frac: float) -> dict[str, str]:
    takes = sorted({m["take"] for m in meta.values()})
    rng = random.Random(seed)
    rng.shuffle(takes)
    n = max(2, round(frac * len(takes)))
    test, val = set(takes[:n]), set(takes[n : 2 * n])
    return {t: "test" if t in test else "val" if t in val else "train" for t in takes}


def stack(demos: dict, ids: list[str], K: int):
    parts = [PO.demo_samples(demos[i], K) for i in ids]
    return tuple(np.concatenate([p[j] for p in parts]) for j in range(3))


def evaluate(pol, st: DemoStore, ids: list[str], extra: int, split: str) -> list[dict]:
    pol.eval()
    out = []
    for i, did in enumerate(ids):
        d = st.load_demo(did)
        a = d["arrays"]
        t = time.time()
        r = PO.rollout(pol, d["xml"], a["state"][0], PO.goal_of(a), len(a["action"]) + extra)
        out.append({"demo_id": did, "split": split, **r})
        print(f"  [{split} {i + 1}/{len(ids)}] {did}: {'OK ' if r['success'] else 'FAIL'} "
              f"{r['why']:24s} place {r['place_err_cm']} cm, lift {r['lift_cm']} cm "
              f"({time.time() - t:.1f} s)", flush=True)  # fmt: skip
    return out


def summary(eps: list[dict]) -> dict:
    if not eps:
        return {}
    ok = [e for e in eps if e["success"]]
    return {"n": len(eps), "success": len(ok), "rate": round(len(ok) / len(eps), 3),
            "place_err_cm_median": float(np.median([e["place_err_cm"] for e in eps])),
            "why": {w: sum(e["why"] == w for e in eps)
                    for w in {e["why"] for e in eps}}}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--use-vel", action="store_true", help="give the policy joint velocities")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--split-frac", type=float, default=0.15, help="share of takes per val/test")
    ap.add_argument("--extra-steps", type=int, default=60, help="rollout = demo length + this")
    ap.add_argument("--max-val", type=int, default=20)
    ap.add_argument("--max-test", type=int, default=0, help="0 = every test demo")
    ap.add_argument("--no-db", action="store_true", help="do not write results to Tiger")
    ap.add_argument("--skip-test", action="store_true",
                    help="development: validation only (the test split stays unseen)")  # fmt: skip
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    st = DemoStore(args.url)
    assert st.pg, "needs Tiger Data ($TIGER_DATABASE_URL)"
    data = load_all(st)
    split = split_takes(data["meta"], args.seed, args.split_frac)
    ids = {s: sorted(d for d, m in data["meta"].items() if split[m["take"]] == s
                     and d in data["demos"]) for s in ("train", "val", "test")}  # fmt: skip
    print({s: f"{len(v)} demos" for s, v in ids.items()},
          {s: sorted({data["meta"][d]["take"] for d in v}) for s, v in ids.items() if s != "train"},
          flush=True)  # fmt: skip
    Xtr, Atr, Mtr = stack(data["demos"], ids["train"], args.chunk)
    Xva, Ava, Mva = stack(data["demos"], ids["val"], args.chunk)
    pol = PO.Policy(args.chunk, args.hidden, args.layers, args.use_vel)
    pol.set_norm(Xtr, Atr)
    opt = torch.optim.AdamW(pol.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * max(1, len(Xtr) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    tX, tA, tM = (torch.as_tensor(v) for v in (Xtr, Atr, Mtr))
    vX, vA, vM = (torch.as_tensor(v) for v in (Xva, Ava, Mva))
    run = Path("runs") / f"policy_{time.strftime('%Y%m%d-%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    cfg = {**vars(args), "git": git_commit(), "train_demos": len(ids["train"]),
           "train_steps": int(len(Xtr)), "takes": split}  # fmt: skip
    (run / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    best, t0 = 1e9, time.time()
    for ep in range(args.epochs):
        pol.train()
        perm = torch.randperm(len(tX))
        tot = 0.0
        for b in range(0, len(perm) - args.batch + 1, args.batch):
            k = perm[b : b + args.batch]
            loss = PO.loss_fn(pol, tX[k], tA[k], tM[k])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(k)
        pol.eval()
        with torch.no_grad():
            vl = float(PO.loss_fn(pol, vX, vA, vM))
        if vl < best:
            best = vl
            torch.save(pol.state_dict(), run / "model.pt")
        torch.save(pol.state_dict(), run / "model_last.pt")
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep + 1}/{args.epochs}: train {tot / len(tX):.4f}  val {vl:.4f}  "
                  f"(best {best:.4f}, {time.time() - t0:.0f} s)", flush=True)  # fmt: skip
    pol.load_state_dict(torch.load(run / "model.pt"))
    torch.set_num_threads(1)
    print("validation rollouts (MuJoCo):", flush=True)
    val_eps = evaluate(pol, st, ids["val"][: args.max_val], args.extra_steps, "val")
    test_ids = ids["test"][: args.max_test] if args.max_test else ids["test"]
    if args.skip_test:
        test_ids = []
    print("test rollouts (MuJoCo, takes never trained on):", flush=True)
    test_eps = evaluate(pol, st, test_ids, args.extra_steps, "test")
    res = {"val": summary(val_eps), "test": summary(test_eps), "best_val_loss": best,
           "train_seconds": round(time.time() - t0)}  # fmt: skip
    (run / "eval.json").write_text(json.dumps({**res, "episodes": val_eps + test_eps}, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    if not args.no_db and test_eps:
        cur = st.con.cursor()
        for stmt in filter(str.strip, DDL.split(";")):
            cur.execute(stmt)
        rid = uuid.uuid4().hex[:10]
        t = res["test"]
        cur.execute("INSERT INTO policy_runs (run_id, name, n_train_demos, n_train_steps, n_test,"
                    " n_success, metrics, config, git) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (rid, f"BC-MLP chunk {args.chunk} ({run.name})", len(ids["train"]),
                     int(len(Xtr)), t.get("n", 0), t.get("success", 0),
                     json.dumps({**t, "val": res["val"]}), json.dumps(cfg, default=str),
                     cfg["git"]))  # fmt: skip
        for e in val_eps + test_eps:
            m = {k: v for k, v in e.items() if k not in ("demo_id", "split", "success", "why")}
            cur.execute("INSERT INTO policy_episodes (run_id, demo_id, split, success, why, "
                        "metrics) VALUES (%s,%s,%s,%s,%s,%s)",
                        (rid, e["demo_id"], e["split"], e["success"], e["why"],
                         json.dumps(m)))  # fmt: skip
        st.con.commit()
        print(f"results written to Tiger Data: policy_runs {rid}", flush=True)
    print(f"run folder: {run}", flush=True)


if __name__ == "__main__":
    main()
