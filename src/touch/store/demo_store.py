"""DemoStore: robot demonstrations (teleop + MimicGen-style generated) in Tiger Data.

The same demos as the robomimic HDF5 (touch.demogen.append_demos), as SQL:
  demo_scenes  one MuJoCo scene (MJCF) per unique scene, by content hash (shared by demos)
  demos        one demonstration: source (teleop | mimicgen), the teleop demo a generated one
               came from, take, success, cup start / place target, steps, attrs (JSON)
  demo_steps   one 30 Hz control step: actions (joint targets + gripper, TCP command), the
               robot / cup observations and the full MuJoCo state (qpos, qvel); a few scalars
               for SQL (cup_z, eef_speed, gripper_closed). Tiger Data: a hypertable on `time`
               (= the demo's start + step / 30 s), compressed into the columnstore per demo
               (segmentby demo_id, orderby step); demo_steps_1s = a continuous aggregate (per
               demo per second: steps, cup height, gripper, speed) for instant dashboards.

Backends by URL: postgres(ql):// = Tiger Data (TimescaleDB); sqlite:///path = local file (tests,
offline; no hypertable / compression / aggregate). Default: $TIGER_DATABASE_URL, else
sqlite:///data/demo_store.sqlite.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

HZ = 30.0
DEFAULT_SQLITE = "sqlite:///data/demo_store.sqlite"
REPO = Path(__file__).resolve().parents[3]

# per-step array columns: (column, HDF5 path in the demo group)
STEP_ARRAYS = [
    ("action", "actions"),
    ("action_eef", "actions_eef"),
    ("joint_pos", "obs/robot0_joint_pos"),
    ("joint_vel", "obs/robot0_joint_vel"),
    ("gripper_qpos", "obs/robot0_gripper_qpos"),
    ("eef_pos", "obs/robot0_eef_pos"),
    ("eef_quat", "obs/robot0_eef_quat"),
    ("cup_pos", "obs/cup_pos"),
    ("cup_quat", "obs/cup_quat"),
    ("state", "states"),
]
STEP_SCALARS = ["reward", "done", "cup_z", "eef_speed", "gripper_closed"]
STEP_COLS = ["time", "demo_id", "step", *[c for c, _ in STEP_ARRAYS], *STEP_SCALARS]
DEMO_COLS = ["demo_id", "dataset", "demo_key", "created_at", "source", "success", "src_demo",
             "take", "scene_hash", "num_steps", "hz", "cup_xy", "place_xy", "lying", "attrs",
             "content_hash"]  # fmt: skip

SCHEMA = """
CREATE TABLE IF NOT EXISTS demo_scenes (
  scene_hash TEXT PRIMARY KEY, created_at {ts}, mjcf TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS demos (
  demo_id TEXT PRIMARY KEY, dataset TEXT, demo_key TEXT, created_at {ts}, source TEXT,
  success BOOLEAN, src_demo TEXT, take TEXT, scene_hash TEXT, num_steps INTEGER, hz REAL,
  cup_xy {real_arr}, place_xy {real_arr}, lying BOOLEAN, attrs {json}, content_hash TEXT);
CREATE TABLE IF NOT EXISTS demo_steps (
  time {ts} NOT NULL, demo_id TEXT NOT NULL, step INTEGER NOT NULL,
  action {real_arr}, action_eef {real_arr}, joint_pos {real_arr}, joint_vel {real_arr},
  gripper_qpos {real_arr}, eef_pos {real_arr}, eef_quat {real_arr}, cup_pos {real_arr},
  cup_quat {real_arr}, state {dbl_arr}, reward REAL, done BOOLEAN, cup_z REAL, eef_speed REAL,
  gripper_closed BOOLEAN);
CREATE INDEX IF NOT EXISTS demo_steps_demo ON demo_steps (demo_id, step)
"""

PG_TIMESCALE = [
    "SELECT create_hypertable('demo_steps', 'time', chunk_time_interval => INTERVAL '7 days', "
    "if_not_exists => TRUE, migrate_data => TRUE)",
    "ALTER TABLE demo_steps SET (timescaledb.compress, timescaledb.compress_segmentby = "
    "'demo_id', timescaledb.compress_orderby = 'step')",
    # columnstore: chunks older than a day are compressed automatically; compress_now() for
    # the rest (e.g. right after a bulk upload)
    "SELECT add_compression_policy('demo_steps', INTERVAL '1 day', if_not_exists => TRUE)",
]
PG_CAGG = """
CREATE MATERIALIZED VIEW IF NOT EXISTS demo_steps_1s
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT time_bucket(INTERVAL '1 second', time) AS bucket, demo_id,
       count(*) AS steps, max(cup_z) AS cup_z_max, min(cup_z) AS cup_z_min,
       avg(eef_speed) AS eef_speed_avg, max(eef_speed) AS eef_speed_max,
       bool_or(gripper_closed) AS gripper_closed, max(reward) AS reward
FROM demo_steps GROUP BY bucket, demo_id WITH NO DATA
"""
PG_CAGG_POLICY = ("SELECT add_continuous_aggregate_policy('demo_steps_1s', start_offset => NULL,"
                  " end_offset => INTERVAL '1 minute', schedule_interval => INTERVAL '1 minute',"
                  " if_not_exists => TRUE)")  # fmt: skip


def local_mjcf(xml: str) -> str:
    """A stored MJCF with this machine's paths: the mesh / asset directories point into the
    repo's third_party/ (the demo was recorded on another machine / folder)."""
    return re.sub(r'(meshdir|texturedir|assetdir)="[^"]*?(third_party/[^"]*)"',
                  lambda m: f'{m.group(1)}="{REPO / m.group(2)}"', xml)  # fmt: skip


def demo_from_group(g, dataset: str, key: str) -> dict:
    """One robomimic HDF5 demo group -> the dict DemoStore.add_demo takes."""
    attrs = {}
    for k, v in g.attrs.items():
        if k in ("model_file", "num_samples"):
            continue
        v = v.item() if hasattr(v, "item") else v
        if isinstance(v, str) and v[:1] in "[{":
            try:
                v = json.loads(v)
            except ValueError:
                pass
        attrs[k] = v
    return {"dataset": dataset, "key": key, "xml": str(g.attrs["model_file"]), "attrs": attrs,
            "arrays": {c: np.asarray(g[p]) for c, p in STEP_ARRAYS},
            "reward": np.asarray(g["rewards"]), "done": np.asarray(g["dones"])}  # fmt: skip


def _take_time(take: str | None) -> datetime | None:
    try:
        return datetime.strptime(str(take), "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None


class DemoStore:
    def __init__(self, url: str | None = None):
        self.url = url or os.environ.get("TIGER_DATABASE_URL") or DEFAULT_SQLITE
        self.pg = self.url.startswith(("postgres://", "postgresql://"))
        if self.pg:
            import psycopg

            self.con = psycopg.connect(self.url, autocommit=False)
        else:
            path = Path(self.url.removeprefix("sqlite:///"))
            path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(path, check_same_thread=False)
        self.ph = "%s" if self.pg else "?"

    def close(self) -> None:
        self.con.close()

    # ---------------------------------------------------------------- schema
    def create_schema(self) -> None:
        """Tables (+ on Tiger: hypertable, columnstore compression, continuous aggregate);
        safe to run again."""
        if self.pg:
            t = dict(ts="TIMESTAMPTZ", json="JSONB", real_arr="REAL[]",
                     dbl_arr="DOUBLE PRECISION[]")  # fmt: skip
        else:
            t = dict(ts="TEXT", json="TEXT", real_arr="TEXT", dbl_arr="TEXT")
        cur = self.con.cursor()
        for stmt in filter(str.strip, SCHEMA.format(**t).split(";")):
            cur.execute(stmt)
        self.con.commit()
        if not self.pg:
            return
        for stmt in PG_TIMESCALE:
            try:
                cur.execute(stmt)
                self.con.commit()
            except Exception as ex:  # already set (compression on an existing table, ...)
                self.con.rollback()
                if "already" not in str(ex).lower():
                    raise
        self.con.autocommit = True  # a continuous aggregate cannot be created in a transaction
        try:
            cur = self.con.cursor()
            cur.execute(PG_CAGG)
            cur.execute(PG_CAGG_POLICY)
        finally:
            self.con.autocommit = False

    # ---------------------------------------------------------------- write
    def has_demo(self, demo_id: str) -> bool:
        return bool(self._q(f"SELECT 1 FROM demos WHERE demo_id = {self.ph}", (demo_id,)))

    def add_demo(self, d: dict, skip_existing: bool = True) -> str | None:
        """d: demo_from_group() output -> its demo_id ("<dataset>:<key>"); None = stored before."""
        did = f"{d['dataset']}:{d['key']}"
        if skip_existing and self.has_demo(did):
            return None
        a = d["arrays"]
        T = len(a["action"])
        assert all(len(v) == T for v in a.values()), {k: len(v) for k, v in a.items()}
        at = d["attrs"]
        xml = d["xml"]
        sh = hashlib.sha256(xml.encode()).hexdigest()[:16]
        cur = self.con.cursor()
        if not self._q(f"SELECT 1 FROM demo_scenes WHERE scene_hash = {self.ph}", (sh,)):
            cur.execute(f"INSERT INTO demo_scenes VALUES ({self.ph}, {self.ph}, {self.ph})",
                        (sh, self._ts(datetime.now(UTC)), xml))  # fmt: skip
        content = hashlib.sha256(b"".join(np.ascontiguousarray(v).tobytes()
                                          for v in a.values())).hexdigest()[:16]  # fmt: skip
        t0 = _take_time(at.get("take")) or datetime.now(UTC)
        if at.get("source") != "teleop":  # generated demos: after their teleop source
            t0 += timedelta(seconds=1 + int(str(d["key"]).split("_")[-1]) % 10_000 * 1e-3)
        row = (did, d["dataset"], d["key"], self._ts(t0), at.get("source"),
               bool(at.get("success", False)), at.get("src_demo") and f"{d['dataset']}:"
               f"{at['src_demo']}", at.get("take"), sh, T, HZ, self._arr(at.get("cup_xy")),
               self._arr(at.get("place_xy")), bool(at.get("lying", False)),
               json.dumps(_jsonable(at)), content)  # fmt: skip
        cols = ", ".join(DEMO_COLS)
        cur.execute(f"INSERT INTO demos ({cols}) VALUES ({', '.join([self.ph] * len(row))})", row)
        eef = a["eef_pos"].astype(np.float64)
        speed = np.r_[0.0, np.linalg.norm(np.diff(eef, axis=0), axis=1) * HZ]
        closed = a["action"][:, -1] > 0
        rows = []
        for i in range(T):
            rows.append((self._ts(t0 + timedelta(seconds=i / HZ)), did, i,
                         *[self._arr(a[c][i]) for c, _ in STEP_ARRAYS],
                         float(d["reward"][i]), bool(d["done"][i]), float(a["cup_pos"][i][2]),
                         float(speed[i]), bool(closed[i])))  # fmt: skip
        if self.pg:
            with cur.copy(f"COPY demo_steps ({', '.join(STEP_COLS)}) FROM STDIN") as cp:
                for r in rows:
                    cp.write_row(r)
        else:
            cur.executemany(f"INSERT INTO demo_steps ({', '.join(STEP_COLS)}) VALUES "
                            f"({', '.join(['?'] * len(STEP_COLS))})", rows)  # fmt: skip
        self.con.commit()
        return did

    def compress_now(self) -> int:
        """Tiger: compress every chunk of demo_steps into the columnstore now -> chunks done."""
        if not self.pg:
            return 0
        rows = self._q("SELECT compress_chunk(c, if_not_compressed => TRUE) "
                       "FROM show_chunks('demo_steps') c")  # fmt: skip
        self.con.commit()
        return len(rows)

    # ---------------------------------------------------------------- read
    def list_demos(self, source: str | None = None, success: bool | None = None,
                   take: str | None = None, dataset: str | None = None,
                   limit: int = 50) -> list[dict]:  # fmt: skip
        """Newest first: [{demo_id, source, success, take, num_steps, src_demo, created_at}]."""
        w, p = [], []
        for col, v in (("source", source), ("success", success), ("take", take),
                       ("dataset", dataset)):  # fmt: skip
            if v is not None:
                w.append(f"{col} = {self.ph}")
                p.append(v)
        sql = ("SELECT demo_id, source, success, take, num_steps, src_demo, created_at FROM demos"
               + (" WHERE " + " AND ".join(w) if w else "")
               + f" ORDER BY created_at DESC, demo_id LIMIT {int(limit)}")  # fmt: skip
        keys = ["demo_id", "source", "success", "take", "num_steps", "src_demo", "created_at"]
        return [dict(zip(keys, r, strict=True)) for r in self._q(sql, p)]

    def load_demo(self, demo_id: str) -> dict:
        """-> {"demo_id", "xml" (local paths), "attrs", "arrays": {column: [T, ...]}, "reward",
        "done"} (the arrays exactly as stored)."""
        meta = self._q(f"SELECT d.attrs, s.mjcf, d.num_steps FROM demos d JOIN demo_scenes s "
                       f"ON s.scene_hash = d.scene_hash WHERE d.demo_id = {self.ph}",
                       (demo_id,))  # fmt: skip
        if not meta:
            raise KeyError(demo_id)
        attrs, xml, T = meta[0]
        cols = [c for c, _ in STEP_ARRAYS]
        rows = self._q(f"SELECT {', '.join(cols)}, reward, done FROM demo_steps "
                       f"WHERE demo_id = {self.ph} ORDER BY step", (demo_id,))  # fmt: skip
        assert len(rows) == T, (len(rows), T)
        arrays = {}
        for j, (c, _) in enumerate(STEP_ARRAYS):
            dt = np.float64 if c == "state" else np.float32
            arrays[c] = np.array([self._unarr(r[j]) for r in rows], dt)
        return {"demo_id": demo_id, "xml": local_mjcf(xml), "attrs": _loads(attrs),
                "arrays": arrays, "reward": np.array([r[-2] for r in rows], np.float32),
                "done": np.array([bool(r[-1]) for r in rows])}  # fmt: skip

    def stats(self) -> dict:
        """Counts for a dashboard; on Tiger also the columnstore compression of demo_steps."""
        n, ok, tel, gen = self._q("SELECT count(*), sum(CASE WHEN success THEN 1 ELSE 0 END), "
                                  "sum(CASE WHEN source = 'teleop' THEN 1 ELSE 0 END), "
                                  "sum(CASE WHEN source <> 'teleop' THEN 1 ELSE 0 END) "
                                  "FROM demos")[0]  # fmt: skip
        steps = self._q("SELECT count(*) FROM demo_steps")[0][0]
        out = {"demos": int(n or 0), "successful": int(ok or 0), "teleop": int(tel or 0),
               "generated": int(gen or 0), "steps": int(steps or 0)}  # fmt: skip
        if self.pg:
            try:
                r = self._q("SELECT sum(before_compression_total_bytes), "
                            "sum(after_compression_total_bytes) "
                            "FROM chunk_compression_stats('demo_steps') "
                            "WHERE compression_status = 'Compressed'")[0]  # fmt: skip
                self.con.commit()
                if r[0]:
                    out["bytes_before"], out["bytes_after"] = int(r[0]), int(r[1])
                    out["compression_ratio"] = round(float(r[0]) / max(float(r[1]), 1), 1)
                tot = self._q("SELECT hypertable_size('demo_steps')")[0][0]
                out["table_bytes"] = int(tot or 0)
            except Exception:
                self.con.rollback()
        return out

    def demo_summary(self, limit: int = 20) -> list[dict]:
        """Per demo, from the continuous aggregate on Tiger (demo_steps on sqlite): duration
        (s), peak cup height (m), mean / max TCP speed (m/s), gripper ever closed."""
        src = "demo_steps_1s" if self.pg else None
        if src:
            sql = (
                "SELECT a.demo_id, sum(a.steps) / 30.0, max(a.cup_z_max), avg(a.eef_speed_avg),"
                " max(a.eef_speed_max), bool_or(a.gripper_closed), d.source, d.success "
                "FROM demo_steps_1s a JOIN demos d USING (demo_id) GROUP BY a.demo_id, "
                f"d.source, d.success, d.created_at ORDER BY d.created_at DESC LIMIT {limit}"
            )
        else:
            sql = ("SELECT s.demo_id, count(*) / 30.0, max(s.cup_z), avg(s.eef_speed), "
                   "max(s.eef_speed), max(s.gripper_closed), d.source, d.success FROM demo_steps "
                   "s JOIN demos d USING (demo_id) GROUP BY s.demo_id ORDER BY d.created_at DESC "
                   f"LIMIT {limit}")  # fmt: skip
        keys = ["demo_id", "seconds", "cup_z_max", "speed_avg", "speed_max", "gripper_closed",
                "source", "success"]  # fmt: skip
        rows = self._q(sql)
        if self.pg:
            self.con.commit()
        return [dict(zip(keys, r, strict=True)) for r in rows]

    # ---------------------------------------------------------------- helpers
    def _q(self, sql: str, params=()) -> list[tuple]:
        cur = self.con.cursor()
        cur.execute(sql, tuple(params))
        return cur.fetchall() if cur.description else []

    def _ts(self, t: datetime):
        return t if self.pg else t.isoformat()

    def _arr(self, v):
        if v is None:
            return None
        v = np.asarray(v, np.float64).ravel()
        return v.tolist() if self.pg else json.dumps(v.tolist())

    def _unarr(self, v) -> list:
        return json.loads(v) if isinstance(v, str) else list(v)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


def _loads(v):
    return json.loads(v) if isinstance(v, str) else (v or {})
