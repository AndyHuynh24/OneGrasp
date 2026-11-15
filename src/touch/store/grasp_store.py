"""GraspStore: write and read hand recordings for robot learning (docs/TIGER_DATA.md).

Backends (chosen by the URL):
  postgres://... / postgresql://...  Tiger Data (TimescaleDB): hand_frames, events = hypertables
  sqlite:///path.sqlite              local file, same tables (tests, offline, no account)
Default URL: $TIGER_DATABASE_URL, else sqlite:///data/grasp_store.sqlite.

Tables
  scenes       the 3D scene: camera calibration, table plane, objects (shape / mesh, pose),
               markers, and a robot-free MuJoCo scene (MJCF) of table + objects + camera
  sessions     one recording (-> scene_id)
  videos       where the recording's video lives (uri; NOT the video itself) + fps, size, hash;
               hand_frames.frame_idx indexes into it
  hand_frames  one hand in one video frame: model outputs, raw MediaPipe / model / MANO values,
               derived gripper-style pose, grab gate
  events       grab / release / object
  grasps       one grab -> release episode per hand

Conventions
  - metres, seconds, pixels; quaternions w x y z
  - frame = "table" (calibrated webcam: z up, table plane z = 0) or "camera" (camera axes)
  - t = seconds since the session started; time = session start + t (UTC)
  - raw model values (joints_model, mano_*) are the model's own: wrist-relative, camera axes,
    RIGHT-hand MANO (left hands are mirrored in x before the model); to place them:
    x_cam = mirror_if_left(x) + cam_t, x_frame = cam_to_world @ [x_cam, 1]
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

DEFAULT_SQLITE = "sqlite:///data/grasp_store.sqlite"
APPROACH_S = 1.5  # seconds before the grab exported as the approach
AFTER_S = 0.5  # seconds after the release exported with an episode
FLUSH_ROWS = 300
HANDS = ("left", "right")
N_VERTS = 778  # MANO


def gripper_pose(joints: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, float] | None:
    """21 joints [21, 3] -> (grasp point [3], quaternion w x y z [4], width m), None if no hand.
    Grasp point = robot_sim.hand_to_gripper origin + PAD_Z along the approach axis."""
    from touch import robot_sim as S

    if joints is None or not np.isfinite(joints).all():
        return None
    org, R, width = S.hand_to_gripper(np.asarray(joints, np.float64))
    q = Rot.from_matrix(R).as_quat()  # x y z w
    return org + S.PAD_Z * R[:, 2], q[[3, 0, 1, 2]], float(width)


SCHEMA = """
CREATE TABLE IF NOT EXISTS scenes (
    scene_id     TEXT PRIMARY KEY,
    created_at   {ts} NOT NULL,
    name         TEXT,
    frame        TEXT NOT NULL,          -- table | camera
    camera       {json},                 -- K, cam_to_world (3 x 4), image_size
    table_plane  {json},                 -- normal, offset_m (z = 0 for frame = table)
    objects      {json},                 -- name, shape, size_m | mesh_uri, pos, quat_wxyz
    markers      {json},                 -- calibration markers (ArUco ids, size)
    mjcf         TEXT                    -- MuJoCo XML: table + objects + camera, no robot
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    started_at   {ts} NOT NULL,
    source       TEXT NOT NULL,          -- live_webcam | hocap_val | biovl2 | ...
    video        TEXT,                   -- input name (camera index / dataset clip)
    frame        TEXT NOT NULL,          -- table | camera
    model        TEXT,                   -- hand model run
    git          TEXT,
    meta         {json}                  -- K, cam_to_world, objects, gate, notes
);
CREATE TABLE IF NOT EXISTS videos (
    video_id     TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    camera       TEXT,                   -- which camera of the session
    uri          TEXT NOT NULL,          -- local path / https / hf ... (the file is NOT here)
    kind         TEXT,                   -- mp4 | image_folder (uri = printf pattern)
    fps          REAL,                   -- nominal (hand_frames.t = true capture time)
    n_frames     INTEGER,
    width        INTEGER,
    height       INTEGER,
    codec        TEXT,
    size_bytes   BIGINT,
    sha256       TEXT,                   -- to check a downloaded copy is the same file
    t0           DOUBLE PRECISION        -- session time t of frame 0
);
CREATE TABLE IF NOT EXISTS hand_frames (
    time         {ts} NOT NULL,
    session_id   TEXT NOT NULL,
    t            DOUBLE PRECISION NOT NULL,
    hand         TEXT NOT NULL,          -- left | right
    present      BOOLEAN NOT NULL,
    state        SMALLINT,               -- 0 none, 1 touch, 2 hold (model argmax)
    p_none       REAL, p_touch REAL, p_hold REAL,
    gate_closed  BOOLEAN,                -- grab gate decision (holding?)
    finger_p     {real_arr},             -- 5 fingertip contact probabilities (thumb..little)
    grasp_x      DOUBLE PRECISION, grasp_y DOUBLE PRECISION, grasp_z DOUBLE PRECISION,
    quat_w       DOUBLE PRECISION, quat_x DOUBLE PRECISION, quat_y DOUBLE PRECISION,
    quat_z       DOUBLE PRECISION,
    width        DOUBLE PRECISION,
    joints       {real_arr},             -- 21 x 3 hand joints (session frame, m), flattened
    keypoints_2d {real_arr}              -- 21 x 2 image keypoints (px), flattened
);
CREATE INDEX IF NOT EXISTS hand_frames_session ON hand_frames (session_id, hand, t);
CREATE TABLE IF NOT EXISTS events (
    time         {ts} NOT NULL,
    session_id   TEXT NOT NULL,
    t            DOUBLE PRECISION NOT NULL,
    hand         TEXT NOT NULL,
    kind         TEXT NOT NULL,          -- grab | release | object
    object       TEXT,
    detail       {json}
);
CREATE INDEX IF NOT EXISTS events_session ON events (session_id, t);
CREATE TABLE IF NOT EXISTS grasps (
    grasp_id     TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    hand         TEXT NOT NULL,
    t_approach   DOUBLE PRECISION NOT NULL,
    t_grab       DOUBLE PRECISION NOT NULL,
    t_release    DOUBLE PRECISION,       -- NULL: still held when the session ended
    object       TEXT,
    grab_x       DOUBLE PRECISION, grab_y DOUBLE PRECISION, grab_z DOUBLE PRECISION,
    lift_m       DOUBLE PRECISION,       -- grasp-point rise above the grab height (90th pct)
    width_grab   DOUBLE PRECISION,
    n_frames     INTEGER
)
"""
# Columns added after the first release: create_schema() adds any that a database lacks.
# (name, postgres type, sqlite type)
_ARR = ("REAL[]", "TEXT")
ADDED_COLS = {
    "sessions": [("scene_id", "TEXT", "TEXT")],
    "hand_frames": [
        ("frame_idx", "INTEGER", "INTEGER"),  # frame number in the session video
        ("hand_score", "REAL", "REAL"),  # MediaPipe handedness confidence
        ("mp_landmarks", *_ARR),  # 21 x 3 raw MediaPipe: x, y px, relative depth
        ("joints_model", *_ARR),  # 21 x 3 model joints: wrist-relative, camera axes, m
        ("cam_t", *_ARR),  # 3: wrist position in camera coordinates, m
        ("mano_pose", *_ARR),  # 16 x 3 MANO joint rotations, axis-angle (right-hand MANO)
        ("mano_betas", *_ARR),  # 10 MANO shape
        ("contact_p", "BYTEA", "BLOB"),  # 778 per-vertex contact probabilities, uint8 (x / 255)
        ("gate_score", "REAL", "REAL"),  # grab-gate classifier score (before hysteresis)
        ("table_p", "REAL", "REAL"),  # model's hand-on-table probability (if it has one)
    ],
}
RAW_COLS = [c[0] for c in ADDED_COLS["hand_frames"]]
FRAME_COLS = ["time", "session_id", "t", "hand", "present", "state", "p_none", "p_touch",
              "p_hold", "gate_closed", "finger_p", "grasp_x", "grasp_y", "grasp_z", "quat_w",
              "quat_x", "quat_y", "quat_z", "width", "joints", "keypoints_2d",
              *RAW_COLS]  # fmt: skip
SESSION_COLS = ["session_id", "started_at", "source", "video", "frame", "model", "git", "meta",
                "scene_id"]  # fmt: skip
SCENE_COLS = ["scene_id", "created_at", "name", "frame", "camera", "table_plane", "objects",
              "markers", "mjcf"]  # fmt: skip
VIDEO_COLS = ["video_id", "session_id", "camera", "uri", "kind", "fps", "n_frames", "width",
              "height", "codec", "size_bytes", "sha256", "t0"]  # fmt: skip
GRASP_COLS = ["grasp_id", "session_id", "hand", "t_approach", "t_grab", "t_release", "object",
              "grab_x", "grab_y", "grab_z", "lift_m", "width_grab", "n_frames"]  # fmt: skip
# per-row array shapes (NaN-filled when NULL) for frames() / episode()
ARR_SHAPES = {"finger_p": (5,), "joints": (21, 3), "keypoints_2d": (21, 2),
              "mp_landmarks": (21, 3), "joints_model": (21, 3), "cam_t": (3,),
              "mano_pose": (16, 3), "mano_betas": (10,), "contact_p": (N_VERTS,)}  # fmt: skip
JSON_COLS = {"meta", "camera", "table_plane", "objects", "markers", "detail"}


class GraspStore:
    def __init__(self, url: str | None = None):
        self.url = url or os.environ.get("TIGER_DATABASE_URL") or DEFAULT_SQLITE
        self.pg = self.url.startswith(("postgres://", "postgresql://"))
        if self.pg:
            import psycopg

            self.con = psycopg.connect(self.url, autocommit=False)
        else:
            path = Path(self.url.removeprefix("sqlite:///"))
            path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(path)
        self.ph = "%s" if self.pg else "?"
        self._buf: list[tuple] = []
        self._live: dict = {}  # session_id -> per-hand running state

    # ---------------------------------------------------------------- schema
    def create_schema(self) -> None:
        """Create missing tables / columns (safe to run on an existing database)."""
        if self.pg:
            ddl = SCHEMA.format(ts="TIMESTAMPTZ", json="JSONB", real_arr="REAL[]")
        else:
            ddl = SCHEMA.format(ts="TEXT", json="TEXT", real_arr="TEXT")
        cur = self.con.cursor()
        for stmt in filter(str.strip, ddl.split(";")):
            cur.execute(stmt)
        for tab, cols in ADDED_COLS.items():
            have = self._columns(tab)
            for name, pg_t, sq_t in cols:
                if name not in have:
                    cur.execute(f"ALTER TABLE {tab} ADD COLUMN {name} {pg_t if self.pg else sq_t}")
        if self.pg:  # TimescaleDB: time-partition the frame and event tables
            for tab in ("hand_frames", "events"):
                cur.execute(f"SELECT create_hypertable('{tab}', 'time', if_not_exists => TRUE, "
                            "migrate_data => TRUE)")  # fmt: skip
        self.con.commit()

    def _columns(self, table: str) -> set[str]:
        if self.pg:
            rows = self._query("SELECT column_name FROM information_schema.columns "
                               "WHERE table_name = %s", (table,))  # fmt: skip
            return {r[0] for r in rows}
        return {r[1] for r in self._query(f"PRAGMA table_info({table})")}

    # ---------------------------------------------------------------- write
    def add_scene(self, name: str, frame: str, camera: dict | None = None,
                  objects: list[dict] | None = None, table_plane: dict | None = None,
                  markers: dict | None = None, mjcf: str | None = None) -> str:  # fmt: skip
        """The 3D scene of one or more sessions. objects: [{name, shape: cylinder | box | mesh,
        size_m (cylinder: [radius, half_height]; box: half sizes), mesh_uri, pos, quat_wxyz}].
        mjcf defaults to touch.store.scene.scene_mjcf(camera, objects) for frame = table."""
        from touch.store.scene import scene_mjcf

        assert frame in ("table", "camera"), frame
        scid = uuid.uuid4().hex[:12]
        if table_plane is None and frame == "table":
            table_plane = {"normal": [0, 0, 1], "offset_m": 0.0}
        if mjcf is None and frame == "table":
            mjcf = scene_mjcf(camera, objects or [])
        self._insert("scenes", SCENE_COLS,
                     (scid, self._ts(datetime.now(UTC)), name, frame, self._json(camera),
                      self._json(table_plane), self._json(objects or []), self._json(markers),
                      mjcf))  # fmt: skip
        self.con.commit()
        return scid

    def start_session(self, source: str, frame: str, video: str = "", model: str = "",
                      git: str = "", meta: dict | None = None,
                      started_at: datetime | None = None,
                      scene_id: str | None = None) -> str:  # fmt: skip
        assert frame in ("table", "camera"), frame
        sid = uuid.uuid4().hex[:12]
        t0 = started_at or datetime.now(UTC)
        self._insert("sessions", SESSION_COLS,
                     (sid, self._ts(t0), source, video, frame, model, git, self._json(meta or {}),
                      scene_id))  # fmt: skip
        self.con.commit()
        self._live[sid] = {"t0": t0, "hands": {h: _HandTrack() for h in HANDS}}
        return sid

    def add_video(self, sid: str, uri: str, kind: str = "mp4", camera: str = "main",
                  fps: float | None = None, n_frames: int | None = None,
                  width: int | None = None, height: int | None = None, codec: str | None = None,
                  t0: float = 0.0, checksum: bool = True) -> str:  # fmt: skip
        """Reference to the session's video (the file stays where it is). For a local mp4 the
        size and SHA-256 are filled in, so a downloaded copy can be checked."""
        size, sha = None, None
        p = Path(uri)
        if kind == "mp4" and p.is_file():
            size = p.stat().st_size
            if checksum:
                h = hashlib.sha256()
                with p.open("rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        h.update(block)
                sha = h.hexdigest()
        vid = uuid.uuid4().hex[:12]
        self._insert("videos", VIDEO_COLS,
                     (vid, sid, camera, uri, kind, fps, n_frames, width, height, codec, size, sha,
                      float(t0)))  # fmt: skip
        self.con.commit()
        return vid

    def add_frame(self, sid: str, t: float, hand: str, present: bool, state_probs=None,
                  finger_probs=None, gate_closed: bool | None = None, joints=None,
                  object_name: str | None = None, keypoints_2d=None,
                  raw: dict | None = None) -> None:  # fmt: skip
        """One hand in one frame. joints [21, 3] in the session frame (None / NaN = no 3D hand).
        raw: any of RAW_COLS (frame_idx, hand_score, mp_landmarks [21, 3], joints_model [21, 3],
        cam_t [3], mano_pose [16, 3, 3] rotations or [16, 3] axis-angle, mano_betas [10],
        contact_p [778] in 0..1, gate_score, table_p). Grab / release events and grasp
        episodes follow gate_closed."""
        assert hand in HANDS, hand
        raw = raw or {}
        assert set(raw) <= set(RAW_COLS), set(raw) - set(RAW_COLS)
        live = self._live[sid]
        p = None if state_probs is None else np.asarray(state_probs, float)
        pose = gripper_pose(None if joints is None else np.asarray(joints, float))
        j = None if pose is None else np.asarray(joints, float).reshape(-1)
        kp = None if keypoints_2d is None else np.asarray(keypoints_2d, float)[:, :2]
        row = (self._ts(live["t0"] + timedelta(seconds=float(t))), sid, float(t), hand,
               bool(present), None if p is None else int(p.argmax()),
               *([None] * 3 if p is None else map(float, p)),
               None if gate_closed is None else bool(gate_closed),
               self._arr(finger_probs),
               *([None] * 3 if pose is None else map(float, pose[0])),
               *([None] * 4 if pose is None else map(float, pose[1])),
               None if pose is None else pose[2], self._arr(j),
               self._arr(kp), *self._raw_values(raw))  # fmt: skip
        self._buf.append(row)
        tr = live["hands"][hand]
        if object_name:
            tr.object = object_name
        if gate_closed is not None:
            if gate_closed and not tr.closed:
                tr.start(t, pose)
                self._event(sid, t, hand, "grab", tr.object)
            elif not gate_closed and tr.closed:
                self._event(sid, t, hand, "release", tr.object)
                self._grasp(sid, hand, tr.finish(t))
            elif gate_closed:
                tr.update(pose)
            tr.closed = bool(gate_closed)
        if len(self._buf) >= FLUSH_ROWS:
            self.flush()

    def _raw_values(self, raw: dict) -> list:
        out = []
        for c in RAW_COLS:
            v = raw.get(c)
            if v is None:
                out.append(None)
            elif c == "frame_idx":
                out.append(int(v))
            elif c in ("hand_score", "gate_score", "table_p"):
                out.append(float(v) if np.isfinite(v) else None)
            elif c == "contact_p":
                a = np.asarray(v, float).reshape(-1)
                assert a.shape == (N_VERTS,), a.shape
                out.append(np.round(np.clip(a, 0, 1) * 255).astype(np.uint8).tobytes())
            else:
                a = np.asarray(v, float)
                if c == "mano_pose" and a.shape == (16, 3, 3):
                    a = Rot.from_matrix(a).as_rotvec()
                assert a.size == int(np.prod(ARR_SHAPES[c])), (c, a.shape)
                out.append(None if not np.isfinite(a).all() else self._arr(a))
        return out

    def add_object(self, sid: str, t: float, hand: str, name: str, detail: dict | None = None):
        """Object identity for a hand (e.g. from Gemini); attached to its current / next grasp."""
        tr = self._live[sid]["hands"][hand]
        tr.object = name
        if tr.closed and tr.cur is not None:
            tr.cur["object"] = name
        self._event(sid, t, hand, "object", name, detail)

    def end_session(self, sid: str, t_end: float | None = None) -> None:
        for hand, tr in self._live[sid]["hands"].items():
            if tr.closed and tr.cur is not None:
                self._grasp(sid, hand, tr.finish(None))
        self.flush()
        del self._live[sid]

    def delete_sessions(self, source: str) -> int:
        """Remove every session of one source and all its rows (e.g. to re-upload sample data)."""
        sids = [r[0] for r in self._query(f"SELECT session_id FROM sessions WHERE source = "
                                          f"{self.ph}", (source,))]  # fmt: skip
        for sid in sids:
            for tab in ("hand_frames", "events", "grasps", "videos", "sessions"):
                self._exec(f"DELETE FROM {tab} WHERE session_id = {self.ph}", (sid,))
        self._exec("DELETE FROM scenes WHERE scene_id NOT IN (SELECT scene_id FROM sessions "
                   "WHERE scene_id IS NOT NULL)", ())  # fmt: skip
        self.con.commit()
        return len(sids)

    def flush(self) -> None:
        if not self._buf:
            return
        sql = "INSERT INTO hand_frames ({}) VALUES ({})".format(
            ",".join(FRAME_COLS), ",".join([self.ph] * len(FRAME_COLS))
        )
        self.con.cursor().executemany(sql, self._buf)
        self.con.commit()
        self._buf.clear()

    def close(self) -> None:
        self.flush()
        self.con.close()

    # ---------------------------------------------------------------- read
    def sessions(self) -> list[dict]:
        rows = self._query(f"SELECT {','.join(SESSION_COLS)} FROM sessions ORDER BY started_at",
                           dicts=True)  # fmt: skip
        return [self._decode(r) for r in rows]

    def session(self, session_id: str) -> dict:
        r = self._query(f"SELECT {','.join(SESSION_COLS)} FROM sessions WHERE session_id = "
                        f"{self.ph}", (session_id,), dicts=True)  # fmt: skip
        assert r, f"no session {session_id}"
        return self._decode(r[0])

    def scene(self, scene_id: str) -> dict | None:
        r = self._query(f"SELECT {','.join(SCENE_COLS)} FROM scenes WHERE scene_id = {self.ph}",
                        (scene_id,), dicts=True)  # fmt: skip
        return self._decode(r[0]) if r else None

    def videos(self, session_id: str) -> list[dict]:
        return self._query(f"SELECT {','.join(VIDEO_COLS)} FROM videos WHERE session_id = "
                           f"{self.ph}", (session_id,), dicts=True)  # fmt: skip

    def grasps(self, session_id: str | None = None, object_like: str | None = None,
               min_lift: float | None = None) -> list[dict]:  # fmt: skip
        q, args = "SELECT * FROM grasps WHERE 1=1", []
        if session_id:
            q, args = q + f" AND session_id = {self.ph}", [*args, session_id]
        if object_like:
            q, args = q + f" AND LOWER(object) LIKE {self.ph}", [*args, f"%{object_like.lower()}%"]
        if min_lift is not None:
            q, args = q + f" AND lift_m >= {self.ph}", [*args, min_lift]
        return self._query(q + " ORDER BY session_id, t_grab", args, dicts=True)

    def events(self, session_id: str, kind: str | None = None) -> list[dict]:
        q, args = (
            f"SELECT t, hand, kind, object, detail FROM events WHERE session_id = {self.ph}",
            [session_id],
        )
        if kind:
            q, args = q + f" AND kind = {self.ph}", [*args, kind]
        rows = self._query(q + " ORDER BY t", args)
        return [{"t": r[0], "hand": r[1], "kind": r[2], "object": r[3], "detail": _loads(r[4])}
                for r in rows]  # fmt: skip

    def frames(self, session_id: str, hand: str, t0: float = -1e12, t1: float = 1e12) -> dict:
        """Every hand_frames column for one hand, t0 <= t <= t1, as numpy arrays [N, ...]
        (arrays NaN-filled where NULL; contact_p back in 0..1; frame_idx / state -1 = NULL)."""
        cols = [c for c in FRAME_COLS if c not in ("time", "session_id", "hand")]
        ph = self.ph
        rows = self._query(f"SELECT {','.join(cols)} FROM hand_frames WHERE session_id = {ph}"
                           f" AND hand = {ph} AND t >= {ph} AND t <= {ph} ORDER BY t",
                           (session_id, hand, t0, t1))  # fmt: skip
        n = len(rows)
        out = {}
        for k, c in enumerate(cols):
            vals = [r[k] for r in rows]
            if c in ARR_SHAPES:
                a = np.full((n, *ARR_SHAPES[c]), np.nan)
                for i, v in enumerate(vals):
                    if v is None:
                        continue
                    if c == "contact_p":
                        a[i] = np.frombuffer(bytes(v), np.uint8) / 255.0
                    else:
                        a[i] = np.asarray(self._unarr(v), float).reshape(ARR_SHAPES[c])
                out[c] = a
            elif c in ("present", "gate_closed"):
                out[c] = np.array([bool(v) for v in vals], bool)
            elif c in ("state", "frame_idx"):
                out[c] = np.array([-1 if v is None else int(v) for v in vals], int)
            else:
                out[c] = np.array([np.nan if v is None else float(v) for v in vals])
        return out

    def episode(self, grasp_id: str) -> dict:
        """One grasp as arrays: approach (APPROACH_S before the grab) -> hold -> release + AFTER_S.
        All hand_frames columns (see frames()), plus t (s, 0 = grab), t_session, gripper_closed
        (= the gate), grasp_pos [N, 3], quat [N, 4] (w x y z); info: grasp, session, scene,
        videos."""
        g = self._query(f"SELECT * FROM grasps WHERE grasp_id = {self.ph}", (grasp_id,),
                        dicts=True)  # fmt: skip
        assert g, f"no grasp {grasp_id}"
        g = g[0]
        t1 = (g["t_release"] if g["t_release"] is not None else 1e12) + AFTER_S
        out = self.frames(g["session_id"], g["hand"], g["t_approach"], t1)
        out["t_session"] = out["t"]
        out["t"] = out["t"] - g["t_grab"]
        out["gripper_closed"] = out["gate_closed"]
        out["grasp_pos"] = np.stack([out["grasp_x"], out["grasp_y"], out["grasp_z"]], -1)
        out["quat"] = np.stack([out[f"quat_{a}"] for a in "wxyz"], -1)
        sess = self.session(g["session_id"])
        out["info"] = {**g, "frame": sess["frame"], "source": sess["source"],
                       "session_meta": sess["meta"], "units": "m", "quat_order": "w x y z",
                       "t_zero": "grab",
                       "scene": self.scene(sess["scene_id"]) if sess.get("scene_id") else None,
                       "videos": self.videos(g["session_id"])}  # fmt: skip
        return out

    def export_episode(self, grasp_id: str, path: Path) -> Path:
        """episode() -> .npz (arrays) + .json (info) next to it."""
        ep = self.episode(grasp_id)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        info = ep.pop("info")
        np.savez_compressed(path, **ep)
        path.with_suffix(".json").write_text(json.dumps(info, indent=1, default=_jsonable))
        return path

    # ---------------------------------------------------------------- internals
    def _event(self, sid, t, hand, kind, obj, detail=None):
        live = self._live[sid]
        self._exec("INSERT INTO events VALUES ({})".format(",".join([self.ph] * 7)),
                   (self._ts(live["t0"] + timedelta(seconds=float(t))), sid, float(t), hand,
                    kind, obj, self._json(detail or {})))  # fmt: skip

    def _grasp(self, sid, hand, g):
        if g is None:
            return
        self._insert("grasps", GRASP_COLS,
                     (uuid.uuid4().hex[:12], sid, hand, max(0.0, g["t_grab"] - APPROACH_S),
                      g["t_grab"], g["t_release"], g["object"], *g["grab_pos"], g["lift"],
                      g["width"], g["n"]))  # fmt: skip
        self.con.commit()

    def _insert(self, table, cols, values):
        assert len(cols) == len(values), (table, len(cols), len(values))
        self._exec(f"INSERT INTO {table} ({','.join(cols)}) VALUES "
                   f"({','.join([self.ph] * len(cols))})", values)  # fmt: skip

    def _exec(self, sql, args):
        self.con.cursor().execute(sql, args)

    def _query(self, sql, args=(), dicts=False):
        cur = self.con.cursor()
        cur.execute(sql, args)
        rows = cur.fetchall()
        if dicts:
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r, strict=True)) for r in rows]
        return rows

    def _decode(self, row: dict) -> dict:
        return {k: (_loads(v) if k in JSON_COLS else v) for k, v in row.items()}

    def _json(self, v):
        return None if v is None else json.dumps(v, default=_jsonable)

    def _ts(self, dt: datetime):
        return dt if self.pg else dt.isoformat()

    def _arr(self, a):
        if a is None:
            return None
        v = [float(x) for x in np.asarray(a, float).reshape(-1)]
        return v if self.pg else json.dumps(v)

    def _unarr(self, v):
        return v if self.pg else json.loads(v)


class _HandTrack:
    """Running grasp episode of one hand (follows the gate)."""

    def __init__(self):
        self.closed, self.object, self.cur = False, None, None

    def start(self, t, pose):
        p = None if pose is None else pose[0]
        self.cur = {"t_grab": float(t), "object": self.object, "n": 1,
                    "grab_pos": [None] * 3 if p is None else [float(v) for v in p],
                    "z0": None if p is None else float(p[2]), "zs": [],
                    "width": None if pose is None else pose[2]}  # fmt: skip

    def update(self, pose):
        if self.cur is None:
            return
        self.cur["n"] += 1
        if pose is not None:
            z = float(pose[0][2])
            if self.cur["z0"] is None:
                self.cur["z0"] = z
                self.cur["grab_pos"] = [float(v) for v in pose[0]]
            self.cur["zs"].append(z)

    def finish(self, t):
        g, self.cur = self.cur, None
        if g is None:
            return None
        g["t_release"] = None if t is None else float(t)
        # 90th percentile of the rise: single-frame jumps of a predicted hand are common
        g["lift"] = (None if g["z0"] is None or not g["zs"]
                     else max(0.0, float(np.percentile(g["zs"], 90)) - g["z0"]))  # fmt: skip
        return g


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _loads(v):
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return v
    return json.loads(v)
