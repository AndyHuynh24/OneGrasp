# Hand data in Tiger Data (TimescaleDB)

OneGrasp records what it sees of a person's hands into a Tiger Data (TimescaleDB / Postgres)
database: the raw camera / model values, the 3D hand, contact, grab / release, the 3D scene and
a reference to the video. This page is for teammates who **read** that data (e.g. to train or
drive a robot by imitation). Human hand data only: turning it into robot motion (retargeting)
is up to you.

## Connect

You need the connection string (ask Andy privately; never commit it):

```bash
export TIGER_DATABASE_URL="postgres://tsdbadmin:<PASSWORD>@<HOST>:<PORT>/tsdb?sslmode=require"
psql "$TIGER_DATABASE_URL"            # SQL shell
```

Python, inside this repo:

```python
from touch.store import GraspStore
st = GraspStore()                        # reads $TIGER_DATABASE_URL
st.sessions()                            # recordings
st.grasps(object_like="cylinder", min_lift=0.05)   # grab -> release episodes
ep = st.episode(grasp_id)                # numpy arrays [N, ...] for one episode + ep["info"]
f = st.frames(session_id, "right")       # every frame of one hand in a session
st.scene(scene_id); st.videos(session_id)
```

Or export episodes as plain files (no database needed afterwards):

```bash
python scripts/export_episodes.py --object cylinder --min-lift 0.05 --out episodes/
# episodes/index.json, episodes/episode_0000/{data.npz, info.json, scene.xml, clip.mp4}
```

Without the repo: `pip install "psycopg[binary]" numpy` and plain SQL (examples below).

## Get the videos and 3D scene files

The database stores references, not files. This downloads everything it references, checks it
against the stored frame counts / SHA-256, and writes MuJoCo scene files with local mesh paths:

```bash
python scripts/fetch_store_assets.py --out data/store_assets              # all sessions
python scripts/fetch_store_assets.py --session <id> --out data/store_assets
python scripts/fetch_store_assets.py --no-video                            # scenes + meshes only
# data/store_assets/hocap/...            HO-Cap images + object meshes (public, CC BY 4.0)
# data/store_assets/videos/<session>.mp4 uploaded recordings
# data/store_assets/scenes/<scene>.xml   MuJoCo: table + objects + camera (add your robot)
# data/store_assets/manifest.json        per session: uri -> local file, status
python scripts/export_episodes.py --hocap-root data/store_assets/hocap --out episodes/
```

One HO-Cap session is about 1000 frames / ~95 MB. Each video row's `frame_idx` is the image
number (`color_000123.jpg` = frame 123, or frame 123 of the mp4).

## Setup for teammates

```bash
git clone https://github.com/AndyHuynh24/OneGrasp.git && cd OneGrasp
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[store]"                   # psycopg, numpy, scipy, opencv, mujoco, ...
export TIGER_DATABASE_URL="..."              # from Andy, privately (add to ~/.zshrc)
python -c "from touch.store import GraspStore; print(len(GraspStore().grasps()), 'grasps')"
```

## What is stored

Words used here:
- **session**: one recording. **episode** / **grasp**: one grab -> release of one hand, plus the
  1.5 s approach before it and 0.5 s after.
- **trajectory**: the hand's pose over time within an episode.
- **raw**: straight from the camera / models, before our processing (smoothing, gate,
  coordinate change). Use it if you want to redo the processing your own way.

### `scenes`: the 3D scene of a session

| column | meaning |
|---|---|
| `frame` | `table` (metres, z up, table plane z = 0, origin at ArUco marker 0) or `camera` |
| `camera` (JSON) | `K` (3x3 intrinsics), `cam_to_world` (3x4, x_table = R x_cam + t), `image_size` |
| `table_plane` (JSON) | normal and offset of the table plane |
| `objects` (JSON) | per object: `name`, `shape` (cylinder / box / mesh), `size_m` (cylinder: radius, half height) or `mesh_uri`, `pos`, `quat_wxyz` at the start |
| `markers` (JSON) | calibration markers (ArUco ids, size) |
| `mjcf` | a MuJoCo XML of table + objects + calibrated camera, **no robot**: add your Panda to it |

Objects that move during a session: see `events` (kind `object`, `detail` = pose / position).

### `sessions`: one row per recording

`session_id`, `started_at` (UTC), `source` (`live_webcam`, `hocap_val`, ...), `frame`,
`model` + `git` (which model / code version), `meta` (JSON: calibration, gate, notes),
`scene_id`.

### `videos`: where the video is (NOT the video itself)

| column | meaning |
|---|---|
| `uri` | where the file is: a path on the recording laptop (`data/recordings/<session>.mp4`), later a Drive / Hugging Face link; `hocap://...` = an image sequence of the public HO-Cap dataset |
| `kind` | `mp4` or `image_folder` (uri is a printf pattern, e.g. `color_%06d.jpg`) |
| `fps`, `n_frames`, `width`, `height`, `codec` | video properties (fps is nominal: use `hand_frames.t` for time) |
| `size_bytes`, `sha256` | check that your downloaded copy is the same file |

`hand_frames.frame_idx` = frame number in this video, so every row can be matched to its image.

### `hand_frames`: one row per hand per video frame (hypertable on `time`)

Processed (ready to use):

| column | meaning |
|---|---|
| `time`, `t` | timestamp; `t` = seconds since the session started (true capture time) |
| `hand`, `present` | `left` / `right`; hand seen this frame (else most columns are NULL) |
| `state`, `p_none`, `p_touch`, `p_hold` | model's hand state (0 none, 1 touch, 2 hold) and probabilities |
| `gate_closed` | the grab gate: is the hand holding something (steadier than `state`) |
| `finger_p` | 5 fingertip contact probabilities: thumb, index, middle, ring, little |
| `joints` | 21 hand joints x (x, y, z) in the session `frame` (metres), MediaPipe order (0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 little) |
| `keypoints_2d` | the 21 joints in the image (x, y pixels) |
| `grasp_x/y/z`, `quat_w/x/y/z`, `width` | convenience, derived from `joints`: point between the thumb tip and the index / middle tips, a frame with y = thumb -> fingers and z = pointing direction, thumb-finger distance (m). Ignore if you derive your own |

Raw:

| column | meaning |
|---|---|
| `frame_idx` | frame number in the session video |
| `hand_score` | MediaPipe hand-detection / handedness confidence |
| `mp_landmarks` | 21 x 3 raw MediaPipe output: x, y pixels and MediaPipe's relative depth |
| `joints_model` | 21 x 3 model joints: wrist-relative, camera axes, metres; left hands are mirrored in x (the model sees every hand as a right hand) |
| `cam_t` | wrist position in camera coordinates (m), fit from the 2D keypoints |
| `mano_pose`, `mano_betas` | MANO hand model parameters: 16 x 3 joint rotations (axis-angle; first = global) and 10 shape values. Right-hand MANO, `flat_hand_mean=True`, wrist at the origin. They rebuild the full 778-vertex 3D hand mesh |
| `contact_p` | 778 bytes: per mesh vertex, probability (byte / 255) that this point of the hand touches something |
| `gate_score` | the grab gate's score before its on/off thresholds |
| `table_p` | model's probability that the hand rests on the table (if the model has it) |

Placing the raw hand in the scene: `x_cam = (mirror x if left) x_model + cam_t`, then
`x_table = R x_cam + t` with `cam_to_world = [R | t]` from the scene. That gives `joints`.

Rebuilding the MANO mesh (needs the MANO model files from mano.is.tue.mpg.de, own licence):

```python
import smplx, torch
mano = smplx.MANO("path/to/mano", is_rhand=True, use_pca=False, flat_hand_mean=True)
pose = torch.tensor(ep["mano_pose"][i], dtype=torch.float32).reshape(1, 48)
out = mano(global_orient=pose[:, :3], hand_pose=pose[:, 3:],
           betas=torch.tensor(ep["mano_betas"][i:i+1], dtype=torch.float32))
verts = out.vertices[0] - out.joints[0, 0]      # wrist-relative, as stored
```

### `events`: grab / release / object

`kind` = `grab` (gate closed), `release` (gate opened), `object` (which object the hand holds
or where an object is; `detail` JSON may hold `pos`, `quat_wxyz`, `xy`).

### `grasps`: one row per grab -> release (per hand)

`t_grab`, `t_release` (NULL = still held at the end), `t_approach` (= t_grab - 1.5 s), `object`,
grab position `grab_x/y/z`, `lift_m` (how far the hand rose while holding, 90th percentile),
`width_grab`, `n_frames`.

## Example queries

```sql
-- all recordings and their scenes
SELECT s.session_id, s.source, s.started_at, sc.objects
FROM sessions s LEFT JOIN scenes sc USING (scene_id) ORDER BY s.started_at;

-- grasps of a cylinder that were lifted at least 5 cm
SELECT grasp_id, session_id, hand, t_grab, t_release, lift_m
FROM grasps WHERE object ILIKE '%cylinder%' AND lift_m >= 0.05;

-- the trajectory of one grasp (approach -> release + 0.5 s) with video frame numbers
SELECT f.t - g.t_grab AS t_rel, f.frame_idx, f.gate_closed, f.joints, f.mano_pose, f.finger_p
FROM grasps g JOIN hand_frames f
  ON f.session_id = g.session_id AND f.hand = g.hand
 AND f.t BETWEEN g.t_approach AND COALESCE(g.t_release, 1e9) + 0.5
WHERE g.grasp_id = '<id>' ORDER BY f.t;

-- wrist position (joint 0) over time, one session
SELECT t, joints[1] AS x, joints[2] AS y, joints[3] AS z
FROM hand_frames WHERE session_id = '<id>' AND hand = 'right' AND present ORDER BY t;

-- where the video of a session is
SELECT uri, kind, fps, n_frames, sha256 FROM videos WHERE session_id = '<id>';

-- fraction of time holding, per minute (TimescaleDB time_bucket)
SELECT time_bucket('1 minute', time) AS minute, hand, avg(gate_closed::int) AS held_fraction
FROM hand_frames WHERE session_id = '<id>' GROUP BY minute, hand ORDER BY minute;
```

(Postgres arrays are 1-based: `joints[3k+1..3k+3]` = joint k.)

## Units and caveats

- Metres, seconds, pixels. Quaternions w x y z.
- `frame = table` positions are the model's estimate from one camera (typically ~1.5-2 cm
  hand-joint error on the HO-Cap validation set, likely more on a new camera / person).
- `frame = camera` data (lab videos) is good for *when* and *what* is grasped and the finger
  pattern; not for exact metric positions.
- Use `gate_closed` for "holding"; `state` flickers.
- The live demo tracks one hand at a time.
- HO-Cap sample data (`source = hocap_val`): HO-Cap is CC BY 4.0; its object meshes and images
  are not in the database (`mesh_uri`, `hocap://` uris point into the HO-Cap download).
  Raw MANO values are not stored for it.

## Sharing (Andy's side)

- Code + this guide: the GitHub repo is private: add teammates under GitHub -> Settings ->
  Collaborators (or `gh api -X PUT repos/AndyHuynh24/OneGrasp/collaborators/<their-github-name>`).
- Database: better than sharing the admin password, give them a read-only login. In the Tiger
  console (Ops view -> your service -> SQL editor), with a password you choose:

  ```sql
  CREATE ROLE team_reader LOGIN PASSWORD '<a long random password>';
  GRANT CONNECT ON DATABASE tsdb TO team_reader;
  GRANT USAGE ON SCHEMA public TO team_reader;
  GRANT SELECT ON ALL TABLES IN SCHEMA public TO team_reader;
  ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO team_reader;
  ```

  Their URL is yours with `tsdbadmin:<password>` replaced by `team_reader:<that password>`.
  Send it privately (DM), never in the repo or a public channel.
- Live recordings' videos: upload `data/recordings/` (e.g. a private Hugging Face dataset) and
  update `videos.uri` to the `hf://datasets/<user>/<repo>/<file>` or `https://` link; the
  stored SHA-256 lets teammates check their copy.

## How data gets in (Andy's side)

```bash
python scripts/live_sim.py --cam 1 --store     # live webcam -> $TIGER_DATABASE_URL,
                                               # video -> data/recordings/<session>.mp4
python scripts/store_hocap_grasps.py --replace # sample data: HO-Cap val, oblique camera
```

Without `TIGER_DATABASE_URL`, both write to a local file `data/grasp_store.sqlite` (same tables).
Storage: about 2 KB per hand per frame, so 750 MB holds roughly 3 hours of one-hand recording
at 30 fps. Videos are files, not in the database.

# Robot demonstrations in Tiger Data

The simulated Franka Panda demonstrations (teleop from a camera take + MimicGen-style generated
scenes, `data/panda_demos/cup_pick_place.hdf5`) are in the same database
(`src/touch/store/demo_store.py`). `scripts/panda_demo.py` uploads each new demo live in a
background thread (~0.3 s per demo; off with `--no-tiger` or another `--dataset`).

| table | what |
|---|---|
| `demo_scenes` | one MuJoCo scene (full MJCF) per unique scene, by content hash |
| `demos` | one demo: `source` (teleop / mimicgen), `src_demo` (the teleop demo a generated one came from), `take`, `success`, cup start / place target, `num_steps`, `attrs` (JSON: metrics, cup pose, ...) |
| `demo_steps` | **hypertable**, one row per 30 Hz step: `action` [8] (7 joint targets + gripper ±1), `action_eef` [7], `joint_pos/vel` [7], `gripper_qpos` [2], `eef_pos` [3], `eef_quat` [4] xyzw, `cup_pos` [3], `cup_quat` [4], `state` [31] (MuJoCo qpos + qvel, float64), `reward`, `done`, plus `cup_z`, `eef_speed`, `gripper_closed` for SQL |
| `demo_steps_1s` | **continuous aggregate**: per demo per second, steps, cup height min/max, TCP speed mean/max, gripper closed |

`demo_steps` is compressed into the columnstore (`segmentby demo_id, orderby step`). Measured on
the 198 demos uploaded on 2026-11-14: 64.7 MB -> 35.3 MB (1.8x). One column per value instead of
arrays gave 1.4x, so the arrays stay.

```bash
python scripts/demos_to_tiger.py --compress          # upload the HDF5 (skips stored demos)
python scripts/replay_from_tiger.py --list           # what is stored
python scripts/replay_from_tiger.py --source teleop  # MuJoCo window, real time (30 Hz)
python scripts/replay_from_tiger.py --check --n 200  # physics re-run of each, no window
```

Replay modes: `states` sets the recorded MuJoCo state each step (exact). `actions` re-runs the
physics from the stored commands only. Across all 198 demos the cup ends within 1 cm of the
recording for 92% (median 0.1 cm). The rest drift, up to 29 cm, because the gripper's open
width is not stored (only open / closed).

```sql
-- success and size per source
SELECT source, count(*), avg(num_steps) / 30.0 AS seconds FROM demos GROUP BY source;
-- generated demos per teleop demo
SELECT src_demo, count(*) FROM demos WHERE source = 'mimicgen' GROUP BY src_demo;
-- per-demo summary from the continuous aggregate (instant)
SELECT demo_id, sum(steps) / 30.0 AS s, max(cup_z_max) AS peak_cup_z, max(eef_speed_max)
FROM demo_steps_1s GROUP BY demo_id ORDER BY demo_id LIMIT 10;
-- training batch: every successful step, ordered
SELECT s.demo_id, s.step, s.joint_pos, s.cup_pos, s.action FROM demo_steps s
JOIN demos d USING (demo_id) WHERE d.success ORDER BY s.demo_id, s.step;
```
