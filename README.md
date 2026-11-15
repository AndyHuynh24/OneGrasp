# OneGrasp

**Teach robots by just doing the task.** OneGrasp turns one human demonstration, filmed with a
normal webcam, into a batch of verified robot training data in a few minutes. No gloves, no
controller, no sensors.

1. **See.** A webcam films your hand doing the task (here: pick up a can, pour, set it down on a
   target). OneGrasp rebuilds the hand in 3D every frame and detects when you grab and let go.
   Double-click the object and SAM 2.1 segments it; ArUco markers on the table give real-world
   coordinates.
2. **Transfer.** A simulated Franka Panda (MuJoCo) repeats your motion in sync with you. We store
   what the hand *does* (gripper position and rotation), not joint angles, so the recording can
   be mapped to other robot arms through their own inverse kinematics.
3. **Scale.** One demo becomes twelve (MimicGen-style): the can and the target move up to 10 cm,
   about half the scenes start with the can lying down, and the robot starts in a random pose.
4. **Verify and store.** Every scene runs in full physics; only successes are kept. They are saved
   in robomimic format (HDF5) and streamed into Tiger Data (Postgres + TimescaleDB).

## Install

Python 3.11, macOS or Linux.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[grasp,store,robot,dev]"

# Franka Panda model (MuJoCo Menagerie, Apache 2.0)
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie /tmp/menagerie
mkdir -p third_party && cp -r /tmp/menagerie/franka_emika_panda third_party/mujoco_panda

# SAM 2.1 Hiera-L (Apache 2.0), for the object double-click
pip install "git+https://github.com/facebookresearch/sam2.git"
mkdir -p third_party/sam
curl -L -o third_party/sam/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

**Trained weights** (grasp model, grab gate, MediaPipe hand model) are a GitHub release download,
not in git:

```bash
scripts/download_weights.sh   # needs `gh auth login`; verifies SHA-256 checksums
```

**MANO** is not included (its licence forbids redistribution): register at
https://mano.is.tue.mpg.de and put `MANO_RIGHT.pkl` / `MANO_LEFT.pkl` in `data/mano/` (or set
`$MANO_DIR`).

## Run the demo

```bash
python scripts/make_markers.py --clean        # print at 100 % ("Actual size"); marker 0 = table origin
python scripts/calib_rig.py --cam 1 --size-mm 100   # once per camera placement
python scripts/panda_demo.py --cam 1
```

In the window:

| Key / action | What happens |
| --- | --- |
| double-click the object | SAM 2.1 segments it and fits it in 3D |
| do the task | your hand is tracked in 3D; grab / let go are found automatically |
| `E` | end the take: the robot motion is planned and the grip is refined in physics |
| `S` / Enter / Space | go for scale: 12 new scenes, simulated side by side |
| `G` | 12 more scenes from the same demo |
| `P` / `V` / `T` | replay your run / the generated runs / a demo read back from Tiger Data |
| `R` / `Q` | new take / quit |

The demo object defaults to a drink can (13.4 x 5.2 cm); set `--obj-h` / `--obj-d` (metres) for
another cylinder. Successful runs are appended to `data/panda_demos/cup_pick_place.hdf5`
(robomimic layout: `actions`, `actions_eef`, `obs`, `states`, `rewards`, `dones`). Each take's
videos, MuJoCo scenes (MJCF) and summary go to `results/robot_sim/panda_demo/<time>/`.

## Tiger Data

Set the database URL (keep it private: it holds the password):

```bash
export TIGER_DATABASE_URL="postgres://USER:PASSWORD@HOST:PORT/tsdb?sslmode=require"
```

With it set, the demo streams every take's hand frames (30 Hz) and every robot demo into Tiger
Data:

- **Hypertables** `hand_frames` and `demo_steps` (30 Hz time series), next to relational tables
  `sessions`, `demos`, `demo_scenes`, joined with plain SQL.
- **Columnstore compression** per demo (segmented by demo, ordered by step).
- **Continuous aggregate** `demo_steps_1s` (per-second stats) behind the live dashboard.

```bash
python scripts/tiger_dashboard.py              # live dashboard, open the printed address
python scripts/replay_from_tiger.py --list     # what is stored
python scripts/replay_from_tiger.py --n 3      # pull demos from Tiger and replay them in MuJoCo
```

Tables and example queries: [docs/TIGER_DATA.md](docs/TIGER_DATA.md). Exported demo format:
[docs/ROBOT_EXPORT.md](docs/ROBOT_EXPORT.md).

## Layout

```
src/touch/        hand tracking, grasp model, 3D hand path, Panda IK / simulation, demo generation,
                  Tiger Data store
scripts/          demo (panda_demo.py), calibration, training / evaluation, dashboard, replay
configs/          training configs
tests/            pytest
video/            the demo video (Remotion)
```

Tests: `pytest -q`. Lint: `ruff check . && ruff format --check .`

## Licences

The grasp model was trained partly on DexYCB (CC BY-NC 4.0), so the trained weights are for
non-commercial use only. MediaPipe and SAM 2.1 are Apache 2.0;
MuJoCo Menagerie's Panda model is Apache 2.0; MANO has its own licence (not redistributed).
