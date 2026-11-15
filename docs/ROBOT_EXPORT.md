# Robot demo export (format v1)

What a recorded human demo becomes after the OneGrasp grasp pipeline, for robot learning
(MuJoCo 4 replay, behaviour cloning, keyframe / waypoint policies). Everything except the `gt_*`
keys is **predicted from one RGB camera** (no gloves, markers or depth).

Produced by `scripts/export_demo.py` (code: `src/touch/robot_export.py`); consumed by
`scripts/replay_export.py` (reference reader, MuJoCo 4.0.0).

## Pipeline (per camera frame, 30 fps)

1. MediaPipe hand landmarks (2D, 21 points).
2. Grasp model (GraspNet, causal, 14 M params, ~17 ms / hand on a laptop CPU): per-finger contact,
   hand state none / touch / hold, per-vertex contact on a MANO hand mesh, wrist-relative mesh.
3. Object gate (`touch.interact.object_gate`, fixed camera, optional): contact may only start
   where an object lies under the hand (background model of the table, hands/forearms masked).
   Used for the exports; g5 does not need it on the HO-Cap rig.
4. 3D hand in the table frame (`touch.handpath`): the predicted mesh placed by fitting it to the
   2D keypoints with the camera calibration, smoothed over 5 frames.
5. Close command: hold, debounced (on 2 / off 3 frames), +3 frames, gaps <= 10 frames filled.
6. Gripper (`touch.robot_sim.retarget` + `pregrasp`): a parallel-jaw grasp from the predicted
   contact vertices (thumb vs fingers), rigidly attached to the hand while closed; before each
   close the gripper comes in straight along its own approach axis from 8 cm back.
7. Keyframes (below).

## Files

`<demo>.npz` (arrays, N = frames) and `<demo>.json` (metadata + keyframes).

### Frames and units
- World = table frame, metres, **z up** (HO-Cap: the table tag; own rig: your calibration).
- Quaternions are **w, x, y, z** (MuJoCo order).
- Gripper pose = the **pad centre** (midpoint between the finger pads): z = approach axis
  (points from the palm towards the object), y = closing axis, x = y × z. The gripper body
  origin is 6 cm behind it along −z (`robot_sim.PAD_Z`).

### Arrays (`.npz`)
| key | shape | meaning |
|---|---|---|
| `t` | [N] | seconds |
| `hand_ok` | [N] bool | a hand pose exists this frame |
| `hand_joints` | [N, 21, 3] | predicted 3D joints (MediaPipe / MANO order), NaN when not ok |
| `hand_pose` | [N, 4, 4] | palm frame: origin = palm centre, x = wrist→middle knuckle, y = pinky→index knuckle |
| `state` | [N] int8 | 0 none, 1 touch, 2 hold (after the object gate) |
| `state_prob` | [N, 3] | model probabilities before the gate |
| `finger_contact` | [N, 5] bool | thumb … pinky touching anything |
| `contact_verts` | [N, 778] bool | MANO vertices in contact |
| `object_near` | [N] bool | the object gate (object under the hand) |
| `gripper_pos` | [N, 3] | pad centre |
| `gripper_quat` | [N, 4] | wxyz |
| `gripper_width` | [N] | opening at the grasp (m) |
| `gripper_close` | [N] bool | close command |
| `grasp_source` | [episodes] | how each grasp was built: `contact` (vertices) or `tips` (fallback) |
| `gt_*` | | HO-Cap only, ground truth for reference: `gt_object_pos` [N, n_obj, 3], `gt_object_quat` [N, n_obj, 4], `gt_hand_joints` [N, 21, 3] |

### Metadata (`.json`)
`format_version`, `source`, `sequence`, `camera`, `camera_group`, `hand`, `fps`, `frames`,
`frame` (world frame description), `K` (3×3), `cam_to_world` (3×4, x_w = R x_c + t), `objects`
(id + mesh path), `model`, `gate`, `close_rule`, `gripper_approach`, `keyframes`.

### Keyframes
Sorted list of `{"type", "frame", "t"}`; per grasp episode:

| type | when |
|---|---|
| `reach_start` | first tracked frame of the demo |
| `pregrasp` | 12 frames before the close: the straight-in approach starts here |
| `contact` | first touch / hold frame leading into this grasp |
| `grasp` | close command on |
| `lift` | gripper 2 cm above its height at the grasp |
| `transport_end` | gripper slows below 5 cm/s while still closed (placed / held still) |
| `release` | close command off |
| `retreat_end` | last tracked frame |

A keyframe policy can use (`pregrasp` pose, `grasp` pose + width, `lift`, `transport_end` pose,
`release`) as waypoints, and the dense arrays between them as the demonstration path.

## Replay

```
python scripts/replay_export.py results/robot_export/<run>/<demo>.npz --render out.mp4
```
Builds the scene from `objects` at their `gt_object_*` poses 1.5 s before each grasp (own rig:
add your object poses the same way), drives the force-limited gripper along `gripper_*`, reports
whether the grasped object was lifted >= 50% of the human's lift.

## Known limits (2026-11-12)
- Model g5 (runs/g5_grasp_hocap_20261112-014324), HO-Cap test top-down: grasp events precision
  0.88, recall 0.64 (it misses about a third of the real holds; what it reports is mostly right).
  Hold onset is often late by more than 5 frames. Per export: `grasp_events.json`.
- The 3D hand position comes from the predicted hand size (one camera): palm error 7.7 cm
  top-down, 12 cm over all cameras (`path_err_cm` in runs/hocap/<run>/sim_final.json). Too coarse
  to grasp by itself: use the object pose (known scene / object tracker) or depth for the final
  approach, and the hand path for the route and timing.
- MuJoCo replay of a human grasp with a parallel-jaw gripper lifts the object in only 21% of
  lifts even with perfect timing: treat the gripper fields as a starting point, not a policy.
- Objects are not tracked by the pipeline; HO-Cap exports carry the GT object poses.
- The gripper retargeting assumes a parallel-jaw gripper; the hand trajectory itself
  (`hand_joints`, `hand_pose`) is gripper-independent.
