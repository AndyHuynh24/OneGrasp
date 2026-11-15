"""Robot demo export: one recorded human demo -> per-frame trajectories + keyframes, in a format a
robot-learning pipeline (MuJoCo 4 replay, behaviour cloning, keyframe / waypoint policies) can
read without this repo's model code. Spec: docs/ROBOT_EXPORT.md.

Frames: world / table frame, metres, z up (HO-Cap: table tag; own rig: the calibration's frame).
Quaternions: w, x, y, z (MuJoCo order). Gripper = parallel-jaw pad pose (centre between the
finger pads; z = approach axis, y = closing axis) from touch.robot_sim.retarget + pregrasp.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from touch import robot_sim as S

FORMAT_VERSION = 1
LIFT_M = 0.02  # gripper rise after closing that marks "lift"
STILL_MPS = 0.05  # gripper speed below which the hand counts as still (m/s)


def to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    return Rot.from_matrix(R).as_quat()[..., [3, 0, 1, 2]]


def keyframes(pad_pos: np.ndarray, close: np.ndarray, state: np.ndarray, ok: np.ndarray,
              fps: float, n_app: int = 12) -> list[dict]:  # fmt: skip
    """Segment a demo into keyframes from the close command [N] bool, hand state [N] (0 none,
    1 touch, 2 hold) and the gripper path [N, 3]:
      reach_start   first frame with a tracked hand
      pregrasp      n_app frames before each close (the straight-in approach starts here)
      contact       first touch/hold frame of the episode (the hand reaches the object)
      grasp         close command on
      lift          gripper z rises LIFT_M above its height at grasp
      transport_end gripper slows below STILL_MPS while still closed (object placed / held still)
      release       close command off
      retreat_end   last tracked frame
    -> [{"type", "frame", "t"}] sorted by frame."""
    n = len(close)
    out = []
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return out
    out.append(("reach_start", int(idx[0])))
    c = np.r_[False, close.astype(bool), False]
    starts = np.flatnonzero(~c[:-1] & c[1:])
    ends = np.flatnonzero(c[:-1] & ~c[1:])
    speed = np.r_[0, np.linalg.norm(np.diff(pad_pos, axis=0), axis=1) * fps]
    speed = np.where(np.isfinite(speed), speed, 0)
    prev = 0
    for a, e in zip(starts, ends, strict=True):
        out.append(("pregrasp", int(max(prev, a - n_app))))
        on = np.flatnonzero(state[prev : a + 1] > 0)
        if len(on):  # the contact run that leads into this grasp
            k = prev + int(on[-1])
            while k > prev and state[k - 1] > 0:
                k -= 1
            out.append(("contact", k))
        out.append(("grasp", int(a)))
        z0 = pad_pos[a, 2]
        up = np.flatnonzero(pad_pos[a:e, 2] - z0 >= LIFT_M)
        if len(up):
            lift = a + int(up[0])
            out.append(("lift", lift))
            moving = speed[lift:e] >= STILL_MPS
            last = np.flatnonzero(moving)
            if len(last) and lift + int(last[-1]) + 1 < e:
                out.append(("transport_end", lift + int(last[-1]) + 1))
        out.append(("release", int(min(e, n - 1))))
        prev = e
    out.append(("retreat_end", int(idx[-1])))
    out.sort(key=lambda x: x[1])
    return [{"type": t, "frame": int(f), "t": round(float(f) / fps, 3)} for t, f in out]


def build(joints: np.ndarray, verts: np.ndarray, ok: np.ndarray, state: np.ndarray,
          state_prob: np.ndarray, finger: np.ndarray, contact_verts: np.ndarray,
          near: np.ndarray, close: np.ndarray, parts: np.ndarray, fps: float,
          pre: bool = True) -> dict:  # fmt: skip
    """World-frame hand [N, 21, 3] / [N, 778, 3] (NaN where not ok), model outputs and the close
    command -> arrays of the export (see docs/ROBOT_EXPORT.md)."""
    n = len(joints)
    assert joints.shape == (n, 21, 3) and verts.shape == (n, 778, 3) and close.shape == (n,)
    pp, RR, W, how = S.retarget(joints, ok, close, verts, contact_verts, parts)
    if pre:
        pp, RR = S.pregrasp(pp, RR, close)
    wrist = np.full((n, 4, 4), np.nan)
    for i in np.flatnonzero(ok):
        wrist[i] = S.hand_frame(joints[i])
    return {
        "t": np.arange(n) / fps,
        "hand_ok": ok.astype(bool),
        "hand_joints": joints.astype(np.float32),
        "hand_pose": wrist.astype(np.float32),
        "state": state.astype(np.int8),
        "state_prob": state_prob.astype(np.float32),
        "finger_contact": finger.astype(bool),
        "contact_verts": contact_verts.astype(bool),
        "object_near": near.astype(bool),
        "gripper_pos": pp.astype(np.float32),
        "gripper_quat": to_quat_wxyz(RR).astype(np.float32),
        "gripper_width": W.astype(np.float32),
        "gripper_close": close.astype(bool),
        "grasp_source": np.array(how),
    }


def save(path: Path, arrays: dict, meta: dict, keys: list[dict]) -> None:
    """path.npz (arrays) + path.json (meta + keyframes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    meta = {"format_version": FORMAT_VERSION, **meta, "keyframes": keys}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
