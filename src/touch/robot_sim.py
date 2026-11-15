# ruff: noqa: E501  (embedded MJCF XML and paths read better unwrapped)
"""Human-to-robot replay of DexYCB grasps in MuJoCo (no special hardware: DexYCB recordings only).

A floating parallel-jaw gripper follows the human hand and closes / opens on a grasp command; the
objects start at their DexYCB poses on the table and are simulated. Did the robot lift the object
the human lifted? That is the robot-level test of the hand-state model.

Frames (verified 2026-11-11 on subject 7):
  master  = DexYCB "world": the master camera's frame. pose.npz pose_y = [qx qy qz qw tx ty tz]
            (scipy order) per object, equal to the master camera's labels (0.0 error).
  table   = extrinsics.yml "apriltag" (tag -> master, x_m = R x_t + t); z is up and the objects'
            lowest points sit at z = 0 +- 6 mm, so the table is the plane z = 0.
  camera c: extrinsics[c] maps camera c -> master (x_m = R x_c + t).

Hand -> gripper (per frame, from the 21 hand joints, MediaPipe / DexYCB order):
  thumb tip p_t = j4, finger tip p_f = mean(j8, j12) (index, middle); grasp centre c = mid-point;
  closing axis y = p_f - p_t; approach axis z = c - palm centre (wrist + 4 MCPs), made orthogonal
  to y; x = y cross z. The pads' centre is placed at c; open width = grasp width + 4 cm.
The gripper (15 cm stroke, 40 N per finger) is generic, not a model of a specific robot. It is a
free body welded to a mocap target (a mocap body itself has no velocity, so friction from it cannot
carry an object: verified, the pads slid up a can).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from touch.data import dexycb as D

PAD_Z = 0.06  # gripper origin -> grasp point (pad centre line) along the approach axis (m)
TIP_PAST = 0.03  # fingertips past the grasp point (val ablation: 0.012 -> 0.26, 0.03 -> 0.46)
FINGER_LEN = 0.06
MAX_HALF = 0.075  # finger travel per side (15 cm stroke; a 10 cm can needs room)
MARGIN = 0.15  # open width = grasp width + margin, capped by the stroke: opens fully
# (val oracle, force drive, close delay 3: margin 0.03 -> 0.40, 0.04 -> 0.48, 0.08 -> 0.61, full -> 0.63)
SUB = 17  # physics steps per 30 fps frame (timestep 2 ms)
# Gripper drive (replay(drive="force")): Cartesian impedance like a torque-controlled arm, clipped
# so the gripper cannot push objects through the table (a weld to the mocap target can).
KP, KD, F_MAX = 2000.0, 90.0, 40.0  # N/m, N s/m, N  (1 kg gripper: ~45 rad/s, critically damped)
KPR, KDR, T_MAX = 7.2, 0.24, 3.0  # N m/rad, N m s/rad, N m  (~2e-3 kg m^2)


# ---------------------------------------------------------------- frames


def _tuple_block(txt: str, key: str) -> np.ndarray:
    m = re.search(rf"{key}: !!python/tuple\n((?:  - .*\n){{12}})", txt)
    assert m, key
    return np.array([float(x) for x in m.group(1).split("- ")[1:]]).reshape(3, 4)


def extrinsics(seq: D.Seq, root: Path = D.ROOT) -> dict[str, np.ndarray]:
    """{'apriltag': tag->master [3, 4], serial: camera->master [3, 4]}"""
    txt = (
        root / "calibration" / f"extrinsics_{seq.meta['extrinsics']}" / "extrinsics.yml"
    ).read_text()
    out = {"apriltag": _tuple_block(txt, "apriltag")}
    for s in seq.meta["serials"]:
        out[s] = _tuple_block(txt, f"'{s}'")
    return out


def master_to_table(x: np.ndarray, tag: np.ndarray) -> np.ndarray:
    """points [..., 3] in master -> table frame (R^T (x - t))."""
    return (x - tag[:, 3]) @ tag[:, :3]


def cam_to_master(x: np.ndarray, T: np.ndarray) -> np.ndarray:
    return x @ T[:, :3].T + T[:, 3]


def object_poses_table(seq: D.Seq, tag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-> positions [N, n_obj, 3], rotation matrices [N, n_obj, 3, 3] in the table frame."""
    py = np.load(seq.root / "pose.npz")["pose_y"]  # [N, n_obj, 7]
    R = Rot.from_quat(py[..., :4].reshape(-1, 4)).as_matrix().reshape(*py.shape[:2], 3, 3)
    t = py[..., 4:]
    Rt = tag[:, :3].T @ R  # broadcast [3,3] @ [N, n, 3, 3]
    return master_to_table(t, tag), Rt


def gt_hand_joints_table(seq: D.Seq, tag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GT MANO joints [N, 21, 3] in the table frame and a valid mask [N]."""
    pm = np.load(seq.root / "pose.npz")["pose_m"][:, 0]  # [N, 51] master frame
    valid = np.abs(pm).sum(1) > 0
    j = np.full((len(pm), 21, 3), np.nan)
    if valid.any():
        _, jj = D.mano_forward(pm[valid], D.mano_betas(seq), seq.side)
        j[valid] = master_to_table(jj, tag)
    return j, valid


# ---------------------------------------------------------------- metric hand position


def fit_translation(J: np.ndarray, uv: np.ndarray, K: np.ndarray, w: np.ndarray | None = None):
    """Root-relative joints J [21, 3] (metres, camera axes) + their 2D image points uv [21, 2]
    -> camera-frame translation T [3] minimising the reprojection error of J + T (linear least
    squares on the perspective equations fx (X+Tx) - (u-cx)(Z+Tz) = 0)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    du, dv = uv[:, 0] - cx, uv[:, 1] - cy
    n = len(J)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    A[:n, 0], A[:n, 2], b[:n] = fx, -du, du * J[:, 2] - fx * J[:, 0]
    A[n:, 1], A[n:, 2], b[n:] = fy, -dv, dv * J[:, 2] - fy * J[:, 1]
    if w is not None:
        ww = np.concatenate([w, w])
        A, b = A * ww[:, None], b * ww
    return np.linalg.lstsq(A, b, rcond=None)[0]


# ---------------------------------------------------------------- hand -> gripper


OPPOSE = (8, 12)  # fingertips opposing the thumb: index, middle (ring tried 2026-11-14: no
# lower on the cup; the "grasp at the rim" was the monocular hand depth, see cup_pose.depth_scale)


def hand_to_gripper(j: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """21 joints [21, 3] (table frame) -> gripper origin [3], rotation [3, 3] (columns x y z),
    open width (m). Thumb tip vs the index + middle tips (OPPOSE)."""
    pt = j[4]
    pf = j[list(OPPOSE)].mean(0)
    c = (pt + pf) / 2
    y = pf - pt
    width = float(np.linalg.norm(y))
    y = y / max(width, 1e-6)
    palm = j[[0, 5, 9, 13, 17]].mean(0)
    a = c - palm
    z = a - (a @ y) * y
    z = z / max(np.linalg.norm(z), 1e-6)
    x = np.cross(y, z)
    R = np.stack([x, y, z], 1)
    return c - PAD_Z * z, R, width


def hand_frame(j: np.ndarray) -> np.ndarray:
    """21 joints -> 4x4 hand pose: origin = palm centre (wrist + 4 MCPs), x = wrist -> middle MCP,
    y = pinky MCP -> index MCP (orthogonalised), z = x cross y."""
    o = j[[0, 5, 9, 13, 17]].mean(0)
    x = j[9] - j[0]
    x = x / max(np.linalg.norm(x), 1e-6)
    y = j[5] - j[17]
    y = y - (y @ x) * x
    y = y / max(np.linalg.norm(y), 1e-6)
    H = np.eye(4)
    H[:3, :3] = np.stack([x, y, np.cross(x, y)], 1)
    H[:3, 3] = o
    return H


def contact_grasp(j: np.ndarray, verts: np.ndarray, contact: np.ndarray, parts: np.ndarray):
    """Grasp from the hand's contact vertices (one or a few frames averaged by the caller):
    closing axis from the thumb's contact centroid to the other parts' (fingers + palm) contact
    centroid; centre = their mid-point; approach = palm centre -> centre (orthogonalised).
    -> 4x4 gripper-pad pose (origin = pad centre), width; None without thumb-vs-other opposition.
    """
    th = contact & (parts == 1)
    ot = contact & (parts != 1)
    if th.sum() < 3 or ot.sum() < 3:
        return None
    a, b = verts[th].mean(0), verts[ot].mean(0)
    y = b - a
    width = float(np.linalg.norm(y))
    if width < 0.01:
        return None
    y = y / width
    c = (a + b) / 2
    palm = j[[0, 5, 9, 13, 17]].mean(0)
    z = c - palm
    z = z - (z @ y) * y
    z = z / max(np.linalg.norm(z), 1e-6)
    G = np.eye(4)
    G[:3, :3] = np.stack([np.cross(y, z), y, z], 1)
    G[:3, 3] = c
    return G, width


def tip_grasp(j: np.ndarray):
    """Fallback grasp from the thumb / index + middle tips (pad pose, width)."""
    p, R, w = hand_to_gripper(j)
    G = np.eye(4)
    G[:3, :3], G[:3, 3] = R, p + PAD_Z * R[:, 2]
    return G, w


def retarget(joints, ok, close, verts=None, contact=None, parts=None):
    """OFFLINE retargeting of a recorded demo to a parallel gripper.
    For each closing episode (a run of `close`), the grasp G is taken from the contact vertices of
    the episode frame tg with the best thumb-vs-other opposition (tip fallback at its first frame),
    and the gripper is rigidly attached to the hand from the previous episode's end to this
    episode's end: pad pose(t) = H(t) H(tg)^-1 G.
    -> pad positions [N, 3], rotations [N, 3, 3], open widths [N], how each grasp was made."""
    n = len(joints)
    H = np.array([hand_frame(j) if o else np.eye(4) for j, o in zip(joints, ok, strict=True)])
    eps = []
    c = np.concatenate([[False], close, [False]])
    d = np.flatnonzero(np.diff(c.astype(np.int8)))
    for a, e in zip(d[::2], d[1::2], strict=True):
        tc = next((t for t in range(a, e) if ok[t]), None)
        if tc is None:
            continue
        g, how, tg = None, "tips", tc
        if verts is not None:  # the episode frame with the best thumb-vs-other opposition
            fr = [t for t in range(a, e) if ok[t]]
            cb = contact[fr] if contact.dtype == bool else contact[fr] >= 0.5
            score = np.minimum((cb & (parts == 1)).sum(1), (cb & (parts != 1)).sum(1))
            if score.max() >= 3:
                tg = fr[int(np.argmax(score))]
                g = contact_grasp(joints[tg], verts[tg], cb[int(np.argmax(score))], parts)
                how = "contact" if g is not None else how
        if g is None:
            tg = tc
            g = tip_grasp(joints[tc])
        eps.append((a, e, tg, g[0], g[1], how))
    if not eps:  # no grasp: follow the tips
        tc = int(np.flatnonzero(ok)[0])
        G, w = tip_grasp(joints[tc])
        eps = [(0, n, tc, G, w, "none")]
    X = np.zeros((n, 4, 4))
    W = np.zeros(n)
    start = 0
    for k, (_, e, tc, G, w, _) in enumerate(eps):
        end = e if k < len(eps) - 1 else n
        rel = np.linalg.inv(H[tc]) @ G
        X[start:end] = rel
        W[start:end] = w
        start = end
    pose = H @ X
    return pose[:, :3, 3], pose[:, :3, :3], W, [ep[5] for ep in eps]


def pregrasp(
    pos: np.ndarray,
    R: np.ndarray,
    close: np.ndarray,
    n_app: int = 12,
    back: float = 0.08,
    n_tr: int = 15,
    floor: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """OFFLINE robot-side approach (the robot replays a recorded demo, so it knows where the
    grasp will be): before each close episode starting at frame a, instead of copying the
    human's (often sideways) path, the gripper blends over n_tr frames to a pre-grasp pose `back`
    m behind the grasp along its own approach axis (pad z), then moves straight in over n_app
    frames with the grasp orientation. floor=True: while travelling to the pre-grasp pose the
    gripper stays at or above its height (does not sweep low across the table).
    pad poses [N, 3] / [N, 3, 3] -> new arrays."""
    from scipy.spatial.transform import Slerp

    pos, R = pos.copy(), R.copy()
    c = np.r_[False, close.astype(bool)]
    starts = np.flatnonzero(~c[:-1] & c[1:])
    ends = np.flatnonzero(c[:-1] & ~c[1:])
    prev_end = 0
    for a in starts:
        Ga, Ra = pos[a].copy(), R[a].copy()
        pre = Ga - back * Ra[:, 2]
        for t in range(max(prev_end, a - n_app), a):
            u = (t - (a - n_app)) / n_app
            pos[t], R[t] = pre + u * (Ga - pre), Ra
        t0 = max(prev_end, a - n_app - n_tr)
        for t in range(t0, max(prev_end, a - n_app)):
            u = (t - (a - n_app - n_tr) + 1) / (n_tr + 1)
            sl = Slerp([0, 1], Rot.from_matrix(np.stack([R[t], Ra])))
            pos[t], R[t] = (1 - u) * pos[t] + u * pre, sl(u).as_matrix()
            if floor:
                pos[t, 2] = max(pos[t, 2], pre[2])
        later = ends[ends >= a]
        prev_end = int(later[0]) if len(later) else len(close)
    return pos, R


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


def object_grasp(G0: np.ndarray, pts: np.ndarray, n_rot: int = 12, open_margin: float = 0.02,
                 min_pts: int = 20, table_clear: float = 0.003):  # fmt: skip
    """Object-centred parallel-jaw grasp near the human's: G0 = the human-derived pad pose (4x4,
    table frame), pts = the target object's surface points [P, 3] (table frame, from the robot's
    perception / known scene). Candidates: approach = the human's, straight down, or between;
    12 rotations about it; centre moved 0 / 30 / 60% towards the object's centroid and 0 / 1.5 /
    3 cm deeper along the approach. Kept if >= min_pts object points lie between the finger pads,
    the object is not wider at the fingertips than mid-finger (no wedge), it fits the stroke, the open
    fingers (half opening = object half-width + open_margin) and the palm touch no object point,
    and the fingertips stay above the table. Cost: distance to the human grasp (per 3 cm) +
    closing-axis and approach-axis angle to the human's (per 45 deg) - 0.5 x approach downwards.
    -> (4x4 pad pose with the centre on the object's mid-line, grasp width) or None."""
    c0, y0, z0 = G0[:3, 3], G0[:3, 1], G0[:3, 2]
    down = np.array([0.0, 0.0, -1.0])
    cen = pts.mean(0)
    best, best_cost = None, np.inf
    for z in {tuple(_unit(z0)), tuple(down), tuple(_unit(_unit(z0) + down))}:
        z = np.array(z)
        u = _unit(np.cross(z, [1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0]))
        v = np.cross(z, u)
        for ang in np.linspace(0, np.pi, n_rot, endpoint=False):
            y = np.cos(ang) * u + np.sin(ang) * v
            R = np.stack([np.cross(y, z), y, z], 1)
            for s, dz in [(s, dz) for s in (0.0, 0.3, 0.6) for dz in (0.0, 0.015, 0.03)]:
                c = c0 + s * (cen - c0) + dz * z  # dz: deeper onto the object
                q = (pts - c) @ R
                slab = (np.abs(q[:, 0]) <= 0.011) & (q[:, 2] >= TIP_PAST - FINGER_LEN) & (
                    q[:, 2] <= TIP_PAST)  # fmt: skip
                if slab.sum() < min_pts:
                    continue
                lo, hi = q[slab, 1].min(), q[slab, 1].max()
                width = float(hi - lo)
                # no wedge: the object must not be wider at the fingertips than mid-finger (the
                # jaws would touch only the tips and squeeze it out, like pinching a cone)
                qs = q[slab]
                tips = qs[qs[:, 2] >= TIP_PAST - 0.015]
                mid = qs[np.abs(qs[:, 2]) <= 0.01]
                if len(mid) < 5 or (len(tips) and np.ptp(tips[:, 1]) > np.ptp(mid[:, 1]) + 0.003):
                    continue
                half = width / 2 + open_margin
                if half > MAX_HALF:
                    continue
                cy = (lo + hi) / 2
                q2 = q - [0.0, cy, 0.0]
                fing = ((np.abs(q2[:, 0]) <= 0.014) & (np.abs(q2[:, 1]) >= half - 0.002)
                        & (np.abs(q2[:, 1]) <= half + 0.015)
                        & (q2[:, 2] >= TIP_PAST - FINGER_LEN - 0.003)
                        & (q2[:, 2] <= TIP_PAST + 0.003))  # fmt: skip
                palm = ((np.abs(q2[:, 0]) <= 0.017) & (np.abs(q2[:, 1]) <= 0.095)
                        & (np.abs(q2[:, 2] + PAD_Z) <= 0.02))  # fmt: skip
                if fing.any() or palm.any():
                    continue
                c2 = c + cy * y
                tip = c2 + TIP_PAST * z
                low = min((tip + a * 0.011 * R[:, 0] + b * (half + 0.012) * y)[2]
                          for a in (-1, 1) for b in (-1, 1))  # fmt: skip
                if low < table_clear:
                    continue
                cost = (np.linalg.norm(c2 - c0) / 0.03
                        + np.arccos(min(1.0, abs(float(y @ y0)))) / (np.pi / 4)
                        + np.arccos(np.clip(float(z @ _unit(z0)), -1, 1)) / (np.pi / 4)
                        - 0.5 * float(z @ down))  # fmt: skip
                if cost < best_cost:
                    G = np.eye(4)
                    G[:3, :3], G[:3, 3] = R, c2
                    best, best_cost = (G, width), cost
    return best


V2 = {"margin": 0.04, "close_speed": 0.2, "wait_close": 15}  # robot v2 replay settings


def object_refine(pos, R, W, close, objects_at, max_dist: float = 0.08):
    """Move each grasp episode onto the object it grasps (object_grasp), in the ROBOT's scene.
    objects_at(t) -> per object (points [P, 3] in the demo at frame t, M [4, 4]) where M maps
    demo-frame-t coordinates to the robot's scene (the object's pose there times its demo pose
    at t inverted; identity when the robot acts in the demo scene at that moment). For the
    episode closing at frame a: the object nearest the human pad centre (demo) is taken, the
    human grasp G0 is carried along with it (M G0), the object grasp G' is searched on its scene
    points, and C = G0^-1 G' is applied to every pad pose of the episode's hand attachment
    (previous episode end .. this episode end): the robot keeps the human's motion relative to
    the grasp. Episodes with no object within max_dist or no valid grasp are kept.
    -> pos, R, W (new arrays), list of "object" / "human" per episode."""
    pos, R, W = pos.copy(), R.copy(), W.copy()
    n = len(close)
    c = np.r_[False, close.astype(bool), False]
    starts = np.flatnonzero(~c[:-1] & c[1:])
    ends = np.flatnonzero(c[:-1] & ~c[1:])
    how, prev = [], 0
    for k, (a, e) in enumerate(zip(starts, ends, strict=True)):
        stop = e if k < len(starts) - 1 else n
        G0 = np.eye(4)
        G0[:3, :3], G0[:3, 3] = R[a], pos[a]
        objs = objects_at(int(a))
        d = [np.linalg.norm(o - pos[a], axis=1).min() for o, _ in objs]
        g = None
        if d and min(d) <= max_dist:
            pts, M = objs[int(np.argmin(d))]
            g = object_grasp(M @ G0, pts @ M[:3, :3].T + M[:3, 3])
        if g is None:
            how.append("human")
        else:
            C = np.linalg.inv(G0) @ g[0]
            for t in range(prev, stop):
                P = np.eye(4)
                P[:3, :3], P[:3, 3] = R[t], pos[t]
                P = P @ C
                R[t], pos[t] = P[:3, :3], P[:3, 3]
            W[prev:stop] = g[1]
            how.append("object")
        prev = stop
    return pos, R, W, how


def fill_gaps(close: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """Merge close episodes separated by <= max_gap open frames (a flickering hold signal must
    not make the robot open and re-grasp). OFFLINE (needs the next episode's start); live use
    would delay the release by max_gap frames instead."""
    out = close.astype(bool).copy()
    c = np.r_[False, out, False]
    starts = np.flatnonzero(~c[:-1] & c[1:])
    ends = np.flatnonzero(c[:-1] & ~c[1:])
    for e, a in zip(ends[:-1], starts[1:], strict=True):
        if a - e <= max_gap:
            out[e:a] = True
    return out


def debounce(raw: np.ndarray, on: int = 2, off: int = 3) -> np.ndarray:
    """bool [N] -> close command: closes after `on` consecutive True, opens after `off` False."""
    out = np.zeros(len(raw), bool)
    state, run = False, 0
    for i, r in enumerate(raw):
        if r != state:
            run += 1
            if run >= (on if r else off):
                state, run = bool(r), 0
        else:
            run = 0
        out[i] = state
    return out


def grasp_signal(state: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Model hand-state probabilities [N, 3] -> raw close signal [N]: TOUCH or HOLD. (DexYCB's
    HOLD label needs the thumb within 3 mm, which misses e.g. mug-handle grasps, so any contact
    closes the gripper.) Frames without a hand keep the previous decision (causal hold)."""
    out = np.zeros(len(state), bool)
    prev = False
    for i in range(len(state)):
        prev = bool(state[i].argmax() >= 1) if present[i] else prev
        out[i] = prev
    return out


def gt_grasp_signal(state: np.ndarray) -> np.ndarray:
    """GT states (-1 = unsure: keep the previous value) -> raw close signal [N]."""
    out = np.zeros(len(state), bool)
    prev = False
    for i, s in enumerate(state):
        prev = bool(s >= 1) if s >= 0 else prev
        out[i] = prev
    return out


# ---------------------------------------------------------------- MuJoCo scene


@dataclass
class Scene:
    xml: str
    names: list[str]  # YCB object names, body "obj{i}"
    grasp_index: int


def build_scene(seq: D.Seq, p0: np.ndarray, R0: np.ndarray, root: Path = D.ROOT,
                stiff: bool = False) -> Scene:  # fmt: skip
    """Table plane, the sequence's YCB objects (free bodies at frame-0 poses, lifted so that each
    rests on z = 0; collision = MuJoCo's convex hull of the mesh, density 300 kg/m^3: an
    assumption, YCB masses are not used), and a mocap parallel gripper."""
    models = sorted(p for p in (root / "models").iterdir() if p.is_dir())
    names = [models[i - 1].name for i in seq.meta["ycb_ids"]]
    assets, bodies = [], []
    for i, nm in enumerate(names):
        pts = D.object_points(nm, 4000)
        z = (pts @ R0[i].T + p0[i])[:, 2].min()
        pos = p0[i] - [0, 0, z - 0.001]
        q = Rot.from_matrix(R0[i]).as_quat()[[3, 0, 1, 2]]  # wxyz
        col = "0.85 0.3 0.25 1" if i == seq.meta["ycb_grasp_ind"] else "0.55 0.6 0.7 1"
        assets.append(
            f'<mesh name="m{i}" file="{(root / "models" / nm).resolve()}/textured_simple.obj"/>'
        )
        bodies.append(
            f'<body name="obj{i}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{q[0]} {q[1]} {q[2]} {q[3]}">'
            f'<freejoint/><geom type="mesh" mesh="m{i}" density="300" rgba="{col}" '
            f'friction="1.0 0.02 0.001" condim="4"/></body>'
        )
    return Scene(scene_xml("".join(assets), "".join(bodies), stiff), names,
                 int(seq.meta["ycb_grasp_ind"]))  # fmt: skip


def scene_xml(assets: str, bodies: str, stiff: bool = False) -> str:
    """MJCF: table plane at z = 0, the given object bodies ("obj{i}", free joints) and the
    gripper (free body welded to the mocap "target", two position-controlled slide fingers).
    stiff=True: contact time constant 4 ms (2 x timestep) instead of MuJoCo's default 20 ms:
    with the soft default a closing finger sinks ~3 cm into the object and squirts it out."""
    dflt = '<default><geom solref="0.004 1" solimp="0.95 0.99 0.001"/></default>' if stiff else ""
    return f"""<mujoco model="dexycb_replay">
  <option timestep="0.002" cone="elliptic" impratio="10"/>{dflt}
  <visual><global offwidth="960" offheight="720"/><quality shadowsize="2048"/></visual>
  <asset>{assets}</asset>
  <worldbody>
    <light pos="0.5 0.3 1.6" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.3 -0.5 1.2" dir="0.3 0.5 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="table" type="plane" pos="0.5 0.3 0" size="1 1 0.01" rgba="0.32 0.3 0.28 1"
          friction="0.8 0.02 0.001"/>
    {bodies}
    <body name="target" mocap="true" pos="0 0 1"/>
    <body name="gripper" pos="0 0 1" gravcomp="1">
      <freejoint name="gfree"/>
      <geom name="palm" mass="0.8" type="box" size="0.012 0.09 0.015" rgba="0.15 0.15 0.18 1"/>
      <body name="finger_l" gravcomp="1">
        <joint name="jl" type="slide" axis="0 1 0" range="0 {MAX_HALF}" damping="20"/>
        <geom type="box" pos="0 0.006 {PAD_Z + TIP_PAST - FINGER_LEN / 2}" size="0.011 0.006 {FINGER_LEN / 2}" mass="0.1"
              rgba="0.2 0.6 0.9 1" friction="1.5 0.05 0.002" condim="4"/>
      </body>
      <body name="finger_r" gravcomp="1">
        <joint name="jr" type="slide" axis="0 -1 0" range="0 {MAX_HALF}" damping="20"/>
        <geom type="box" pos="0 -0.006 {PAD_Z + TIP_PAST - FINGER_LEN / 2}" size="0.011 0.006 {FINGER_LEN / 2}" mass="0.1"
              rgba="0.2 0.6 0.9 1" friction="1.5 0.05 0.002" condim="4"/>
      </body>
    </body>
  </worldbody>
  <equality><weld body1="gripper" body2="target" solref="0.004 1"/></equality>
  <contact><exclude body1="gripper" body2="finger_l"/><exclude body1="gripper" body2="finger_r"/></contact>
  <actuator>
    <position name="al" joint="jl" kp="2000" forcerange="-40 40" ctrlrange="0 {MAX_HALF}"/>
    <position name="ar" joint="jr" kp="2000" forcerange="-40 40" ctrlrange="0 {MAX_HALF}"/>
  </actuator>
</mujoco>"""


def _interp_poses(pos: np.ndarray, R: np.ndarray, ok: np.ndarray):
    """Fill frames without a hand pose: linear / slerp between known frames, hold at the ends."""
    from scipy.spatial.transform import Slerp

    idx = np.flatnonzero(ok)
    n = len(pos)
    t = np.arange(n)
    tc = np.clip(t, idx[0], idx[-1])
    P = np.stack([np.interp(tc, idx, pos[idx, k]) for k in range(3)], 1)
    Q = Slerp(idx, Rot.from_matrix(R[idx]))(tc).as_matrix() if len(idx) > 1 else R[idx[[0] * n]]
    return P, Q


def smooth(x: np.ndarray, ok: np.ndarray, k: int = 5) -> np.ndarray:
    """Centred moving average over valid frames (OFFLINE post-processing of a recorded demo)."""
    out = x.copy()
    h = k // 2
    for i in np.flatnonzero(ok):
        a, b = max(0, i - h), min(len(x), i + h + 1)
        m = ok[a:b]
        out[i] = x[a:b][m].mean(0)
    return out


def object_between_pads(pad_pos, pad_R, width, obj_pts, min_pts: int = 20, slack: float = 0.02):
    """Robot-side close gate: is an object between the fingers? pad poses [N, 3] / [N, 3, 3],
    grasp width [N], object surface points per frame [N, P, 3] (table frame; from the robot's
    perception) -> bool [N]: >= min_pts points inside the jaw slab |x| <= 1.1 cm (finger width),
    |y| <= width / 2 + slack (closing axis), z in [TIP_PAST - FINGER_LEN, TIP_PAST] (finger span)."""
    out = np.zeros(len(pad_pos), bool)
    for t in range(len(pad_pos)):
        q = (obj_pts[t] - pad_pos[t]) @ pad_R[t]  # into the pad frame (columns = axes)
        inside = (
            (np.abs(q[:, 0]) <= 0.011)
            & (np.abs(q[:, 1]) <= width[t] / 2 + slack)
            & (q[:, 2] >= TIP_PAST - FINGER_LEN)
            & (q[:, 2] <= TIP_PAST)
        )
        out[t] = inside.sum() >= min_pts
    return out


def gate_close(close: np.ndarray, ready: np.ndarray) -> np.ndarray:
    """A close episode may only START on a frame where `ready` (object between the pads); once
    closed it stays closed until the command ends (the object is in the hand by then)."""
    out = np.zeros(len(close), bool)
    for t in range(len(close)):
        out[t] = close[t] and (ready[t] or (t > 0 and out[t - 1]))
    return out


def above_table(pad_pos, pad_R, half_open, clear: float = 0.004) -> np.ndarray:
    """Lift the gripper (world z) where its fingertips would go below the table: the human's
    fingers can slide under an object's edge, a rigid gripper cannot."""
    out = pad_pos.copy()
    for t in range(len(pad_pos)):
        R = pad_R[t]
        tip = pad_pos[t] + TIP_PAST * R[:, 2]
        corners = [tip + a * 0.011 * R[:, 0] + b * half_open[t] * R[:, 1] for a in (-1, 1)
                   for b in (-1, 1)]  # fmt: skip
        low = min(c[2] for c in corners)
        if low < clear:
            out[t, 2] += clear - low
    return out


def _quat_err(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking orientation qb to qa (wxyz quaternions)."""
    import mujoco

    qe, qn, v = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(qn, qb)
    mujoco.mju_mulQuat(qe, qa, qn)
    if qe[0] < 0:
        qe = -qe
    mujoco.mju_quat2Vel(v, qe, 1.0)
    return v


def replay(scene: Scene, pad_pos: np.ndarray, pad_R: np.ndarray, width: np.ndarray,
           ok: np.ndarray, close: np.ndarray, render=None, lift_table: bool = False,
           drive: str = "force", margin: float = MARGIN,
           close_speed: float | None = None, wait_close: int = 0) -> dict:  # fmt: skip
    """Drive the gripper: pad-centre poses [N, 3] / [N, 3, 3] (table frame; ok [N] = known),
    open width [N], close [N] -> per-frame object positions [N, n_obj, 3] (table frame), finger
    opening [N], and frames from `render(model, data, t)` if given.
    drive="force": clipped PD force / torque on the gripper towards the target (default);
    drive="weld": the old stiff weld to the mocap target (unbounded force: can crush objects
    into the table). margin: open width = grasp width + margin (capped by the stroke).
    close_speed (m/s per finger): ramp the finger targets at most this fast (a real gripper
    closes at ~0.1-0.3 m/s); None = jump (the position servo then slams them shut).
    wait_close (frames): while the fingers are still closing, the arm holds its pose (the demo
    clock pauses) for up to this many frames, then continues the demo path late by that much."""
    import mujoco

    m = mujoco.MjModel.from_xml_string(scene.xml)
    d = mujoco.MjData(m)
    assert drive in ("force", "weld")
    if drive == "force":
        d.eq_active[:] = 0
    gid = m.body("gripper").id
    vel6 = np.zeros(6)
    n = len(pad_pos)
    half = np.where(close, 0.5 * width, np.minimum(MAX_HALF, (width + margin) / 2))
    if lift_table:  # off by default: the table collides with the fingers anyway (val ablation)
        pad_pos = above_table(pad_pos, pad_R, half)
    pos = pad_pos - PAD_Z * pad_R[:, :, 2]  # gripper origin
    P, Q = _interp_poses(pos, pad_R, ok)
    first = int(np.flatnonzero(ok)[0])
    obj_ids = [m.body(f"obj{i}").id for i in range(len(scene.names))]
    jl, jr = m.jnt_qposadr[m.joint("jl").id], m.jnt_qposadr[m.joint("jr").id]
    traj = np.full((n, len(obj_ids), 3), np.nan)
    opening = np.zeros(n)
    frames = []
    mujoco.mj_forward(m, d)
    for _ in range(200):  # let the objects settle on the table (gripper parked above)
        mujoco.mj_step(m, d)
    q_prev = p_prev = None
    lag, open_prev = 0, None
    for t in range(n):
        ti = t
        if wait_close:
            ti = max(first, t - lag) if t >= first else t
        if t >= first:
            q1 = Rot.from_matrix(Q[ti]).as_quat()[[3, 0, 1, 2]]
            if p_prev is None:  # place the gripper (free body, welded to the mocap target)
                p_prev, q_prev = P[ti], q1
                adr = m.jnt_qposadr[m.joint("gfree").id]
                d.qpos[adr : adr + 3], d.qpos[adr + 3 : adr + 7] = P[ti], q1
                d.mocap_pos[0], d.mocap_quat[0] = P[ti], q1
                mujoco.mj_forward(m, d)
            half = 0.0 if close[ti] else min(MAX_HALF, (width[ti] + margin) / 2)
            if close_speed is None:
                d.ctrl[:] = half
            if np.dot(q_prev, q1) < 0:
                q1 = -q1
            dt = SUB * m.opt.timestep
            v_t = (P[ti] - p_prev) / dt  # feed-forward target velocities
            w_t = _quat_err(q1, q_prev) / dt
            for s in range(SUB):  # interpolate the target pose within the frame
                if close_speed is not None:
                    step = close_speed * m.opt.timestep
                    d.ctrl[:] = np.clip(half, d.ctrl - step, d.ctrl + 4 * step)  # opens faster
                a = (s + 1) / SUB
                d.mocap_pos[0] = (1 - a) * p_prev + a * P[ti]
                qa = (1 - a) * q_prev + a * q1
                d.mocap_quat[0] = qa / np.linalg.norm(qa)
                if drive == "force":
                    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, gid, vel6, 0)
                    f = KP * (d.mocap_pos[0] - d.xpos[gid]) + KD * (v_t - vel6[3:])
                    tq = KPR * _quat_err(d.mocap_quat[0], d.xquat[gid]) + KDR * (w_t - vel6[:3])
                    d.xfrc_applied[gid, :3] = f * min(1.0, F_MAX / max(np.linalg.norm(f), 1e-9))
                    d.xfrc_applied[gid, 3:] = tq * min(1.0, T_MAX / max(np.linalg.norm(tq), 1e-9))
                mujoco.mj_step(m, d)
            p_prev, q_prev = P[ti], q1
        traj[t] = d.xpos[obj_ids]
        opening[t] = d.qpos[jl] + d.qpos[jr]
        if wait_close and t >= first:
            if close[ti] and open_prev is not None and open_prev - opening[t] > 0.001:
                lag = min(lag + 1, wait_close)
            open_prev = opening[t]
        if render is not None:
            frames.append(render(m, d, t))
    return {"obj": traj, "opening": opening, "frames": frames}


def lift_outcome(gt_pos: np.ndarray, sim_pos: np.ndarray, lift_m: float = 0.05) -> dict | None:
    """Grasped object: GT [N, 3] vs sim [N, 3] positions (table frame). None when the human did
    not lift it >= lift_m. success = the sim object rose >= half the human's peak lift, at the
    frame of that peak; held_end = same test at the last frame."""
    gain = gt_pos[:, 2] - gt_pos[0, 2]
    k = int(np.argmax(gain))
    if gain[k] < lift_m:
        return None
    sg = sim_pos[:, 2] - sim_pos[0, 2]
    end_gain = gain[-1]
    return {
        "gt_lift_m": float(gain[k]),
        "success": bool(sg[k] >= 0.5 * gain[k]),
        "held_end": bool(sg[-1] >= 0.5 * end_gain) if end_gain >= lift_m else None,
        "pos_err_peak_m": float(np.linalg.norm(sim_pos[k] - gt_pos[k])),
    }
