"""Pick a camera angle where the robot arm does not hide the cup, then render a run from its
recorded states (Andy 2026-11-14: "if the arm blocks the view, use an angle that does not").

How: after the simulation, candidate free cameras around the scene (the scene's own "view"
camera turned by AZ_OFFS degrees about the vertical, at its elevation and a higher one) are
scored by casting rays from the camera to points on the cup (centre, top, rim) at sampled
steps of the run (mj_ray on visible geoms): a point counts when it is inside the image and the
first thing the ray hits is the cup. Score = mean visibility + 0.5 x the 10th percentile (no
long blocked stretches) - a small penalty for turning away from the original view. The original
view is kept when it is already clear (CLEAR_MEAN / CLEAR_P10).
"""

from __future__ import annotations

import math

import numpy as np

AZ_OFFS = (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)  # deg
EL_UP = 15.0  # deg: the higher candidate elevation, above the original
CLEAR_MEAN, CLEAR_P10 = 0.75, 0.5  # the gripper's own fingers hide ~20% of a held cup
TURN_PENALTY = 0.0008  # score per degree turned away from the original view
VIS_GROUPS = np.array([1, 0, 1, 0, 0, 0], np.uint8)  # geom groups rays see: 0 (scene, cup), 2
# (robot visual meshes); not 1 (the drawn goal) or 3 (robot collision geoms, not rendered)


def free_from_named(m, d, name: str = "view") -> dict:
    """The named camera (after mj_forward) as free-camera parameters."""
    cid = m.camera(name).id
    p = d.cam_xpos[cid].copy()
    f = -d.cam_xmat[cid].reshape(3, 3)[:, 2]  # a camera looks along its -z
    try:
        look = d.xpos[m.body("look").id].copy()
    except KeyError:
        look = p + 0.6 * f
    dist = float(np.linalg.norm(look - p))
    f = (look - p) / max(dist, 1e-9)
    return {"lookat": look, "distance": dist, "azimuth": math.degrees(math.atan2(f[1], f[0])),
            "elevation": math.degrees(math.asin(np.clip(f[2], -1, 1))),
            "fovy": float(m.cam_fovy[cid])}  # fmt: skip


def cam_pose(c: dict) -> tuple[np.ndarray, np.ndarray]:
    """Free camera -> (position, forward unit vector) (MuJoCo's free-camera convention)."""
    az, el = math.radians(c["azimuth"]), math.radians(c["elevation"])
    f = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return np.asarray(c["lookat"], float) - c["distance"] * f, f


def cup_points(m, d, body: str = "obj0") -> np.ndarray:
    """Points on the cup: centre, top centre, top rim and mid-height rim (8 each)."""
    b = m.body(body).id
    c, R = d.xpos[b], d.xmat[b].reshape(3, 3)
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
    r, hh = (float(m.geom_size[gs[0]][0]), float(m.geom_size[gs[0]][1])) if gs else (0.03, 0.045)
    a = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ring = np.stack([r * np.cos(a), r * np.sin(a), np.zeros(8)], 1)
    loc = np.vstack([[0, 0, 0], [0, 0, hh], ring + [0, 0, hh], ring])
    return c + loc @ R.T


def visibility(m, d, c: dict, aspect: float, body: str = "obj0") -> float:
    """Share of the cup's points the camera sees (in the image, nothing in front)."""
    import mujoco

    pos, f = cam_pose(c)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(f, up)
    x /= max(np.linalg.norm(x), 1e-9)
    y = np.cross(x, f)
    t = math.tan(math.radians(c["fovy"]) / 2)
    b = m.body(body).id
    gid = np.zeros(1, np.int32)
    seen = 0
    pts = cup_points(m, d, body)
    for p in pts:
        v = p - pos
        z = v @ f
        if z <= 0 or abs(v @ y) / z > t or abs(v @ x) / z > t * aspect:
            continue
        dist = float(np.linalg.norm(v))
        hit = mujoco.mj_ray(m, d, pos, v / dist, VIS_GROUPS, 1, -1, gid)
        if gid[0] >= 0 and m.geom_bodyid[gid[0]] == b and hit <= dist + 0.05:
            seen += 1
    return seen / len(pts)


def best_view(m, states, aspect: float, name: str = "view", n_samples: int = 24,
              body: str = "obj0") -> tuple[dict, dict]:  # fmt: skip
    """-> (free camera, info): the clearest candidate over sampled recorded states [T, nq+nv]."""
    import mujoco

    d = mujoco.MjData(m)
    d.qpos[:] = states[0][: m.nq]
    mujoco.mj_forward(m, d)
    base = free_from_named(m, d, name)
    idx = np.unique(np.linspace(0, len(states) - 1, n_samples).astype(int))
    cands = []
    for el in (base["elevation"], max(base["elevation"] - EL_UP, -80.0)):
        for off in AZ_OFFS:
            cands.append({**base, "azimuth": base["azimuth"] + off, "elevation": el,
                          "_turn": abs(off) + abs(el - base["elevation"])})  # fmt: skip
    vis = np.zeros((len(cands), len(idx)))
    for j, t in enumerate(idx):
        d.qpos[:] = states[t][: m.nq]
        mujoco.mj_forward(m, d)
        for i, c in enumerate(cands):
            vis[i, j] = visibility(m, d, c, aspect, body)
    mean, p10 = vis.mean(1), np.percentile(vis, 10, axis=1)
    score = mean + 0.5 * p10 - TURN_PENALTY * np.array([c["_turn"] for c in cands])
    k = 0 if (mean[0] >= CLEAR_MEAN and p10[0] >= CLEAR_P10) else int(np.argmax(score))
    info = {"turned_deg": round(cands[k]["azimuth"] - base["azimuth"], 1),
            "raised_deg": round(base["elevation"] - cands[k]["elevation"], 1),
            "cup_visible": round(float(mean[k]), 2),
            "cup_visible_original": round(float(mean[0]), 2)}  # fmt: skip
    return {kk: v for kk, v in cands[k].items() if kk != "_turn"}, info


def render_states(m, states, cam: dict, w: int, h: int, every: int = 1):
    """Yield (step, BGR image) of the recorded states seen from the free camera."""
    import mujoco

    mc = mujoco.MjvCamera()
    if "fixed" in cam:  # a model camera, moved (real_view)
        m.cam_pos[cam["fixed"]] = cam["pos"]  # worldbody camera: its pose is in world
        m.cam_quat[cam["fixed"]] = cam["quat"]
        mc.type = mujoco.mjtCamera.mjCAMERA_FIXED
        mc.fixedcamid = cam["fixed"]
    else:
        m.vis.global_.fovy = cam["fovy"]
        mc.type = mujoco.mjtCamera.mjCAMERA_FREE
        mc.lookat[:] = cam["lookat"]
        mc.distance, mc.azimuth, mc.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, h, w)
    try:
        for t in range(0, len(states), every):
            d.qpos[:] = states[t][: m.nq]
            d.qvel[:] = states[t][m.nq : m.nq + m.nv]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=mc)
            yield t, np.ascontiguousarray(r.render()[..., ::-1])
    finally:
        r.close()


# ---- the real camera's perspective (Andy 2026-11-14: the robot seen from the same side as the
# webcam, so the cup moves in the same screen direction as the hand on the left)
FIT = 0.62  # the robot, the cup and their paths fill at most this share of the image
REAL = "rec"  # the calibrated webcam camera (touch.live_robot.camera_xml)


VIEW_ELEV = -32.0  # deg: the real camera tilted down to this (Andy 2026-11-14: not top view)


HEAD = "vhead"  # optional model camera whose forward direction (horizontal part) is the render
# heading, chosen per take by match_heading so the cup's path looks as in the webcam video


def heading_xml(hd) -> str:
    """Render heading (table-plane unit vector) -> a camera line for the scene (HEAD)."""
    hd = np.asarray(hd, float)[:2] / max(float(np.linalg.norm(np.asarray(hd)[:2])), 1e-9)
    x = np.r_[hd[1], -hd[0], 0.0]  # right of the heading, so -z (forward) = heading
    return f'<camera name="{HEAD}" pos="0 0 -1" xyaxes="{x[0]} {x[1]} 0 0 0 1"/>'


def view_axes(hd, elev: float = VIEW_ELEV):
    """Heading (table plane) + elevation -> camera rotation [right, up, back] (upright)."""
    hd = np.asarray(hd, float)[:2] / max(float(np.linalg.norm(np.asarray(hd)[:2])), 1e-9)
    e = math.radians(elev)
    f = np.array([math.cos(e) * hd[0], math.cos(e) * hd[1], math.sin(e)])
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return np.stack([x, y, -f], 1)


def match_heading(R0, p0, pts, dirs=(), elev: float = VIEW_ELEV, dist: float = 1.2):
    """The upright render heading whose image shows `pts` (cup positions on its path, the goal)
    and `dirs` [(point, direction)] (e.g. the pour: the cup top's lean) in the same on-screen
    directions as the real camera (R0 = its [right, up, back] axes, p0 = its position; mirror-free
    both). -> (heading [2], mean on-screen direction mismatch in deg)."""
    pts = np.asarray(pts, float)

    def screen(R, c, P):
        v = (P - c) @ R
        return v[:, :2] / np.maximum(-v[:, 2:3], 1e-3)

    def disp(R, c):  # on-screen unit displacements: point pairs + the directions
        S = screen(R, c, pts)
        D = [S[i] - S[0] for i in range(1, len(S))]
        for p, u in dirs:
            a, b = screen(R, c, np.array([p, np.asarray(p) + 0.05 * np.asarray(u)]))
            D.append(b - a)
        D = np.array(D)
        n = np.linalg.norm(D, axis=1, keepdims=True)
        return D / np.maximum(n, 1e-9), n[:, 0]

    real, w = disp(np.asarray(R0, float), np.asarray(p0, float))
    w = w / max(w.sum(), 1e-9)
    w[len(pts) - 1 :] = max(w.max(), 1.0 / max(len(w), 1))  # the directions count fully
    cen = pts.mean(0)
    best = (None, 1e9)
    for a in np.radians(np.arange(0, 360, 2)):
        hd = np.array([math.cos(a), math.sin(a)])
        R = view_axes(hd, elev)
        rend, _ = disp(R, cen + R[:, 2] * dist)
        ang = np.degrees(np.arccos(np.clip((rend * real).sum(1), -1, 1)))
        err = float((w * ang).sum() / w.sum())
        if err < best[1]:
            best = (hd, err)
    return best


def real_view(m, states, aspect: float, name: str = REAL, n_samples: int = 24) -> dict | None:
    """-> a fixed-camera spec {"fixed", "pos", "quat", "fovy"}: looking the way the real camera's
    image "up" points (its top = away), lowered to VIEW_ELEV, image right = the real camera's
    (left / right on screen as in the webcam image), field of view the real one's; placed
    (sideways in its image plane, then back along its axis) so every robot body and the cup
    over sampled recorded states fit in FIT of the image. None: no such camera."""
    import mujoco

    try:
        cid = m.camera(name).id
    except KeyError:
        return None
    d = mujoco.MjData(m)
    d.qpos[:] = states[0][: m.nq]
    mujoco.mj_forward(m, d)
    R0 = d.cam_xmat[cid].reshape(3, 3)  # columns: image right, image up, backwards
    try:  # the take's matched heading (match_heading), the same for its generated scenes
        Rh = d.cam_xmat[m.camera(HEAD).id].reshape(3, 3)
        hd = -Rh[:2, 2]
    except KeyError:
        hd = R0[:2, 1] - R0[:2, 2]  # horizontal: image up + forward
    R = view_axes(hd)
    t = math.tan(math.radians(m.cam_fovy[cid]) / 2) * FIT
    tx = t * aspect
    pts = []
    for k in np.unique(np.linspace(0, len(states) - 1, n_samples).astype(int)):
        d.qpos[:] = states[k][: m.nq]
        mujoco.mj_forward(m, d)
        pts.append(d.xpos[1:].copy())  # every body (robot links, cup), not the world
    P = np.concatenate(pts)
    c = P.mean(0)
    Q = (P - c) @ R  # camera axes, origin at the content's centre
    cx = 0.5 * (Q[:, 0].max() + Q[:, 0].min())
    cy = 0.5 * (Q[:, 1].max() + Q[:, 1].min())
    # distance in front of the centre so every point is inside the frustum
    dist = max(float(np.max(np.abs(Q[:, 0] - cx) / tx + Q[:, 2])),
               float(np.max(np.abs(Q[:, 1] - cy) / t + Q[:, 2])), 0.2)  # fmt: skip
    pos = c + R @ np.array([cx, cy, dist])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return {"fixed": cid, "pos": pos, "quat": quat, "fovy": float(m.cam_fovy[cid]),
            "dist_m": round(dist, 2)}  # fmt: skip


def view_for(m, states, aspect: float) -> tuple[dict, dict]:
    """The camera to show a run with: the real camera's perspective when the scene has it,
    else the clearest angle around the scene's "view" camera (best_view)."""
    rv = real_view(m, states, aspect)
    if rv is not None:
        return rv, {"real_camera": True, "dist_m": rv["dist_m"]}
    return best_view(m, states, aspect)
