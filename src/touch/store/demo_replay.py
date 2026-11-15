"""Reproduce a stored robot demonstration (touch.store.demo_store) in MuJoCo.

Two ways:
  states   set the recorded MuJoCo state (qpos, qvel) step by step: exactly what was recorded
  actions  re-run the physics from the stored commands only (the arm's joint targets + the
           gripper), from the recorded first state, with the same controller as the recording
           (touch.panda.simulate: 17 x 2 ms physics steps per 30 Hz step, ctrl = interpolated
           target + 0.1 x target velocity): shows the commands alone reproduce the result
"""

from __future__ import annotations

import numpy as np

from touch.panda import SUB

GRIP_OPEN_MAX = 255.0


def build(xml: str):
    import mujoco

    m = mujoco.MjModel.from_xml_string(xml)
    return m, mujoco.MjData(m)


def set_state(m, d, state) -> None:
    import mujoco

    d.qpos[:] = state[: m.nq]
    d.qvel[:] = state[m.nq : m.nq + m.nv]
    mujoco.mj_forward(m, d)


def _grip_ctrl(a: dict, t: int) -> float:
    """Gripper command: closed -> 0; open -> the opening the recording had a step later (the
    recorder commanded the object width + 2 cm, not stored as such)."""
    if a["action"][t, -1] > 0:
        return 0.0
    t1 = min(t + 1, len(a["action"]) - 1)
    w = float(a["gripper_qpos"][t1].sum())
    return float(np.clip(w / 2 / 0.04 * 255 + 8, 0, GRIP_OPEN_MAX))


def steps(m, d, demo: dict, mode: str = "states"):
    """Yield the step index after the model is at that step (state set / physics stepped)."""
    import mujoco

    a = demo["arrays"]
    T = len(a["action"])
    set_state(m, d, a["state"][0])
    if mode == "states":
        for t in range(T):
            set_state(m, d, a["state"][t])
            yield t
        return
    assert mode == "actions", mode
    dt = SUB * m.opt.timestep
    q_prev = a["action"][0, :7].astype(np.float64)
    for t in range(T):
        q = a["action"][t, :7].astype(np.float64)
        qd = (q - q_prev) / dt
        d.ctrl[7] = _grip_ctrl(a, t)
        yield t  # the state BEFORE step t's command (as the recording logged it)
        for s in range(SUB):
            d.ctrl[:7] = q_prev + (q - q_prev) * (s + 1) / SUB + 0.1 * qd
            mujoco.mj_step(m, d)
        q_prev = q


def cup_pos(m, d) -> np.ndarray:
    return d.xpos[m.body("obj0").id].copy()


def compare(m, d, demo: dict) -> dict:
    """After an actions re-run: where the cup ended vs the recording (cm), its peak height."""
    a = demo["arrays"]
    end = cup_pos(m, d)
    return {"cup_end_vs_recorded_cm": round(100 * float(np.linalg.norm(end - a["cup_pos"][-1])),
                                            1),
            "cup_end_xy": [round(float(v), 3) for v in end[:2]],
            "recorded_end_xy": [round(float(v), 3) for v in a["cup_pos"][-1][:2]]}  # fmt: skip


def render_worker(args) -> dict:
    """Pool worker for the live demo: load one demo from the database (its own connection),
    reproduce it in MuJoCo (mode: "states" = the recorded states, exact; "actions" = physics
    re-run from the stored commands) and render camera "view" every `every` steps -> summary +
    "frames" (jpeg bytes; the caller plays them back at 30 Hz / every).
    args = (url, demo_id, k, tile_wh, every, mode)."""
    import time

    import cv2
    import mujoco

    from touch.store.demo_store import DemoStore

    url, demo_id, k, (tw, th), every, mode = args
    t0 = time.perf_counter()
    st = DemoStore(url)
    demo = st.load_demo(demo_id)
    st.close()
    t_load = time.perf_counter() - t0
    from touch import render_view as RV

    m, d = build(demo["xml"])
    cam, _ = RV.view_for(m, demo["arrays"]["state"], tw / th)  # the webcam's perspective
    mc = mujoco.MjvCamera()
    if "fixed" in cam:
        m.cam_pos[cam["fixed"]] = cam["pos"]
        m.cam_quat[cam["fixed"]] = cam["quat"]
        mc.type = mujoco.mjtCamera.mjCAMERA_FIXED
        mc.fixedcamid = cam["fixed"]
    else:  # older demos without the webcam camera: the clearest angle (best_view)
        m.vis.global_.fovy = cam["fovy"]
        mc.type = mujoco.mjtCamera.mjCAMERA_FREE
        mc.lookat[:] = cam["lookat"]
        mc.distance, mc.azimuth, mc.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
    r = mujoco.Renderer(m, th, tw)
    T = len(demo["arrays"]["action"])
    frames = []
    for t in steps(m, d, demo, mode):
        if t % every == 0 or t == T - 1:
            r.update_scene(d, camera=mc)
            img = r.render()[..., ::-1]
            frames.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
    r.close()
    a = demo["arrays"]
    at = demo["attrs"]
    cmp = compare(m, d, demo) if mode == "actions" else {"cup_end_vs_recorded_cm": 0.0}
    return {
        **cmp,
        "k": k,
        "mode": mode,
        "demo_id": demo_id,
        "steps": T,
        "frames": frames,
        "load_ms": round(1e3 * t_load),
        "source": at.get("source"),
        "success": bool(at.get("success")),
        "metrics": at.get("metrics") or {},
        "lift_cm": round(100 * float(a["cup_pos"][:, 2].max() - a["cup_pos"][0, 2]), 1),
    }
