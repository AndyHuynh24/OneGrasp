"""Replay an exported robot demo (docs/ROBOT_EXPORT.md) in MuJoCo 4 with the force-limited
parallel gripper (touch.robot_sim), around each grasp keyframe. Reads only the export files (no
model), so it is the reference for consuming the format.

Scene: table plane + the export's objects at their poses 1.5 s before the grasp (gt_object_* when
present, else no objects). For every grasp: did the robot lift the grasped object >= 50% of what
the human did?

    python scripts/replay_export.py results/robot_export/<run>/<demo>.npz [--render out.mp4]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch import robot_sim as S  # noqa: E402

PRE, POST = 45, 30  # frames replayed before the grasp keyframe / after the release keyframe


def obj_vertices(path: str) -> np.ndarray:
    with open(path) as fh:
        return np.array([[float(x) for x in ln.split()[1:4]] for ln in fh if ln.startswith("v ")])


def scene(meta: dict, A: dict, f0: int, cam: bool) -> S.Scene:
    assets, bodies = [], []
    if "gt_object_pos" in A:
        for i, o in enumerate(meta["objects"]):
            p, q = A["gt_object_pos"][f0, i], A["gt_object_quat"][f0, i]
            R = Rot.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            z = (obj_vertices(o["mesh"]) @ R.T + p)[:, 2].min()
            p = p - [0, 0, z - 0.001]  # rest on the table
            assets.append(f'<mesh name="m{i}" file="{Path(o["mesh"]).resolve()}"/>')
            bodies.append(f'<body name="obj{i}" pos="{p[0]} {p[1]} {p[2]}" '
                          f'quat="{q[0]} {q[1]} {q[2]} {q[3]}"><freejoint/><geom type="mesh" '
                          f'mesh="m{i}" density="300" rgba="0.55 0.6 0.7 1" '
                          f'friction="1.0 0.02 0.001" condim="4"/></body>')  # fmt: skip
    xml = S.scene_xml("".join(assets), "".join(bodies))
    if cam and "cam_to_world" in meta:  # render from the recording camera's viewpoint
        T, K = np.array(meta["cam_to_world"]), np.array(meta["K"])
        x, y = T[:, 0], -T[:, 1]
        fovy = np.degrees(2 * np.arctan(K[1, 2] / K[1, 1]))
        xml = xml.replace("<worldbody>", f'<worldbody><camera name="rec" pos="{T[0, 3]} {T[1, 3]} '
                          f'{T[2, 3]}" xyaxes="{x[0]} {x[1]} {x[2]} {y[0]} {y[1]} {y[2]}" '
                          f'fovy="{fovy}"/>')  # fmt: skip
    return S.Scene(xml, [o["id"] for o in meta.get("objects", [])], 0)


def main() -> None:
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("demo", type=Path)
    ap.add_argument("--render", type=Path, default=None)
    args = ap.parse_args()
    A = dict(np.load(args.demo.with_suffix(".npz")))
    meta = json.loads(args.demo.with_suffix(".json").read_text())
    keys = meta["keyframes"]
    grasps = [k["frame"] for k in keys if k["type"] == "grasp"]
    releases = [k["frame"] for k in keys if k["type"] == "release"]
    R = Rot.from_quat(A["gripper_quat"][:, [1, 2, 3, 0]]).as_matrix()
    n = len(A["t"])
    frames, results = [], []
    for g, r in zip(grasps, releases, strict=True):
        a, b = max(0, g - PRE), min(n, r + POST)
        sl = slice(a, b)
        ok = A["hand_ok"][sl] & np.isfinite(A["gripper_pos"][sl]).all(1)
        if ok.sum() < 5:
            continue
        rnd: dict = {}

        def render(m, d, t, rnd=rnd):
            if "r" not in rnd:
                rnd["r"] = mujoco.Renderer(m, 480, 640)
            rnd["r"].update_scene(d, camera="rec")
            return rnd["r"].render()[..., ::-1].copy()

        sc = scene(meta, A, a, args.render is not None)
        res = S.replay(sc, A["gripper_pos"][sl].astype(np.float64), R[sl],
                       A["gripper_width"][sl].astype(np.float64), ok, A["gripper_close"][sl],
                       render=render if args.render else None)  # fmt: skip
        row = {"grasp_frame": g, "release_frame": r}
        if "gt_object_pos" in A:
            human = A["gt_object_pos"][sl, :, 2] - A["gt_object_pos"][a, :, 2]
            k = int(human.max(0).argmax())  # the object the human lifted most in this window
            robot = res["obj"][:, k, 2] - res["obj"][0, k, 2]
            hk = float(human[:, k].max())
            row |= {"object": meta["objects"][k]["id"], "human_lift_m": round(hk, 3),
                    "robot_lift_m": round(float(robot.max()), 3),
                    "success": bool(robot.max() >= 0.5 * hk)}  # fmt: skip
        results.append(row)
        print(row, flush=True)
        frames += res["frames"]
    if args.render and frames:
        raw = args.render.with_suffix(".raw.mp4")
        args.render.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (640, 480))
        for f in frames:
            vw.write(f)
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(args.render)], check=True)  # fmt: skip
        raw.unlink()
        print("wrote", args.render)


if __name__ == "__main__":
    main()
