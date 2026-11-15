"""Raw camera | MuJoCo robot sim on OakInk2 human lifts (EXTRA VALIDATION, never trained on).

Per lift (an object raised >= 5 cm): left = the raw allocentric_top camera with the GT and the
model's state for the lifting hand; right = the sim seen from the same calibrated camera. Robot v2
as on DexYCB (scripts/render_sim_demo.py --robot v2): the gripper follows the GT 3D hand path,
the grasp is planned on the object (touch.robot_sim.object_refine), stiff contacts, slow close;
it closes on the model's contact signal with the DexYCB-val rule (TOUCH or HOLD, debounced,
+3 frames). Objects start at their lift-window-start poses, dropped onto the table.

    python scripts/render_sim_oakink2.py --preds runs/oakink2/<run>/preds.npz [--n 10]
writes results/oakink2/sim_vs_camera_<run>.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
import sim_replay as SR  # noqa: E402
from eval_oakink2 import Seq, gt_labels, object_poses  # noqa: E402
from render_sim_demo import STATE_COL, STATES, label  # noqa: E402
from sim_hocap import POST, PRE  # noqa: E402

from touch import robot_sim as S  # noqa: E402
from touch.data import dexycb as D  # noqa: E402
from touch.data import oakink2 as O  # noqa: E402

CAM = "allocentric_top"
W, Hh = 848, 480
A = np.asarray(O.Y_UP_TO_Z_UP)  # world (y up) -> table frame (z up); table at y = 0


def table_poses(seq: Seq) -> tuple[np.ndarray, np.ndarray]:
    """Object -> table frame: R [K, N, 3, 3], t [K, N, 3]."""
    R, t = object_poses(seq)
    return A @ R, t @ A.T


def mesh_obj(obj_id: str) -> Path:
    """MuJoCo-readable OBJ copy of the object mesh (cached)."""
    out = O.ROOT / "cache" / "mesh" / f"{obj_id}.obj"
    if not out.exists():
        p = O.mesh_path(obj_id)
        v, f = O.read_ply(p) if p.suffix == ".ply" else D.load_obj(p)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]
        out.write_text("\n".join(lines) + "\n")
    return out.resolve()


def lifts(seq: Seq, R, t) -> list[tuple[int, int, int]]:
    """(object, window start, end) per lift >= 5 cm above where the object started the lift."""
    n, out = t.shape[1], []
    for k in range(t.shape[0]):
        up = np.flatnonzero(t[k, :, 2] - np.minimum.accumulate(t[k, :, 2]) > 0.01)
        for r in np.split(up, np.flatnonzero(np.diff(up) > 1) + 1) if len(up) else []:
            a = max(0, r[0] - PRE)
            if (t[k, r, 2] - t[k, a, 2]).max() >= 0.05:
                out.append((k, a, min(n, r[-1] + POST)))
    return out


def camera_xml(seq: Seq) -> str:
    E = seq.world_to_cam(CAM)
    c2w = np.linalg.inv(E)
    pos, Rc = A @ c2w[:3, 3], A @ c2w[:3, :3]
    x, y = Rc[:, 0], -Rc[:, 1]  # OpenCV -> MuJoCo camera axes
    K = seq.K(CAM)
    fovy = np.degrees(2 * np.arctan(Hh / 2 / K[1, 1]))
    return (f'<camera name="rec" pos="{pos[0]} {pos[1]} {pos[2]}" xyaxes="{x[0]} {x[1]} {x[2]} '
            f'{y[0]} {y[1]} {y[2]}" fovy="{fovy}"/>')  # fmt: skip


def scene(seq: Seq, R, t, a0: int) -> tuple[S.Scene, list[float]]:
    """Objects at their frame-a0 poses, each dropped onto the table (z = 0)."""
    assets, bodies, drops = [], [], []
    for i, o in enumerate(seq.anno["obj_list"]):
        pts = O.object_points(o, 4000)
        z = float((pts @ R[i, a0].T + t[i, a0])[:, 2].min()) - 0.001
        drops.append(z)
        p = t[i, a0] - [0, 0, z]
        q = Rot.from_matrix(R[i, a0]).as_quat()[[3, 0, 1, 2]]
        assets.append(f'<mesh name="m{i}" file="{mesh_obj(o)}"/>')
        bodies.append(f'<body name="obj{i}" pos="{p[0]} {p[1]} {p[2]}" '
                      f'quat="{q[0]} {q[1]} {q[2]} {q[3]}"><freejoint/>'
                      f'<geom type="mesh" mesh="m{i}" density="300" friction="1.0 0.02 0.001" '
                      f'condim="4"/></body>')  # fmt: skip
    xml = S.scene_xml("".join(assets), "".join(bodies), stiff=True)
    xml = xml.replace("<worldbody>", "<worldbody>" + camera_xml(seq))
    return S.Scene(xml, list(seq.anno["obj_list"]), 0), drops


def run_lift(seq, R, t, k, a0, b0, side, close):
    import mujoco

    L = gt_labels(seq, side)
    parts = D.vertex_parts(side)
    sl = slice(a0, b0)
    J, V, ok = L["joints"][sl], L["verts"][sl], L["valid"][sl]
    c = close[sl]
    sc, drops = scene(seq, R, t, a0)
    local = [O.object_points(o, 4000) for o in seq.anno["obj_list"]]

    def pose(i, f, dz=0.0):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R[i, f], t[i, f] - [0, 0, dz]
        return T

    def objects_at(tt):  # window frame tt: demo points + map to the sim scene (objects at a0)
        f = a0 + tt
        return [(p @ R[i, f].T + t[i, f], pose(i, a0, drops[i]) @ np.linalg.inv(pose(i, f)))
                for i, p in enumerate(local)]  # fmt: skip

    pp, RR, Wd, _ = S.retarget(J, ok, c, V, L["vert"][sl] == 1, parts)
    pp, RR, Wd, _ = S.object_refine(pp, RR, Wd, c, objects_at)
    pp, RR = S.pregrasp(pp, RR, c, floor=True)
    rnd: dict = {}

    def render(m, d, _t):
        if "r" not in rnd:
            rnd["r"] = mujoco.Renderer(m, Hh, W)
        rnd["r"].update_scene(d, camera="rec")
        return rnd["r"].render()[..., ::-1].copy()

    res = S.replay(sc, pp, RR, Wd, ok, c, render, **S.V2)
    out = S.lift_outcome(t[k, sl], res["obj"][:, k])  # relative lift: the drop offset cancels
    return res["frames"], out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10, help="lifts to render (<= 2 per sequence)")
    ap.add_argument("--fps", type=int, default=15, help="15 = half speed")
    args = ap.parse_args()
    P = np.load(args.preds)
    run = args.preds.parent.name
    model = run.split("_")[0]
    out = Path("results/oakink2") / f"sim_vs_camera_{run}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2 * W, Hh))
    n_ok = n_all = 0
    for name in O.list_sequences():
        seq = Seq(name)
        R, t = table_poses(seq)
        done = 0
        for k, a0, b0 in lifts(seq, R, t):
            if n_all >= args.n or done >= 2:
                break
            hold = {s: int((gt_labels(seq, s)["state_any2"][a0:b0] == 2).sum())
                    for s in ("right", "left")}  # fmt: skip
            side = max(hold, key=hold.get)
            key = f"{name}/{CAM}/{side}"
            if hold[side] == 0 or f"{key}/state" not in P.files:
                continue
            st = P[f"{key}/state"].astype(np.float32)
            close = SR.shift(S.debounce(S.grasp_signal(st, P[f"{key}/present"])), 3)
            frames, res = run_lift(seq, R, t, k, a0, b0, side, close)
            if res is None:
                continue
            n_all, n_ok = n_all + 1, n_ok + int(res["success"])
            done += 1
            obj = seq.anno["obj_list"][k]
            print(f"{name[:40]} obj {obj} {side} hand frames {a0}-{b0}: success={res['success']}",
                  flush=True)  # fmt: skip
            gs = gt_labels(seq, side)["state_any2"]
            for j, i in enumerate(range(a0, b0)):
                img = cv2.imread(str(seq.image(CAM, i)))
                simf = frames[j].copy() if j < len(frames) else np.zeros((Hh, W, 3), np.uint8)
                s = int(st[i].argmax()) if P[f"{key}/present"][i] else 0
                g = int(gs[i])
                label(img, "camera (raw), OakInk2", (16, 34), 0.7)
                label(
                    img, f"GT: {STATES[g] if g >= 0 else '?'}", (16, 70), 0.8, STATE_COL[max(g, 0)]
                )
                label(img, f"{model}: {STATES[s]}", (16, 104), 0.8, STATE_COL[s])
                label(img, f"{side} hand lifts {obj}", (16, Hh - 16), 0.6)
                label(simf, "robot sim (v2), same camera", (16, 34), 0.7)
                cl = bool(close[i])
                label(simf, f"gripper: {'CLOSED' if cl else 'OPEN'}", (16, 70), 0.8,
                      (0, 0, 255) if cl else (255, 255, 255))  # fmt: skip
                label(simf, f"GT hand path + {model} timing", (16, Hh - 16), 0.6)
                frame = np.hstack([img, simf])
                vw.write(frame)
            ok_txt = ("LIFTED", (80, 220, 80)) if res["success"] else ("FAILED", (60, 60, 255))
            for _ in range(args.fps):
                f2 = frame.copy()
                label(f2, f"robot: {ok_txt[0]}", (W + 40, Hh // 2), 1.4, ok_txt[1], 3)
                vw.write(f2)
        if n_all >= args.n:
            break
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(out)], check=True)  # fmt: skip
    raw.unlink()
    print("wrote", out, f"lifted {n_ok}/{n_all}")


if __name__ == "__main__":
    main()
