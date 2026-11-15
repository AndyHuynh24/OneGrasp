"""Side-by-side demo video: raw DexYCB val camera | robot sim seen from the same calibrated camera.

The sim gripper follows the GT 3D hand path and opens / closes on the grasp model's contact signal
(sim_replay condition B, same close rule). One sequence per object; val subject only.

    python scripts/render_sim_demo.py --preds runs/<run>/val_preds.npz
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_replay as SR  # noqa: E402

from touch import robot_sim as S  # noqa: E402
from touch.data import dexycb as D  # noqa: E402

OBJECTS = ["025_mug", "011_banana", "006_mustard_bottle", "002_master_chef_can", "035_power_drill",
           "019_pitcher_base", "024_bowl", "036_wood_block",
           "007_tuna_fish_can", "037_scissors"]  # fmt: skip
STATES = ["NONE", "TOUCH", "HOLD"]
STATE_COL = [(200, 200, 200), (0, 200, 255), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def camera_xml(seq, cam: str, tag: np.ndarray) -> str:
    """MuJoCo camera at the real camera's pose (table frame) with its vertical field of view."""
    Tc = S.extrinsics(seq)[cam]  # camera -> master
    pos = S.master_to_table(Tc[:, 3], tag)
    R = tag[:, :3].T @ Tc[:, :3]  # camera axes (OpenCV: x right, y down, z forward) in table frame
    x, y = R[:, 0], -R[:, 1]  # MuJoCo camera: x right, y up, looks along -z
    fovy = np.degrees(2 * np.arctan(240.0 / D.intrinsics(cam)[1, 1]))
    return (f'<camera name="human" pos="{pos[0]} {pos[1]} {pos[2]}" '
            f'xyaxes="{x[0]} {x[1]} {x[2]} {y[0]} {y[1]} {y[2]}" fovy="{fovy}"/>')  # fmt: skip


def label(img, text, org, scale=0.8, col=(255, 255, 255), th=2):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, col, th, cv2.LINE_AA)


def main() -> None:
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default results/robot_sim/<run>/sim_vs_camera_<robot>.mp4",
    )
    ap.add_argument("--robot", default="v2", choices=["v1", "v2"],
                    help="v2: object-centred grasp + safe approach + fixed physics")  # fmt: skip
    ap.add_argument("--cam", type=int, default=1, help="index into dexycb.CAMS")
    ap.add_argument("--delay", type=int, default=3)
    ap.add_argument("--fps", type=int, default=15, help="15 = half speed")
    args = ap.parse_args()
    cam = D.CAMS[args.cam]
    P = np.load(args.preds)
    run = args.preds.parent.name
    model = run.split("_")[0]
    if args.out is None:
        args.out = Path("results/robot_sim") / run / f"sim_vs_camera_{args.robot}.mp4"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_all = 0
    seqs = [s for s in D.list_sequences() if s.split == "val"]
    assert seqs and all(s.split == "val" for s in seqs)
    raw = args.out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 480))
    done = set()
    for seq in seqs:
        ex = S.extrinsics(seq)
        tag = ex["apriltag"]
        p_obj, R_obj = S.object_poses_table(seq, tag)
        gi = int(seq.meta["ycb_grasp_ind"])
        scene = S.build_scene(seq, p_obj[0], R_obj[0], stiff=args.robot == "v2")
        SR.ROBOT["v2"], SR.ROBOT["R_obj"] = args.robot == "v2", R_obj
        obj = scene.names[gi]
        key = f"{seq.name}/{cam}"
        if obj not in OBJECTS or obj in done or f"{key}/state" not in P.files:
            continue
        if (p_obj[:, gi, 2] - p_obj[0, gi, 2]).max() < 0.05:
            continue
        done.add(obj)
        C = np.load(SR.PROC / seq.subject / seq.name / "contact.npz")
        parts = D.vertex_parts(seq.side)
        v, j, ok = SR.gt_hand(seq, tag)
        st = P[f"{key}/state"].astype(np.float32)
        close = SR.shift(S.debounce(S.grasp_signal(st, P[f"{key}/present"])), args.delay)
        scene = S.Scene(scene.xml.replace("<worldbody>", "<worldbody>" + camera_xml(seq, cam, tag)),
                        scene.names, scene.grasp_index)  # fmt: skip
        rnd = {}

        def render(m, d, t, rnd=rnd):
            if "r" not in rnd:
                rnd["r"] = mujoco.Renderer(m, 480, 640)
            rnd["r"].update_scene(d, camera="human")
            return rnd["r"].render()[..., ::-1].copy()

        r, extra = SR.run(scene, p_obj, gi, j, ok, close, v, C["vert"] == 1, parts, render)
        if r is None:
            continue
        ok_txt = ("LIFTED", (80, 220, 80)) if r["success"] else ("FAILED", (60, 60, 255))
        frames = extra[1]["frames"]
        print(f"{seq.name} {obj} success={r['success']}", flush=True)
        n_all, n_ok = n_all + 1, n_ok + int(r["success"])
        for t, simf in enumerate(frames):
            img = cv2.imread(str(seq.image(cam, t)))
            k = int(st[t].argmax())
            label(img, "camera (raw)", (16, 34), 0.7)
            label(img, f"{model}: {STATES[k]}", (16, 70), 0.9, STATE_COL[k])
            label(img, obj[4:].replace("_", " "), (16, 462), 0.7)
            label(simf, f"robot sim ({args.robot}), same camera", (16, 34), 0.7)
            g = "CLOSED" if close[t] else "OPEN"
            label(
                simf, f"gripper: {g}", (16, 70), 0.9, (0, 0, 255) if close[t] else (255, 255, 255)
            )
            label(simf, f"GT hand path + {model} timing", (16, 462), 0.6)
            frame = np.hstack([img, simf])
            vw.write(frame)
        for _ in range(args.fps):  # hold the last frame with the outcome for 1 s
            f2 = frame.copy()
            label(f2, f"robot: {ok_txt[0]}", (660, 260), 1.4, ok_txt[1], 3)
            vw.write(f2)
        if len(done) == len(OBJECTS):
            break
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(args.out)], check=True)  # fmt: skip
    raw.unlink()
    print("wrote", args.out, "objects:", sorted(done), f"lifted {n_ok}/{n_all}")


if __name__ == "__main__":
    main()
