"""Store HO-Cap VAL recordings in the grasp store, as the live pipeline would see them: the model's
own 3D hand (predicted mesh placed by its MediaPipe keypoints + the camera calibration, no GT),
its hand-state probabilities and the learned grab gate (probabilities-only variant), in the
table frame (HO-Cap world: table plane z = 0, metres). Object names and poses are HO-Cap GT
(known scene, like the live demo's clicked cylinder). Validation subject only (never test).

    python scripts/store_hocap_grasps.py [--run runs/<run>] [--cam 105322251225] [--store URL]
One session per (sequence, camera); default store = $TIGER_DATABASE_URL or local SQLite.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from sim_hocap import lifter, lifts  # noqa: E402

from touch.data import hocap as H  # noqa: E402
from touch.handpath import world_hand  # noqa: E402
from touch.interact.grab_gate import FusionGate  # noqa: E402
from touch.store import GraspStore  # noqa: E402
from touch.train import git_hash  # noqa: E402

RUN = Path("runs/g9_grasp_oblique_20261113-093840")


def gate_dir(run: Path) -> Path:
    """The run's probabilities-only gate, else g8's (the g9 gate may still be building)."""
    for d in (Path("runs/grab_fusion") / run.name,
              Path("runs/grab_fusion/g8_grasp_webcam_20261113-011442")):  # fmt: skip
        if (d / "fusion_start.json").exists():
            return d
    raise FileNotFoundError("no fusion_start gate")


def obj_pose(R, t, i, f) -> dict:
    return {"pos": t[i, f].tolist(), "quat_wxyz": Rot.from_matrix(R[i, f]).as_quat()[[3, 0, 1, 2]]
            .tolist()}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--cam", default="105322251225", help="HO-Cap camera (Andy's oblique angle)")
    ap.add_argument("--store", default=None)
    ap.add_argument(
        "--replace", action="store_true", help="first delete earlier hocap_val sessions"
    )
    args = ap.parse_args()
    P = np.load(Path("runs/hocap_val") / args.run.name / "preds.npz")
    gdir = gate_dir(args.run)
    st = GraspStore(args.store)
    st.create_schema()
    if args.replace:
        print(f"deleted {st.delete_sessions('hocap_val')} earlier hocap_val sessions")
    K, c2w = H.intrinsics(args.cam), H.cam_to_world(args.cam)
    base = datetime(2023, 10, 27, tzinfo=UTC)
    for seq in H.list_sequences(video_only=True, split="val"):
        R, t = H.object_poses(seq)
        lift_by_side: dict[str, list] = {}
        for k, a0, b0 in lifts(seq):
            lift_by_side.setdefault(lifter(seq, a0, b0), []).append((k, a0, b0))
        objects = [{"name": o, "shape": "mesh",
                    "mesh_uri": f"hocap://models/{o}/cleaned_mesh_2000.obj",
                    **obj_pose(R, t, i, 0)} for i, o in enumerate(seq.objects)]  # fmt: skip
        img = Path(seq.subject) / seq.name / args.cam  # HO-Cap release layout
        h_px, w_px = cv2.imread(str(seq.image(args.cam, 0))).shape[:2]
        scene_id = st.add_scene(f"hocap {seq.name}", "table",
                                camera={"K": K, "cam_to_world": c2w, "image_size": [w_px, h_px],
                                        "serial": args.cam},
                                objects=objects)  # fmt: skip
        hh, mm, ss = (int(seq.name.split("_")[1][i : i + 2]) for i in (0, 2, 4))
        sid = st.start_session(
            "hocap_val", "table", video=f"{seq.name}/{args.cam}", model=args.run.name,
            git=git_hash(), started_at=base + timedelta(hours=hh, minutes=mm, seconds=ss),
            meta={"K": K, "cam_to_world": c2w, "fps": H.FPS, "objects": objects,
                  "gate": f"{gdir}/fusion_start", "hand_3d": "model (no GT)",
                  "dataset": "HO-Cap (CC BY 4.0), https://irvlutd.github.io/HOCap/"},
            scene_id=scene_id)  # fmt: skip
        st.add_video(sid, f"hocap://{img}/color_%06d.jpg", kind="image_folder", camera=args.cam,
                     fps=H.FPS, n_frames=seq.n_frames, width=w_px, height=h_px)  # fmt: skip
        n_grasp = 0
        for side in seq.sides:
            key = f"subject_9/{seq.name}/{args.cam}/{side}"
            if f"{key}/state" not in P.files:
                continue
            with np.errstate(all="ignore"):
                wh = world_hand(P[f"{key}/joints"], P[f"{key}/verts"].astype(np.float32),
                                P[f"{key}/lm"], K, c2w, side == "left", smooth_k=1)  # fmt: skip
            gate = FusionGate.from_dir(gdir, "start")
            starts = {a0: (k, b0) for k, a0, b0 in lift_by_side.get(side, [])}
            for f in range(len(P[f"{key}/state"])):
                if f in starts:
                    k, b0 = starts[f]
                    st.add_object(sid, f / H.FPS, side, seq.objects[k],
                                  {"index": k, "lift_window_s": [f / H.FPS, b0 / H.FPS],
                                   **obj_pose(R, t, k, f)})  # fmt: skip
                pres = bool(P[f"{key}/present"][f])
                closed = gate.step(P[f"{key}/state"][f], P[f"{key}/finger"][f], pres)
                ok = bool(wh["ok"][f])
                raw = {"frame_idx": f, "gate_score": gate.score}
                if pres:
                    raw |= {"mp_landmarks": P[f"{key}/lm"][f],
                            "joints_model": P[f"{key}/joints"][f],
                            "cam_t": wh["cam_t"][f] if ok else None,
                            "contact_p": P[f"{key}/vert"][f]}  # fmt: skip
                st.add_frame(sid, f / H.FPS, side, pres, P[f"{key}/state"][f],
                             P[f"{key}/finger"][f], closed,
                             wh["joints"][f] if wh["ok"][f] else None,
                             keypoints_2d=P[f"{key}/lm"][f] if pres else None,
                             raw=raw)  # fmt: skip
        st.end_session(sid)
        n_grasp = len(st.grasps(session_id=sid))
        print(f"{seq.name} cam {args.cam}: session {sid}, {n_grasp} grasps", flush=True)
    st.close()


if __name__ == "__main__":
    main()
