"""Export grasp episodes from the store (Tiger Data or SQLite) as self-contained folders for robot
learning / imitation: per episode the hand data (all stored columns), its info (grasp, session,
3D scene incl. MuJoCo MJCF, video reference) and the matching video clip.

    python scripts/export_episodes.py [--store URL] [--object cylinder] [--min-lift 0.05]
        [--session ID] [--out results/robot_data/episodes] [--no-video]

out/
  index.json                       one entry per episode (grasp id, object, times, lift, files)
  episode_0000/
    data.npz                       arrays [N, ...] of every hand_frames column, t (0 = grab), ...
    info.json                      grasp + session + scene (objects, camera, MJCF) + video ref
    scene.xml                      robot-free MuJoCo scene (table + objects + camera), if any
    clip.mp4                       the episode's camera frames (data.npz frame_idx -> clip frame)
Video / mesh uris: local paths are read directly; hocap://<rel> is resolved under --hocap-root
(data/store_assets/hocap after scripts/fetch_store_assets.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch.store import GraspStore  # noqa: E402
from touch.store.grasp_store import _jsonable  # noqa: E402
from touch.store.scene import localize_mjcf, resolve_uri  # noqa: E402


def frame_reader(video: dict, hocap_root: Path):
    """video row -> function(frame_idx) -> BGR image or None."""
    uri = video["uri"]
    if video["kind"] == "image_folder":
        pat = resolve_uri(uri, hocap_root)
        return lambda i: cv2.imread(pat % i)
    cap = cv2.VideoCapture(uri)
    state = {"next": 0}

    def read(i):
        if i != state["next"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, im = cap.read()
        state["next"] = i + 1
        return im if ok else None

    return read


def write_clip(ep: dict, video: dict, path: Path, hocap_root: Path) -> int:
    idx = ep["frame_idx"]
    idx = idx[idx >= 0]
    if not len(idx):
        return 0
    read = frame_reader(video, hocap_root)
    vw, n = None, 0
    for i in range(int(idx.min()), int(idx.max()) + 1):
        im = read(i)
        if im is None:
            continue
        if vw is None:
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(video["fps"] or 30), (im.shape[1], im.shape[0]))  # fmt: skip
        vw.write(im)
        n += 1
    if vw is not None:
        vw.release()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=None)
    ap.add_argument("--object", default=None, help="substring of the object name")
    ap.add_argument("--min-lift", type=float, default=None, help="m")
    ap.add_argument("--session", default=None)
    ap.add_argument("--out", type=Path, default=Path("results/robot_data/episodes"))
    ap.add_argument("--hocap-root", type=Path, default=Path("data/hocap/ds"))
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    st = GraspStore(args.store)
    gs = st.grasps(session_id=args.session, object_like=args.object, min_lift=args.min_lift)
    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for k, g in enumerate(gs):
        d = args.out / f"episode_{k:04d}"
        d.mkdir(exist_ok=True)
        ep = st.episode(g["grasp_id"])
        info = ep.pop("info")
        np.savez_compressed(d / "data.npz", **ep)
        files = ["data.npz", "info.json"]
        scene = info.get("scene") or {}
        if scene.get("mjcf"):
            (d / "scene.xml").write_text(localize_mjcf(scene["mjcf"], scene.get("objects") or [],
                                                       args.hocap_root, d))  # fmt: skip
            files.append("scene.xml")
        clip_frames = 0
        if not args.no_video and info["videos"]:
            clip_frames = write_clip(ep, info["videos"][0], d / "clip.mp4", args.hocap_root)
            if clip_frames:
                files.append("clip.mp4")
                info["clip"] = {"file": "clip.mp4",
                                "first_frame_idx": int(ep["frame_idx"][ep["frame_idx"] >= 0].min()),
                                "n_frames": clip_frames}  # fmt: skip
        (d / "info.json").write_text(json.dumps(info, indent=1, default=_jsonable))
        index.append({"episode": d.name, "grasp_id": g["grasp_id"], "session_id": g["session_id"],
                      "hand": g["hand"], "object": g["object"], "n_frames": len(ep["t"]),
                      "t_grab": g["t_grab"], "t_release": g["t_release"], "lift_m": g["lift_m"],
                      "files": files})  # fmt: skip
        print(f"{d.name}: {g['object']} {g['hand']} {len(ep['t'])} frames, clip {clip_frames}",
              flush=True)  # fmt: skip
    (args.out / "index.json").write_text(json.dumps(index, indent=1, default=_jsonable))
    print(f"{len(index)} episodes -> {args.out}")
    st.close()


if __name__ == "__main__":
    main()
