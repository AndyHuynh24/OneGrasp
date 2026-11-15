"""Live hackathon demo: webcam -> grasp model (g6) -> MuJoCo robot copying your grasp of one
cylinder (9 cm tall, 6.2 cm diameter), shown side by side: camera | sim from the same viewpoint.

Setup (once per camera placement):
  1. python scripts/calib_rig.py --flip hv     (marker 0 flat on the table, press s to save)
  2. put the real cylinder on the table inside the camera view
  3. python scripts/live_sim.py
     click the centre of the cylinder's BASE in the left (camera) image: the sim cylinder moves
     there and the green outline shows where the sim thinks it is; adjust until the outline sits
     on the real cylinder.
Keys: q / Esc quit, r reset the sim (cylinder back to its spot, gripper parked).
The robot: touch.live_robot (assist on by default: the robot knows the cylinder pose).
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from live_grasp import (  # noqa: E402
    STATES,
    HandTracker,
    LiveGrasp,
    default_ckpt,
    draw,
    frames_with_hands,
    threaded,
)

from touch.handpath import world_hand  # noqa: E402
from touch.interact.grab_gate import FusionGate  # noqa: E402
from touch.live_panda import LivePanda  # noqa: E402
from touch.live_robot import CYL_D, CYL_H, LiveRobot, camera_xml  # noqa: E402

PANEL_W = 640  # each half of the window


def project(pts_w: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    R, t = c2w[:, :3], c2w[:, 3]
    pc = (pts_w - t) @ R  # world -> camera
    return (pc[:, :2] / pc[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def click_to_table(u: float, v: float, K: np.ndarray, c2w: np.ndarray) -> np.ndarray | None:
    """Image pixel -> point on the table plane z = 0 (table frame)."""
    ray = c2w[:, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    o = c2w[:, 3]
    if abs(ray[2]) < 1e-9 or -o[2] / ray[2] <= 0:
        return None
    return o + (-o[2] / ray[2]) * ray


def raw_fields(out: dict, lm, fidx: int, score, cam_t, fusion) -> dict:
    """The raw per-frame values stored with --store (touch.store.grasp_store.RAW_COLS)."""
    raw = {"frame_idx": fidx if fidx >= 0 else None, "hand_score": score,
           "gate_score": getattr(fusion, "score", None), "table_p": out.get("table")}  # fmt: skip
    if lm is not None:
        raw |= {"mp_landmarks": lm, "joints_model": out["joints"], "cam_t": cam_t,
                "mano_pose": out.get("mano_rot"), "mano_betas": out.get("mano_betas"),
                "contact_p": out["vert"]}  # fmt: skip
    return {k: (None if v is None else np.asarray(v, float) if k not in ("frame_idx",) else v)
            for k, v in raw.items()}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", type=Path, default=default_ckpt(), help="default g10, else g9, g8, g6"
    )
    ap.add_argument("--calib", type=Path, default=Path("results/rig/calib.json"))
    ap.add_argument("--cam", default="0", help="webcam index, or a video file (testing)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (testing)")
    ap.add_argument(
        "--obj",
        type=float,
        nargs=2,
        default=[0.20, 0.0],
        help="cylinder base centre in the table frame (m); or click it live",
    )
    ap.add_argument("--no-assist", action="store_true", help="gripper follows the hand only")
    ap.add_argument(
        "--close-on",
        default="hold",
        choices=["hold", "contact"],
        help="close after 2 HOLD frames (default) or 2 TOUCH/HOLD frames",
    )
    ap.add_argument(
        "--release-frames",
        type=int,
        default=20,
        help="open after this many consecutive NONE frames (TOUCH never releases)",
    )
    ap.add_argument("--fast-p", type=float, default=0.99,
                    help="release at once when P(NONE) >= this ...")  # fmt: skip
    ap.add_argument("--fast-frames", type=int, default=10, help="... for this many frames in a row")
    ap.add_argument("--gate", default="fusion", choices=["fusion", "rule"],
                    help="fusion: learned gate (runs/grab_fusion/<run>/, validated on HO-Cap val); "
                    "rule: --close-on / --release-frames / --fast-* rule")  # fmt: skip
    ap.add_argument(
        "--fusion-variant",
        default="auto",
        help="hold_own: adapted to you (eval_own_grabs.py --adapt); "
        "hold: tonight's gate; start: probabilities only; auto: hold else start",
    )
    # fmt: skip
    ap.add_argument("--fusion-dir", type=Path, default=None,
                    help="default runs/grab_fusion/<ckpt run name>")  # fmt: skip
    ap.add_argument("--release", default="fast", choices=["normal", "fast", "fastest"],
                    help="learned gate's release speed (fusion_hold.json release_profiles, tuned "
                         "on HO-Cap val): normal 13 / fast 9 / fastest 7 frames median delay, at "
                         "0.7 / 1.4 / 2.9%% of holds dropped early")  # fmt: skip
    ap.add_argument(
        "--smoothing",
        default="one_euro",
        choices=["one_euro", "exp"],
        help="hand -> gripper smoothing (exp = previous fixed exponential)",
    )
    ap.add_argument("--upright", action="store_true",
                    help="held cylinder keeps standing: follow only the hand's turn about the "
                         "vertical (default: follow the full hand rotation)")  # fmt: skip
    ap.add_argument("--robot", default="panda", choices=["panda", "gripper"],
                    help="Franka Panda arm (default) or the floating parallel gripper")  # fmt: skip
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--provider", choices=["cpu", "coreml"], default="cpu",
                    help="coreml = Mac GPU / Neural Engine")  # fmt: skip
    ap.add_argument("--out", type=Path, default=None, help="also record the window (mp4)")
    ap.add_argument("--video-dir", default="data/recordings",
                    help="with --store: raw camera video saved here (<session>.mp4)")  # fmt: skip
    ap.add_argument("--store", nargs="?", const="", default=None,
                    help="record hand + gripper data for robots (touch.store.GraspStore); "
                         "alone = $TIGER_DATABASE_URL or local SQLite, or give a URL")  # fmt: skip
    args = ap.parse_args()  # fmt: skip
    if not args.calib.exists():
        sys.exit(f"no calibration at {args.calib}: run first (marker 0 flat, in view)\n"
                 f"  python scripts/calib_rig.py --cam {args.cam} --flip hv --size-mm 100\n"
                 "then press s in its window to save")  # fmt: skip
    cal = json.loads(args.calib.read_text())
    K, c2w = np.asarray(cal["K"]), np.asarray(cal["cam_to_world"])
    W, H = cal["image_size"]
    flip_s = cal.get("flip", "")
    flip = {"h": "h" in flip_s, "v": "v" in flip_s}
    src = int(args.cam) if args.cam.isdigit() else args.cam
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    live = LiveGrasp(args.ckpt, args.threads, args.provider)
    ph = round(H * PANEL_W / W)
    fusion = None
    if args.gate == "fusion":
        fdir = args.fusion_dir or Path("runs/grab_fusion") / args.ckpt.parent.name
        variant = args.fusion_variant
        if variant == "auto":  # hold_own only when asked: it is fit to one person
            variant = "hold" if (fdir / "fusion_hold.json").exists() else "start"
        if (fdir / f"fusion_{variant}.json").exists():
            fusion = FusionGate.from_dir(fdir, variant)
            prof = json.loads((fdir / f"fusion_{variant}.json").read_text()).get(
                "release_profiles", {}
            )
            if args.release in prof:  # release speed: overrides off / k_off / p_none / k_none
                for k, v in prof[args.release]["release"].items():
                    setattr(fusion.hyst, k, v)
            print(f"grab gate: learned fusion ({fdir}), release {args.release}, "
                  f"gate {fusion.hyst.__dict__}")  # fmt: skip
        else:
            print(f"grab gate: no fusion in {fdir} -> rule gate")
    robot_kw = {"cam_xy": c2w[:2, 3]} if args.robot == "panda" else {}
    robot = (LivePanda if args.robot == "panda" else LiveRobot)(
        args.obj,
        **robot_kw,
        fusion=fusion,
        smoothing=args.smoothing,
        upright=args.upright,
        assist=not args.no_assist,
        camera=camera_xml(K, c2w, H),
        close_on=args.close_on,
        release_frames=args.release_frames,
        fast_p=args.fast_p,
        fast_frames=args.fast_frames,
    )
    import mujoco

    renderer = mujoco.Renderer(robot.m, ph, PANEL_W)
    store, sid, rec, t_start = None, None, None, time.perf_counter()
    if args.store is not None:
        from touch.store import GraspStore, VideoRecorder
        from touch.train import git_hash

        store = GraspStore(args.store or None)
        store.create_schema()
        scene_id = store.add_scene(
            "live_rig", "table",
            camera={"K": K, "cam_to_world": c2w, "image_size": [W, H], "flip": flip_s,
                    "reproj_px": cal.get("reproj_px"), "calib_file": str(args.calib)},
            objects=[{"name": "cylinder", "shape": "cylinder", "size_m": [CYL_D / 2, CYL_H / 2],
                      "pos": [*robot.obj_xy, CYL_H / 2], "quat_wxyz": [1, 0, 0, 0]}],
            markers={"aruco_ids": cal.get("markers_seen"), "size_m": cal.get("marker_size_m"),
                     "origin": "marker 0 (table frame)"})  # fmt: skip
        sid = store.start_session(
            "live_webcam", "table", video=str(args.cam), model=args.ckpt.parent.name,
            git=git_hash(), meta={"K": K, "cam_to_world": c2w, "image_size": [W, H],
                                  "object": {"name": "cylinder", "height_m": CYL_H,
                                             "diameter_m": CYL_D, "xy": robot.obj_xy},
                                  "gate": "fusion" if fusion is not None else "rule"},
            scene_id=scene_id)  # fmt: skip
        rec = VideoRecorder(Path(args.video_dir) / f"{sid}.mp4", fps=30)
        print(f"recording session {sid} -> {store.url.split('@')[-1]}, video {rec.path}")
    stop = threading.Event()
    items: queue.Queue = queue.Queue(maxsize=4)
    n_done = [0]
    tracker = HandTracker(W)  # the same hand every frame (a second hand must not take over)

    def perception() -> None:  # capture + MediaPipe thread -> model (this thread) -> main
        for frame, hands, handedness, _, _ in threaded(frames_with_hands(cap, src, 30.0,
                                                                         stop, flip)):  # fmt: skip
            if stop.is_set():
                break
            assert frame.shape[1] == W and frame.shape[0] == H, (
                f"camera gives {frame.shape[1]}x{frame.shape[0]}, calibration is {W}x{H}"
            )
            fidx = rec.write(frame) if rec is not None else -1  # raw frame, before drawing
            lm, left = tracker.pick(hands, handedness)
            score = next((handedness[i][0].score for i, h in enumerate(hands) if h is lm), None)
            out = live.step(live.prep(frame, lm, left))
            jw, cam_t = None, None
            if lm is not None:
                with np.errstate(all="ignore"):
                    wh = world_hand(out["joints"][None], out["verts"][None], lm[None], K, c2w,
                                    left, smooth_k=1)  # fmt: skip
                jw = wh["joints"][0] if wh["ok"][0] else None
                cam_t = wh["cam_t"][0] if wh["ok"][0] else None
            items.put((frame, lm, left, out, jw, time.perf_counter(), fidx, score, cam_t))
            n_done[0] += 1
            if args.max_frames and n_done[0] >= args.max_frames:
                break
        stop.set()

    worker = threading.Thread(target=perception, daemon=True)
    worker.start()
    title = "live grasp -> MuJoCo (q quit, r reset, click = cylinder base)"
    cv2.namedWindow(title)

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and x < PANEL_W:
            p = click_to_table(x * W / PANEL_W, y * H / ph, K, c2w)
            if p is not None:
                robot.reset(p[:2])
                if store is not None:
                    store.add_object(
                        sid,
                        time.perf_counter() - t_start,
                        "right",
                        "cylinder",
                        {"xy": p[:2], "height_m": CYL_H, "diameter_m": CYL_D},
                    )
                print(f"cylinder at x {p[0]:+.3f} y {p[1]:+.3f} m (table frame)", flush=True)

    cv2.setMouseCallback(title, on_mouse)
    ring = np.stack([np.cos(np.linspace(0, 2 * np.pi, 48)), np.sin(np.linspace(0, 2 * np.pi, 48)),
                     np.zeros(48)], 1) * (CYL_D / 2)  # fmt: skip
    writer, info, fps, t_prev = None, None, 0.0, time.perf_counter()
    t_loop, n_win = time.perf_counter(), 0
    while not stop.is_set() or not items.empty():
        try:
            batch = [items.get(timeout=0.1)]
        except queue.Empty:
            continue
        while not items.empty():  # physics sees every model frame, the window the newest
            batch.append(items.get_nowait())
        for frm, lm, left_i, out, jw, t_cap, fidx, score, cam_t in batch:
            extra = {}
            if fusion is not None and fusion.needs_frames:  # richer gate: frame + hand inputs
                extra = {"joints_pred": out.get("joints"), "lm": lm, "bgr": frm,
                         "gray": cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)}  # fmt: skip
            info = robot.step(jw, out["state"], lm is not None, out["finger"], t=t_cap, **extra)
            if store is not None:
                store.add_frame(sid, t_cap - t_start, "left" if left_i else "right",
                                lm is not None, out["state"], out["finger"],
                                bool(info["close"]), jw, object_name="cylinder",
                                keypoints_2d=lm, raw=raw_fields(out, lm, fidx, score, cam_t,
                                                                fusion))  # fmt: skip
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
        frame, lm, left, out = batch[-1][:4]
        draw(frame, lm, out, left, fps, decision=bool(info["close"]))
        c = np.array([*robot.obj_xy, 0.0])  # where the sim cylinder started (= the real one)
        for z in (0.0, CYL_H):
            pts = project(c + ring + [0, 0, z], K, c2w).astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 220, 0), 2)
        if info["pad_target"] is not None:
            u = project(info["pad_target"][None], K, c2w)[0]
            if np.isfinite(u).all():
                cv2.circle(frame, tuple(int(a) for a in u), 8, (255, 0, 255), 2)
        left_img = cv2.resize(frame, (PANEL_W, ph), interpolation=cv2.INTER_AREA)
        renderer.update_scene(robot.data, camera=robot.view_camera)
        sim = renderer.render()[..., ::-1].copy()
        lift = (info["object"][2] - CYL_H / 2) * 100
        txt = f"gripper {'CLOSED' if info['close'] else 'open'}  lift {lift:+.1f} cm"
        txt += "  (assisted)" if info["assisted"] and info["close"] else ""
        cv2.putText(sim, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if info["close"] else (255, 255, 255), 2)  # fmt: skip
        raw = STATES[int(np.argmax(out["state"]))] if lm is not None else "-"
        txt2 = f"model (this frame): {raw}"
        if fusion is not None and np.isfinite(fusion.score):
            txt2 += f"   grab score {fusion.score:.2f}"
        cv2.putText(sim, txt2, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        col = (0, 0, 255) if info["close"] else (200, 200, 200)
        cv2.putText(sim, f"decision: {'HOLD' if info['close'] else 'not holding'}", (12, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)  # fmt: skip
        both = np.hstack([left_img, sim])
        if args.out is not None:
            if writer is None:
                writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                         both.shape[1::-1])  # fmt: skip
            writer.write(both)
        cv2.imshow(title, both)
        n_win += 1
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            stop.set()
        elif key == ord("r"):
            robot.reset()
    stop.set()
    cap.release()
    dt = time.perf_counter() - t_loop
    print(
        f"throughput: model + physics {n_done[0] / dt:.1f} fps, window {n_win / dt:.1f} fps "
        f"({n_done[0]} frames, {dt:.1f} s, provider {args.provider})",
        flush=True,
    )
    if store is not None:
        n_vid = rec.close()
        store.add_video(sid, str(rec.path), fps=30, n_frames=n_vid, width=W, height=H,
                        codec="mp4v")  # fmt: skip
        store.end_session(sid)
        n = len(store.grasps(session_id=sid))
        print(f"stored session {sid}: {n} grasps -> {store.url.split('@')[-1]}")
        store.close()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
