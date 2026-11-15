"""OakInk2 EXTRA VALIDATION (never trained / tuned on): the live grasp model on the 3 fixed
third-person cameras, scored against GT touch / hold labels derived from the official MANO hands
and object poses with the HO-Cap rules (touch.data.hocap: 3 mm contact, 10 mm ignore band,
episode-relative object motion; hold = any 2 hand parts in contact + the object moved > 1 cm).

Same path and scoring as scripts/eval_hocap.py: MediaPipe -> association to the projected GT
hand -> LiveGrasp (causal, one frame at a time) -> eval_hocap.score.
OakInk2 is bimanual; both hands are scored. Its world frame is y-up (touch.data.oakink2.
Y_UP_TO_Z_UP); the table height is the lowest resting object's lowest point, so "hand resting on
the table" uses the same 1 cm rule as HO-Cap in a z-up table frame.

    python scripts/eval_oakink2.py --ckpt runs/<run>/best.pt [--workers 4]
writes runs/oakink2/<run>/{metrics.json, preds.npz}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from eval_hocap import associate, run_track, score  # noqa: E402

from touch.data import dexycb as D  # noqa: E402
from touch.data import hocap as H  # noqa: E402
from touch.data import oakink2 as O  # noqa: E402

CACHE = O.ROOT / "cache"
FPS = 30  # annotated frames = every 4th 120 Hz mocap frame


class Seq:
    """Adapter so eval_hocap.run_track can read OakInk2 frames: image(cam, i) by frame index."""

    def __init__(self, name: str):
        self.name = name
        self.anno = O.load_anno(O.anno_path(name))
        self.frames = [int(f) for f in self.anno["frame_id_list"]]

    def image(self, cam: str, i: int) -> Path:
        return O.image_path(self.name, cam, self.frames[i])

    def K(self, cam: str) -> np.ndarray:
        return np.asarray(self.anno["cam_intr"][cam][self.frames[0]], np.float64)

    def world_to_cam(self, cam: str) -> np.ndarray:
        return np.asarray(self.anno["cam_extr"][cam][self.frames[0]], np.float64)


def object_poses(seq: Seq) -> tuple[np.ndarray, np.ndarray]:
    """R [K, N, 3, 3], t [K, N, 3] (object -> world) at the annotated frames."""
    T = np.stack([[seq.anno["obj_transf"][o][f] for f in seq.frames] for o in seq.anno["obj_list"]])
    return T[..., :3, :3], T[..., :3, 3]


def table_height(seq: Seq) -> float:
    """Table height along the world y (up) axis = 0: the OakInk2 world origin is on the table.
    Checked on the 8 downloaded sequences: the lowest resting object's lowest point is within
    7 mm of y = 0 in 7 of 8 (the 8th, 2.4 cm, starts with objects stacked)."""
    return 0.0


def gt_labels(seq: Seq, side: str) -> dict:
    """Same quantities and rules as eval_hocap.gt_labels (both hold rules, vertex labels)."""
    f = CACHE / "labels" / f"{seq.name}_{side}.npz"
    if f.exists():
        return dict(np.load(f))
    from scipy.spatial import cKDTree

    verts, joints = O.gt_hand(seq.anno, seq.frames, side)
    R, t = object_poses(seq)
    trees = [cKDTree(O.object_points(o)) for o in seq.anno["obj_list"]]
    n = len(seq.frames)
    vd_all = np.full((len(trees), n, 778), np.inf, np.float32)
    for i in range(n):
        for k, tree in enumerate(trees):
            vd_all[k, i] = tree.query((verts[i] - t[k, i]) @ R[k, i])[0]
    parts = D.vertex_parts(side)
    pd_all = np.stack([vd_all[:, :, parts == p].min(2) for p in range(6)], 2)
    part_dist, part_obj = pd_all.min(0), pd_all.argmin(0)
    valid = np.isfinite(verts).all((1, 2))
    disp = H.episode_displacement(part_dist, part_obj, t)
    L1 = D.labels_from_dist(part_dist, disp, valid)
    L2 = D.labels_from_dist(part_dist, disp, valid, hold_rule="any2")
    vd = vd_all.min(0) * 1000
    vert = np.where(vd <= 3, 1, np.where(vd > 10, 0, -1)).astype(np.int8)
    vert[~valid] = -1
    y0 = table_height(seq)
    verts_t = O.to_table_frame(verts, y0)  # z up, table at z = 0 (eval_hocap.score's rest rule)
    joints_t = O.to_table_frame(joints, y0)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, finger=L1["finger"], state=L1["state"], state_any2=L2["state"],
                        vert=vert, part_dist=part_dist, valid=valid,
                        joints=joints_t.astype(np.float32), verts=verts_t.astype(np.float32),
                        world_joints=joints.astype(np.float32),
                        world_verts=verts.astype(np.float32), obj_disp=disp,
                        table_y=y0)  # fmt: skip
    return dict(np.load(f))


def mediapipe(seq: Seq, cam: str) -> np.ndarray:
    f = CACHE / "mp" / f"{seq.name}_{cam}.npz"
    if f.exists():
        return np.load(f)["lm"]
    from mediapipe import Image, ImageFormat

    from touch.data.landmarks import DEFAULT_MODEL, _make_landmarker

    n = len(seq.frames)
    lm = np.full((n, 2, 21, 3), np.nan, np.float32)
    with _make_landmarker(DEFAULT_MODEL, 0.3) as det:
        for i in range(n):
            img = cv2.imread(str(seq.image(cam, i)))
            if img is None:
                continue
            h, w = img.shape[:2]
            r = det.detect_for_video(
                Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                int(i * 1000 / FPS),
            )
            for k, hand in enumerate(r.hand_landmarks[:2]):
                lm[i, k] = [[p.x * w, p.y * h, p.z * w] for p in hand]
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, lm=lm)
    return lm


def cam_points(x: np.ndarray, E: np.ndarray) -> np.ndarray:
    return x @ E[:3, :3].T + E[:3, 3]


def work(job) -> dict:
    ckpt, name, cam = job
    from live_grasp import LiveGrasp

    seq = Seq(name)
    n, K, E = len(seq.frames), seq.K(cam), seq.world_to_cam(cam)
    lm = mediapipe(seq, cam)
    gt_uv, gt_ok, Ls = {}, {}, {}
    for side in ("right", "left"):
        L = Ls[side] = gt_labels(seq, side)
        jc = cam_points(L["world_joints"], E)
        gt_uv[side] = D.project(jc.reshape(-1, 3), K).reshape(n, 21, 2)
        gt_ok[side] = L["valid"] & (jc[:, :, 2] > 0.05).all(1)
    tracks = associate(lm, gt_uv, gt_ok)
    live = LiveGrasp(Path(ckpt), threads=2)
    res = {}
    for side, tlm in tracks.items():
        r = run_track(live, seq, cam, tlm, side == "left")
        r["lm"] = tlm
        res[f"{name}/{cam}/{side}"] = r
    return res


def gt_cam_mesh(L: dict, E: np.ndarray, left: bool) -> tuple[np.ndarray, np.ndarray]:
    """GT joints / verts in the camera frame, wrist-relative, left hands mirrored (as the model)."""
    j, v = cam_points(L["world_joints"], E), cam_points(L["world_verts"], E)
    v, j = v - j[:, :1], j - j[:, :1]
    if left:
        j[..., 0] *= -1
        v[..., 0] *= -1
    return j.astype(np.float32), v.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N (sequence, camera) jobs (smoke)")
    args = ap.parse_args()
    seqs = O.list_sequences()
    for name in seqs:  # GT once, serially (cached)
        seq = Seq(name)
        for side in ("right", "left"):
            L = gt_labels(seq, side)
        print(f"GT {name}: table y {float(L['table_y']):.3f} m, frames {len(seq.frames)}",
              flush=True)  # fmt: skip
    jobs = [(str(args.ckpt), s, c) for s in seqs for c in O.CAMS]
    if args.limit:
        jobs = jobs[: args.limit]
    t0, preds = time.time(), {}
    with get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs)):
            preds.update(r)
            print(f"{i + 1}/{len(jobs)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    items, per_cam = [], {c: [] for c in O.CAMS}
    for key, P in preds.items():
        name, cam, side = key.split("/")
        seq = Seq(name)
        L = gt_labels(seq, side)
        gj, gv = gt_cam_mesh(L, seq.world_to_cam(cam), side == "left")
        it = {"P": P, "L": L, "gj": gj, "gv": gv}
        items.append(it)
        per_cam[cam].append(it)
    metrics = {"all": score(items), **{c: score(v) for c, v in per_cam.items() if v},
               "ckpt": str(args.ckpt), "sequences": seqs}  # fmt: skip
    out = Path("runs/oakink2") / args.ckpt.parent.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    flat = {f"{k}/{f}": v for k, P in preds.items() for f, v in P.items()}
    np.savez_compressed(out / "preds.npz", **flat)
    a = metrics["all"]
    print(json.dumps({k: a[k] for k in ("state_any2_raw", "finger", "table_rest", "present_rate",
                                        "mesh")}, indent=1))  # fmt: skip
    print("wrote", out)


if __name__ == "__main__":
    main()
