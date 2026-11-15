"""Per-frame CAUSAL grab features from a grasp model's stored validation predictions, computed the
way the live demo can (scripts/live_sim.py): the predicted MANO mesh placed in the table frame
(touch.handpath.world_hand, translation fitted to the MediaPipe keypoints, no smoothing) against
known object poses, plus the model's own probabilities and short causal history.

Datasets (validation / never-trained-on only): HO-Cap val (subject 9), DexYCB val (subject 7),
OakInk2 (extra validation). HO-Cap test only with --test (final check, once).
Object poses, two variants (the live demo knows the object at rest; while carried it assumes the
object moves with the hand):
  cur    the true pose at each frame (optimistic while carried)
  start  frozen at the track's first frame (pessimistic: a moved object is "lost")

    python scripts/grab_features.py --run runs/g6_grasp_table_20261112-101602 [--test]
writes runs/grab_fusion/<run>/<dataset>_<variant>.npz  (X [n, F], names, y_any2, y_air, tid,
present, state [n, 3])
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]

from touch.interact.grab_gate import prob_features, rolling  # noqa: E402

OUT = Path("runs/grab_fusion")
PALM = [0, 5, 9, 13, 17]
FAR = 0.15  # m: distances are clipped here
GEOM = ["d_palm", "d_thumb", "d_index", "d_middle", "d_ring", "d_pinky", "n_1cm", "n_2cm",
        "opp_cos", "opp_ok", "ap_ti", "ap_tm", "palm_z", "speed", "vz", "obj_dz"]  # fmt: skip


def geometry(Vw, Jw, objs, parts, variant: str) -> np.ndarray:
    """Vw [N, 778, 3], Jw [N, 21, 3] (table frame, NaN = none); objs: list of (R [N, 3, 3],
    t [N, 3], cKDTree of object-frame surface points, object-frame centroid) -> [N, len(GEOM)]."""
    N = len(Vw)
    G = np.full((N, len(GEOM)), np.nan)
    G[:, :6] = FAR
    for i in range(N):
        if not np.isfinite(Vw[i]).all():
            continue
        best = None
        for R, t, tree, c0 in objs:
            k = 0 if variant == "start" else i
            if not (np.isfinite(t[k]).all() and np.isfinite(R[k]).all()):
                continue
            q = (Vw[i] - t[k]) @ R[k]  # into the object frame
            d, _ = tree.query(q, distance_upper_bound=FAR)
            if best is None or d.min() < best[0].min():
                best = (d, q, c0, t[k][2] - t[0][2] if variant == "cur" else 0.0)
        palm = Jw[i, PALM].mean(0)
        G[i, 10] = np.linalg.norm(Jw[i, 4] - Jw[i, 8])
        G[i, 11] = np.linalg.norm(Jw[i, 4] - Jw[i, 12])
        G[i, 12] = palm[2]
        if i >= 3 and np.isfinite(Jw[i - 3]).all():
            v = (palm - Jw[i - 3, PALM].mean(0)) * 10.0  # m/s over 3 frames at 30 fps
            G[i, 13], G[i, 14] = np.linalg.norm(v), v[2]
        if best is None:
            continue
        d, q, c0, dz = best
        d = np.minimum(d, FAR)
        for p in range(6):
            G[i, p] = d[parts == p].min()
        G[i, 6], G[i, 7] = (d <= 0.01).sum(), (d <= 0.02).sum()
        th, fi = np.flatnonzero(parts == 1), np.flatnonzero(parts >= 2)
        a, b = th[d[th].argmin()], fi[d[fi].argmin()]
        ok = d[a] <= 0.03 and d[b] <= 0.03
        u, w = q[a] - c0, q[b] - c0
        G[i, 8] = float(u @ w / max(np.linalg.norm(u) * np.linalg.norm(w), 1e-9)) if ok else 1.0
        G[i, 9] = float(ok)
        G[i, 15] = dz
    return G


def features(state, finger, present, G) -> tuple[np.ndarray, list[str]]:
    X, names = prob_features(state, finger, present)
    cols, names = [X], list(names)
    cols.append(G)
    names += GEOM
    key = [GEOM.index(n) for n in ("d_thumb", "d_index", "d_middle", "n_1cm", "opp_cos")]
    for w in (5, 15):
        cols.append(rolling(G[:, key], w, np.nanmean))
        names += [f"{GEOM[k]}_m{w}" for k in key]
    z = G[:, GEOM.index("palm_z")]
    cols.append((z - rolling(z[:, None], 30, np.nanmin)[:, 0])[:, None])
    names.append("rise30")
    return np.concatenate(cols, 1).astype(np.float32), names


def pack(state, finger, present, Jw, Vw, objs, parts, y_any2, y_air):
    out = {}
    for variant in ("cur", "start"):
        G = geometry(Vw, Jw, objs, parts, variant)
        X, names = features(state, finger, present, G)
        out[variant] = (X, names)
    return out, y_any2, y_air


def _tree(pts):
    from scipy.spatial import cKDTree

    return cKDTree(pts), pts.mean(0)


def hocap_job(job):
    run, key, split = job
    from eval_hocap import gt_labels

    from touch.data import dexycb as D
    from touch.data import hocap as H
    from touch.handpath import world_hand

    root = Path("runs/hocap_val" if split == "val" else "runs/hocap") / run
    P = np.load(root / "preds.npz")
    _, name, cam, side = key.split("/")
    seq = next(s for s in H.list_sequences(video_only=True, split=split) if s.name == name)
    L = gt_labels(seq, side)
    with np.errstate(all="ignore"):
        wh = world_hand(P[f"{key}/joints"], P[f"{key}/verts"].astype(np.float32),
                        P[f"{key}/lm"], H.intrinsics(cam), H.cam_to_world(cam), side == "left",
                        smooth_k=1)  # fmt: skip
    R, t = H.object_poses(seq)
    objs = [(R[k], t[k], *_tree(H.object_points(o, 3000)[:3000])) for k, o in
            enumerate(seq.objects)]  # fmt: skip
    return key, pack(P[f"{key}/state"], P[f"{key}/finger"], P[f"{key}/present"], wh["joints"],
                     wh["verts"], objs, D.vertex_parts(side), L["state_any2"],
                     L["state_air"])  # fmt: skip


def dexycb_job(job):
    run, subj, name = job
    from touch import robot_sim as S
    from touch.data import dexycb as D
    from touch.handpath import world_hand

    P = np.load(Path("runs") / run / "val_preds.npz")
    seq = next(s for s in D.list_sequences() if s.name == name and s.subject == subj)
    proc = Path("data/dexycb_proc") / subj / name
    c = np.load(proc / "contact.npz")
    y = D.labels_from_dist(c["part_dist"], c["obj_disp"], c["valid"], hold_rule="any2")["state"]
    ex = S.extrinsics(seq)
    tag = ex["apriltag"]
    tt, Rt = S.object_poses_table(seq, tag)
    models = sorted(p for p in (D.ROOT / "models").iterdir() if p.is_dir())
    names = [models[i - 1].name for i in seq.meta["ycb_ids"]]
    objs = [(Rt[:, k], tt[:, k], *_tree(D.object_points(nm, 3000)[:3000])) for k, nm in
            enumerate(names)]  # fmt: skip
    out = {}
    for cam in D.CAMS:
        key = f"{name}/{cam}"
        if f"{key}/present" not in P.files:
            continue
        with np.load(proc / f"{cam}.npz") as z:
            lm, flip = z["lm_px"].astype(np.float32), bool(z["flip"])
        Rc = tag[:, :3].T @ ex[cam][:, :3]
        c2w = np.concatenate([Rc, (tag[:, :3].T @ (ex[cam][:, 3] - tag[:, 3]))[:, None]], 1)
        with np.errstate(all="ignore"):
            wh = world_hand(P[f"{key}/joints"], P[f"{key}/verts"].astype(np.float32), lm,
                            D.intrinsics(cam), c2w, flip, smooth_k=1)  # fmt: skip
        out[key] = pack(P[f"{key}/state"], P[f"{key}/finger"], P[f"{key}/present"], wh["joints"],
                        wh["verts"], objs, D.vertex_parts(seq.side), y, y)  # fmt: skip
    return out


def oakink2_job(job):
    run, name = job
    from eval_oakink2 import Seq, gt_labels, object_poses

    from touch.data import dexycb as D
    from touch.data import oakink2 as O
    from touch.handpath import world_hand

    P = np.load(Path("runs/oakink2") / run / "preds.npz")
    seq = Seq(name)
    A = np.asarray(O.Y_UP_TO_Z_UP)
    R, t = object_poses(seq)  # [K, N, 3, 3], [K, N, 3] world (y up)
    objs = [(A @ R[k], t[k] @ A.T, *_tree(O.object_points(o, 3000)[:3000])) for k, o in
            enumerate(seq.anno["obj_list"])]  # fmt: skip
    out = {}
    for cam in O.CAMS:
        for side in ("right", "left"):
            key = f"{name}/{cam}/{side}"
            if f"{key}/present" not in P.files:
                continue
            c2w = A @ np.linalg.inv(seq.world_to_cam(cam))[:3]
            L = gt_labels(seq, side)
            with np.errstate(all="ignore"):
                wh = world_hand(P[f"{key}/joints"], P[f"{key}/verts"].astype(np.float32),
                                P[f"{key}/lm"], seq.K(cam), c2w, side == "left",
                                smooth_k=1)  # fmt: skip
            out[key] = pack(P[f"{key}/state"], P[f"{key}/finger"], P[f"{key}/present"],
                            wh["joints"], wh["verts"], objs, D.vertex_parts(side),
                            L["state_any2"], L["state_any2"])  # fmt: skip
    return out


def save(ds: str, run: str, results: dict, P_state: dict) -> None:
    d = OUT / run
    d.mkdir(parents=True, exist_ok=True)
    keys = sorted(results)
    for variant in ("cur", "start"):
        X = np.concatenate([results[k][0][variant][0] for k in keys])
        names = results[keys[0]][0][variant][1]
        ya = np.concatenate([results[k][1] for k in keys]).astype(np.int8)
        yr = np.concatenate([results[k][2] for k in keys]).astype(np.int8)
        tid = np.concatenate([np.full(len(results[k][1]), i) for i, k in enumerate(keys)])
        np.savez_compressed(d / f"{ds}_{variant}.npz", X=X, names=np.array(names), y_any2=ya,
                            y_air=yr, tid=tid, keys=np.array(keys),
                            state=X[:, :3], present=X[:, names.index("present")])  # fmt: skip
    print(f"{ds}: {len(keys)} tracks, {len(X)} frames -> {d}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--test", action="store_true", help="HO-Cap TEST (final check only)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="", help="comma list: hocap_val,dexycb_val,oakink2")
    args = ap.parse_args()
    run = args.run.name
    todo = args.only.split(",") if args.only else ["hocap_val", "dexycb_val", "oakink2"]
    if args.test:
        todo = ["hocap_test"]
    t0 = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        for ds in todo:
            res = {}
            if ds.startswith("hocap"):
                split = "val" if ds == "hocap_val" else "test"
                root = Path("runs/hocap_val" if split == "val" else "runs/hocap") / run
                P = np.load(root / "preds.npz")
                keys = sorted({k.rsplit("/", 1)[0] for k in P.files})
                for k, r in pool.imap_unordered(hocap_job, [(run, k, split) for k in keys]):
                    res[k] = r
            elif ds == "dexycb_val":
                from touch.data import dexycb as D

                jobs = [(run, s.subject, s.name) for s in D.list_sequences() if s.split == "val"]
                for r in pool.imap_unordered(dexycb_job, jobs):
                    res.update(r)
            elif ds == "oakink2":
                from touch.data import oakink2 as O

                for r in pool.imap_unordered(oakink2_job, [(run, n) for n in O.list_sequences()]):
                    res.update(r)
            save(ds, run, res, {})
            print(f"  {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
