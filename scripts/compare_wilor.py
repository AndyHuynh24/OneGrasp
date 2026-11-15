"""Our grasp model vs WiLoR (scripts/run_wilor.py) on DexYCB val: hand-mesh accuracy + a render.

Same frames for both (our MediaPipe hand present, GT annotated, both predictions finite), same
convention: wrist-relative, camera axes, left hands mirrored to right. Reports MPJPE / PA-MPJPE /
MPVPE (mm), overall and split by GT state (none / touch / hold: hand-object occlusion grows).

    python scripts/compare_wilor.py --ours runs/<run>/val_preds.npz
writes runs/baselines/compare_wilor.json and results/baselines/wilor_vs_ours.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch.data import dexycb as D  # noqa: E402

PROC = Path("data/dexycb_proc")


def procrustes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Align a [K, 3] to b with scale, rotation and translation."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(A.T @ B)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
        S[-1] *= -1
    return (S.sum() / (A**2).sum()) * A @ (U @ Vt) + mb


def overlay(img, verts, faces, K, color):
    """Painter's-algorithm mesh (camera frame, m) blended onto img."""
    p = verts[:, :2] / verts[:, 2:] * [K[0, 0], K[1, 1]] + K[:2, 2]
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    shade = 0.35 + 0.65 * np.abs(n[:, 2])
    layer = img.copy()
    for f in np.argsort(-tri[:, :, 2].mean(1)):
        c = tuple(int(x) for x in np.asarray(color) * shade[f])
        cv2.fillConvexPoly(layer, p[faces[f]].astype(np.int32), c, cv2.LINE_AA)
    return cv2.addWeighted(layer, 0.65, img, 0.35, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ours", type=Path, default=Path("runs/g1_grasp_repvit_20261109-154049/val_preds.npz")
    )
    ap.add_argument("--wilor", type=Path, default=Path("runs/baselines/wilor_val.npz"))
    ap.add_argument("--video", type=Path, default=Path("results/baselines/wilor_vs_ours.mp4"))
    ap.add_argument("--n-clips", type=int, default=6)
    args = ap.parse_args()
    Op, W = np.load(args.ours), np.load(args.wilor)
    seqs = {s.name: s for s in D.list_sequences() if s.split == "val"}
    err = {m: {k: [] for k in ("mpjpe", "pa_mpjpe", "mpvpe")} for m in ("ours", "wilor")}
    by_state = {m: {s: [] for s in (0, 1, 2)} for m in ("ours", "wilor")}
    keys = sorted(k[: -len("/joints")] for k in W.files if k.endswith("/joints"))
    for key in keys:
        name, cam = key.split("/")
        p = next(PROC.glob(f"*/{name}/{cam}.npz"))
        z, c = np.load(p), np.load(p.parent / "contact.npz")
        gj, gv = z["joints"].astype(np.float32), z["verts"].astype(np.float32)
        pred = {"ours": (Op[f"{key}/joints"], Op[f"{key}/verts"].astype(np.float32)),
                "wilor": (W[f"{key}/joints"], W[f"{key}/verts"].astype(np.float32))}  # fmt: skip
        ok = z["present"] & c["valid"] & np.isfinite(pred["wilor"][0]).all((1, 2))
        ok &= np.isfinite(pred["ours"][0]).all((1, 2))
        for i in np.flatnonzero(ok):
            for m, (pj, pv) in pred.items():
                e = float(np.linalg.norm(pj[i] - gj[i], axis=-1).mean()) * 1000
                err[m]["mpjpe"].append(e)
                err[m]["pa_mpjpe"].append(
                    float(np.linalg.norm(procrustes(pj[i], gj[i]) - gj[i], axis=-1).mean()) * 1000
                )
                err[m]["mpvpe"].append(float(np.linalg.norm(pv[i] - gv[i], axis=-1).mean()) * 1000)
                if c["state"][i] >= 0:
                    by_state[m][int(c["state"][i])].append(e)
    res = {
        m: {k: round(float(np.mean(v)), 2) for k, v in e.items()}
        | {
            "mpjpe_by_gt_state": {
                n: round(float(np.mean(by_state[m][s])), 2)
                for s, n in enumerate(("none", "touch", "hold"))
            }
        }
        for m, e in err.items()
    }
    res["frames"] = len(err["ours"]["mpjpe"])
    res["speed_ms_per_hand"] = {
        "wilor_model_batch16_mps": round(float(W["sec_per_hand_model"]) * 1000, 1),
        "ours_model_onnx_cpu": "17.2 (live benchmark, docs/DECISIONS.md)",
    }
    out = Path("runs/baselines/compare_wilor.json")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

    # render: camera | ours | WiLoR | GT (each mesh placed at the GT wrist position)
    faces = {s: np.asarray(D.mano_model(s)["f"]).astype(np.int64) for s in ("right", "left")}
    cam = "840412060917"
    K = D.intrinsics(cam)
    picks, seen = [], set()
    for name, s in seqs.items():
        if f"{name}/{cam}/joints" in W.files and s.grasp_object not in seen:
            seen.add(s.grasp_object)
            picks.append(s)
        if len(picks) >= args.n_clips:
            break
    args.video.parent.mkdir(parents=True, exist_ok=True)
    raw = args.video.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 960))
    cols = {"ours": (60, 170, 255), "wilor": (255, 160, 60), "GT": (90, 220, 90)}
    for s in picks:
        key = f"{s.name}/{cam}"
        p = next(PROC.glob(f"*/{s.name}/{cam}.npz"))
        z, c = np.load(p), np.load(p.parent / "contact.npz")
        flip, root = bool(z["flip"]), z["root"]
        fc = faces["left" if flip else "right"]
        meshes = {"ours": Op[f"{key}/verts"].astype(np.float32),
                  "wilor": W[f"{key}/verts"].astype(np.float32),
                  "GT": z["verts"].astype(np.float32)}  # fmt: skip
        for i in range(len(root)):
            img = cv2.imread(str(s.image(cam, i)))
            tiles = [img.copy()]
            cv2.putText(
                tiles[0],
                f"{s.grasp_object[4:]}  ({s.side} hand)",
                (14, 462),
                0,
                0.7,
                (255, 255, 255),
                2,
            )
            for m, V in meshes.items():
                t = img.copy()
                good = c["valid"][i] and np.isfinite(V[i]).all() and z["present"][i]
                if good:
                    v = V[i].copy()
                    if flip:
                        v[:, 0] *= -1
                    t = overlay(t, v + root[i], fc, K, cols[m])
                    if m != "GT":
                        e = float(np.linalg.norm(V[i] - meshes["GT"][i], axis=-1).mean()) * 1000
                        cv2.putText(
                            t, f"vertex error {e:.1f} mm", (14, 462), 0, 0.7, (255, 255, 255), 2
                        )
                else:
                    cv2.putText(t, "no hand / no GT", (14, 462), 0, 0.7, (200, 200, 200), 2)
                label = {
                    "ours": "OURS (17 ms, CPU)",
                    "wilor": "WiLoR (102 ms, GPU)",
                    "GT": "ground truth",
                }[m]
                cv2.putText(t, label, (14, 34), 0, 0.9, cols[m], 2)
                tiles.append(t)
            vw.write(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt",
                    "yuv420p", "-crf", "23", str(args.video)], check=True)  # fmt: skip
    raw.unlink()
    print("wrote", args.video)


if __name__ == "__main__":
    main()
