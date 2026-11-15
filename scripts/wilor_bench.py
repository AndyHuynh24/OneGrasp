"""Is WiLoR's 3D hand (CVPR 2025, github.com/rolpotamias/WiLoR, CC-BY-NC-ND) more accurate than our
grasp model's on HO-Cap val (subject 9)? Same frames, same MediaPipe box, same handedness; error vs
the HO-Cap GT hand in the convention of scripts/eval_hocap.py (wrist-relative joints in camera
axes, left hands mirrored to right; MPJPE and Procrustes-aligned PA-MPJPE, mm).

    python scripts/wilor_bench.py export --run runs/<run> [--every 15]   # main venv
    cd third_party/WiLoR && PYOPENGL_PLATFORM=darwin ../wilor_venv/bin/python \\
        ../../scripts/wilor_run.py ../../runs/wilor_bench/jobs.npz          # WiLoR venv
    python scripts/wilor_bench.py compare                                 # main venv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]

OUT = Path("runs/wilor_bench")


def export(run: str, every: int) -> None:
    from eval_hocap import gt_cam_mesh, gt_labels

    from touch.data import hocap as H

    P = np.load(Path("runs/hocap_val") / run / "preds.npz")
    keys = sorted({k.rsplit("/", 1)[0] for k in P.files})
    seqs = {s.name: s for s in H.list_sequences(video_only=True, split="val")}
    rows = {k: [] for k in ("img", "box", "right", "gj", "ours", "key", "frame")}
    for key in keys:
        subj, name, cam, side = key.split("/")
        seq = seqs[name]
        L = gt_labels(seq, side)
        gj, _ = gt_cam_mesh(L, cam, side == "left")
        pres, lm, pj = P[f"{key}/present"], P[f"{key}/lm"], P[f"{key}/joints"]
        for i in range(0, len(pres), every):
            if not (pres[i] and L["valid"][i] and np.isfinite(lm[i]).all()):
                continue
            lo, hi = lm[i, :, :2].min(0), lm[i, :, :2].max(0)
            rows["img"].append(str(seq.image(cam, i).resolve()))
            rows["box"].append(np.r_[lo, hi])
            rows["right"].append(side == "right")
            rows["gj"].append(gj[i])
            rows["ours"].append(pj[i])
            rows["key"].append(key)
            rows["frame"].append(i)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "jobs.npz", **{k: np.asarray(v) for k, v in rows.items()})
    print(f"{len(rows['img'])} crops from {len(keys)} tracks -> {OUT / 'jobs.npz'}")


def _errs(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from eval_hocap import procrustes_err

    p, g = pred - pred[:, :1], gt - gt[:, :1]
    mpjpe = np.linalg.norm(p - g, axis=-1).mean(1) * 1000
    pa = np.array([procrustes_err(a, b) for a, b in zip(p, g, strict=True)]) * 1000
    return mpjpe, pa


def ray_correct(joints: np.ndarray, J) -> np.ndarray:
    """WiLoR (like HaMeR) predicts the hand's rotation in a virtual camera looking straight at the
    crop; rotate it so that camera's z axis points along the real ray through the box centre
    (HO-Cap intrinsics). Left hands: the mirrored frame mirrors the ray too."""
    from touch.data import hocap as H

    out = np.empty_like(joints)
    z = np.array([0.0, 0.0, 1.0])
    for i, (key, box, right) in enumerate(zip(J["key"], J["box"], J["right"], strict=True)):
        d = np.linalg.inv(H.intrinsics(key.split("/")[2])) @ np.r_[(box[:2] + box[2:]) / 2, 1]
        d /= np.linalg.norm(d)
        if not right:
            d[0] *= -1
        v, c = np.cross(z, d), z @ d
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / max(v @ v, 1e-12))
        out[i] = joints[i] @ R.T
    return out


def compare() -> None:
    J = np.load(OUT / "jobs.npz")
    W = dict(np.load(OUT / "wilor.npz"))
    W["joints_raw"], W["joints"] = W["joints"], ray_correct(W["joints"], J)
    ok = W["ok"]
    res = {"n": int(ok.sum()), "n_jobs": len(ok)}
    for name, pred in (("ours", J["ours"]), ("wilor", W["joints"])):
        m, pa = _errs(pred[ok], J["gj"][ok])
        res[name] = {"mpjpe_mm": round(float(m.mean()), 2),
                     "mpjpe_median": round(float(np.median(m)), 2),
                     "pa_mpjpe_mm": round(float(pa.mean()), 2)}  # fmt: skip
    cams = np.array([k.split("/")[2] for k in J["key"]])
    for cam in sorted(set(cams)):
        sel = ok & (cams == cam)
        res[cam] = {n: round(float(_errs(p[sel], J["gj"][sel])[0].mean()), 2)
                    for n, p in (("ours", J["ours"]), ("wilor", W["joints"]))}  # fmt: skip
    print(json.dumps(res, indent=1))
    (OUT / "compare.json").write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["export", "compare"])
    ap.add_argument("--run", type=Path, default=Path("runs/g8_grasp_webcam_20261113-011442"))
    ap.add_argument("--every", type=int, default=15)
    args = ap.parse_args()
    export(args.run.name, args.every) if args.stage == "export" else compare()


if __name__ == "__main__":
    main()
