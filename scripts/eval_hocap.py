"""Zero-shot HO-Cap TEST benchmark of a grasp model: the live, causal ONNX path on every camera.

HO-Cap is never used for training or tuning (src/touch/data/hocap.py). For each video sequence,
camera and GT hand:
  1. MediaPipe (2 hands) on every frame, cached in data/hocap/cache/mp/.
  2. Association: the MediaPipe hand nearest the GT hand's projected joints is that hand's track
     (uses GT only to pick WHICH hand, like DexYCB's GT handedness; the model never sees GT).
  3. LiveGrasp (scripts/live_grasp.py) runs causally over the track; left hands mirrored by the
     GT side, as on DexYCB val.
Scores use train_grasp.evaluate's definitions (frames without a MediaPipe hand count as a miss)
and are reported overall and by camera group (top-down = optical axis within 35 deg of straight
down: the overhead-webcam setup; low = below the table edge looking up; oblique = the rest), for
the original hold label ("thumb") and the corrected one ("any2"). Extra HO-Cap-only scores:
  table_rest: hand touching the table (lowest GT vertex <= 1 cm above it) with no object within
              10 mm; GT is "none", reported is how often the model says touch / hold.
  mesh:       MPJPE / PA-MPJPE / MPVPE (mm), wrist-relative, camera axes, left mirrored.

    python scripts/eval_hocap.py --ckpt runs/<run>/best.pt [--workers 4]
writes runs/hocap/<run>/preds.npz (per-track predictions + associated landmarks for WiLoR) and
runs/hocap/<run>/metrics.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
from touch.data import dexycb as D  # noqa: E402
from touch.data import hocap as H  # noqa: E402
from touch.eval.metrics import interval_timing  # noqa: E402
from touch.train_grasp import _f1, _runs, _timing  # noqa: E402

CACHE = H.ROOT.parent / "cache"
ASSOC_FRAC = 0.5  # accept a MediaPipe hand if mean joint distance < 0.5 x GT 2D hand size
STATE_NAMES = ("none", "touch", "hold")


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()  # fmt: skip
    except Exception:
        return "unknown"


def cam_group(serial: str) -> str:
    T = H.cam_to_world(serial)
    ang = np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1)))  # optical axis vs straight down
    return "top-down" if ang < 35 else ("low" if ang > 90 else "oblique")


def world_to_cam(x: np.ndarray, serial: str) -> np.ndarray:
    T = H.cam_to_world(serial)
    return (x - T[:, 3]) @ T[:, :3]


def gt_labels(seq: H.Seq, side: str) -> dict:
    """Cached GT per (sequence, hand): distances, both label rules, vertex labels, world mesh."""
    f = CACHE / "labels" / f"{seq.subject}_{seq.name}_{side}.npz"
    if not f.exists():
        r = H.labels(seq, side)
        R, t = H.object_poses(seq)
        L2 = D.labels_from_dist(r["part_dist"], r["obj_disp"], r["valid"], hold_rule="any2")
        vd = r["vert_dist"] * 1000
        vert = np.where(vd <= 3, 1, np.where(vd > 10, 0, -1)).astype(np.int8)
        vert[~r["valid"]] = -1
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, finger=r["finger"], state=r["state"], state_any2=L2["state"],
                            vert=vert, part_dist=r["part_dist"], valid=r["valid"],
                            joints=r["joints"].astype(np.float32),
                            verts=r["verts"].astype(np.float32),
                            obj_disp=r["obj_disp"])  # fmt: skip
    L = dict(np.load(f))
    if "state_air" not in L:  # hold = held AND off the table (added 2026-11-12)
        touched = H.labels(seq, side)["touched_obj"]
        L["state_air"] = H.hold_air(L["state_any2"], touched, H.object_low_z(seq))
        L["touched_obj"] = touched
        np.savez_compressed(f, **L)
    return L


def mediapipe_cache(seq: H.Seq, cam: str, n: int) -> dict:
    f = CACHE / "mp" / f"{seq.subject}_{seq.name}_{cam}.npz"
    if f.exists():
        return dict(np.load(f))
    from mediapipe import Image, ImageFormat

    from touch.data.landmarks import DEFAULT_MODEL, _make_landmarker

    lm = np.full((n, 2, 21, 3), np.nan, np.float32)
    with _make_landmarker(DEFAULT_MODEL, 0.3) as det:
        for i in range(n):
            img = cv2.imread(str(seq.image(cam, i)))
            if img is None:
                continue
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            r = det.detect_for_video(
                Image(image_format=ImageFormat.SRGB, data=rgb), int(i * 1000 / H.FPS)
            )
            for k, hand in enumerate(r.hand_landmarks[:2]):
                lm[i, k] = [[p.x * w, p.y * h, p.z * w] for p in hand]
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, lm=lm)
    return {"lm": lm}


def associate(lm: np.ndarray, gt_uv: dict[str, np.ndarray], gt_ok: dict[str, np.ndarray]):
    """lm [N, 2, 21, 3] MediaPipe; gt_uv side -> [N, 21, 2] -> side -> [N, 21, 3] (NaN = none).
    Each MediaPipe hand goes to at most one GT hand (the closer one)."""
    n = len(lm)
    out = {s: np.full((n, 21, 3), np.nan, np.float32) for s in gt_uv}
    for i in range(n):
        cand = []
        for s, uv in gt_uv.items():
            if not gt_ok[s][i]:
                continue
            size = (uv[i].max(0) - uv[i].min(0)).max()
            for k in range(2):
                if np.isfinite(lm[i, k]).all():
                    d = np.linalg.norm(lm[i, k, :, :2] - uv[i], axis=1).mean()
                    if d < ASSOC_FRAC * size:
                        cand.append((d, s, k))
        used_s, used_k = set(), set()
        for _d, s, k in sorted(cand):
            if s not in used_s and k not in used_k:
                out[s][i] = lm[i, k]
                used_s.add(s)
                used_k.add(k)
    return out


def run_track(live, seq: H.Seq, cam: str, lm: np.ndarray, left: bool) -> dict:
    from collections import deque

    live.hist = deque([(live.pad_tok, live.pad_he, False)] * (live.T - 1), maxlen=live.T - 1)
    live.lm_hist.clear()
    live.last_imgs = None
    live.sticky.prev = 0
    n = len(lm)
    out = {"state": np.zeros((n, 3), np.float32), "finger": np.zeros((n, 5), np.float32),
           "vert": np.zeros((n, 778), np.float16),
           "joints": np.full((n, 21, 3), np.nan, np.float32),
           "verts": np.full((n, 778, 3), np.nan, np.float16), "sticky": np.zeros(n, np.int8),
           "present": np.isfinite(lm).all((1, 2))}  # fmt: skip
    for i in range(n):
        img = cv2.imread(str(seq.image(cam, i)))
        x = lm[i] if out["present"][i] else None
        r = live.step(live.prep(img, x, left))
        out["state"][i], out["finger"][i], out["sticky"][i] = (
            r["state"],
            r["finger"],
            r["state_sticky"],
        )
        out["vert"][i] = r["vert"]
        if out["present"][i]:
            out["joints"][i], out["verts"][i] = r["joints"], r["verts"]
    return out


def work(job) -> dict:
    """One (sequence, camera): MediaPipe, association, model per GT hand."""
    ckpt, subject, name, cam = job
    from live_grasp import LiveGrasp

    seq = next(s for s in H.list_sequences(video_only=True) if s.name == name)
    n = seq.n_frames
    K = H.intrinsics(cam)
    lm = mediapipe_cache(seq, cam, n)["lm"]
    gt_uv, gt_ok = {}, {}
    for side in seq.sides:
        L = gt_labels(seq, side)
        jc = world_to_cam(L["joints"], cam)
        gt_uv[side] = D.project(jc.reshape(-1, 3), K).reshape(n, 21, 2)
        gt_ok[side] = L["valid"] & (jc[:, :, 2] > 0.05).all(1)
    tracks = associate(lm, gt_uv, gt_ok)
    live = LiveGrasp(Path(ckpt), threads=2)
    res = {}
    for side, tlm in tracks.items():
        r = run_track(live, seq, cam, tlm, side == "left")
        r["lm"] = tlm
        res[f"{subject}/{name}/{cam}/{side}"] = r
    return res


def gt_cam_mesh(L: dict, cam: str, left: bool) -> tuple[np.ndarray, np.ndarray]:
    j, v = world_to_cam(L["joints"], cam), world_to_cam(L["verts"], cam)
    v, j = v - j[:, :1], j - j[:, :1]
    if left:
        j[..., 0] *= -1
        v[..., 0] *= -1
    return j.astype(np.float32), v.astype(np.float32)


def procrustes_err(a: np.ndarray, b: np.ndarray) -> float:
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(A.T @ B)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
        S[-1] *= -1
    al = (S.sum() / (A**2).sum()) * A @ (U @ Vt) + mb
    return float(np.linalg.norm(al - b, axis=1).mean())


def score(items: list[dict]) -> dict:
    """items: per track {pred dict, L (GT), gj, gv}. Same definitions as train_grasp.evaluate."""
    acc = {k: [] for k in ("Fp", "Fg", "Vp", "Vg", "tf", "pres", "rest", "hover", "mj", "pa", "mv")}
    S = {rule: ([], [], [], []) for rule in ("thumb", "any2", "air")}  # ps, gs, sticky, timing
    for it in items:
        P, L = it["P"], it["L"]
        pres = P["present"]
        pf = (P["finger"] >= 0.5) & pres[:, None]
        ps = np.where(pres, P["state"].argmax(1), 0)
        fm = L["finger"] >= 0
        acc["Fp"].append(pf[fm])
        acc["Fg"].append(L["finger"][fm] == 1)
        vm = (L["vert"] >= 0) & pres[:, None]
        acc["Vp"].append((P["vert"] >= 0.5)[vm])
        acc["Vg"].append(L["vert"][vm] == 1)
        acc["pres"].append(pres[L["valid"]])
        for rule, key in (("thumb", "state"), ("any2", "state_any2"), ("air", "state_air")):
            if key not in L:  # e.g. OakInk2 has no air labels
                continue
            gs = L[key]
            sm = gs >= 0
            S[rule][0].append(ps[sm])
            S[rule][1].append(gs[sm])
            S[rule][2].append(P["sticky"][sm])
            for a, e in _runs(sm):
                g2 = gs[a:e] == 2
                S[rule][3].append((interval_timing(ps[a:e] == 2, g2),
                                   interval_timing(P["sticky"][a:e] == 2, g2)))  # fmt: skip
        sm = L["state"] >= 0
        for a, e in _runs(sm):
            for f in range(5):
                acc["tf"].append(interval_timing(pf[a:e, f], L["finger"][a:e, f] == 1))
        d = L["part_dist"].min(1) * 1000
        on_table = (L["verts"][:, :, 2].min(1) <= 0.01) & (d > 10) & L["valid"]
        acc["rest"].append(ps[on_table])
        near = (d <= 30) & (d > 10) & L["valid"]
        acc["hover"].append(pf[near].any(1))
        ok = pres & L["valid"]
        pj, pv = P["joints"][ok], P["verts"][ok].astype(np.float32)
        gj, gv = it["gj"][ok], it["gv"][ok]
        acc["mj"].append(np.linalg.norm(pj - gj, axis=-1).mean(1) * 1000)
        acc["mv"].append(np.linalg.norm(pv - gv, axis=-1).mean(1) * 1000)
        acc["pa"].append(
            np.array([procrustes_err(a, b) for a, b in zip(pj[::5], gj[::5], strict=True)]) * 1000
        )
    cat = {k: np.concatenate(v) if v and k != "tf" else v for k, v in acc.items()}
    out = {"finger": _f1(cat["Fp"], cat["Fg"]), "vert": _f1(cat["Vp"], cat["Vg"]),
           "finger_timing": _timing(cat["tf"]),
           "present_rate": round(float(cat["pres"].mean()), 4)}  # fmt: skip
    for rule, (ps, gs, st, tm) in S.items():
        if not ps:
            continue
        ps, gs, st = np.concatenate(ps), np.concatenate(gs), np.concatenate(st)
        for name, p in (("raw", ps), ("sticky", st)):
            per = {n: _f1(p == c, gs == c) for c, n in enumerate(STATE_NAMES)}
            out[f"state_{rule}_{name}"] = {
                **per,
                "macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 4),
            }
        out[f"hold_timing_{rule}_raw"] = _timing([t[0] for t in tm])
        out[f"hold_timing_{rule}_sticky"] = _timing([t[1] for t in tm])
    r = cat["rest"]
    out["table_rest"] = {"frames": int(len(r)), "says_touch": round(float((r == 1).mean()), 4),
                         "says_hold": round(float((r == 2).mean()), 4)}  # fmt: skip
    out["near_hover"] = {"frames": int(len(cat["hover"])),
                         "false_contact_rate": round(float(cat["hover"].mean()), 4)}  # fmt: skip
    out["mesh"] = {"mpjpe_mm": round(float(cat["mj"].mean()), 2),
                   "pa_mpjpe_mm": round(float(cat["pa"].mean()), 2),
                   "mpvpe_mm": round(float(cat["mv"].mean()), 2)}  # fmt: skip
    out["score_thumb"] = round((out["finger"]["f1"] + out["state_thumb_raw"]["macro_f1"]) / 2, 4)
    out["n_tracks"] = len(items)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N (sequence, camera) jobs (smoke)")
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="test = subjects 5 + 8, val = 9 (runs/hocap_val/)")  # fmt: skip
    ap.add_argument(
        "--rescore",
        action="store_true",
        help="score the saved preds.npz again (new label rules); no model run",
    )
    args = ap.parse_args()
    seqs = H.list_sequences(video_only=True, split=args.split)
    assert seqs, "no HO-Cap video sequences under data/hocap/ds"
    root = Path("runs/hocap") if args.split == "test" else Path(f"runs/hocap_{args.split}")
    out_dir = root / args.ckpt.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for s in seqs:  # GT once, in the parent (workers read the cache)
        for side in s.sides:
            gt_labels(s, side)
    print(f"GT labels ready ({time.time() - t0:.0f} s)", flush=True)
    jobs = [
        (str(args.ckpt), s.subject, s.name, c)
        for s in seqs
        for c in s.serials
        if (s.root / c / f"color_{s.n_frames - 1:06d}.jpg").exists()
    ]  # downloaded cameras
    if args.limit:
        jobs = jobs[: args.limit]
    preds = {}
    if args.rescore:
        Z = np.load(out_dir / "preds.npz")
        for k in Z.files:
            key, f = k.rsplit("/", 1)
            preds.setdefault(key, {})[f] = Z[k]
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(work, jobs)):
                preds.update(r)
                print(f"{k + 1}/{len(jobs)} {list(r)} | {(time.time() - t0) / 60:.1f} min",
                      flush=True)  # fmt: skip
        flat = {f"{key}/{f}": v for key, r in preds.items() for f, v in r.items()}
        np.savez(out_dir / "preds.npz", **flat)

    groups: dict[str, list] = {"all": []}
    seq_by = {s.name: s for s in seqs}
    for key, P in sorted(preds.items()):
        subject, name, cam, side = key.split("/")
        L = gt_labels(seq_by[name], side)
        gj, gv = gt_cam_mesh(L, cam, side == "left")
        it = {"P": P, "L": L, "gj": gj, "gv": gv}
        groups["all"].append(it)
        groups.setdefault(cam_group(cam), []).append(it)
    metrics = {g: score(v) for g, v in groups.items()}
    metrics["meta"] = {"ckpt": str(args.ckpt), "git": git_hash(),
                       "sequences": [s.name for s in seqs],
                       "cameras": {c: cam_group(c) for c in seqs[0].serials},
                       "minutes": round((time.time() - t0) / 60, 1)}  # fmt: skip
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    for g in groups:
        m = metrics[g]
        st2, sa = m["state_any2_raw"]["macro_f1"], m["state_air_raw"]["macro_f1"]
        ha = m["state_air_raw"]["hold"]
        print(f"{g:9s} tracks {m['n_tracks']:3d} | finger F1 {m['finger']['f1']:.3f} | "
              f"state macro any2 {st2:.3f} air {sa:.3f} | air hold P/R {ha['precision']:.2f}/"
              f"{ha['recall']:.2f} | MPJPE {m['mesh']['mpjpe_mm']}")  # fmt: skip
    print("wrote", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
