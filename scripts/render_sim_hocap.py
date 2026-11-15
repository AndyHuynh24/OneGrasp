"""Raw camera | MuJoCo robot sim, on HO-Cap lifts (--split test | val), two models side by side.

Per human lift (random sample, fixed seed): left = the raw top-down camera with the GT and each
model's hand state; then one sim panel per model, rendered from the same calibrated camera. The
sim gripper follows the GT 3D hand path and closes on that model's hold signal (sim_hocap.py
variant "model": debounced, +3 frames). Text under each sim panel: robot lift vs human lift.

    python scripts/render_sim_hocap.py --preds runs/hocap/<a>/preds.npz runs/hocap/<b>/preds.npz
writes results/hocap/sim_camera_<a>_vs_<b>.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from eval_hocap import gt_labels  # noqa: E402
from sim_hocap import delay, lifter, lifts, scene  # noqa: E402

from touch import robot_sim as S  # noqa: E402
from touch.data import dexycb as D  # noqa: E402
from touch.data import hocap as H  # noqa: E402

NAMES = ["none", "touch", "HOLD"]
COL = [(170, 170, 170), (0, 165, 255), (0, 0, 230)]
W, Hh = 640, 480


def one_handed(seq, k, a0, b0) -> bool:
    """True if every hand except the lifter has no GT touch / hold in the lift window."""
    side = lifter(seq, a0, b0)
    return all(not (gt_labels(seq, o)["state_any2"][a0:b0] >= 1).any()
               for o in seq.sides if o != side)  # fmt: skip


def gate_closed(run: str, split: str) -> dict[str, np.ndarray]:
    """Per track key: the saved live gate's closed / open per frame (hold_study.py features for
    hocap_test / hocap_val must exist; gate hysteresis runs from the track start)."""
    import json

    import grab_fusion as G
    import hold_study as HS

    from touch.interact.grab_gate import Ensemble, Fusion

    fdir = Path("runs/grab_fusion") / run
    sel = json.loads((fdir / "fusion_hold.json").read_text())
    d = HS.load(run, "hocap_test" if split == "test" else "hocap_val")
    clf = Ensemble([Fusion.load(f) for f in sorted(fdir.glob("fusion_hold_m*.npz"))])
    sc = clf.predict_proba(d["X"][:, [d["names"].index(n) for n in sel["features"]]])[:, 1]
    closed = G.run_gate(d, sc, **sel["gate"])
    return {str(d["keys"][t]): closed[d["tid"] == t] for t in np.unique(d["tid"])}


def label(img, text, org, scale=0.6, col=(255, 255, 255), th=2):
    cv2.putText(img, text, org, 0, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, 0, scale, col, th, cv2.LINE_AA)


def with_camera(xml: str, cam: str) -> str:
    T, K = H.cam_to_world(cam), H.intrinsics(cam)
    x, y = T[:, 0], -T[:, 1]  # OpenCV (x right, y down) -> MuJoCo camera (x right, y up)
    fovy = np.degrees(2 * np.arctan(K[1, 2] / K[1, 1]))
    return xml.replace("<worldbody>", f'<worldbody><camera name="rec" pos="{T[0, 3]} {T[1, 3]} '
                       f'{T[2, 3]}" xyaxes="{x[0]} {x[1]} {x[2]} {y[0]} {y[1]} {y[2]}" '
                       f'fovy="{fovy}"/>')  # fmt: skip


def sim(seq, k, a0, b0, L, close, cam):
    import mujoco

    pp, RR, Wd, _ = S.retarget(L["joints"], L["valid"], close, L["verts"], L["vert"] == 1,
                               L["parts"])  # fmt: skip
    sc = scene(seq, a0)
    sc = S.Scene(with_camera(sc.xml, cam), sc.names, 0)
    rnd: dict = {}

    def render(m, d, t):
        if "r" not in rnd:
            rnd["r"] = mujoco.Renderer(m, Hh, W)
        rnd["r"].update_scene(d, camera="rec")
        return rnd["r"].render()[..., ::-1].copy()

    sl = slice(a0, b0)
    res = S.replay(sc, pp[sl], RR[sl], Wd[sl], L["valid"][sl], close[sl], render=render)
    _, t = H.object_poses(seq)
    human = float((t[k, sl, 2] - t[k, a0, 2]).max())
    robot = float((res["obj"][:, k, 2] - res["obj"][0, k, 2]).max())
    return res["frames"], human, robot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, nargs="+", required=True, help="1 or 2 models")
    ap.add_argument(
        "--latch",
        action="store_true",
        help="also show each model + the causal hold latch (touch.interact.hold_latch, "
        "parameters from <run>/hold_latch.json, tuned on validation)",
    )
    ap.add_argument("--gate", action="store_true",
                    help="also show each model + its learned live grab gate (fusion_hold.*, "
                    "fit on validation; scores over the whole track, causal)")  # fmt: skip
    ap.add_argument(
        "--gt",
        default="any2",
        choices=["any2", "air"],
        help="GT hold rule shown (air = held AND off the table)",
    )
    ap.add_argument("--cam", default="105322251564")
    ap.add_argument("--n", type=int, default=6, help="lifts to render")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--split",
        default="test",
        choices=["test", "val"],
        help="val = subject 9 (used for checkpoint selection: not a held-out score)",
    )
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--one-hand", action="store_true",
                    help="only lifts where the other hand has no GT contact")  # fmt: skip
    ap.add_argument("--successes", action="store_true",
                    help="only lifts the sim lifted for >= 1 model (sim_final.json, variant "
                         "'model'): a SELECTED set, not representative")  # fmt: skip
    args = ap.parse_args()
    variants = []  # (name, preds, latch params or None)
    for p in args.preds:
        P, nm = np.load(p), p.parent.name.split("_")[0]
        variants.append((nm, P, None))
        if args.latch:
            import json

            run = Path("runs") / p.parent.name
            prm = json.loads((run / "hold_latch.json").read_text())["best"]["params"]
            variants.append((f"{nm}+latch", P, prm))
        if args.gate:
            variants.append((f"{nm}+gate", P, {"gate": gate_closed(p.parent.name, args.split)}))
    Ps = [v[1] for v in variants]
    names = [v[0] for v in variants]
    have = {f.split("/")[1] for f in Ps[0].files}  # sequences with predictions
    jobs = [(seq, k, a0, b0) for seq in H.list_sequences(video_only=True, split=args.split)
            if seq.name in have for k, a0, b0 in lifts(seq)]  # fmt: skip
    if args.one_hand:
        jobs = [j for j in jobs if one_handed(*j)]
        print(len(jobs), "one-handed lifts", flush=True)
    if args.successes:
        import json

        won = set()
        for p in args.preds:
            for r in json.loads((p.parent / "sim_final.json").read_text())["trials"]:
                if r["variant"] == "model" and r["cam"] == args.cam and r["success"]:
                    won.add((r["seq"], r["object"], r["frames"][0]))
        jobs = [j for j in jobs if (j[0].name, j[0].objects[j[1]], j[2]) in won]
        print(len(jobs), "lifts the sim lifted for at least one model", flush=True)
    rng = np.random.default_rng(args.seed)
    jobs = [jobs[i] for i in sorted(rng.choice(len(jobs), min(args.n, len(jobs)), replace=False))]
    tag = "_successes" if args.successes else f"_random{args.seed}"
    tag += "_onehand" if args.one_hand else ""
    tag += "_val" if args.split == "val" else ""
    tag += f"_{args.cam}" if args.cam != "105322251564" else ""
    out = Path("results/hocap") / f"sim_camera_{'_vs_'.join(names)}{tag}.mp4"
    raw = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                         (W * (1 + len(Ps)), Hh + 30))  # fmt: skip
    for seq, k, a0, b0 in jobs:
        side = lifter(seq, a0, b0)
        key = f"{seq.subject}/{seq.name}/{args.cam}/{side}"
        if f"{key}/present" not in Ps[0].files:
            continue
        L = {**gt_labels(seq, side), "parts": D.vertex_parts(side)}
        gt = L[f"state_{args.gt}"]
        st, sims = [], []
        for _, P, prm in variants:
            s = np.where(P[f"{key}/present"], P[f"{key}/state"].argmax(1), 0)
            if prm is not None and "gate" in prm:  # learned gate: closed -> HOLD, else <= touch
                s = np.where(prm["gate"][key], 2, np.minimum(s, 1))
                st.append(s)
                sims.append(sim(seq, k, a0, b0, L, delay(s == 2), args.cam))
                continue
            if prm is not None:  # whole track from its start (causal), then the window
                from touch.handpath import world_hand
                from touch.interact.hold_latch import HoldLatch, palm_height

                with np.errstate(all="ignore"):
                    wh = world_hand(
                        P[f"{key}/joints"],
                        P[f"{key}/verts"].astype(np.float32),
                        P[f"{key}/lm"],
                        H.intrinsics(args.cam),
                        H.cam_to_world(args.cam),
                        side == "left",
                        smooth_k=1,
                    )
                    s = HoldLatch(**prm).run(s, P[f"{key}/present"], palm_height(wh["joints"]))
            st.append(s)
            sims.append(sim(seq, k, a0, b0, L, delay(S.debounce(s == 2)), args.cam))
        lm = Ps[0][f"{key}/lm"]
        obj = seq.objects[k]
        print(
            seq.name,
            obj,
            side,
            [
                f"{n}: robot {r:.3f} / human {h:.3f} m"
                for n, (_, h, r) in zip(names, sims, strict=True)
            ],
            flush=True,
        )
        for j, i in enumerate(range(a0, b0)):
            cam = cv2.imread(str(seq.image(args.cam, i)))
            if np.isfinite(lm[i]).all():
                x0, y0 = lm[i, :, :2].min(0).astype(int)
                x1, y1 = lm[i, :, :2].max(0).astype(int)
                cv2.rectangle(cam, (x0, y0), (x1, y1), (255, 255, 255), 2)
            g = int(gt[i])
            label(cam, f"GT ({args.gt}) {NAMES[g] if g >= 0 else '?'}", (12, 28), 0.7,
                  COL[max(g, 0)])  # fmt: skip
            for m, (n, s) in enumerate(zip(names, st, strict=True)):
                label(cam, f"{n}: {NAMES[int(s[i])]}", (12, 58 + 28 * m), 0.7, COL[int(s[i])])
            label(cam, f"camera  {seq.name}  {side} hand lifts {obj}", (12, Hh - 12), 0.45)
            tiles = [cam]
            for n, s, (fr, h, r) in zip(names, st, sims, strict=True):
                t = fr[j].copy() if j < len(fr) else np.zeros((Hh, W, 3), np.uint8)
                c = int(s[i])
                label(t, f"robot driven by {n}: {'CLOSED' if c == 2 else 'open'}", (12, 28), 0.7,
                      COL[c])  # fmt: skip
                ok = "LIFTED" if r >= 0.5 * h else "failed"
                label(t, f"{ok}: robot {r * 100:.1f} cm / human {h * 100:.1f} cm", (12, Hh - 12),
                      0.55)  # fmt: skip
                tiles.append(t)
            row = np.hstack(tiles)
            bar = np.zeros((30, row.shape[1], 3), np.uint8)
            label(
                bar,
                "sim: gripper follows the GT hand path, closes on the model's HOLD "
                "(debounced, +3 frames); objects start at their true poses",
                (8, 20),
                0.5,
            )
            vw.write(np.vstack([row, bar]))
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "26", str(out)], check=True)  # fmt: skip
    raw.unlink()
    print("wrote", out)


if __name__ == "__main__":
    main()
