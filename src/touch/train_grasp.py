"""Train the grasp model (touch.models.graspnet) on DexYCB; select on the s1 val subject only.

    python -m touch.train_grasp --config configs/g1_grasp_repvit.yaml

Writes runs/<name>_<timestamp>/: config.yaml, run_info.json (git hash, seed), log.jsonl, best.pt
(best val score = mean of finger-contact F1 and hand-state macro F1), last.pt, eval_val.json.
The test subjects (s1 "test") are never loaded here.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from touch.data.grasp_store import (
    PVDB_PROC,
    PVDB_VAL_PROC,
    GraspTrack,
    GraspWindowDataset,
    StoreReader,
    grasp_index,
    grasp_weights,
    load_egop_tracks,
    load_grasp_tracks,
    load_hocap_tracks,
    load_own_tracks,
    to_input,
)
from touch.data.patches import LM_FEAT_DIMS
from touch.eval.metrics import interval_timing
from touch.models.graspnet import GraspNet, count_params, init_from_causal, init_from_touchnet
from touch.train import _tolerant, git_hash, pick_device, seed_everything


@dataclass
class GraspConfig:
    name: str = "g1_grasp"
    seed: int = 0
    tip_backbone: str = "mobilenetv3_large_100.ra_in1k"
    hand_backbone: str = "repvit_m1_1.dist_450e_in1k"
    pretrained: bool = True
    init_touchnet: str = ""  # OneTouch checkpoint for the fingertip branch
    bidir: bool = False  # OFFLINE model (sees future frames): for labelling recorded demos only
    init_causal: str = ""  # bidir: start from this trained causal grasp checkpoint
    init_grasp: str = ""  # start from this trained grasp checkpoint (same architecture)
    hold_rule: str = "thumb"  # state label: "thumb" (original) | "any2" (dexycb.labels_from_dist)
    egop_frac: float = 0.0  # share of sampled windows from EgoPressure surface-no-object tracks
    pvdb_frac: float = 0.0  # share from PVDB overhead tabletop tracks (state none + finger labels)
    surface_val_w: float = 0.0  # selection = (1-w) * DexYCB val score + w * PVDB val-fold score
    hocap_frac: float = 0.0  # share of windows from HO-Cap train subjects (1-4, 6, 7)
    hocap_val_w: float = 0.0  # selection weight of the HO-Cap val subject (9) score
    hocap_hold_rule: str = "any2"  # "air": HO-Cap hold only while the object is off the table
    hocap_cam_boost: str = ""  # "cam:factor,..." e.g. "105322251225:3": oversample a camera
    own_frac: float = 0.0  # share of windows from our own key-labelled recordings (state only)
    own_val_w: float = 0.0  # selection weight of our own val session(s) (evaluate_state)
    own_touch_frac: float = 0.0  # share of windows from own TOUCH sessions (fingertip on the
    #   table, per-finger labels); own_frac is then the share of own grasp sessions
    own_train: str = ""  # comma list: only these own train sessions (k-fold); "" = all
    own_eval: str = ""  # comma list of own sessions (any split) scored after every epoch (logged)
    quick_val: bool = False  # skip the DexYCB / surface / HO-Cap validation (~40 min / epoch):
    #   own_eval only, best.pt = last epoch (k-fold runs: nothing is selected on held-out data)
    aug: str = "mild"  # "webcam": + white balance / gamma / blur / low-res / noise / JPEG
    table: bool = False  # "hand touches the table" head (scripts/make_table_labels.py labels)
    w_table: float = 0.5
    T: int = 8
    emb: int = 128
    lm_emb: int = 64
    hidden: int = 128
    hand_emb: int = 256
    hand_hidden: int = 256
    img_drop: float = 0.2
    hand_drop: float = 0.1
    mesh: bool = True
    grad_frames: int = 2  # encoder gradients from the newest k frames of each window (0 = all)
    epochs: int = 6
    samples_per_epoch: int = 24000
    batch_size: int = 16
    lr: float = 3e-4
    tip_lr_mult: float = 0.3  # the pretrained fingertip branch moves slower
    weight_decay: float = 1e-4
    warmup_steps: int = 300
    num_workers: int = 6
    max_minutes: float = 480.0
    w_finger: float = 1.0
    w_state: float = 1.0
    w_vert: float = 1.0
    w_mesh: float = 0.5
    vert_pos_w: float = 5.0
    pos_w_max: float = 20.0


def load_grasp_config(path_or_dict) -> dict:
    d = (
        path_or_dict
        if isinstance(path_or_dict, dict)
        else yaml.safe_load(Path(path_or_dict).read_text())
    )
    unknown = set(d) - {f.name for f in fields(GraspConfig)}
    assert not unknown, f"unknown config keys: {unknown}"
    return asdict(GraspConfig(**d))


def build_grasp_model(cfg: dict) -> GraspNet:
    return GraspNet(
        cfg["tip_backbone"], cfg["hand_backbone"], cfg["pretrained"], LM_FEAT_DIMS["v2"],
        emb=cfg["emb"], lm_emb=cfg["lm_emb"], hidden=cfg["hidden"], hand_emb=cfg["hand_emb"],
        hand_hidden=cfg["hand_hidden"], img_drop=cfg["img_drop"], hand_drop=cfg["hand_drop"],
        mesh=cfg["mesh"], grad_frames=cfg["grad_frames"], bidir=cfg["bidir"],
        table=cfg.get("table", False),
    )  # fmt: skip


def masked_bce(logit, target, mask, pos_w) -> torch.Tensor:
    if mask.sum() == 0:
        return logit.sum() * 0
    return nn.functional.binary_cross_entropy_with_logits(
        logit[mask], target[mask], pos_weight=torch.as_tensor(pos_w, device=logit.device)
    )


def losses(out: dict, b: dict, cfg: dict) -> dict[str, torch.Tensor]:
    fm = b["finger_mask"]
    pos = (b["finger"][fm] > 0.5).sum().clamp(min=1)
    pw = ((fm.sum() - pos) / pos).clamp(1.0, cfg["pos_w_max"])
    L = {
        "finger": masked_bce(out["finger"], b["finger"], fm, pw),
        "state": nn.functional.cross_entropy(
            out["state"].flatten(0, 1), b["state"].flatten(), ignore_index=-1
        )
        if (b["state"] >= 0).any()
        else out["state"].sum() * 0,
        "vert": masked_bce(out["vert"], b["vlab"], b["vlab_mask"], cfg["vert_pos_w"]),
    }
    if "table" in out:
        tm = b["table_mask"]
        tpos = (b["table"][tm] > 0.5).sum().clamp(min=1)
        tpw = ((tm.sum() - tpos) / tpos).clamp(1.0, cfg["pos_w_max"])
        L["table"] = masked_bce(out["table"], b["table"], tm, tpw)
    if "verts" in out:
        mm = b["mesh_mask"]
        if mm.any():
            dv = (out["verts"][mm] - b["verts"][mm]).abs().mean()
            dj = (out["joints"][mm] - b["joints"][mm]).abs().mean()
            L["mesh"] = 100.0 * (dv + dj) + 1e-3 * (out["betas"][mm] ** 2).mean()  # cm
        else:
            L["mesh"] = out["verts"].sum() * 0
    return L


# ---------------------------------------------------------------- validation (whole tracks)


@torch.no_grad()
def predict_track(model: GraspNet, tr, mesh_cache, T: int, device, chunk: int = 64) -> dict:
    """Score every frame t of a track with its window (from zero GRU state), like training:
    t-T+1..t (causal) or centred on t (bidir). Frame features are computed once, then each
    window re-runs the GRUs. Frames before the track start are pad frames (black images, no
    hand); frames after its end (bidir only) have no hand and hold the last images."""
    n, reader = len(tr), StoreReader()
    tgt = model.window_target(T)
    pad, post = tgt, T - 1 - tgt
    N = n + pad + post
    tips = np.zeros((N, 5, model.tip_size, model.tip_size, 3), np.uint8)
    hand = np.zeros((N, model.hand_size, model.hand_size, 3), np.uint8)
    lm = np.zeros((N, tr.lm_feat.shape[1]), np.float32)
    present = np.zeros(N, bool)
    last = None
    for i in range(n + post):
        if i >= n:
            if last is not None:
                tips[pad + i], hand[pad + i] = np.stack(last[:5]), last[5]
            continue
        if tr.present[i]:
            last = [reader.get(int(tr.offset[i, s]), int(tr.length[i, s]), tr.store)
                    for s in range(6)]  # fmt: skip
            present[pad + i], lm[pad + i] = True, tr.lm_feat[i]
        if last is not None:
            tips[pad + i], hand[pad + i] = np.stack(last[:5]), last[5]
    feats = []
    for a in range(0, N, chunk):
        sl = slice(a, a + chunk)
        enc = model.encode(
            to_input(torch.from_numpy(tips[sl]).to(device))[None],
            to_input(torch.from_numpy(hand[sl]).to(device))[None],
            torch.from_numpy(lm[sl])[None].to(device),
            torch.from_numpy(present[sl])[None].to(device),
        )
        feats.append({k: v[0] for k, v in enc.items()})
    full = {k: torch.cat([f[k] for f in feats]) for k in feats[0]}
    idx = torch.arange(n, device=device)[:, None] + torch.arange(T, device=device)[None]  # [n, T]
    win = {k: v[idx] for k, v in full.items()}
    out = model.temporal(win)
    res = {
        "finger": torch.sigmoid(out["finger"][:, tgt]).cpu().numpy(),
        "state": torch.softmax(out["state"][:, tgt], -1).cpu().numpy(),
        "vert": torch.sigmoid(out["vert"][:, tgt]).cpu().numpy(),
    }
    if "table" in out:
        res["table"] = torch.sigmoid(out["table"][:, tgt]).cpu().numpy()
    if model.use_mesh:
        m = model.mesh(full["hand_feat"][pad : pad + n], full["lm"][pad : pad + n])
        res["verts"] = m["verts"].cpu().numpy()
        res["joints"] = m["joints"].cpu().numpy()
    return res


def _f1(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = int((pred & gt).sum())
    fp, fn = int((pred & ~gt).sum()), int((~pred & gt).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {
        "f1": round(2 * p * r / max(p + r, 1e-9), 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], mask, [False]])
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[::2], d[1::2], strict=True))


def evaluate_surface(model, tracks, T: int, device, thr: float = 0.5) -> dict:
    """Held-out "hand on a surface, no object" tracks (PVDB val fold, overhead view): how often
    the state head claims contact (any state > 0 is wrong: there is no object), and fingertip
    contact F1 against the pressure pad."""
    model.eval()
    false, F_p, F_g, T_p, T_g = [], [], [], [], []
    for tr in tracks:
        r = predict_track(model, tr, None, T, device)
        if "table" in r:
            tab = tr.table_or_unknown()
            tm = (tab >= 0) & tr.present
            T_p.append(r["table"][tm] >= thr)
            T_g.append(tab[tm] == 1)
        ps = r["state"].argmax(1)[tr.present]
        false.append(ps > 0)
        pf = (r["finger"] >= thr) & tr.present[:, None]
        fm = tr.finger >= 0
        F_p.append(pf[fm])
        F_g.append(tr.finger[fm] == 1)
    fr = float(np.concatenate(false).mean())
    fin = _f1(np.concatenate(F_p), np.concatenate(F_g))
    out = {"state_false_contact": round(fr, 4), "finger": fin,
           "score": round((fin["f1"] + 1 - fr) / 2, 4)}  # fmt: skip
    if T_p:
        out["table"] = _f1(np.concatenate(T_p), np.concatenate(T_g))
    return out


def evaluate_state(model, tracks, T: int, device, thr: float = 0.5) -> dict:
    """Our own recordings: per-state F1 on labelled frames (no hand = predicted none), hold
    timing, score = mean of none F1 and hold F1 (touch is rare there). Touch sessions (per-finger
    labels) add fingertip contact F1 + tap timing ("finger", "finger_timing"); with no hold in
    the labels the score is then the mean of none F1 and finger F1."""
    model.eval()
    S_p, S_g, timing, F_p, F_g, timing_f = [], [], [], [], [], []
    for tr in tracks:
        r = predict_track(model, tr, None, T, device)
        ps = np.where(tr.present, r["state"].argmax(1), 0)
        sm = tr.state >= 0
        S_p.append(ps[sm])
        S_g.append(tr.state[sm])
        for a, e in _runs(sm):
            timing.append(interval_timing(ps[a:e] == 2, tr.state[a:e] == 2))
        fm = tr.finger >= 0
        if fm.any():
            pf = (r["finger"] >= thr) & tr.present[:, None]
            F_p.append(pf[fm])
            F_g.append(tr.finger[fm] == 1)
            for f in range(5):
                for a, e in _runs(fm[:, f]):
                    timing_f.append(interval_timing(pf[a:e, f], tr.finger[a:e, f] == 1))
    S_p, S_g = np.concatenate(S_p), np.concatenate(S_g)
    per = {name: _f1(S_p == c, S_g == c) for c, name in enumerate(["none", "touch", "hold"])}
    out = {"state": per, "hold_timing": _timing(timing),
           "score": round((per["none"]["f1"] + per["hold"]["f1"]) / 2, 4)}  # fmt: skip
    if F_p:
        out["finger"] = _f1(np.concatenate(F_p), np.concatenate(F_g))
        out["finger_timing"] = _timing(timing_f)
        if not (S_g == 2).any():
            out["score"] = round((per["none"]["f1"] + out["finger"]["f1"]) / 2, 4)
    return out


def touch_weights(tracks: list[GraspTrack], index: np.ndarray) -> np.ndarray:
    """Own touch sessions: windows with the labelled fingertip in contact and without it get
    equal total weight (taps are rare), x3 within +-3 frames of a contact change (grasp_weights
    does the x3 near label changes; its state balancing is a no-op here, all state none)."""
    w = grasp_weights(tracks, index)
    c = np.array([(tracks[k].finger[t] == 1).any() for k, t in index])
    for v in (True, False):
        if (c == v).any():
            w[c == v] *= 0.5 / w[c == v].sum()
    return w / w.mean()


def with_surface(ev: dict, model, surf, cfg: dict, device, hval=None, oval=None) -> dict:
    """Add the PVDB val-fold surface metrics, the HO-Cap val-subject metrics and the selection
    score = (1 - ws - wh) * DexYCB val + ws * PVDB surface + wh * HO-Cap val."""
    parts = []
    if surf:
        ev["surface"] = evaluate_surface(model, surf, cfg["T"], device)
        parts.append((cfg["surface_val_w"], ev["surface"]["score"]))
    if hval:
        tracks, cache = hval
        ev["hocap_val"] = evaluate(model, tracks, cache, cfg["T"], device)
        parts.append((cfg["hocap_val_w"], ev["hocap_val"]["score"]))
    if oval:
        ev["own_val"] = evaluate_state(model, oval, cfg["T"], device)
        parts.append((cfg["own_val_w"], ev["own_val"]["score"]))
    w0 = 1 - sum(w for w, _ in parts)
    ev["select"] = round(w0 * ev["score"] + sum(w * x for w, x in parts), 4)
    return ev


def evaluate(model, tracks, mesh_cache, T: int, device, thr: float = 0.5, part_dist=None) -> dict:
    """Per-frame scores on validation. Frames without a MediaPipe hand count as predicted
    "no contact / none" (a miss), so tracking failures are part of the score."""
    model.eval()
    F_p, F_g, S_p, S_g, V_p, V_g, mpjpe, mpvpe, hover = [], [], [], [], [], [], [], [], []
    timing_f, timing_hold, T_p, T_g = [], [], [], []
    for tr in tracks:
        r = predict_track(model, tr, mesh_cache, T, device)
        pf = (r["finger"] >= thr) & tr.present[:, None]
        ps = np.where(tr.present, r["state"].argmax(1), 0)
        fm = tr.finger >= 0
        F_p.append(pf[fm])
        F_g.append(tr.finger[fm] == 1)
        sm = tr.state >= 0
        S_p.append(ps[sm])
        S_g.append(tr.state[sm])
        if "table" in r:
            tab = tr.table_or_unknown()
            tm = (tab >= 0) & tr.present
            T_p.append(r["table"][tm] >= thr)
            T_g.append(tab[tm] == 1)
        vl = np.asarray(mesh_cache[1][tr.start : tr.start + len(tr)])
        vm = (vl >= 0) & tr.present[:, None]
        V_p.append((r["vert"] >= thr)[vm])
        V_g.append(vl[vm] == 1)
        for a, e in _runs(sm):
            timing_hold.append(interval_timing(ps[a:e] == 2, tr.state[a:e] == 2))
            for f in range(5):
                timing_f.append(interval_timing(pf[a:e, f], tr.finger[a:e, f] == 1))
        if "verts" in r:
            ok = tr.present & tr.valid
            g = np.asarray(mesh_cache[0][tr.start : tr.start + len(tr)], np.float32)
            mpvpe.append(np.linalg.norm(r["verts"][ok] - g[ok, :778], axis=-1).mean(1) * 1000)
            mpjpe.append(np.linalg.norm(r["joints"][ok] - g[ok, 778:], axis=-1).mean(1) * 1000)
        if part_dist is not None:  # near-object hover: nothing within 10 mm, something within 30
            d = part_dist[(tr.subject, tr.seq)] * 1000
            near = (d.min(1) <= 30) & (d.min(1) > 10)
            if near.any():
                hover.append(pf[near].any(1))
    S_p, S_g = np.concatenate(S_p), np.concatenate(S_g)
    per_state = {name: _f1(S_p == c, S_g == c) for c, name in enumerate(["none", "touch", "hold"])}
    out = {
        "finger": _f1(np.concatenate(F_p), np.concatenate(F_g)),
        "state": {**per_state,
                  "macro_f1": round(float(np.mean([v["f1"] for v in per_state.values()])), 4),
                  "accuracy": round(float((S_p == S_g).mean()), 4)},
        "vert": _f1(np.concatenate(V_p), np.concatenate(V_g)),
        "finger_timing": _timing(timing_f),
        "hold_timing": _timing(timing_hold),
        "present_rate": round(float(np.mean(np.concatenate([t.present for t in tracks]))), 4),
    }  # fmt: skip
    if mpjpe:
        out["mesh"] = {"mpjpe_mm": round(float(np.concatenate(mpjpe).mean()), 2),
                       "mpvpe_mm": round(float(np.concatenate(mpvpe).mean()), 2)}  # fmt: skip
    if hover:
        h = np.concatenate(hover)
        out["near_hover"] = {"frames": int(len(h)), "false_contact_rate": round(float(h.mean()), 4)}
    if T_p:  # reported, not part of the selection score
        out["table"] = _f1(np.concatenate(T_p), np.concatenate(T_g))
    out["score"] = round((out["finger"]["f1"] + out["state"]["macro_f1"]) / 2, 4)
    return out


def _timing(res: list[dict]) -> dict:
    n_gt = sum(r["n_gt"] for r in res)
    out = {"n_gt": n_gt, "matched": sum(r["matched"] for r in res)}
    for b in ("onset", "release"):
        err = np.concatenate([r[b]["errors_frames"] for r in res] or [np.zeros(0)]).astype(float)
        out[b] = {
            f"sr@{k}": round(float((np.abs(err) <= k).sum()) / max(n_gt, 1), 3) for k in (3, 5, 10)
        }
        out[b]["mae_frames"] = round(float(np.abs(err).mean()), 2) if len(err) else None
    return out


def load_part_dist(tracks) -> dict:
    from touch.data.grasp_store import PROC

    out = {}
    for tr in tracks:
        key = (tr.subject, tr.seq)
        if key not in out:
            with np.load(PROC / tr.subject / tr.seq / "contact.npz") as c:
                out[key] = c["part_dist"]
    return out


# ---------------------------------------------------------------- training


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="a few steps + a tiny val, to test")
    args = ap.parse_args()
    cfg = load_grasp_config(args.config)
    if args.smoke:
        cfg.update(epochs=1, samples_per_epoch=64, num_workers=0, warmup_steps=2)
    seed_everything(cfg["seed"])
    device = pick_device()
    run = Path("runs") / f"{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    (run / "run_info.json").write_text(json.dumps({"git": git_hash(), "seed": cfg["seed"],
                                                   "device": str(device)}))  # fmt: skip

    train, mesh, vlab = load_grasp_tracks("train", hold_rule=cfg["hold_rule"])
    val, _, _ = load_grasp_tracks("val", hold_rule=cfg["hold_rule"])
    assert train and val and all(t.split == "val" for t in val)
    assert not {t.subject for t in train} & {t.subject for t in val}
    if args.smoke:
        val = val[:2]
    part_dist = load_part_dist(val)
    surf = []
    if cfg["surface_val_w"] > 0:
        surf = load_egop_tracks(PVDB_VAL_PROC, "pvdb_grasp_val", split="val")
        assert surf and not {t.subject for t in surf} & {
            t.subject for t in load_egop_tracks(PVDB_PROC, "pvdb_grasp")
        }
        if args.smoke:
            surf = surf[:2]
    idx = grasp_index(train)
    w = grasp_weights(train, idx)
    # surface, no object: each source gets its share of the total sampling weight
    hval, caches = None, {}
    if cfg["hocap_frac"] > 0 or cfg["hocap_val_w"] > 0:
        hr = cfg.get("hocap_hold_rule", "any2")
        htr, hmesh, hvlab = load_hocap_tracks("train", hold_rule=hr)
        hv, _, _ = load_hocap_tracks("val", hold_rule=hr)
        assert htr and hv and not {t.subject for t in htr} & {t.subject for t in hv}
        caches["hocap_grasp"] = (hmesh, hvlab)
        if cfg["hocap_val_w"] > 0:
            hval = (hv[:2] if args.smoke else hv, (hmesh, hvlab))
        if cfg["hocap_frac"] > 0:  # own state balancing, then its share of the total weight
            hidx = grasp_index(htr) + np.array([len(train), 0])
            hw = grasp_weights(htr, hidx - np.array([len(train), 0]))
            for item in filter(None, cfg.get("hocap_cam_boost", "").split(",")):
                cam, b = item.split(":")
                m = np.array([htr[t].cam.startswith(cam) for t in hidx[:, 0] - len(train)])
                assert m.any(), f"no HO-Cap train tracks from camera {cam}"
                hw[m] *= float(b)
                print(f"HO-Cap camera {cam}: x{b} ({m.mean():.1%} of windows)", flush=True)
            f = cfg["hocap_frac"]
            hw = hw / hw.sum() * f / (1 - f) * w.sum()
            train, idx, w = train + htr, np.concatenate([idx, hidx]), np.concatenate([w, hw])
    oval = None
    if cfg.get("own_val_w", 0) > 0:
        oval = load_own_tracks("val")
        assert oval and not {t.seq for t in oval} & {t.seq for t in load_own_tracks("train")}
    oeval = []
    for name in filter(None, cfg.get("own_eval", "").split(",")):
        found = [t for sp in ("train", "val", "test") for t in load_own_tracks(sp) if t.seq == name]
        assert len(found) == 1, f"own_eval session {name}: {len(found)} found"
        oeval += found
    if cfg.get("own_frac", 0) > 0:  # own state balancing, then its share of the total weight
        otr = load_own_tracks("train")
        keep = set(filter(None, cfg.get("own_train", "").split(",")))
        if keep:
            assert keep <= {t.seq for t in otr}, keep - {t.seq for t in otr}
            otr = [t for t in otr if t.seq in keep]
        assert not {t.seq for t in otr} & {t.seq for t in oeval}, "own_eval session in training"
        assert otr and all(t.split == "train" for t in otr)
        groups = [(otr, cfg["own_frac"], grasp_weights, "own recordings")]
        if cfg["own_touch_frac"] > 0:  # touch sessions (per-finger labels) get their own share
            tt = [t for t in otr if (t.finger >= 0).any()]
            gg = [t for t in otr if not (t.finger >= 0).any()]
            assert tt and gg, "own_touch_frac needs own touch AND grasp sessions in training"
            groups = [(gg, cfg["own_frac"], grasp_weights, "own grasp sessions"),
                      (tt, cfg["own_touch_frac"], touch_weights, "own touch sessions")]  # fmt: skip
        base = w.sum() / (1 - sum(f for _, f, _, _ in groups))  # DexYCB (+ HO-Cap) keep the rest
        for trs, f, wfn, what in groups:
            oidx = grasp_index(trs)
            ow = wfn(trs, oidx)
            ow = ow / ow.sum() * f * base
            oidx = oidx + np.array([len(train), 0])
            train, idx, w = train + trs, np.concatenate([idx, oidx]), np.concatenate([w, ow])
            print(f"{what}: {len(trs)} tracks, {len(oidx)} windows, share {f}", flush=True)
    sources = [(cfg["egop_frac"], load_egop_tracks),
               (cfg["pvdb_frac"], lambda: load_egop_tracks(PVDB_PROC, "pvdb_grasp"))]  # fmt: skip
    extra = [(f, ld) for f, ld in sources if f > 0]
    total = w.sum() / (1 - sum(f for f, _ in extra))  # DexYCB (+ HO-Cap) keep the rest
    for f, ld in extra:
        eg = ld()
        assert eg and all(t.split == "train" for t in eg)
        eidx = grasp_index(eg) + np.array([len(train), 0])
        ew = np.full(len(eidx), f * total / len(eidx))
        train, idx, w = train + eg, np.concatenate([idx, eidx]), np.concatenate([w, ew])
    model = build_grasp_model(cfg)
    ds = GraspWindowDataset(train, mesh, vlab, cfg["T"], idx, augment=True, seed=cfg["seed"],
                            caches=caches, aug=cfg.get("aug", "mild"),
                            target=model.window_target(cfg["T"]))  # fmt: skip
    g = torch.Generator().manual_seed(cfg["seed"])
    sampler = WeightedRandomSampler(w, cfg["samples_per_epoch"], replacement=True, generator=g)
    dl = DataLoader(ds, cfg["batch_size"], sampler=sampler, num_workers=cfg["num_workers"],
                    persistent_workers=cfg["num_workers"] > 0, drop_last=True)  # fmt: skip

    info = {"params_M": round(count_params(model), 2), "train_tracks": len(train),
            "val_tracks": len(val), "train_windows": len(idx)}  # fmt: skip
    if cfg["init_touchnet"]:
        info["init_copied"] = len(init_from_touchnet(model, cfg["init_touchnet"]))
    if cfg["init_causal"]:
        info["init_copied"] = len(init_from_causal(model, cfg["init_causal"]))
    if cfg["init_grasp"]:
        sd = torch.load(cfg["init_grasp"], map_location="cpu", weights_only=False)["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # only a newly added table head may be missing; anything else is an architecture mismatch
        assert not unexpected and all(k.startswith("table_head.") for k in missing), (
            missing,
            unexpected,
        )
        info["init_copied"], info["init_new"] = len(sd), missing
    print(json.dumps(info), flush=True)
    model.to(device)
    tip_params = [
        p for n, p in model.named_parameters() if n.split(".")[0] in ("encoder", "proj", "lm_mlp")
    ]
    tip_ids = {id(p) for p in tip_params}
    rest = [p for p in model.parameters() if id(p) not in tip_ids]
    opt = torch.optim.AdamW(
        [{"params": tip_params, "lr": cfg["lr"] * cfg["tip_lr_mult"]},
         {"params": rest, "lr": cfg["lr"]}],
        weight_decay=cfg["weight_decay"],
    )  # fmt: skip
    base = [gr["lr"] for gr in opt.param_groups]
    total = cfg["epochs"] * (cfg["samples_per_epoch"] // cfg["batch_size"])
    step, best, t0 = 0, -1.0, time.monotonic()
    log = open(run / "log.jsonl", "a")
    if (cfg["init_touchnet"] or cfg["init_causal"] or cfg["init_grasp"]) and not cfg["quick_val"]:
        # epoch -1: the untrained model's finger output equals the OneTouch checkpoint's, so this
        # is the OneTouch baseline on objects in the same harness (state/vertex/mesh: untrained)
        ev = evaluate(model, val, (mesh, vlab), cfg["T"], device, part_dist=part_dist)
        ev = with_surface(ev, model, surf, cfg, device, hval, oval)
        src = "OneTouch fingertip branch"
        if cfg["init_grasp"] or cfg["init_causal"]:
            src = "grasp checkpoint" if cfg["init_grasp"] else "causal grasp model"
        rec = {"epoch": -1, "note": f"init = {src}", "val": ev}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
    for ep in range(cfg["epochs"]):
        model.train()
        agg: dict[str, float] = {}
        for b in _tolerant(dl):
            f = (
                min(1.0, (step + 1) / cfg["warmup_steps"])
                * 0.5
                * (1 + math.cos(math.pi * step / total))
            )
            for gr, lr in zip(opt.param_groups, base, strict=True):
                gr["lr"] = lr * f
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            b["tips"], b["hand"] = to_input(b["tips"]), to_input(b["hand"])
            out = model(b["tips"], b["hand"], b["lm"], b["present"], with_mesh=cfg["mesh"])
            L = losses(out, b, cfg)
            loss = sum(cfg[f"w_{k}"] * v for k, v in L.items())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            step += 1
            for k, v in L.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            if step % 50 == 0:
                msg = " ".join(f"{k} {v / 50:.3f}" for k, v in agg.items())
                print(f"ep {ep} step {step}/{total} {msg} | {(time.monotonic() - t0) / 60:.1f} min",
                      flush=True)  # fmt: skip
                agg = {}
            if (time.monotonic() - t0) / 60 > cfg["max_minutes"]:
                break
        if cfg["quick_val"]:
            ev = {"select": float(step)}  # the last epoch is kept
        else:
            ev = evaluate(model, val, (mesh, vlab), cfg["T"], device, part_dist=part_dist)
            ev = with_surface(ev, model, surf, cfg, device, hval, oval)
        if oeval:
            ev["own_eval"] = {t.seq: evaluate_state(model, [t], cfg["T"], device) for t in oeval}
        mins = round((time.monotonic() - t0) / 60, 1)
        rec = {"epoch": ep, "step": step, "minutes": mins, "val": ev}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        ck = {"model": model.state_dict(), "config": cfg, "epoch": ep, "val": ev}
        torch.save(ck, run / "last.pt")
        if ev["select"] > best:
            best = ev["select"]
            torch.save(ck, run / "best.pt")
            (run / "eval_val.json").write_text(json.dumps(ev, indent=1))
        if (time.monotonic() - t0) / 60 > cfg["max_minutes"]:
            print("time budget reached", flush=True)
            break
    print("best val score", best, "->", run / "best.pt", flush=True)


if __name__ == "__main__":
    main()
