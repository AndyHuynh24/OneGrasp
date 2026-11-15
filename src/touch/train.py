"""Train a TouchNet on EgoPressure fingertip patches; select on validation only.

Usage:
    python -m touch.train --config configs/c4_mnv3_gru.yaml
Writes runs/<name>_<timestamp>/: config.yaml, run_info.json (git hash, seed), log.jsonl,
best.pt (best val touch-down F1 +-2 frames; post-processing tuned on val), last.pt, eval_val.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from touch.data.catalog import load_catalog
from touch.data.patch_store import IN_CHANS, WindowDataset, all_index, load_tracks, sample_weights
from touch.data.patches import LM_FEAT_DIMS
from touch.eval.run import evaluate, predict, tune_postprocessing
from touch.models.touchnet import TouchNet


@dataclass
class Config:
    name: str = "c4_mnv3_gru"
    seed: int = 0
    backbone: str = "mobilenetv3_small_100.lamb_in1k"
    pretrained: bool = True
    temporal: str = "gru"  # gru | none | bigru (offline only, non-causal)
    use_landmarks: bool = True
    use_image: bool = True
    input_repr: str = "rgb"  # rgb | ndiff | rgb+ndiff
    T: int = 8
    views: list[str] = field(default_factory=lambda: ["ego", "overhead"])
    train_sources: list[str] = field(default_factory=lambda: ["egopressure"])
    epochs: int = 12
    samples_per_epoch: int = 60000
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 300
    loss_steps: str = "all"  # all | last
    num_workers: int = 6
    max_minutes: float = 240.0
    init_ckpt: str = ""  # optional warm start (e.g. PVDB-pretrained)
    proj_aug: float = 0.0  # probability of simulated projector light per window
    lm_feats: str = "v1"  # v1 | v2 (touch.data.patches.landmark_features)
    lm_emb: int = 32  # landmark branch width
    img_drop: float = 0.0  # training-only image-branch dropout per window (TouchNet)
    pos_w_max: float = 20.0  # cap of the per-batch BCE positive weight (neg/pos ratio)
    hover_weight: float = 1.0  # sampling weight multiplier for hover-only (no-contact) sequences


def load_config(path_or_dict) -> dict:
    d = (
        path_or_dict
        if isinstance(path_or_dict, dict)
        else yaml.safe_load(Path(path_or_dict).read_text())
    )
    known = {f.name for f in fields(Config)}
    unknown = set(d) - known
    assert not unknown, f"unknown config keys: {unknown}"
    return asdict(Config(**d))


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_hash() -> str:
    """Read HEAD from .git without running git."""
    g = Path(".git")
    try:
        head = (g / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            p = g / ref
            if p.exists():
                return p.read_text().strip()
            for line in (g / "packed-refs").read_text().splitlines():
                if line.endswith(ref):
                    return line.split()[0]
        return head
    except OSError:
        return "unknown"


def build_model(cfg: dict) -> TouchNet:
    return TouchNet(
        backbone=cfg["backbone"],
        pretrained=cfg["pretrained"],
        in_chans=IN_CHANS[cfg["input_repr"]],
        temporal=cfg["temporal"],
        use_landmarks=cfg["use_landmarks"],
        use_image=cfg["use_image"],
        n_lm=LM_FEAT_DIMS[cfg["lm_feats"]],
        lm_emb=cfg["lm_emb"],
        img_drop=cfg["img_drop"],
    )


SHM_TIMEOUT = "Shared memory manager connection has timed out"


def _tolerant(loader, max_skips: int = 20):
    """Iterate the loader, skipping batches lost to macOS's torch_shm_manager timing out under
    heavy machine load (2026-11-08: this killed a 5 h run after epoch 0). Other errors, or more
    than max_skips such batches in one epoch, still raise."""
    it, skips = iter(loader), 0
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except RuntimeError as e:
            if SHM_TIMEOUT not in str(e) or skips >= max_skips:
                raise
            skips += 1
            print(f"skipped a batch ({skips}/{max_skips} this epoch): {SHM_TIMEOUT}", flush=True)


def _worker_init(wid: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2**32)


def train(cfg: dict) -> Path:
    seed_everything(cfg["seed"])
    device = pick_device()
    run_dir = Path("runs") / f"{cfg['name']}_s{cfg['seed']}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    info = {
        "git_hash": git_hash(),
        "start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    train_seqs = [
        s for src in cfg["train_sources"] for s in load_catalog(source=src, split="train")
    ]
    val_seqs = load_catalog(source="egopressure", split="val")
    assert not {s.participant for s in train_seqs} & {s.participant for s in val_seqs}
    assert all(s.split == "train" for s in train_seqs) and all(s.split == "val" for s in val_seqs)
    train_tracks = load_tracks(train_seqs, cfg["views"], lm_feats=cfg["lm_feats"])
    val_tracks = load_tracks(val_seqs, cfg["views"], lm_feats=cfg["lm_feats"])
    index = all_index(train_tracks)
    weights = sample_weights(train_tracks, index, hover_weight=cfg["hover_weight"])
    T = cfg["T"]
    ds = WindowDataset(
        train_tracks,
        T,
        index,
        cfg["input_repr"],
        augment=True,
        seed=cfg["seed"],
        load_images=cfg["use_image"],
        proj_aug=cfg["proj_aug"],
    )
    gen = torch.Generator().manual_seed(cfg["seed"])
    sampler = WeightedRandomSampler(
        weights, cfg["samples_per_epoch"], replacement=True, generator=gen
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        num_workers=cfg["num_workers"],
        persistent_workers=cfg["num_workers"] > 0,
        worker_init_fn=_worker_init,
        drop_last=True,
    )
    print(
        f"train windows {len(index)}, tracks {len(train_tracks)}, val tracks {len(val_tracks)}",
        flush=True,
    )

    model = build_model(cfg).to(device)
    if cfg["init_ckpt"]:
        sd = torch.load(cfg["init_ckpt"], map_location="cpu", weights_only=False)["model"]
        print("warm start:", model.load_state_dict(sd, strict=False), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(loader)
    total = steps_per_epoch * cfg["epochs"]

    def lr_at(step: int) -> float:
        if step < cfg["warmup_steps"]:
            return (step + 1) / cfg["warmup_steps"]
        prog = (step - cfg["warmup_steps"]) / max(1, total - cfg["warmup_steps"])
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    best, step, t_start = -1.0, 0, time.monotonic()  # awake time: stops during sleep
    log = open(run_dir / "log.jsonl", "a")
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, losses = time.time(), []
        for batch in _tolerant(loader):
            img = batch["img"].to(device, non_blocking=True)
            lm = batch["lm"].to(device)
            present = batch["present"].to(device)
            finger = batch["finger"].to(device)
            y = batch["label"].to(device)
            m = batch["label_mask"].to(device)
            if cfg["loss_steps"] == "last":
                m = m.clone()
                m[:, :-1] = False
            logits = model(img, lm, present, finger)
            yv, lv = y[m], logits[m]
            if yv.numel() == 0:
                continue
            n_pos = yv.sum()
            pos_w = ((yv.numel() - n_pos) / n_pos.clamp(min=1)).clamp(1.0, cfg["pos_w_max"])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lv, yv, pos_weight=pos_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(loss.item())
            if step % 200 == 0:
                rate = step / (time.monotonic() - t_start)
                print(
                    f"ep {epoch} step {step}/{total} loss {np.mean(losses[-200:]):.4f} "
                    f"{rate:.1f} it/s",
                    flush=True,
                )
        train_min = (time.time() - t0) / 60
        t1 = time.time()
        probs = predict(model, val_tracks, T, cfg["input_repr"], device)
        if device.type == "mps":  # give back the validation-batch cache (~3 GB) before training
            torch.mps.empty_cache()
        post, _ = tune_postprocessing(val_tracks, probs, "event")
        post_frame, _ = tune_postprocessing(val_tracks, probs, "frame")
        res = evaluate(val_tracks, probs, post)
        res_frame = evaluate(val_tracks, probs, post_frame)
        rec = {
            "epoch": epoch,
            "step": step,
            "train_loss": float(np.mean(losses)),
            "train_min": round(train_min, 2),
            "val_min": round((time.time() - t1) / 60, 2),
            "post": post,
            "post_frame": post_frame,
            "val": res,
            "frame_op": res_frame,
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        a, fa = res["all"], res_frame["all"]
        print(
            f"== ep {epoch} loss {rec['train_loss']:.4f} | val auroc {a['finger_auroc']:.4f} | "
            f"event op {post}: click F1 ±2 {a['click_f1_pm2']:.3f} exact {a['click_f1_exact']:.3f} "
            f"hoverFP/min {a['hover_fp_per_min']:.2f} timing {a['timing_down_mae_ms']:.0f} ms "
            f"acc {a['finger_acc']:.4f} | frame op: bal {fa['finger_bal_acc']:.4f} "
            f"acc {fa['finger_acc']:.4f} hand_acc {fa['hand_acc']:.4f} "
            f"| ego/overhead auroc {res.get('ego', a)['finger_auroc']:.4f}/"
            f"{res.get('overhead', a)['finger_auroc']:.4f} "
            f"| {train_min:.1f}+{rec['val_min']:.1f} min",
            flush=True,
        )
        ck = {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "post": post,
            "post_frame": post_frame,
            "val": res,
        }
        torch.save(ck, run_dir / "last.pt")
        # model selection on validation: touch-down F1 at +-2 frames (the target metric)
        if a["click_f1_pm2"] > best:
            best = a["click_f1_pm2"]
            torch.save(ck, run_dir / "best.pt")
            (run_dir / "eval_val.json").write_text(json.dumps(rec, indent=2))
        if (time.monotonic() - t_start) / 60 > cfg["max_minutes"]:
            print("time budget reached", flush=True)
            break
    log.close()
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value (YAML values)")
    args = ap.parse_args()
    d = yaml.safe_load(Path(args.config).read_text())
    for kv in args.set:
        k, v = kv.split("=", 1)
        d[k] = yaml.safe_load(v)
    if args.seed is not None:
        d["seed"] = args.seed
    cfg = load_config(d)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print(train(cfg))


if __name__ == "__main__":
    main()
