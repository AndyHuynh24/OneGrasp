"""Build the canonical per-sequence format shared by every data source.

data/processed/<source>/<participant>/<sequence>/
    meta.json       source, participant, gesture fields, hand, fps, frame ids, views (image path
                    template, size, intrinsics, static extrinsics), raw_dir, split folder (PVDB)
    pressure.npz    kpa [N, 105, 185] float16, PV-convention kPa, uncleaned (raw values)
    labels.npz      frame_id [N], t [N] (s from first frame), max_kpa, area_mm2, total_kpa,
                    centroid_rc [N, 2], contact [N] (bool; floor 0.5 then >= 1.0 kPa)
                    EgoPressure only: has_anno [N], joints_m [N, 21, 3] (world, NaN if missing),
                    gt2d_<view> [N, 21, 2] (GT joints projected into each view, NaN if missing)
    fingers.npz     max_kpa [N, 5], area_mm2 [N, 5], contact [N, 5] (bool), valid [N] (bool),
                    other_max_kpa [N]; meta key "finger_source" says how fingers were located
    landmarks_<view>.npz   MediaPipe output (written by touch.data.landmarks)

Images are not copied: meta.json points at the raw files.
Participant ids are prefixed by source ("ep_001", "pv_9008") so they never collide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from touch.data import egopressure as ep
from touch.data import pvdb as pv
from touch.data.names import parse_sequence_name
from touch.data.pressure import (
    CONTACT_KPA,
    FINGER_DIPS,
    FINGER_TIPS,
    MAX_TIP_HEIGHT_MM,
    NOISE_FLOOR_KPA,
    clean,
    finger_pressure,
    frame_stats,
)

FORMAT_VERSION = 3  # 2: EgoPressure fingers gated by tip height (MAX_TIP_HEIGHT_MM)
# 3: EgoPressure pad origin refit on train participants (egopressure.PAD_ORIGIN_XY_MM)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def _base_meta(
    source: str, participant: str, sequence: str, fps: float, frames: list[int], raw_dir: Path
) -> dict:
    name = parse_sequence_name(sequence)
    return {
        "format_version": FORMAT_VERSION,
        "source": source,
        "participant": participant,
        "sequence": sequence,
        "gesture": name.gesture,
        "family": name.family,
        "variant": name.variant,
        "reps": name.reps,
        "take": name.take,
        "hand": name.hand,
        "fps": fps,
        "n_frames": len(frames),
        "frame_ids": frames,
        "raw_dir": str(raw_dir),
        "pressure_units": "pv_convention_kpa",
        "noise_floor_kpa": NOISE_FLOOR_KPA,
        "contact_kpa": CONTACT_KPA,
    }


def _finger_arrays(
    kpa: np.ndarray,
    tips_rc: np.ndarray,
    dips_rc: np.ndarray,
    valid: np.ndarray,
    tip_height_mm: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """kpa [N,105,185]; tips/dips [N,5,2] pad (row, col); valid [N] -> fingers.npz arrays.

    tip_height_mm [N, 5] (optional, needs 3D joints): fingers higher than MAX_TIP_HEIGHT_MM
    above the pad cannot take pressure.
    """
    n = kpa.shape[0]
    out = {
        "max_kpa": np.full((n, 5), np.nan, np.float32),
        "area_mm2": np.full((n, 5), np.nan, np.float32),
        "other_max_kpa": np.full(n, np.nan, np.float32),
    }
    cleaned = clean(kpa)
    for i in np.flatnonzero(valid):
        eligible = None if tip_height_mm is None else tip_height_mm[i] <= MAX_TIP_HEIGHT_MM
        r = finger_pressure(cleaned[i], tips_rc[i], dips_rc[i], eligible=eligible)
        out["max_kpa"][i] = r["max_kpa"]
        out["area_mm2"][i] = r["area_mm2"]
        out["other_max_kpa"][i] = r["other_max_kpa"]
    out["contact"] = np.nan_to_num(out["max_kpa"], nan=0) >= CONTACT_KPA
    out["valid"] = np.asarray(valid, bool)
    out["tip_height_mm"] = (
        np.full((n, 5), np.nan, np.float32)
        if tip_height_mm is None
        else np.asarray(tip_height_mm, np.float32)
    )
    return out


def build_egopressure(seq_dir: Path, out_root: Path) -> Path:
    seq_dir = Path(seq_dir)
    name = parse_sequence_name(seq_dir.name)
    participant = f"ep_{name.participant:03d}"
    out = Path(out_root) / "egopressure" / participant / seq_dir.name
    out.mkdir(parents=True, exist_ok=True)

    frames = ep.list_frames(seq_dir)
    n = len(frames)
    config = ep.load_config(seq_dir)
    kpa = np.stack([ep.load_force(ep.frame_path(seq_dir, f, "force.bin")) for f in frames])

    views = {}
    intr = {}
    for view in ("ego", "overhead"):
        K, dist, size = ep.camera_intrinsics(config, view)
        intr[view] = (K, dist)
        views[view] = {
            "image": str(seq_dir / f"{{frame:06d}}.cam-{ep.VIEW_FILE_CAM[view]}_rgb.jpeg"),
            "mask": str(seq_dir / f"{{frame:06d}}.cam-{ep.VIEW_FILE_CAM[view]}_mask.png"),
            "width": size[0],
            "height": size[1],
            "K": K.tolist(),
            "dist": dist.tolist(),
        }
    R4, t4 = ep.static_extrinsics_m(config, "overhead")
    views["overhead"]["R"], views["overhead"]["t_m"] = R4.tolist(), t4.tolist()
    views["ego"]["extrinsics"] = "per-frame (labels.npz ego_R, ego_t_m)"

    has_anno = np.zeros(n, bool)
    joints = np.full((n, 21, 3), np.nan, np.float32)
    ego_R = np.full((n, 3, 3), np.nan, np.float32)
    ego_t = np.full((n, 3), np.nan, np.float32)
    gt2d = {v: np.full((n, 21, 2), np.nan, np.float32) for v in views}
    for i, f in enumerate(frames):
        a = ep.load_anno(ep.frame_path(seq_dir, f, "anno.pkl"))
        if "ego_camera_pose" in a:
            R, t = ep.ego_extrinsics_m(a)
            ego_R[i], ego_t[i] = R, t
        if not a.get("has_annotation") or "joint_position" not in a:
            continue
        has_anno[i] = True
        J = np.asarray(a["joint_position"], np.float64)
        joints[i] = J
        gt2d["overhead"][i] = ep.project(J, R4, t4, *intr["overhead"])
        if np.isfinite(ego_R[i]).all():
            gt2d["ego"][i] = ep.project(
                J, ego_R[i].astype(np.float64), ego_t[i].astype(np.float64), *intr["ego"]
            )

    stats = frame_stats(kpa)
    labels = {
        "frame_id": np.asarray(frames, np.int32),
        "t": (np.asarray(frames, np.float64) - frames[0]) / ep.FPS,
        **stats,
        "has_anno": has_anno,
        "joints_m": joints,
        "ego_R": ego_R,
        "ego_t_m": ego_t,
        **{f"gt2d_{v}": a for v, a in gt2d.items()},
    }
    pad_rc = ep.world_to_pad_rc(joints)  # [N, 21, 2]
    tip_height = -joints[:, list(FINGER_TIPS), 2] * 1000.0  # hand is on the -z side of the pad
    fingers = _finger_arrays(
        kpa, pad_rc[:, list(FINGER_TIPS)], pad_rc[:, list(FINGER_DIPS)], has_anno, tip_height
    )

    # overhead-only downloads (VIEWS="4") have no ego images: list only the views on disk
    # (if no view has images, e.g. in tests, keep both)
    on_disk = [v for v in views if any(seq_dir.glob(f"*.cam-{ep.VIEW_FILE_CAM[v]}_rgb.jpeg"))]
    if on_disk:
        views = {v: views[v] for v in on_disk}

    meta = _base_meta("egopressure", participant, seq_dir.name, ep.FPS, frames, seq_dir)
    meta.update(
        {
            "views": views,
            "finger_source": "gt_joints (MANO joints projected onto the pad; tip height <= 12 mm)",
            "participant_info": config.get("participant", {}),
        }
    )
    _write_npz(out / "pressure.npz", kpa=kpa.astype(np.float16))
    _write_npz(out / "labels.npz", **labels)
    _write_npz(out / "fingers.npz", **fingers)
    _write_json(out / "meta.json", meta)
    return out


def build_pvdb(seq_dir: Path, out_root: Path) -> Path:
    """PVDB sequence -> canonical. fingers.npz needs landmarks: see build_pvdb_fingers."""
    seq_dir = Path(seq_dir)
    split, person = seq_dir.parent.parent.name, seq_dir.parent.name
    participant = f"pv_{person.split('_')[0]}"
    out = Path(out_root) / "pvdb" / participant / seq_dir.name
    out.mkdir(parents=True, exist_ok=True)

    meta_raw = pv.load_metadata(seq_dir)
    frames = pv.list_frames(seq_dir)
    kpa = np.stack([pv.load_force_kpa(seq_dir, f) for f in frames])
    ts = np.asarray(meta_raw["timesteps"], np.float64)
    t = ts[frames] - ts[frames[0]] if len(ts) >= len(frames) else np.asarray(frames) / pv.FPS

    views = {}
    for view in pv.VIEW_CAMERA:
        K, dist, R, tv, size = pv.camera_params(meta_raw, view)
        views[view] = {
            "image": str(seq_dir / f"camera_{pv.VIEW_CAMERA[view]}" / "{frame:05d}.jpg"),
            "width": size[0],
            "height": size[1],
            "K": K.tolist(),
            "dist": dist.tolist(),
            "R": R.tolist(),
            "t_m": tv.tolist(),
            "sensel_homography": pv.sensel_homography(meta_raw, view).tolist(),
        }
    labels = {"frame_id": np.asarray(frames, np.int32), "t": t, **frame_stats(kpa)}
    meta = _base_meta("pvdb", participant, seq_dir.name, pv.FPS, frames, seq_dir)
    meta.update(
        {
            "views": views,
            "pvdb_split_folder": split,
            "pvdb_person": person,
            "participant_info": meta_raw.get("participant", {}),
            "finger_source": "pending (needs MediaPipe landmarks on the overhead view)",
        }
    )
    _write_npz(out / "pressure.npz", kpa=kpa.astype(np.float16))
    _write_npz(out / "labels.npz", **labels)
    _write_json(out / "meta.json", meta)
    return out


def build_pvdb_fingers(out_dir: Path, view: str = "overhead") -> None:
    """Per-finger labels for a PVDB sequence from its MediaPipe landmarks (primary hand)."""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    lm = np.load(out_dir / f"landmarks_{view}.npz")
    kpa = np.load(out_dir / "pressure.npz")["kpa"].astype(np.float32)
    raw_meta = pv.load_metadata(Path(meta["raw_dir"]))
    n = kpa.shape[0]
    prim = lm["primary"]
    valid = prim >= 0
    pts = np.zeros((n, 21, 2), np.float64)
    for i in np.flatnonzero(valid):
        pts[i] = lm["lm_px"][i, prim[i], :, :2]
    pad_rc = np.zeros((n, 21, 2))
    if valid.any():
        pad_rc[valid] = pv.pixels_to_pad_rc(pts[valid].reshape(-1, 2), raw_meta, view).reshape(
            -1, 21, 2
        )
    fingers = _finger_arrays(kpa, pad_rc[:, list(FINGER_TIPS)], pad_rc[:, list(FINGER_DIPS)], valid)
    _write_npz(out_dir / "fingers.npz", **fingers)
    meta["finger_source"] = f"mediapipe_{view} (primary hand landmarks mapped to the pad)"
    _write_json(out_dir / "meta.json", meta)
