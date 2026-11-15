"""Fixtures that write tiny fake EgoPressure / PVDB sequences (no real data needed)."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from touch.data.egopressure import PAD_ORIGIN_XY_MM

N_FRAMES = 6
TIP_RC = (50.0, 90.0)  # where the fake index fingertip sits on the pad (row, col)


def _pad_rc_to_world_m(row: float, col: float, z: float = -0.005) -> list[float]:
    return [(col * 1.25 + PAD_ORIGIN_XY_MM[0]) / 1000, (row * 1.25 + PAD_ORIGIN_XY_MM[1]) / 1000, z]


def _fake_joints() -> np.ndarray:
    """21 joints over the pad; index DIP->tip along +row ends at TIP_RC, other tips far away."""
    J = np.zeros((21, 3), np.float32)
    J[:] = _pad_rc_to_world_m(80, 90, -0.04)  # wrist-ish default, 4 cm above the pad
    far = {4: (60, 20), 12: (10, 150), 16: (10, 170), 20: (10, 180)}
    for tip, (r, c) in far.items():
        J[tip] = _pad_rc_to_world_m(r, c)
        J[tip - 1] = _pad_rc_to_world_m(r + 10, c)
    J[8] = _pad_rc_to_world_m(*TIP_RC)
    J[7] = _pad_rc_to_world_m(TIP_RC[0] + 12, TIP_RC[1])
    return J


def _blob(value: float) -> np.ndarray:
    F = np.zeros((105, 185), np.float32)
    r0, c0 = int(TIP_RC[0] + 4), int(TIP_RC[1])  # contact sits behind the tip, toward the DIP
    F[r0 - 3 : r0 + 4, c0 - 3 : c0 + 4] = value
    return F


def _calib(fx: float, w: int, h: int, with_extrinsics: bool) -> dict:
    c = {"fx": fx, "fy": fx, "cx": w / 2, "cy": h / 2, "ImageSizeX": w, "ImageSizeY": h}
    c.update({k: 0.0 for k in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "p3", "p4")})
    if with_extrinsics:  # camera 0.5 m on the hand side (-z) looking at the pad (+z)
        c["ModelViewMatrix"] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 500.0], [0, 0, 0, 1]]
    return c


# Pressure per frame: none, press (3 frames), none, then a tiny sub-floor reading.
PRESSURE_VALUES = [0.0, 5.0, 8.0, 3.0, 0.0, 0.3]


@pytest.fixture
def fake_ep_sequence(tmp_path):
    name = "p_007_index_press_high_x5_right"
    d = tmp_path / "egopressure" / "raw" / name
    d.mkdir(parents=True)
    J = _fake_joints()
    for i, v in enumerate(PRESSURE_VALUES, start=1):
        _blob(v).tofile(d / f"{i:06d}.force.bin")
        anno = {
            "ego_camera_pose": {"R": np.eye(3).tolist(), "T": [0.0, 0.0, 0.5]},
            "hand_side": "right",
            "has_annotation": i != 1,
        }
        if i != 1:
            anno["joint_position"] = J.copy()
        with open(d / f"{i:06d}.anno.pkl", "wb") as f:
            pickle.dump(anno, f, protocol=4)
    cfg = {
        "num_cameras": 8,
        "camera_calibrations": {
            "0": _calib(900, 1920, 1080, False),
            "4": _calib(1800, 2560, 1440, True),
        },
        "participant": {"id": "007", "hand_side": "right"},
    }
    (d / f"{name}.json").write_text(json.dumps(cfg))
    (d / ".done").write_text("fake\n")
    return d


@pytest.fixture
def fake_pv_sequence(tmp_path):
    d = tmp_path / "pvdb" / "raw" / "val_fold_5" / "9008_beryl_cabric" / "index_press_high_x5_right"
    (d / "force").mkdir(parents=True)
    (d / "camera_1").mkdir()
    for i, v in enumerate(PRESSURE_VALUES):
        counts = _blob(v) / 0.3686635944700461  # kPa -> Sensel counts (inverse of PV formula)
        with open(d / "force" / f"{i:05d}.pkl", "wb") as f:
            pickle.dump(counts.astype(np.float32), f, protocol=4)
    # Camera 85 cm above the pad (world y = height), looking straight down.
    cam = _calib(2190, 1920, 1080, False)
    cam["ModelViewMatrix"] = [[1, 0, 0, -80.0], [0, 0, -1, 850.0], [0, 1, 0, -150.0], [0, 0, 0, 1]]
    meta = {
        "num_cameras": 4,
        "num_frames": len(PRESSURE_VALUES),
        "camera_calibrations": {"1": cam},
        "participant": {"Active Hand": "right"},
        "timesteps": [100.0 + i / 15 for i in range(len(PRESSURE_VALUES))],
    }
    (d / "metadata.json").write_text(json.dumps(meta))
    (d / ".done").write_text("fake\n")
    return d
