import json
import os
import pickle

import numpy as np
import pytest
import yaml

from tests.conftest import N_FRAMES, TIP_RC
from touch.data import canonical, egopressure, pvdb
from touch.data.catalog import load_catalog
from touch.data.io import load_numpy_pickle


def test_numpy_only_unpickler_refuses_other_globals(tmp_path):
    p = tmp_path / "evil.pkl"
    p.write_bytes(pickle.dumps(os.getcwd))  # a non-numpy global
    with pytest.raises(pickle.UnpicklingError):
        load_numpy_pickle(p)


def test_build_egopressure_shapes_and_labels(fake_ep_sequence, tmp_path):
    out = canonical.build_egopressure(fake_ep_sequence, tmp_path / "processed")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["participant"] == "ep_007" and meta["family"] == "index_press"
    assert meta["n_frames"] == N_FRAMES and set(meta["views"]) == {"ego", "overhead"}

    kpa = np.load(out / "pressure.npz")["kpa"]
    assert kpa.shape == (N_FRAMES, 105, 185) and kpa.dtype == np.float16

    L = np.load(out / "labels.npz")
    assert L["contact"].tolist() == [False, True, True, True, False, False]  # 0.3 is below floor
    assert L["t"][1] == pytest.approx(1 / 30)
    assert L["joints_m"].shape == (N_FRAMES, 21, 3)
    assert L["has_anno"].tolist() == [False] + [True] * (N_FRAMES - 1)
    for view in ("ego", "overhead"):
        g = L[f"gt2d_{view}"]
        assert g.shape == (N_FRAMES, 21, 2)
        assert np.isnan(g[0]).all() and np.isfinite(g[1:]).all()

    F = np.load(out / "fingers.npz")
    assert F["max_kpa"].shape == (N_FRAMES, 5) and F["contact"].shape == (N_FRAMES, 5)
    assert F["valid"].tolist() == L["has_anno"].tolist()
    assert F["contact"][:, 1].tolist() == [False, True, True, True, False, False]
    assert not F["contact"][:, [0, 2, 3, 4]].any()
    assert F["tip_height_mm"].shape == (N_FRAMES, 5)
    assert F["tip_height_mm"][1, 1] == pytest.approx(5.0, abs=1e-3)  # fake index tip z = -5 mm


def test_world_to_pad_roundtrip():
    x0, y0 = egopressure.PAD_ORIGIN_XY_MM
    world = np.array([[(90 * 1.25 + x0) / 1000, (50 * 1.25 + y0) / 1000, 0.0]])
    rc = egopressure.world_to_pad_rc(world)
    assert rc[0] == pytest.approx([50.0, 90.0])
    assert egopressure.pad_rc_to_world_m(rc)[0, :2] == pytest.approx(world[0, :2])


def test_project_marks_points_behind_camera_nan():
    K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], float)
    uv = egopressure.project(
        np.array([[0, 0, 1.0], [0, 0, -1.0]]), np.eye(3), np.zeros(3), K, np.zeros(8)
    )
    assert uv[0] == pytest.approx([500, 500]) and np.isnan(uv[1]).all()


def test_build_pvdb_converts_counts_to_kpa(fake_pv_sequence, tmp_path):
    out = canonical.build_pvdb(fake_pv_sequence, tmp_path / "processed")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["participant"] == "pv_9008" and meta["pvdb_split_folder"] == "val_fold_5"
    kpa = np.load(out / "pressure.npz")["kpa"].astype(np.float32)
    assert kpa.shape == (N_FRAMES, 105, 185)
    assert kpa.max() == pytest.approx(8.0, rel=1e-3)
    L = np.load(out / "labels.npz")
    assert L["contact"].tolist() == [False, True, True, True, False, False]
    assert L["t"][-1] == pytest.approx((N_FRAMES - 1) / 15)


def test_pvdb_pixels_to_pad_roundtrip(fake_pv_sequence):
    meta = pvdb.load_metadata(fake_pv_sequence)
    H = pvdb.sensel_homography(meta, "overhead")
    colrow = np.array([[90.0, 50.0, 1.0]])
    uv = (H @ colrow.T).T
    uv = uv[:, :2] / uv[:, 2:]
    rc = pvdb.pixels_to_pad_rc(uv, meta, "overhead")
    assert rc[0] == pytest.approx([50.0, 90.0], abs=1e-3)


def test_pvdb_fingers_from_landmarks(fake_pv_sequence, tmp_path):
    out = canonical.build_pvdb(fake_pv_sequence, tmp_path / "processed")
    meta = pvdb.load_metadata(fake_pv_sequence)
    H = pvdb.sensel_homography(meta, "overhead")

    def to_px(rc):
        v = H @ np.array([rc[1], rc[0], 1.0])
        return v[:2] / v[2]

    lm = np.full((N_FRAMES, 2, 21, 3), np.nan, np.float32)
    hand = np.zeros((21, 3), np.float32)
    for j in range(21):
        hand[j, :2] = to_px((95, 20 + j))  # park every joint far away first
    hand[8, :2], hand[7, :2] = to_px(TIP_RC), to_px((TIP_RC[0] + 12, TIP_RC[1]))
    lm[:, 0] = hand
    np.savez(out / "landmarks_overhead.npz", lm_px=lm, primary=np.zeros(N_FRAMES, np.int8))
    canonical.build_pvdb_fingers(out)
    F = np.load(out / "fingers.npz")
    assert F["contact"][:, 1].tolist() == [False, True, True, True, False, False]


def test_catalog_assigns_splits(fake_ep_sequence, fake_pv_sequence, tmp_path):
    root = tmp_path / "processed"
    canonical.build_egopressure(fake_ep_sequence, root)
    canonical.build_pvdb(fake_pv_sequence, root)
    splits = tmp_path / "splits.yaml"
    splits.write_text(
        yaml.safe_dump({"egopressure": {"train": ["ep_007"]}, "pvdb": {"val": ["pv_9008"]}})
    )
    seqs = load_catalog(root, splits)
    assert {(s.source, s.split) for s in seqs} == {("egopressure", "train"), ("pvdb", "val")}
    ep_seq = load_catalog(root, splits, source="egopressure")[0]
    assert len(ep_seq) == N_FRAMES
    assert ep_seq.image_path("ego", 0).name == "000001.cam-d_rgb.jpeg"
    assert ep_seq.pressure().shape == (N_FRAMES, 105, 185)


def test_build_egopressure_overhead_only_download(fake_ep_sequence, tmp_path):
    """VIEWS="4" downloads have only cam-4 images: meta lists only the overhead view."""
    (fake_ep_sequence / "000001.cam-4_rgb.jpeg").write_bytes(b"")
    out = canonical.build_egopressure(fake_ep_sequence, tmp_path / "processed")
    meta = json.loads((out / "meta.json").read_text())
    assert list(meta["views"]) == ["overhead"]
