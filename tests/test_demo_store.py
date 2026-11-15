"""DemoStore round trip on the sqlite backend: an HDF5 demo in, the same arrays out."""

import h5py
import numpy as np

from touch.store.demo_store import STEP_ARRAYS, DemoStore, demo_from_group, local_mjcf

SHAPES = {"actions": 8, "actions_eef": 7, "obs/robot0_joint_pos": 7, "obs/robot0_joint_vel": 7,
          "obs/robot0_gripper_qpos": 2, "obs/robot0_eef_pos": 3, "obs/robot0_eef_quat": 4,
          "obs/cup_pos": 3, "obs/cup_quat": 4, "states": 31}  # fmt: skip


def _demo_file(path, T=12, source="teleop", key="demo_0"):
    rng = np.random.default_rng(0)
    with h5py.File(path, "a") as f:
        g = f.require_group("data").create_group(key)
        for p, n in SHAPES.items():
            g[p] = rng.normal(size=(T, n)).astype(np.float64 if p == "states" else np.float32)
        g["rewards"] = np.r_[np.zeros(T - 1), 1.0].astype(np.float32)
        g["dones"] = (np.arange(T) == T - 1).astype(np.int64)
        g.attrs["model_file"] = '<mujoco><compiler meshdir="/elsewhere/third_party/x/assets"/>'
        g.attrs["num_samples"] = T
        g.attrs["source"] = source
        g.attrs["success"] = True
        g.attrs["take"] = "20261114-050550"
        g.attrs["cup_xy"] = "[0.1, 0.2]"
        g.attrs["place_xy"] = "[0.3, 0.1]"
        if source != "teleop":
            g.attrs["src_demo"] = "demo_0"


def test_round_trip(tmp_path):
    h5 = tmp_path / "demos.hdf5"
    _demo_file(h5)
    _demo_file(h5, source="mimicgen", key="demo_1")
    st = DemoStore(f"sqlite:///{tmp_path / 'store.sqlite'}")
    st.create_schema()
    with h5py.File(h5, "r") as f:
        ids = [st.add_demo(demo_from_group(f["data"][k], "demos", k)) for k in ("demo_0", "demo_1")]
        again = st.add_demo(demo_from_group(f["data"]["demo_0"], "demos", "demo_0"))
    assert ids == ["demos:demo_0", "demos:demo_1"]
    assert again is None  # stored before: skipped
    d = st.load_demo("demos:demo_0")
    with h5py.File(h5, "r") as f:
        g = f["data/demo_0"]
        for col, p in STEP_ARRAYS:
            assert d["arrays"][col].shape == g[p].shape, col
            np.testing.assert_allclose(d["arrays"][col], g[p][()], rtol=1e-6, err_msg=col)
        np.testing.assert_array_equal(d["done"], g["dones"][()].astype(bool))
    assert d["arrays"]["state"].dtype == np.float64
    rows = st.list_demos(source="mimicgen")
    assert [r["demo_id"] for r in rows] == ["demos:demo_1"]
    assert rows[0]["src_demo"] == "demos:demo_0"
    s = st.stats()
    assert (s["demos"], s["teleop"], s["generated"], s["steps"]) == (2, 1, 1, 24)
    summ = st.demo_summary()
    assert len(summ) == 2 and abs(summ[0]["seconds"] - 12 / 30) < 1e-6


def test_local_mjcf_paths():
    xml = '<compiler meshdir="/Users/someone/proj/third_party/mujoco_panda/assets"/>'
    out = local_mjcf(xml)
    assert "/Users/someone" not in out and out.endswith('third_party/mujoco_panda/assets"/>')
