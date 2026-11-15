import cv2
import numpy as np
import torch

from touch.data.dexycb import SUBJECTS
from touch.data.grasp_store import (
    GraspWindowDataset,
    grasp_index,
    grasp_weights,
    load_grasp_tracks,
    to_input,
)

N, F = 6, 207


def make_proc(tmp_path):
    """One val-subject sequence, one camera; the image value encodes the frame index."""
    proc, root = tmp_path / "proc", tmp_path / "patches"
    d = proc / SUBJECTS[6] / "seq0"
    d.mkdir(parents=True)
    root.mkdir()
    present = np.array([True, True, False, True, True, True])
    offset = np.full((N, 6), -1, np.int64)
    length = np.zeros((N, 6), np.int32)
    blob, pos = [], 0
    for i in np.flatnonzero(present):
        for s in range(6):
            size = 160 if s == 5 else 112
            ok, b = cv2.imencode(".png", np.full((size, size, 3), 10 * i + s, np.uint8))
            offset[i, s], length[i, s] = pos, len(b)
            blob.append(b.reshape(-1))
            pos += len(b)
    np.concatenate(blob).tofile(root / "dexycb_store.bin")
    finger = np.zeros((N, 5), np.int8)
    finger[3:, 1] = 1
    finger[0, 2] = -1
    state = np.array([0, 0, 1, 1, 2, -1], np.int8)
    vert = np.zeros((N, 778), np.int8)
    vert[4, :10] = 1
    np.savez(d / "contact.npz", finger=finger, state=state, vert=vert, valid=np.ones(N, bool))
    verts = np.random.default_rng(0).normal(size=(N, 778, 3)).astype(np.float16)
    np.savez(d / "cam.npz", offset=offset, length=length, present=present,
             lm_feat=np.ones((N, F), np.float32), flip=np.bool_(False),
             mp_right=np.ones(N, np.int8), verts=verts,
             joints=np.zeros((N, 21, 3), np.float16))  # fmt: skip
    (d / "done").touch()
    return proc, root, verts


def test_window_shapes_and_causal_hold(tmp_path):
    proc, root, verts = make_proc(tmp_path)
    tracks, mesh, vlab = load_grasp_tracks("val", proc, root)
    assert len(tracks) == 1 and load_grasp_tracks("train", proc, root)[0] == []
    idx = grasp_index(tracks)
    assert idx[:, 1].tolist() == [0, 1, 3, 4, 5]  # frames with a hand and some label
    w = grasp_weights(tracks, idx)
    assert w.shape == (5,) and np.isclose(w.mean(), 1.0)
    ds = GraspWindowDataset(tracks, mesh, vlab, T=4, index=idx, root=root)
    b = ds[3]  # t = 4: window frames 1, 2, 3, 4
    assert b["tips"].shape == (4, 5, 112, 112, 3) and b["tips"].dtype == torch.uint8
    assert b["hand"].shape == (4, 160, 160, 3)
    assert b["lm"].shape == (4, F) and b["present"].tolist() == [True, False, True, True]
    assert b["finger"].shape == (4, 5) and b["vlab"].shape == (4, 778)
    assert b["verts"].shape == (778, 3) and b["joints"].shape == (21, 3)
    # frame 2 has no hand: it repeats frame 1's images (past only), with no labels
    assert b["tips"][1, 0, 0, 0, 0] == 10 and b["hand"][1, 0, 0, 0] == 15
    assert b["tips"][3, 2, 0, 0, 0] == 42
    assert b["state"].tolist() == [0, -1, 1, 2]
    assert not b["finger_mask"][1].any() and b["finger"][2, 1] == 1
    assert b["vlab"][3, :10].all() and b["vlab_mask"][3].all()
    assert np.allclose(b["verts"].numpy(), verts[4].astype(np.float32))
    assert bool(b["mesh_mask"])
    b0 = ds[0]  # t = 0: three pad frames before the start
    assert b0["present"].tolist() == [False, False, False, True]
    assert (b0["tips"][:3] == 0).all() and b0["tips"][3, 0, 0, 0, 0] == 0
    assert not b0["finger_mask"][3, 2] and b0["finger_mask"][3, 1]


def test_to_input():
    x = torch.full((2, 8, 8, 3), 255, dtype=torch.uint8)
    y = to_input(x)
    assert y.shape == (2, 3, 8, 8) and y.dtype == torch.float32
    ref = (1 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    assert torch.allclose(y[0, :, 0, 0], ref, atol=1e-5)


def test_window_target_centre_and_track_end():
    from touch.data.grasp_store import GraspTrack, GraspWindowDataset

    n, F = 6, 4
    tr = GraspTrack(
        subject="s",
        seq="q",
        cam="c",
        split="train",
        offset=np.full((n, 6), -1),
        length=np.zeros((n, 6), int),
        present=np.zeros(n, bool),
        lm_feat=np.arange(n * F, dtype=np.float32).reshape(n, F),
        finger=np.zeros((n, 5), np.int8),
        state=np.arange(n, dtype=np.int8) % 3,
        valid=np.ones(n, bool),
        start=0,
        flip=False,
        mp_right=np.ones(n, np.int8),
    )
    mesh = np.zeros((n, 799, 3), np.float16)
    vlab = np.zeros((n, 778), np.int8)

    class DS(GraspWindowDataset):  # no image store needed: every frame "present" with fake images
        def _imgs(self, tr, i):
            if i < 0 or i >= len(tr):
                return None
            return [np.full((112, 112, 3), i, np.uint8)] * 5 + [np.full((160, 160, 3), i, np.uint8)]

    idx = np.array([[0, 4]])
    tr.present[:] = True
    ds = DS([tr], mesh, vlab, 5, idx, target=2)  # frames 2..6; frame 6 is past the end
    b = ds[0]
    assert b["state"].tolist() == [2, 0, 1, 2, -1]
    assert b["present"].tolist() == [True, True, True, True, False]
    assert b["tips"].shape == (5, 5, 112, 112, 3) and int(b["tips"][4, 0, 0, 0, 0]) == 5
    ds_c = DS([tr], mesh, vlab, 5, idx)  # causal default: frames 0..4
    assert ds_c[0]["state"].tolist() == [0, 1, 2, 0, 1]


def test_egop_track_has_state_none_and_no_mesh(tmp_path):
    import cv2

    from touch.data.grasp_store import GraspWindowDataset, load_egop_tracks

    n, T = 5, 3
    bufs = []
    for i in range(n * 6):
        size = 160 if i % 6 == 5 else 112
        ok, b = cv2.imencode(".jpg", np.full((size, size, 3), i, np.uint8))
        bufs.append(b.reshape(-1))
    lengths = np.array([len(b) for b in bufs]).reshape(n, 6)
    offsets = np.concatenate([[0], np.cumsum(lengths.reshape(-1))[:-1]]).reshape(n, 6)
    (tmp_path / "egop_grasp_store.bin").write_bytes(np.concatenate(bufs).tobytes())
    d = tmp_path / "proc" / "ep_001" / "seq"
    d.mkdir(parents=True)
    present = np.array([1, 1, 0, 1, 1], bool)
    np.savez(d / "overhead.npz", offset=offsets, length=lengths.astype(np.int32), present=present,
             lm_feat=np.zeros((n, 7), np.float32), flip=np.bool_(False),
             mp_right=np.ones(n, np.int8), split=np.str_("train"))  # fmt: skip
    tracks = load_egop_tracks(tmp_path / "proc")
    assert len(tracks) == 1 and tracks[0].start == -1 and tracks[0].store == "egop_grasp"
    assert tracks[0].state.tolist() == [0, 0, -1, 0, 0]
    ds = GraspWindowDataset(tracks, None, None, T, np.array([[0, 4]]), root=tmp_path)
    b = ds[0]
    assert b["tips"].shape == (T, 5, 112, 112, 3) and b["hand"].shape == (T, 160, 160, 3)
    assert b["state"].tolist() == [-1, 0, 0]  # frame 2 has no hand -> ignored
    assert not b["finger_mask"].any() and not b["vlab_mask"].any() and not bool(b["mesh_mask"])
    assert b["verts"].shape == (778, 3) and b["joints"].shape == (21, 3)


def test_webcam_aug_shapes_and_effect():
    import numpy as np

    from touch.data.grasp_store import GraspWindowDataset

    ds = GraspWindowDataset.__new__(GraspWindowDataset)
    ds.rng = np.random.default_rng(0)
    rng = np.random.default_rng(1)
    tips = rng.integers(0, 256, (4, 5, 112, 112, 3), dtype=np.uint8)
    hand = rng.integers(0, 256, (4, 160, 160, 3), dtype=np.uint8)
    changed = 0
    for _ in range(10):
        t2, h2 = ds._webcam(tips, hand)
        assert t2.shape == (4, 5, 112, 112, 3) and t2.dtype == np.uint8
        assert h2.shape == (4, 160, 160, 3) and h2.dtype == np.uint8
        changed += int(not np.array_equal(t2, tips))
    assert changed >= 5  # applied with p = 0.8
