import numpy as np
import pytest
import torch

from touch.data.patch_store import represent
from touch.data.patches import (
    LM_FEAT_DIMS,
    LM_HISTORY,
    N_KIN,
    N_LM_FEAT,
    PATCH_SHIFT,
    extract_patch,
    fingertip_matrix,
    landmark_features,
    normalize_hand,
)
from touch.models.touchnet import TouchNet


def _apply(M, px, py):
    return M @ np.array([px, py, 1.0])


@pytest.mark.parametrize("direction", [(1, 0), (0, 1), (-1, -1), (0.3, -0.9)])
def test_fingertip_matrix_puts_finger_up_and_tip_above_centre(direction):
    dip = np.array([200.0, 150.0])
    u = np.array(direction, float) / np.hypot(*direction)
    tip = dip + 40 * u
    S = 112
    M = fingertip_matrix(tip, dip, size=S)
    # patch centre maps to tip - shift*L*u
    np.testing.assert_allclose(_apply(M, S / 2, S / 2), tip - PATCH_SHIFT * 40 * u, atol=1e-9)
    # moving up in the patch (-y) moves along the finger direction in the image
    step = _apply(M, S / 2, S / 2 - 1) - _apply(M, S / 2, S / 2)
    np.testing.assert_allclose(step / np.hypot(*step), u, atol=1e-9)
    assert np.linalg.det(M[:, :2]) > 0  # rotation, no mirror


def test_extract_patch_shape_and_content():
    img = np.zeros((300, 400, 3), np.uint8)
    img[140:160, 190:210] = 255  # bright square at the tip
    M = fingertip_matrix(np.array([200.0, 150.0]), np.array([160.0, 150.0]), size=112)
    p = extract_patch(img, M, 112)
    assert p.shape == (112, 112, 3) and p.dtype == np.uint8
    # tip sits PATCH_SHIFT*L above the centre: around row 56 - 0.3*40/(2.2*40/112) ~ 41
    assert p[41, 56].mean() > 200


def test_landmark_features_invariant_to_translation_rotation_scale():
    rng = np.random.default_rng(0)
    lm = rng.normal(size=(21, 3)) * 30
    lm[9, :2] = lm[0, :2] + np.array([5.0, -60.0])
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    lm2 = lm.copy()
    lm2[:, :2] = (lm[:, :2] @ R.T) * 2.0 + np.array([500.0, -40.0])
    lm2[:, 2] = lm[:, 2] * 2.0
    q1, _ = normalize_hand(lm)
    q2, _ = normalize_hand(lm2)
    np.testing.assert_allclose(q1, q2, atol=1e-5)
    np.testing.assert_allclose(q1[9, :2], [0, -1], atol=1e-6)


def test_landmark_features_shape_and_missing_frames():
    seq = np.random.default_rng(1).normal(size=(5, 21, 3)).astype(np.float32) * 50
    seq[2] = np.nan
    f, present = landmark_features(seq)
    assert f.shape == (5, N_LM_FEAT) and present.tolist() == [True, True, False, True, True]
    assert (f[2] == 0).all() and f[3, 126] == 0 and f[4, 126] == 1  # velocity resets after gap


@pytest.mark.parametrize("temporal", ["gru", "none"])
@pytest.mark.parametrize("repr_", ["rgb", "rgb+ndiff"])
def test_touchnet_output_shape(temporal, repr_):
    c = 6 if repr_ == "rgb+ndiff" else 3
    m = TouchNet("mobilenetv3_small_100", pretrained=False, in_chans=c, temporal=temporal).eval()
    B, T = 2, 8
    x = torch.randn(B, T, c, 112, 112)
    out = m(
        x, torch.randn(B, T, N_LM_FEAT), torch.ones(B, T, dtype=torch.bool), torch.tensor([1, 3])
    )
    assert out.shape == (B, T)


def test_touchnet_is_causal():
    torch.manual_seed(0)
    m = TouchNet("mobilenetv3_small_100", pretrained=False, temporal="gru").eval()
    B, T = 1, 6
    x = torch.randn(B, T, 3, 112, 112)
    lm = torch.randn(B, T, N_LM_FEAT)
    pres = torch.ones(B, T, dtype=torch.bool)
    f = torch.tensor([1])
    with torch.no_grad():
        a = m(x, lm, pres, f)
        x2 = x.clone()
        x2[:, -1] = torch.randn(3, 112, 112)  # change only the last (newest) frame
        b = m(x2, lm, pres, f)
    torch.testing.assert_close(a[:, :-1], b[:, :-1])
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_represent_shapes():
    x = np.random.default_rng(0).random((4, 112, 112, 3)).astype(np.float32)
    assert represent(x, x, "rgb").shape == (4, 3, 112, 112)
    assert represent(x, x, "rgb+ndiff").shape == (4, 6, 112, 112)
    assert np.abs(represent(x, x, "ndiff")).max() == 0  # no change -> zero difference


def _hand_seq(n=8, seed=2):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(21, 3)) * 30
    base[9, :2] = base[0, :2] + np.array([5.0, -60.0])
    steps = rng.normal(size=(n, 21, 3)) * 2
    return (base + np.cumsum(steps, 0)).astype(np.float32)


def test_landmark_features_v2_shape_and_v1_prefix():
    seq = _hand_seq()
    seq[3] = np.nan
    f1, _ = landmark_features(seq, "v1")
    f2, present = landmark_features(seq, "v2")
    assert f2.shape == (8, LM_FEAT_DIMS["v2"]) and LM_FEAT_DIMS["v2"] == N_LM_FEAT + N_KIN
    np.testing.assert_array_equal(f2[:, :N_LM_FEAT], f1)
    assert (f2[3] == 0).all()
    flags = f2[:, N_LM_FEAT + 48 : N_LM_FEAT + 52]  # has lag 1, 2, 4, has accel
    assert flags[4].tolist() == [0, 1, 1, 0] and flags[7].tolist() == [1, 1, 0, 1]


def test_landmark_features_v2_causal_and_window():
    """Frame i depends only on frames i-LM_HISTORY+1..i (the live demo keeps exactly that many)."""
    seq = _hand_seq(10)
    full, _ = landmark_features(seq, "v2")
    seq2 = seq.copy()
    seq2[8:] += 50.0  # change the future
    np.testing.assert_array_equal(landmark_features(seq2, "v2")[0][:8], full[:8])
    h = LM_HISTORY["v2"]
    for i in range(h - 1, 10):
        win, _ = landmark_features(seq[i - h + 1 : i + 1], "v2")
        np.testing.assert_allclose(win[-1], full[i], atol=1e-6)


def test_landmark_features_v2_invariant_but_sees_hand_motion():
    seq = _hand_seq()
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    seq2 = seq.copy()
    seq2[..., :2] = (seq[..., :2] @ R.T) * 2.0 + np.array([500.0, -40.0])
    seq2[..., 2] *= 2.0
    np.testing.assert_allclose(
        landmark_features(seq, "v2")[0], landmark_features(seq2, "v2")[0], atol=1e-4
    )
    # translating the whole hand over time: v1 velocity stays 0, v2 displacement does not
    still = np.repeat(seq[:1], 3, 0)
    moving = still + np.array([0.0, 6.0, 0.0])[None, None] * np.arange(3)[:, None, None]
    f1, _ = landmark_features(moving.astype(np.float32), "v1")
    f2, _ = landmark_features(moving.astype(np.float32), "v2")
    np.testing.assert_allclose(f1[2, 63:126], 0, atol=1e-6)
    assert np.abs(f2[2, N_LM_FEAT : N_LM_FEAT + 12]).max() > 0.1


def test_touchnet_v2_landmarks_and_img_drop_shape():
    m = TouchNet(
        "mobilenetv3_small_100", pretrained=False, n_lm=LM_FEAT_DIMS["v2"], lm_emb=64, img_drop=0.5
    )
    B, T = 3, 8
    args = (
        torch.randn(B, T, 3, 112, 112),
        torch.randn(B, T, LM_FEAT_DIMS["v2"]),
        torch.ones(B, T, dtype=torch.bool),
        torch.tensor([0, 1, 4]),
    )
    assert m.train()(*args).shape == (B, T)
    m.eval()
    with torch.no_grad():  # no dropout at eval: deterministic
        torch.testing.assert_close(m(*args), m(*args))
    assert m.gru.input_size == 128 + 64 + 1


def test_landmark_features_v2_jitter():
    """Jitter is 0 for steady motion and grows with frame-to-frame shake of a point."""
    from touch.data.patches import KIN_PTS

    seq = np.repeat(_hand_seq(1), 6, 0)
    seq[..., 1] += 3.0 * np.arange(6)[:, None]  # whole hand moving at constant speed
    f, _ = landmark_features(seq.astype(np.float32), "v2")
    jit = slice(LM_FEAT_DIMS["v2"] - len(KIN_PTS) - 1, LM_FEAT_DIMS["v2"])
    assert f[3, jit][-1] == 0  # not enough history yet
    np.testing.assert_allclose(f[5, jit][:-1], 0, atol=1e-5)
    assert f[5, jit][-1] == 1
    shaky = seq.copy()
    shaky[:, 8, :2] += np.random.default_rng(0).normal(size=(6, 2)) * 3  # index tip shakes
    g, _ = landmark_features(shaky.astype(np.float32), "v2")
    j = g[5, jit][:-1]
    assert j[1] > 0.1 and np.abs(j[[0, 2, 3, 4, 5]]).max() < 1e-5
