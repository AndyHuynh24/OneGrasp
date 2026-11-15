import numpy as np
import pytest
import torch

from touch.data.patches import LM_FEAT_DIMS, hand_crop_matrix
from touch.models.graspnet import MANO_DIR, GraspNet, init_from_touchnet, rot6d_to_matrix
from touch.models.touchnet import TouchNet

needs_mano = pytest.mark.skipif(not (MANO_DIR / "MANO_RIGHT.pkl").exists(), reason="no MANO")
F = LM_FEAT_DIMS["v2"]


def small(mesh=True, **kw):
    torch.manual_seed(0)
    return GraspNet("mobilenetv3_small_100", "mobilenetv3_small_100", pretrained=False,
                    mesh=mesh, **kw).eval()  # fmt: skip


def inputs(B=2, T=4, s=112, S=160):
    g = torch.Generator().manual_seed(1)
    return (
        torch.randn(B, T, 5, 3, s, s, generator=g),
        torch.randn(B, T, 3, S, S, generator=g),
        torch.randn(B, T, F, generator=g),
        torch.ones(B, T, dtype=torch.bool),
    )


@needs_mano
def test_output_shapes():
    m = small()
    out = m(*inputs())
    assert out["finger"].shape == (2, 4, 5)
    assert out["state"].shape == (2, 4, 3)
    assert out["vert"].shape == (2, 4, 778)
    assert out["verts"].shape == (2, 778, 3)
    assert out["joints"].shape == (2, 21, 3)
    assert torch.allclose(out["joints"][:, 0], torch.zeros(2, 3), atol=1e-6)  # root-relative


def test_causal():
    m = small(mesh=False)
    tips, hand, lm, pr = inputs(T=6)
    a = m(tips, hand, lm, pr, with_mesh=False)
    tips2, hand2 = tips.clone(), hand.clone()
    tips2[:, 4:] += 1.0
    hand2[:, 4:] -= 1.0
    b = m(tips2, hand2, lm, pr, with_mesh=False)
    for k in ("finger", "state", "vert"):
        assert torch.allclose(a[k][:, :4], b[k][:, :4], atol=1e-5), k
        assert not torch.allclose(a[k][:, 4:], b[k][:, 4:]), k


def test_init_matches_touchnet(tmp_path):
    torch.manual_seed(0)
    tn = TouchNet("mobilenetv3_small_100", pretrained=False, temporal="gru", n_lm=F,
                  lm_emb=64).eval()  # fmt: skip
    p = tmp_path / "tn.pt"
    torch.save({"model": tn.state_dict()}, p)
    m = small(mesh=False)
    copied = init_from_touchnet(m, p)
    assert "gru.weight_ih_l0" in copied and "encoder.conv_stem.weight" in copied
    tips, hand, lm, pr = inputs(T=5)
    out = m(tips, hand, lm, pr, with_mesh=False)["finger"]
    for f in range(5):
        ref = tn(tips[:, :, f], lm, pr, torch.full((2,), f))
        assert torch.allclose(out[..., f], ref, atol=1e-4), f


def test_rot6d_orthonormal():
    R = rot6d_to_matrix(torch.randn(10, 6))
    eye = torch.eye(3).expand(10, 3, 3)
    assert torch.allclose(R.transpose(1, 2) @ R, eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(10), atol=1e-5)
    I6 = torch.tensor([1.0, 0, 0, 0, 1, 0])
    assert torch.allclose(rot6d_to_matrix(I6), torch.eye(3))


@needs_mano
def test_mano_layer_matches_dexycb_forward():
    from smplx.lbs import batch_rodrigues

    from touch.data import dexycb as D

    rng = np.random.default_rng(0)
    pose_m = np.zeros((3, 51), np.float32)
    pose_m[:, :3] = rng.normal(0, 0.5, (3, 3))
    pose_m[:, 3:48] = rng.normal(0, 0.5, (3, 45))
    betas = rng.normal(0, 0.5, 10).astype(np.float32)
    v_ref, j_ref = D.mano_forward(pose_m, betas, "right")
    mp = D.mano_model("right")
    full = pose_m[:, 3:48] @ np.asarray(mp["hands_components"]) + np.asarray(mp["hands_mean"])
    aa = torch.from_numpy(np.concatenate([pose_m[:, :3], full], 1).astype(np.float32))
    R = batch_rodrigues(aa.view(-1, 3)).view(3, 16, 3, 3)
    m = small()
    v, j = m.mano(R, torch.from_numpy(np.tile(betas, (3, 1))))
    root = j_ref[:, :1]
    assert np.abs(v.numpy() - (v_ref - root)).max() < 1e-5
    assert np.abs(j.numpy() - (j_ref - root)).max() < 1e-5


def test_hand_crop_flip_mirrors():
    lm = np.array([[100.0, 50.0], [140.0, 90.0]] + [[120.0, 70.0]] * 19)
    M = hand_crop_matrix(lm, size=100, expand=1.0)
    Mf = hand_crop_matrix(lm, size=100, expand=1.0, flip=True)
    # crop pixel (u, v) of the flipped crop = crop pixel (size - u, v) of the plain crop
    for u, v in [(0, 0), (30, 70), (100, 100)]:
        a = Mf @ np.array([u, v, 1.0])
        b = M @ np.array([100 - u, v, 1.0])
        assert np.allclose(a, b)
    assert np.allclose(M @ np.array([50, 50, 1.0]), [120, 70])  # centred on the landmark box


def test_grad_frames_same_forward_fewer_grads():
    torch.manual_seed(0)
    m = small(mesh=False, img_drop=0.0, hand_drop=0.0)
    m.train()
    for mod in m.modules():  # deterministic forward: no dropout, BN uses running stats
        if isinstance(mod, (torch.nn.Dropout, torch.nn.BatchNorm2d)):
            mod.eval()
    x = inputs(T=5)
    m.grad_frames = 0
    a = m(*x, with_mesh=False)["finger"]
    m.grad_frames = 2
    b = m(*x, with_mesh=False)["finger"]
    assert torch.allclose(a, b, atol=1e-5)
    tips = x[0].clone().requires_grad_(True)
    m(tips, *x[1:], with_mesh=False)["finger"].sum().backward()
    g = tips.grad.abs().sum(dim=(0, 2, 3, 4, 5))
    assert (g[:3] == 0).all() and (g[3:] > 0).all()


@needs_mano
def test_step_equals_window():
    from touch.models.graspnet import GraspStep

    m = small()
    tips, hand, lm, pr = inputs(B=1, T=4)
    pr[0, 1] = False
    full = m(tips, hand, lm, pr)
    enc = m.encode(tips, hand, lm, pr)
    step = GraspStep(m).eval()
    out = step(tips[0, -1], hand[0, -1], lm[0, -1], pr[0, -1], enc["tok"][0, :-1],
               enc["hand_emb"][0, :-1], pr[0, :-1])  # fmt: skip
    assert out[0].shape == (5,) and out[1].shape == (3,) and out[2].shape == (778,)
    assert out[3].shape == (778, 3) and out[4].shape == (21, 3)
    assert out[-2].shape == (16, 3, 3) and out[-1].shape == (10,)  # MANO rotations, betas
    v, j = m.mano(out[-2][None], out[-1][None])  # the params rebuild the output mesh
    assert torch.allclose(v[0], out[3], atol=1e-5) and torch.allclose(j[0], out[4], atol=1e-5)
    assert torch.allclose(out[0], torch.sigmoid(full["finger"][0, -1]), atol=1e-5)
    assert torch.allclose(out[1], torch.softmax(full["state"][0, -1], -1), atol=1e-5)
    assert torch.allclose(out[3], full["verts"][0], atol=1e-5)


# ---------------------------------------------------------------- offline bidirectional variant


def test_bidir_shapes_and_window_target():
    m = small(mesh=False, bidir=True)
    assert not m.causal and m.window_target(9) == 4 and small(mesh=False).window_target(9) == 8
    out = m(*inputs(T=9), with_mesh=False)
    assert out["finger"].shape == (2, 9, 5)
    assert out["state"].shape == (2, 9, 3)
    assert out["vert"].shape == (2, 9, 778)


def test_bidir_init_equals_causal(tmp_path):
    from touch.models.graspnet import init_from_causal

    c = small(mesh=False)
    for p in c.parameters():  # non-trivial weights everywhere
        p.data += 0.01 * torch.randn_like(p)
    p = tmp_path / "c.pt"
    torch.save({"model": c.state_dict()}, p)
    torch.manual_seed(5)
    b = GraspNet("mobilenetv3_small_100", "mobilenetv3_small_100", pretrained=False, mesh=False,
                 bidir=True).eval()  # fmt: skip
    init_from_causal(b, p)
    x = inputs(T=7)
    oc = c(*x, with_mesh=False)
    ob = b(*x, with_mesh=False)
    for k in ("finger", "state", "vert"):  # zero backward readout -> identical at every step
        assert torch.allclose(oc[k], ob[k], atol=1e-5), k


def test_bidir_sees_future():
    m = small(mesh=False, bidir=True)
    tips, hand, lm, pr = inputs(T=7)
    a = m(tips, hand, lm, pr, with_mesh=False)["state"][:, 3]
    tips2 = tips.clone()
    tips2[:, 5:] += 1.0
    b = m(tips2, hand, lm, pr, with_mesh=False)["state"][:, 3]
    assert not torch.allclose(a, b)


def test_bidir_grad_frames_centre():
    torch.manual_seed(0)
    m = small(mesh=False, bidir=True, img_drop=0.0, hand_drop=0.0, grad_frames=3)
    m.train()
    for mod in m.modules():
        if isinstance(mod, (torch.nn.Dropout, torch.nn.BatchNorm2d)):
            mod.eval()
    x = inputs(T=7)
    tips = x[0].clone().requires_grad_(True)
    m(tips, *x[1:], with_mesh=False)["finger"].sum().backward()
    g = tips.grad.abs().sum(dim=(0, 2, 3, 4, 5))
    assert (g[[0, 1, 5, 6]] == 0).all() and (g[2:5] > 0).all()


def test_table_head_shape_and_off_by_default():
    m = small(mesh=False, table=True)
    out = m(*inputs(), with_mesh=False)
    assert out["table"].shape == (2, 4)  # one logit per frame
    assert "table" not in small(mesh=False)(*inputs(), with_mesh=False)
