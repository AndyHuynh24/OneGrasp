import numpy as np
import pytest
import torch

from touch.data.patch_store import represent
from touch.data.patches import N_LM_FEAT
from touch.models.streaming import StreamingStep, example_inputs, export_onnx, pad_features
from touch.models.touchnet import TouchNet


def _model(seed=0):
    torch.manual_seed(seed)
    return TouchNet("mobilenetv3_small_100", pretrained=False, temporal="gru").eval()


def test_streaming_step_shapes():
    m = _model()
    with torch.no_grad():
        prob, feat = StreamingStep(m)(*example_inputs(m, "rgb", T=8))
    assert prob.shape == (5,)
    assert feat.shape == (5, m.gru.input_size)
    assert ((prob >= 0) & (prob <= 1)).all()


def test_streaming_equals_windowed_forward():
    """Live frame-by-frame scores == full forward on the causal window (zero frames before t=0)."""
    m, T, N = _model(), 4, 7
    g = torch.Generator().manual_seed(1)
    x = torch.randn(N, 3, 112, 112, generator=g)
    lm = torch.randn(N, N_LM_FEAT, generator=g)
    pres = torch.tensor([True, True, False, True, True, True, False])
    f = torch.tensor([2])
    step = StreamingStep(m)
    with torch.no_grad():
        hist = pad_features(m, "rgb", f)[:, None].expand(-1, T - 1, -1)
        live = []
        for t in range(N):
            p, feat = step(x[t : t + 1], lm[t : t + 1], pres[t : t + 1], f, hist)
            hist = torch.cat([hist[:, 1:], feat[:, None]], 1)
            live.append(p.item())
        # frames before the start: black patch (as eval's represent(zeros)), no landmarks
        zero = np.zeros((112, 112, 3), np.float32)
        pad_img = torch.from_numpy(represent(zero, zero, "rgb"))
        ref = []
        for t in range(N):
            idx = range(t - T + 1, t + 1)
            xi = torch.stack([x[i] if i >= 0 else pad_img for i in idx])
            li = torch.stack([lm[i] if i >= 0 else torch.zeros(N_LM_FEAT) for i in idx])
            pi = torch.stack([pres[i] if i >= 0 else torch.tensor(False) for i in idx])
            logit = m(xi[None], li[None], pi[None], f)[0, -1]
            ref.append(torch.sigmoid(logit).item())
    np.testing.assert_allclose(live, ref, rtol=1e-4, atol=1e-5)


def test_onnx_export_matches_torch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    m = _model()
    path = tmp_path / "step.onnx"
    export_onnx(m, "rgb", 8, path)
    inp = example_inputs(m, "rgb", 8)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {k.name: t.numpy() for k, t in zip(sess.get_inputs(), inp, strict=True)})
    with torch.no_grad():
        ref = StreamingStep(m)(*inp)
    assert out[0].shape == (5,) and out[1].shape == tuple(ref[1].shape)
    np.testing.assert_allclose(out[0], ref[0].numpy(), atol=1e-5)
