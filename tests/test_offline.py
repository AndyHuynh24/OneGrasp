import pytest
import torch

from touch.eval.run import window_logits
from touch.models.streaming import StreamingStep
from touch.models.touchnet import TouchNet


def _model(temporal, seed=0):
    torch.manual_seed(seed)
    return TouchNet("mobilenetv3_small_100", pretrained=False, temporal=temporal).eval()


def test_bigru_output_shape_and_target():
    m = _model("bigru")
    B, T = 2, 12
    with torch.no_grad():
        out = m(
            torch.randn(B, T, 3, 112, 112),
            torch.randn(B, T, m.n_lm),
            torch.ones(B, T, dtype=torch.bool),
            torch.tensor([1, 3]),
        )
    assert out.shape == (B, T)
    assert not m.causal and m.window_target(T) == 6
    assert _model("gru").causal and _model("gru").window_target(T) == T - 1


def _changed(m, T, n=20, j=10):
    """Frames whose score changes when only frame j's feature changes."""
    d = m.gru.input_size
    g = torch.Generator().manual_seed(0)
    feats, pad = torch.randn(n, d, generator=g), torch.randn(d, generator=g)
    with torch.no_grad():
        a = window_logits(m, feats, pad, T, batch=7)
        feats[j] += 5.0
        b = window_logits(m, feats, pad, T, batch=7)
    assert a.shape == (n,)
    return set(torch.nonzero((a - b).abs() > 1e-6).flatten().tolist())


def test_causal_window_never_uses_future_frames():
    T, j = 8, 10
    assert _changed(_model("gru"), T, j=j) == set(range(j, j + T))  # j..j+T-1 only


def test_offline_window_uses_past_and_future_frames():
    T, j = 12, 10
    c = T // 2  # frame i sees i-c .. i+T-1-c
    assert _changed(_model("bigru"), T, j=j) == set(range(j - (T - 1 - c), j + c + 1))


def test_offline_model_cannot_stream_live():
    with pytest.raises(ValueError, match="offline"):
        StreamingStep(_model("bigru"))
