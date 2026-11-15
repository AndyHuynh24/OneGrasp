import numpy as np

from touch.interact.grab_gate import (
    WINDOW,
    Ensemble,
    Fusion,
    FusionGate,
    Hysteresis,
    prob_features,
)


def _random_stream(n=120, seed=0):
    rng = np.random.default_rng(seed)
    st = rng.dirichlet([1, 1, 1], n)
    fi = rng.random((n, 5))
    pr = rng.random(n) > 0.2
    return st, fi, pr


def _model(nf, seed=0):
    rng = np.random.default_rng(seed)
    layers = [(rng.normal(0, 0.3, (2 * nf, 8)).astype(np.float32), np.zeros(8, np.float32)),
              (rng.normal(0, 0.3, (8, 1)).astype(np.float32), np.zeros(1, np.float32))]  # fmt: skip
    return Fusion(np.zeros(nf, np.float32), np.ones(nf, np.float32), layers)


def test_prob_features_shape_and_causal():
    st, fi, pr = _random_stream()
    X, names = prob_features(st, fi, pr)
    assert X.shape == (120, 20) and len(names) == 20
    X2, _ = prob_features(st[:60], fi[:60], pr[:60])  # the future does not change the past
    np.testing.assert_array_equal(X[:60], X2)
    assert np.isnan(X[~pr, 0]).all()  # no hand -> NaN probs, present flag 0
    assert (X[~pr, names.index("present")] == 0).all()


def test_streaming_equals_batch():
    st, fi, pr = _random_stream()
    X, names = prob_features(st, fi, pr)
    model = Ensemble([_model(len(names), 0), _model(len(names), 1)])
    p_batch = model.predict_proba(X)[:, 1]
    assert p_batch.shape == (120,)
    gate = FusionGate(model, {"on": 0.5, "off": 0.3, "k_on": 2, "k_off": 3})
    h = Hysteresis(on=0.5, off=0.3, k_on=2, k_off=3)
    for t in range(len(st)):
        closed = gate.step(st[t], fi[t], pr[t])
        assert len(gate.buf) == min(t + 1, WINDOW)
        if pr[t]:
            np.testing.assert_allclose(gate.score, p_batch[t], rtol=1e-5, atol=1e-6)
        assert closed == h.step(p_batch[t] if pr[t] else np.nan, pr[t])


def test_hysteresis():
    h = Hysteresis(on=0.6, off=0.2, k_on=2, k_off=3)
    seq = [0.7, 0.1, 0.7, 0.7, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1]
    out = [h.step(s, True) for s in seq]
    assert out == [False, False, False, True, True, True, True, True, True, False]
    assert h.step(0.0, False) is False  # no hand keeps the decision


def test_hysteresis_fast_release():
    """The score lingers high after the object is put down; the model's NONE opens the gripper."""
    kw = {"on": 0.6, "off": 0.2, "k_on": 2, "k_off": 3}
    score = [0.9, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5]
    pnone = [0.0, 0.0, 0.0, 0.95, 0.95, 0.95, 0.95, 0.95]
    slow = Hysteresis(**kw)
    assert [slow.step(s, True, q) for s, q in zip(score, pnone, strict=True)][-1] is True
    fast = Hysteresis(**kw, p_none=0.9, k_none=3)
    out = [fast.step(s, True, q) for s, q in zip(score, pnone, strict=True)]
    assert out == [False, True, True, True, True, False, False, False]
    h = Hysteresis(**kw, p_none=0.9, k_none=3)  # NAN P(NONE) (old callers) never fast-releases
    assert [h.step(s, True) for s in score][-1] is True


def test_save_load_roundtrip(tmp_path):
    m = _model(20)
    m.save(tmp_path / "m.npz")
    m2 = Fusion.load(tmp_path / "m.npz")
    X = np.random.default_rng(1).random((5, 20)).astype(np.float32)
    np.testing.assert_allclose(m.predict_proba(X), m2.predict_proba(X))
