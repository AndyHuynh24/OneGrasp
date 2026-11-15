from types import SimpleNamespace

import numpy as np

from touch.data.patch_store import sample_weights


def _track(variant, contact):
    n = len(contact)
    return SimpleNamespace(
        contact=np.array(contact, bool),
        valid=np.ones(n, bool),
        seq=SimpleNamespace(meta={"variant": variant}),
    )


def _index(k, n):
    return np.array([(k, t, f) for t in range(n) for f in range(5)])


def test_hover_weight_scales_only_no_contact_sequences():
    press = [[0] * 5] * 4 + [[1, 0, 0, 0, 0]] * 4 + [[0] * 5] * 4
    hover = [[0] * 5] * 12
    tracks = [_track("high", press), _track("no-contact", hover)]
    index = np.vstack([_index(0, 12), _index(1, 12)])
    base = sample_weights(tracks, index)
    w = sample_weights(tracks, index, hover_weight=2.0)
    assert w.shape == (len(index),)
    is_hover = index[:, 0] == 1
    np.testing.assert_allclose(w[is_hover], 2.0 * base[is_hover])
    np.testing.assert_allclose(w[~is_hover], base[~is_hover])
    np.testing.assert_allclose(base[is_hover], 1.0)  # default unchanged
