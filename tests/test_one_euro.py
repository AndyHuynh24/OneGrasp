import numpy as np
from scipy.spatial.transform import Rotation as Rot

from touch.interact.one_euro import OneEuro, OneEuroRotation


def test_constant_signal_passes_unchanged():
    f = OneEuro(min_cutoff=1.0, beta=0.5)
    for _ in range(10):
        y = f([1.0, 2.0, 3.0])
    assert y.shape == (3,) and np.allclose(y, [1, 2, 3])


def test_reduces_jitter_at_rest_and_lags_less_with_beta():
    rng = np.random.default_rng(0)
    noisy = 0.005 * rng.standard_normal((300, 3))
    f = OneEuro(min_cutoff=1.0, beta=0.0)
    out = np.array([f(x) for x in noisy])
    assert out[30:].std() < 0.3 * noisy.std()  # jitter at rest strongly reduced
    ramp = np.linspace(0, 1, 30)[:, None] * [1.0, 0, 0]  # 1 m in 1 s
    lag = {}
    for beta in (0.0, 5.0):
        g = OneEuro(min_cutoff=1.0, beta=beta)
        lag[beta] = float(ramp[-1, 0] - [g(x) for x in ramp][-1][0])
    assert lag[5.0] < 0.5 * lag[0.0]  # speed raises the cutoff: less lag when moving


def test_rotation_filter_returns_rotations_and_converges():
    f = OneEuroRotation(min_cutoff=1.0, beta=0.0)
    R1 = Rot.from_euler("z", 90, degrees=True).as_matrix()
    f(np.eye(3))
    for _ in range(90):
        R = f(R1)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.allclose(R, R1, atol=1e-3)
