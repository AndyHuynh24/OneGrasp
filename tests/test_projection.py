import numpy as np

from touch.data.projection import apply_light, make_screen, random_window_projection, screen_light


def test_make_screen_shape_range_and_deterministic():
    a = make_screen(np.random.default_rng(3), 90, 160)
    b = make_screen(np.random.default_rng(3), 90, 160)
    assert a.shape == (90, 160, 3) and a.dtype == np.float32
    assert a.min() >= 0.02 and a.max() <= 1.0
    np.testing.assert_array_equal(a, b)


def test_uniform_light_leaves_patch_unchanged():
    x = np.random.default_rng(0).random((2, 112, 112, 3)).astype(np.float32)
    np.testing.assert_allclose(apply_light(x, np.ones_like(x), 0.4), x, atol=1e-5)


def test_screen_light_follows_the_fingertip():
    screen = np.full((300, 400, 3), 0.1, np.float32)
    screen[140:160, 190:210] = 1.0  # bright spot under the tip at (200, 150)
    tip_dip = np.array([[200.0, 150.0], [160.0, 150.0]])
    light = screen_light(screen, tip_dip)
    assert light.shape == (112, 112, 3)
    assert light[41, 56].mean() > 0.9 and light[100, 56].mean() < 0.2  # tip near row 41


def test_random_window_projection_shape_and_missing_rows():
    rng = np.random.default_rng(0)
    W, N = 9, 20
    x = rng.random((W, 112, 112, 3)).astype(np.float32)
    geo = np.zeros((N, 5, 2, 2), np.float32)
    geo[:, :, 0] = [200.0, 150.0]  # tip
    geo[:, :, 1] = [160.0, 150.0]  # dip
    src = np.arange(W) + 5
    src[0] = -1  # row without a patch
    out = random_window_projection(rng, x, src, geo, (300, 400), finger=2)
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
    np.testing.assert_array_equal(out[0], x[0])
    assert not np.allclose(out[1:], x[1:])
