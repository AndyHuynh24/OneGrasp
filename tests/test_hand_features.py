import cv2
import numpy as np

from touch.interact.hand_features import (
    COMOTION_NAMES,
    POSE_NAMES,
    TIPCOLOR_NAMES,
    CoMotion,
    image_motion,
    motion3d,
    pose_features,
    tip_color,
)


def _hand(scale=1.0, curl=0.0):
    """A synthetic right hand [21, 3]: palm in the x-y plane, fingers along +y; curl bends the
    distal segments toward -z."""
    J = np.zeros((21, 3))
    bases = {
        1: (-0.03, 0.02),
        5: (-0.02, 0.09),
        9: (0.0, 0.095),
        13: (0.02, 0.09),
        17: (0.035, 0.08),
    }
    for b, (x, y) in bases.items():
        J[b] = [x, y, 0]
        for k in range(1, 4):
            ang = curl * k
            J[b + k] = J[b + k - 1] + 0.025 * np.array([0, np.cos(ang), -np.sin(ang)])
    return J * scale


def test_pose_features_shape_scale_invariant_and_curl():
    X = pose_features(np.stack([_hand(), _hand(scale=3.0), _hand(curl=0.6)]))
    assert X.shape == (3, len(POSE_NAMES)) == (3, 32)
    np.testing.assert_allclose(X[0], X[1], atol=1e-5)  # palm-length normalised
    ci = POSE_NAMES.index("curl_index")
    assert X[2, ci] < X[0, ci] - 0.1  # curled finger: tip closer to its base
    assert np.isnan(pose_features(np.full((1, 21, 3), np.nan))).all()


def test_image_motion_and_motion3d():
    lm = np.stack([_hand()[:, :2] * 1000 + [300, 300 - 5 * t] for t in range(10)])  # moves up
    v, names = image_motion(lm)
    assert v.shape == (10, 3) and len(names) == 3
    assert np.isnan(v[:3]).all() and (v[3:, 1] > 0).all()  # y up is positive
    Jw = np.stack([_hand() + [0, 0, 0.01 * t] for t in range(40)])
    M = motion3d(Jw)
    assert M.shape == (40, 4)
    np.testing.assert_allclose(M[10, 2], 0.3, atol=1e-6)  # 1 cm / frame at 30 fps = 0.3 m/s
    assert M[-1, 3] > 0.25  # rose ~29 cm over the last 30 frames


def _scene(t, carry: bool):
    """Static textured background; a textured hand (thick bones) and, if carry, a textured
    object next to the index fingertip, both moving right by 3 px per frame."""
    from touch.interact.hand_features import CHAINS

    bg = cv2.GaussianBlur((np.random.default_rng(0).random((240, 320)) * 255).astype(np.uint8),
                          (0, 0), 1.0)  # fmt: skip
    mov = cv2.GaussianBlur((np.random.default_rng(1).random((240, 400)) * 255).astype(np.uint8),
                           (0, 0), 1.0)  # fmt: skip
    dx = 3 * t
    moving = mov[:, 40 - dx : 360 - dx]  # texture shifted right by dx
    lm = _hand()[:, :2] * 900 + [100 + dx, 40]
    mask = np.zeros((240, 320), np.uint8)
    pts = np.round(lm).astype(np.int32)
    for ch in CHAINS:
        for a, b in zip(ch[:-1], ch[1:], strict=True):
            cv2.line(mask, tuple(pts[a]), tuple(pts[b]), 255, 12)
    cv2.fillPoly(mask, [pts[[0, 1, 5, 9, 13, 17]]], 255)
    if carry:
        x0, y0 = pts[8]
        cv2.rectangle(mask, (int(x0) + 8, int(y0) - 30), (int(x0) + 45, int(y0) + 30), 255, -1)
    img = np.where(mask > 0, moving, bg)
    return img, lm


def test_comotion_carry_vs_empty():
    out = {}
    for carry in (True, False):
        cm = CoMotion()
        rows = [cm.step(*_scene(t, carry)) for t in range(8)]
        F = np.stack(rows)
        assert F.shape == (8, len(COMOTION_NAMES))
        out[carry] = np.nanmean(F[:, 0])
    assert out[True] > out[False]  # the carried object moves with the hand
    cm = CoMotion()
    assert np.isnan(cm.step(np.zeros((240, 320), np.uint8), None)[:4]).all()


def test_tip_color_shape_and_sign():
    img = np.full((240, 320, 3), (120, 140, 200), np.uint8)  # reddish skin (BGR)
    lm = _hand()[:, :2] * 900 + [100, 40]
    base = tip_color(img, lm)
    assert base.shape == (len(TIPCOLOR_NAMES),) == (13,)
    np.testing.assert_allclose(base[:12], 0, atol=1e-3)  # same colour as the palm
    pale = img.copy()
    for f in (4, 8, 12, 16, 20):  # blanch the fingertips
        c = (lm[f] + lm[f - 1]) / 2
        cv2.circle(pale, tuple(int(v) for v in c), 8, (190, 195, 215), -1)
    x = tip_color(pale, lm)
    assert x[TIPCOLOR_NAMES.index("tip_da_mean")] < -2  # less red than the palm
    assert np.isnan(tip_color(img, None)).all()
