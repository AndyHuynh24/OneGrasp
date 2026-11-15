import numpy as np
import pytest
import torch

from touch.interact.gestures import GestureConfig
from touch.interact.handflow import HandFlowAdapter
from touch.interact.session import FrameResult, LiveTouch, TouchSession, landmarks_px
from touch.interact.surface import SurfaceMap, detect_aruco_corners
from touch.interact.touches import TouchEvent
from touch.train import build_model, load_config

POST = {"k": 3, "on": 0.6, "off": 0.4}


def _hand(dx=0.0):
    lm = np.zeros((21, 3), np.float32)
    lm[0] = (320 + dx, 400, 0)
    for j in range(5):
        for k in range(4):
            lm[1 + 4 * j + k] = (250 + 35 * j + dx, 340 - 30 * k, 0)
    return lm


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    cfg = load_config({"backbone": "mobilenetv3_small_100", "pretrained": False, "T": 4})
    model = build_model(cfg)
    path = tmp_path_factory.mktemp("ck") / "best.pt"
    torch.save({"config": cfg, "model": model.state_dict(), "post": POST}, path)
    return path


def test_live_touch_shapes_and_no_hand(ckpt):
    live = LiveTouch(ckpt, threads=1)
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), np.uint8)
    for i in range(5):
        p = live.model(frame, _hand(i))
        assert p.shape == (5,) and ((p >= 0) & (p <= 1)).all()
        s = live.postprocess(p)
        assert s.shape == (5,) and live.touch.shape == (5,) and live.touch.dtype == bool
    assert (live.model(frame, None) == 0).all()


def test_ensemble_of_identical_members_equals_single(ckpt):
    a, b = LiveTouch(ckpt, threads=1), LiveTouch([ckpt, ckpt], threads=1, post=POST)
    frame = np.random.default_rng(1).integers(0, 255, (480, 640, 3), np.uint8)
    for i in range(6):
        lm = None if i == 3 else _hand(2 * i)
        np.testing.assert_allclose(a.model(frame, lm), b.model(frame, lm), atol=1e-6)


def test_ensemble_requires_post(ckpt):
    with pytest.raises(AssertionError, match="ensemble"):
        LiveTouch([ckpt, ckpt], threads=1)


def test_landmarks_px_scales_z_by_width():
    out = landmarks_px(np.full((21, 3), 0.5), 640, 480)
    assert out.shape == (21, 3) and np.allclose(out[0], (320, 240, 320))


def test_surface_map_corners_and_inside():
    corners = np.array([[100, 50], [500, 80], [520, 400], [80, 380]], np.float32)
    s = SurfaceMap.from_corners(corners, (1920, 1080))
    np.testing.assert_allclose(
        s.map(corners), [[0, 0], [1920, 0], [1920, 1080], [0, 1080]], atol=1e-3
    )
    assert s.inside(s.map(np.array([300, 200]))) and not s.inside(s.map(np.array([10, 10])))


def test_aruco_corners_found_in_order():
    d = cv2_dict()
    img = np.full((600, 800), 255, np.uint8)
    pos = {0: (50, 50), 1: (650, 50), 2: (650, 450), 3: (50, 450)}
    for i, (x, y) in pos.items():
        img[y : y + 100, x : x + 100] = __import__("cv2").aruco.generateImageMarker(d, i, 100)
    c = detect_aruco_corners(img)
    np.testing.assert_allclose(c, [(x + 49.5, y + 49.5) for x, y in pos.values()], atol=1.5)
    assert detect_aruco_corners(img[:, :400]) is None  # markers 1, 2 missing


def cv2_dict():
    import cv2

    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _ev(t, phase, xy, tid=0, finger=1, lost=False):
    return TouchEvent(t, phase, tid, finger, xy, 100.0, 0.9, lost)


def _res(t, events=(), xy=(200.0, 200.0), present=True):
    tips = np.full((5, 2), np.nan) if not present else np.tile(np.array(xy), (5, 1))
    return FrameResult(t, present, np.full(5, 0.5), np.zeros(5, bool), tips, None, list(events))


def test_pointer_ignores_touches_starting_off_surface():
    s = TouchSession.__new__(TouchSession)
    s.surface = SurfaceMap.from_corners([[0, 0], [100, 0], [100, 100], [0, 100]], (1000, 1000))
    s.on_surface = set()
    out = s._pointer([_ev(0, "down", (50, 50), 0), _ev(0, "down", (150, 50), 1, finger=2)])
    assert [p["id"] for p in out] == [0] and out[0]["xy"] == [500.0, 500.0]
    out = s._pointer([_ev(1, "move", (120, 50), 0), _ev(1, "up", (150, 50), 1, finger=2)])
    assert [p["id"] for p in out] == [0]  # slid off but started inside: still reported
    assert s._pointer([_ev(2, "up", (120, 50), 0)])[0]["phase"] == "up" and not s.on_surface


def test_handflow_tap_hold_hover():
    cfg = GestureConfig()
    a = HandFlowAdapter(cfg=cfg)
    assert a.update(_res(0.0)).gesture == "touch_hover"
    assert a.update(_res(0.1, [_ev(0.1, "down", (200.0, 200.0))])).gesture == "touch_hover"
    out = a.update(_res(0.3, [_ev(0.3, "up", (201.0, 200.0))]))
    assert out.gesture == "touch" and out.xy == (200.0, 200.0)  # click where it went down
    a.update(_res(1.0, [_ev(1.0, "down", (200.0, 200.0), tid=1)]))
    assert a.update(_res(1.0 + cfg.tap_max_s + 0.05)).gesture == "touch_hold"  # long press
    assert a.update(_res(2.5, [_ev(2.5, "up", (200.0, 200.0), tid=1)])).gesture == "touch_hover"
    a.update(_res(3.0, [_ev(3.0, "down", (200.0, 200.0), tid=2)]))
    assert a.update(_res(3.1, xy=(260.0, 200.0))).gesture == "touch_hold"  # drag (0.6 hand sizes)
    assert a.update(_res(3.2, present=False)).gesture is None


def test_handflow_flicker_and_lost_hand_are_not_clicks():
    a = HandFlowAdapter()
    a.update(_res(0.0, [_ev(0.0, "down", (200.0, 200.0))]))
    assert a.update(_res(0.02, [_ev(0.02, "up", (200.0, 200.0))])).gesture == "touch_hover"
    a.update(_res(1.0, [_ev(1.0, "down", (200.0, 200.0), tid=1)]))
    out = a.update(_res(1.2, [_ev(1.2, "up", (200.0, 200.0), tid=1, lost=True)], present=False))
    assert out.gesture is None
