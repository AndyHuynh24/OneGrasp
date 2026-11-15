"""Live (streaming) version of every HOLD feature of the offline study (scripts/hold_study.py):
the grasp model's probabilities, 3D hand motion, grasp shape (MANO + MediaPipe), 2D motion and
optical-flow co-motion. Keeps the last WINDOW frames and recomputes the batch definitions on them,
so live values equal the offline ones (tests/test_hold_stream.py)."""

from __future__ import annotations

from collections import deque

import numpy as np

from touch.interact.grab_gate import prob_features, rolling
from touch.interact.hand_features import (
    COMOTION_NAMES,
    MOTION3D_NAMES,
    POSE_NAMES,
    TIPCOLOR_NAMES,
    CoMotion,
    image_motion,
    motion3d,
    pose_features,
    tip_color,
)

WINDOW = 31  # longest window (30) + 1
TIP_WINDOW = 90 + 15  # tip_da_rel90: 15-frame mean vs its max over 90 frames
ROLL15 = ("co_frac", "mano_aperture_min", "mano_curl_index")


class HoldStream:
    """step(...) -> feature row [F] in the order of `names` (a subset of all_names())."""

    def __init__(self, names: list[str]):
        self.names = list(names)
        self.need_flow = any(n.startswith(("co_", "still_frac", "hand_flow")) for n in names)
        self.need_tip = any(n.startswith("tip_") for n in names)
        self.reset()

    @staticmethod
    def all_names() -> list[str]:
        p = prob_features(np.zeros((1, 3)), np.zeros((1, 5)), np.ones(1))[1]
        return (p + MOTION3D_NAMES + [f"mano_{n}" for n in POSE_NAMES]
                + [f"mp_{n}" for n in POSE_NAMES] + ["img_vx", "img_vy_up", "img_speed"]
                + COMOTION_NAMES + [f"{c}_m15" for c in ROLL15] + TIPCOLOR_NAMES
                + ["tip_da_mean_m15", "tip_da_rel90"])  # fmt: skip

    def reset(self) -> None:
        self.buf: deque = deque(maxlen=WINDOW)
        self.flow = CoMotion()
        self.tip: deque = deque(maxlen=TIP_WINDOW)

    def frame_features(self, present: bool, lm=None, gray=None, bgr=None) -> dict:
        """The image-dependent features (optical-flow co-motion, fingertip colour) of one frame,
        in frame order. A recorder computes these live (cheap) and calls step(..., co=, tc=)
        later without the frames: same rows as step(..., gray=, bgr=)."""
        lm_ = np.asarray(lm) if present and lm is not None else None
        co = (self.flow.step(gray, lm_) if self.need_flow and gray is not None
              else np.full(len(COMOTION_NAMES), np.nan))  # fmt: skip
        tc = (tip_color(bgr, lm_) if self.need_tip and bgr is not None
              else np.full(len(TIPCOLOR_NAMES), np.nan, np.float32))  # fmt: skip
        return {"co": co, "tc": tc}

    def step(self, state, finger, present: bool, joints_pred=None, lm=None, gray=None,
             joints_world=None, bgr=None, co=None, tc=None) -> np.ndarray:  # fmt: skip
        """state [3], finger [5], hand present, predicted MANO joints [21, 3] (wrist-relative),
        MediaPipe landmarks [21, 3] px, grayscale frame, world joints [21, 3] (table frame),
        colour frame (BGR, for the fingertip colour features). co / tc: precomputed
        frame_features (then gray / bgr are not used)."""
        nan21 = np.full((21, 3), np.nan)

        def arr(x):
            return nan21 if (x is None or not present) else np.asarray(x, np.float64)

        if co is None:
            ff = self.frame_features(present, lm, gray, bgr)
            co, tc = ff["co"], ff["tc"]
        self.buf.append((np.asarray(state, float), np.asarray(finger, float), bool(present),
                         arr(joints_pred), arr(lm), arr(joints_world), co))  # fmt: skip
        st, fi, pr, jp, lmb, jw, cob = (np.array(x) for x in zip(*self.buf, strict=True))
        P, pnames = prob_features(st, fi, pr)
        lmm = lmb.copy()
        im, imn = image_motion(lmm)
        cols = {**dict(zip(pnames, P[-1], strict=True)),
                **dict(zip(MOTION3D_NAMES, motion3d(jw)[-1], strict=True)),
                **dict(zip([f"mano_{n}" for n in POSE_NAMES], pose_features(jp[-1:])[0],
                           strict=True)),
                **dict(zip([f"mp_{n}" for n in POSE_NAMES], pose_features(lmm[-1:])[0],
                           strict=True)),
                **dict(zip(imn, im[-1], strict=True)),
                **dict(zip(COMOTION_NAMES, cob[-1], strict=True))}  # fmt: skip
        cols["co_frac_m15"] = rolling(cob[:, :1], 15, np.nanmean)[-1, 0]
        self.tip.append(tc)
        cols.update(zip(TIPCOLOR_NAMES, tc, strict=True))
        da = np.array(self.tip)[:, TIPCOLOR_NAMES.index("tip_da_mean"), None]
        m15 = rolling(da, 15, np.nanmean)
        cols["tip_da_mean_m15"] = m15[-1, 0]
        cols["tip_da_rel90"] = m15[-1, 0] - rolling(m15, 90, np.nanmax)[-1, 0]
        for c in ROLL15[1:]:
            src = pose_features(jp[-15:])[:, POSE_NAMES.index(c[len("mano_") :])]
            cols[f"{c}_m15"] = rolling(src[:, None], 15, np.nanmean)[-1, 0]
        return np.array([cols[n] for n in self.names], np.float32)
