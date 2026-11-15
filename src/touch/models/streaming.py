"""Live (frame-by-frame) inference for TouchNet, matching the training/eval windows exactly.

Training and touch.eval.run score frame t with the GRU run from a zero state over the window
t-T+1..t. Live, the encoder runs once per new frame and its features are kept in a history of the
last T-1 frames; the (tiny) GRU is then re-run over that window, so live scores equal eval scores.
Before the first frame the history holds the "pad" feature (zero image, no landmarks, present=0),
as in touch.eval.run.predict_track.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from touch.data.patch_store import IN_CHANS, represent
from touch.data.patches import PATCH_SIZE

INPUTS = ["img", "lm", "present", "finger", "hist"]


class StreamingStep(nn.Module):
    """One live frame for B fingers.

    img [B, C, S, S], lm [B, model.n_lm], present [B] bool, finger [B] long,
    hist [B, T-1, d] (features of the previous T-1 frames, oldest first)
    -> prob [B] (touch probability now), feat [B, d] (append to hist for the next frame).
    """

    def __init__(self, model):
        super().__init__()
        if not model.causal:
            raise ValueError(
                f"temporal={model.temporal!r} is the offline (non-causal) model: it needs future "
                "frames and cannot run live; use scripts/annotate_video.py on recorded video"
            )
        self.model = model

    def forward(self, img, lm, present, finger, hist):
        f = self.model.frame_features(img[:, None], lm[:, None], present[:, None], finger)
        logit = self.model.temporal_head(torch.cat([hist, f], 1))[:, -1]
        return torch.sigmoid(logit), f[:, 0]


@torch.no_grad()
def pad_features(model, input_repr: str, fingers: torch.Tensor) -> torch.Tensor:
    """Feature of a frame before the sequence start, per finger -> [B, d]."""
    zero = np.zeros((len(fingers), PATCH_SIZE, PATCH_SIZE, 3), np.float32)
    img = torch.from_numpy(represent(zero, zero, input_repr)).float()
    lm = torch.zeros(len(fingers), model.n_lm)
    present = torch.zeros(len(fingers), dtype=torch.bool)
    return model.frame_features(img[:, None], lm[:, None], present[:, None], fingers)[:, 0]


def example_inputs(model, input_repr: str, T: int, batch: int = 5) -> tuple[torch.Tensor, ...]:
    fingers = torch.arange(batch) % 5
    pad = pad_features(model, input_repr, fingers)
    return (
        torch.rand(batch, IN_CHANS[input_repr], PATCH_SIZE, PATCH_SIZE),
        torch.randn(batch, model.n_lm),
        torch.ones(batch, dtype=torch.bool),
        fingers,
        pad[:, None].expand(-1, T - 1, -1).contiguous(),
    )


@torch.no_grad()
def export_onnx(model, input_repr: str, T: int, path: Path) -> None:
    """Export StreamingStep to ONNX with a fixed batch of 5 fingers (the GRU export bakes in its
    zero initial state, so the batch size cannot change at run time)."""
    step = StreamingStep(model).eval()
    torch.onnx.export(
        step,
        example_inputs(model, input_repr, T),
        str(path),
        input_names=INPUTS,
        output_names=["prob", "feat"],
        opset_version=17,
        dynamo=False,
    )
