"""Per-fingertip touch model: patch encoder (+ landmark MLP) -> optional GRU -> logit.

temporal="none" is a single-frame model (EgoTouch-style, C2); temporal="gru" is C4 (per-frame
encoder + causal GRU): input [B, T, C, H, W], output logits [B, T], step t only sees frames <= t.
temporal="bigru" is the OFFLINE mode for recorded video only (approved separately from the causal
live model): a bidirectional GRU, so step t also sees the later frames of its window. It is scored
at window position window_target(T) = T // 2, i.e. with T // 2 past and T - 1 - T // 2 future
frames, and cannot be streamed live (touch.models.streaming refuses it).
img_drop: training-only modality dropout -- zero the image embedding of a whole window with this
probability, so the head cannot ignore the landmark branch.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from touch.data.patches import N_LM_FEAT


class TouchNet(nn.Module):
    def __init__(
        self,
        backbone: str = "mobilenetv3_small_100.lamb_in1k",
        pretrained: bool = True,
        in_chans: int = 3,
        temporal: str = "gru",
        use_landmarks: bool = True,
        use_image: bool = True,
        emb: int = 128,
        hidden: int = 128,
        dropout: float = 0.2,
        n_lm: int = N_LM_FEAT,
        lm_emb: int = 32,
        img_drop: float = 0.0,
    ):
        super().__init__()
        self.temporal, self.use_landmarks, self.use_image = temporal, use_landmarks, use_image
        self.n_lm, self.img_drop = n_lm, img_drop
        d = 0
        if use_image:
            self.encoder = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0, in_chans=in_chans
            )
            with torch.no_grad():  # pooled width (num_features is pre-head for MobileNetV3)
                width = self.encoder.eval()(torch.zeros(1, in_chans, 112, 112)).shape[-1]
            self.encoder.train()
            self.proj = nn.Sequential(nn.Linear(width, emb), nn.ReLU(), nn.Dropout(dropout))
            d += emb
        if use_landmarks:
            self.lm_mlp = nn.Sequential(
                nn.Linear(n_lm + 5, 2 * lm_emb), nn.ReLU(), nn.Linear(2 * lm_emb, lm_emb), nn.ReLU()
            )
            d += lm_emb
        d += 1  # present flag
        if temporal == "gru":
            self.gru = nn.GRU(d, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)
        elif temporal == "bigru":
            self.gru = nn.GRU(d, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * hidden, 1)
        elif temporal == "none":
            self.head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            raise ValueError(temporal)

    @property
    def causal(self) -> bool:
        return self.temporal != "bigru"

    def window_target(self, T: int) -> int:
        """Window position whose output scores a frame: the last one for causal models."""
        return T - 1 if self.causal else T // 2

    def frame_features(self, img, lm, present, finger) -> torch.Tensor:
        """Per-frame features [B, T, d] (no temporal mixing)."""
        B, T = present.shape
        parts = []
        if self.use_image:
            e = self.proj(self.encoder(img.flatten(0, 1))).view(B, T, -1)
            if self.training and self.img_drop > 0 and self.use_landmarks:
                keep = torch.rand(B, 1, 1, device=e.device) >= self.img_drop
                e = e * keep.to(e.dtype)
            parts.append(e)
        if self.use_landmarks:
            oh = nn.functional.one_hot(finger, 5).float()[:, None].expand(B, T, 5)
            parts.append(self.lm_mlp(torch.cat([lm, oh], -1)) * present[..., None])
        parts.append(present[..., None].float())
        return torch.cat(parts, -1)

    def temporal_head(self, feats: torch.Tensor) -> torch.Tensor:
        if self.temporal in ("gru", "bigru"):
            feats, _ = self.gru(feats)
        return self.head(feats).squeeze(-1)

    def forward(self, img, lm, present, finger) -> torch.Tensor:
        return self.temporal_head(self.frame_features(img, lm, present, finger))
