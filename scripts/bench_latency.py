"""CPU latency of a trained TouchNet in streaming use, run with ONNX Runtime (deployment path).

Streaming cost per frame = encoder on one 112x112 patch per finger + landmark MLP + one GRU step
(the GRU hidden state is carried, so no window recompute). The step is exported to ONNX and timed
with onnxruntime on CPU; plain PyTorch eager is much slower on CPU for these small networks and
is not what we would deploy. MediaPipe is not included (measure separately). Writes
<run>/latency.json and <run>/step.onnx. Run on an idle machine (no training in the background).

Usage:
    python scripts/bench_latency.py runs/<run>/best.pt [--threads 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from touch.train import build_model  # noqa: E402

NAMES = ["img", "lm", "present", "finger", "h"]


class Step(torch.nn.Module):
    """One streaming step: [B, C, 112, 112] patches + GRU state -> touch prob [B], new state."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img, lm, present, finger, h):
        m = self.model
        f = m.frame_features(img[:, None], lm[:, None], present[:, None], finger)
        if m.temporal == "gru":
            f, h = m.gru(f, h)
        return torch.sigmoid(m.head(f)).flatten(), h


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model({**cfg, "pretrained": False}).eval()
    model.load_state_dict(ck["model"])
    step = Step(model).eval()
    c = 6 if cfg["input_repr"] == "rgb+ndiff" else 3
    hidden = model.gru.hidden_size if model.temporal == "gru" else 1
    res = {"params": sum(p.numel() for p in model.parameters()), "threads": args.threads}
    onnx_path = Path(args.ckpt).parent / "step.onnx"
    for batch in (5, 1):  # all 5 fingers of a hand in one batch, or 1 finger
        inp = (
            torch.rand(batch, c, 112, 112),
            torch.randn(batch, model.n_lm),
            torch.ones(batch, dtype=torch.bool),
            torch.arange(batch) % 5,
            torch.zeros(1, batch, hidden),
        )
        torch.onnx.export(
            step,
            inp,
            onnx_path,
            input_names=NAMES,
            output_names=["p", "h_out"],
            dynamic_axes={k: {0: "B"} for k in NAMES[:4]} | {"h": {1: "B"}},
            opset_version=17,
            dynamo=False,
        )
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        feed = {k: t.numpy() for k, t in zip(NAMES, inp, strict=True)}
        diff = np.abs(sess.run(None, feed)[0] - step(*inp)[0].numpy()).max()
        assert diff < 1e-4, f"ONNX output differs from PyTorch by {diff}"
        ts = []
        for i in range(args.iters + 20):
            t0 = time.perf_counter()
            sess.run(None, feed)
            if i >= 20:
                ts.append(time.perf_counter() - t0)
        res[f"cpu_ms_batch{batch}"] = round(1000 * float(np.median(ts)), 3)
    res["cpu_ms_per_finger_frame"] = res["cpu_ms_batch1"]
    res["runtime"] = f"onnxruntime {ort.__version__} CPU"
    out = Path(args.ckpt).parent / "latency.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
