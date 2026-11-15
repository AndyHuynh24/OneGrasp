#!/usr/bin/env bash
# Download the trained weights used by the live demos (GitHub release "weights-v1") into place:
#   runs/g6_grasp_table_20261112-101602/      grasp model (live_grasp.py, live_sim.py)
#   runs/c4_mnv3L_gru_proj_lm2_s0_20261107-144246/  touch model (live_touch.py)
#   runs/grab_fusion/g6_grasp_table_20261112-101602/ learned grab gate (live_sim.py default)
#   checkpoints/mediapipe/hand_landmarker.task  MediaPipe hands (Apache 2.0)
#   checkpoints/pv2/paper_29.pth               PressureVision++ teacher (MIT)
# Not included: MANO (licence forbids redistribution; register at mano.is.tue.mpg.de and put
# MANO_RIGHT.pkl / MANO_LEFT.pkl in ~/.cache/fah/mano). The grasp model's ONNX file is rebuilt
# from best.pt on first run. Needs the GitHub CLI logged in (the repo is private).
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
gh release download weights-v1 --repo AndyHuynh24/OneTouch --dir "$tmp" \
  --pattern onetouch_weights_v1.tar.gz --pattern SHA256SUMS
tar xzf "$tmp/onetouch_weights_v1.tar.gz"
shasum -a 256 -c "$tmp/SHA256SUMS"
rm -rf "$tmp"
