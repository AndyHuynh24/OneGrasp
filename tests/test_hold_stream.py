"""Live HoldStream features == the offline study's features (one real HO-Cap val track)."""

import sys
from pathlib import Path

import numpy as np
import pytest

RUN = "g6_grasp_table_20261112-101602"
NEED = [Path("runs/grab_fusion") / RUN / "hocap_val_hand.npz",
        Path("runs/hocap_val") / RUN / "preds.npz"]  # fmt: skip


@pytest.mark.skipif(not all(p.exists() for p in NEED), reason="needs the local fusion study files")
def test_stream_matches_offline():
    import cv2

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import hold_study

    from touch.data import hocap as H
    from touch.handpath import world_hand
    from touch.interact.hold_stream import HoldStream

    va = hold_study.load(RUN, "hocap_val")
    names = [n for n in HoldStream.all_names() if n != "palm_z"]
    cols = [va["names"].index(n) for n in names]
    t = 0
    idx = np.flatnonzero(va["tid"] == t)[:200]
    key = str(va["keys"][t])
    _, name, cam, side = key.split("/")
    P = np.load(NEED[1])
    seq = next(s for s in H.list_sequences(video_only=True, split="val") if s.name == name)
    lm = P[f"{key}/lm"]
    wh = world_hand(P[f"{key}/joints"], P[f"{key}/verts"].astype(np.float32), lm,
                    H.intrinsics(cam), H.cam_to_world(cam), side == "left", smooth_k=1)  # fmt: skip
    hs = HoldStream(names)
    rows = []
    for i in range(len(idx)):
        pres = bool(np.isfinite(lm[i]).all())
        bgr = cv2.imread(str(seq.image(cam, i)))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rows.append(hs.step(P[f"{key}/state"][i], P[f"{key}/finger"][i], pres,
                            P[f"{key}/joints"][i], lm[i], gray, wh["joints"][i], bgr))  # fmt: skip
    live = np.stack(rows)
    off = va["X"][idx][:, cols]
    assert live.shape == off.shape == (len(idx), len(names))
    bad = [(n, int(np.sum(~np.isclose(live[:, k], off[:, k], rtol=1e-4, atol=1e-4,
                                      equal_nan=True))))
           for k, n in enumerate(names)]  # fmt: skip
    bad = [b for b in bad if b[1]]
    assert not bad, bad
