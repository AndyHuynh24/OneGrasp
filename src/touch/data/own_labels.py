"""Labels for our own recordings (scripts/collect_labels.py): key presses held during recording
become per-frame labels.

Session folder (data/own/<task>/<session>/):
  video.mp4      camera frames as captured (frame i = row i of frames.npz)
  frames.npz     t [N] float64: time.perf_counter() when frame i was read from the camera
  events.json    key edges [{"t", "key", "down"}] on the same clock, plus "bad" marks
  meta.json      task, finger, participant, camera, latency_s, shifts, dropped, reviewed ...
  labels.npz     label [N] int8, valid [N] bool, edge_dist [N] int16 (frames to the nearest
                 label change; mask a few frames around edges if timing matters)

Tasks and keys (hold the key exactly while it is true):
  touch   F = the chosen finger (meta.finger) touches the surface (tap / short touch),
          H = it presses and stays on the surface (hold / long press); H wins if both are down
          labels 0 hover / none, 1 touch, 2 hold (both are "in contact" for a binary model)
  grasp   F = the hand touches the object (first contact .. hand leaves it),
          H = the object is in the hand AND off the table (lift-off .. set down)
          labels 0 none, 1 touch, 2 hold; same definition as the DexYCB / HO-Cap training
          labels (hold = grasped and lifted; gripped on the table = touch)
          (hold wins if both are down)
  X (any task) marks a mistake: frames within BAD_MARGIN_S of it are invalid.

A frame was exposed about latency_s before it was read, so frame i gets the key state at
t[i] - latency_s. latency_s is unknown at recording time (0); set it in the review mode, where
single edges can also be shifted (meta.shifts, by edge index) or whole presses dropped.
"""

from __future__ import annotations

import numpy as np

# "space" = the first version's key (sessions recorded before 2026-11-13 evening), same meaning as h
FINGERS = ("thumb", "index", "middle", "ring", "pinky")  # labels.npz "finger" index
TASKS = {"touch": {"f": 1, "h": 2, "space": 1}, "grasp": {"h": 2, "space": 2, "f": 1}}
LABEL_NAMES = {"touch": ["none", "touch", "hold"], "grasp": ["none", "touch", "hold"]}
BAD_MARGIN_S = 0.5


def intervals(events: list[dict], t_end: float) -> list[dict]:
    """Key edges -> presses [{"key", "t0", "t1", "i0", "i1"}] (i0 / i1: indices of the edges in
    events; i1 None if still held at the end -> t1 = t_end). Repeated downs are ignored."""
    out, open_ = [], {}
    for i, e in enumerate(events):
        k = e["key"]
        if k in ("bad", "finger"):
            continue
        if e["down"] and k not in open_:
            open_[k] = (i, e["t"])
        elif not e["down"] and k in open_:
            i0, t0 = open_.pop(k)
            out.append({"key": k, "t0": t0, "t1": e["t"], "i0": i0, "i1": i})
    for k, (i0, t0) in open_.items():
        out.append({"key": k, "t0": t0, "t1": t_end, "i0": i0, "i1": None})
    return sorted(out, key=lambda r: r["t0"])


def frame_labels(t: np.ndarray, events: list[dict], task: str, latency_s: float = 0.0,
                 shifts: dict | None = None, dropped: list | None = None,
                 release_pad_s: dict | None = None,
                 invalid: list | None = None, finger: str | None = None) -> dict:  # fmt: skip
    """t [N] frame read times -> {"label" [N] int8, "valid" [N] bool, "edge_dist" [N] int16}.
    shifts: {edge index (str or int): seconds added to that edge}; dropped: edge indices whose
    press is dropped (its frames become invalid); release_pad_s: {key: s} frames from each
    release of that key to s later are invalid (the key was let go before the hand let go);
    invalid: [[t0, t1], ...] time ranges (events clock) that are invalid.
    finger (touch task): the finger labelled at the start; {"key": "finger", "finger": name}
    events switch it -> "finger" [N] int8 (FINGERS index; -1 for the grasp task): the finger
    that label 1 refers to (the other fingers are taken as not touching)."""
    t = np.asarray(t, np.float64)
    n = len(t)
    keys = TASKS[task]
    shifts = {int(k): float(v) for k, v in (shifts or {}).items()}
    dropped = {int(i) for i in (dropped or [])}
    te = t - latency_s  # exposure time of each frame
    label = np.zeros(n, np.int8)
    valid = np.ones(n, bool)
    edges, pads = [], []
    t_end = float(t[-1]) + 1.0 if n else 0.0
    for r in intervals(events, t_end):
        if r["key"] not in keys:
            continue
        t0 = r["t0"] + shifts.get(r["i0"], 0.0)
        t1 = r["t1"] + (shifts.get(r["i1"], 0.0) if r["i1"] is not None else 0.0)
        inside = (te >= t0) & (te < t1)
        if r["i0"] in dropped or (r["i1"] is not None and r["i1"] in dropped):
            valid &= ~inside
            continue
        label[inside] = np.maximum(label[inside], keys[r["key"]])
        edges += [t0] + ([t1] if r["i1"] is not None else [])
        pad = (release_pad_s or {}).get(r["key"], 0.0)
        if pad and r["i1"] is not None:
            pads.append((t1, t1 + pad))
    for a, b in pads + [tuple(x) for x in invalid or []]:
        valid &= ~((te >= a) & (te < b))
    for e in events:
        if e["key"] == "bad":
            valid &= np.abs(te - e["t"]) > BAD_MARGIN_S
    edge_dist = np.full(n, np.iinfo(np.int16).max, np.int16)
    if edges and n:
        k = np.searchsorted(te, np.array(edges))  # first frame at / after each edge
        idx = np.arange(n)
        d = np.min(np.abs(idx[:, None] - k[None, :]), axis=1)
        edge_dist = np.minimum(d, np.iinfo(np.int16).max).astype(np.int16)
    fing = np.full(n, -1, np.int8)
    if task == "touch":
        sw = [(-np.inf, finger or "index")] + [(e["t"], e["finger"]) for e in events
                                              if e["key"] == "finger"]  # fmt: skip
        for (ta, name), (tb, _) in zip(sw, sw[1:] + [(np.inf, None)], strict=True):
            fing[(te >= ta) & (te < tb)] = FINGERS.index(name)
    return {"label": label, "valid": valid, "edge_dist": edge_dist, "finger": fing}


def labels_from_meta(t: np.ndarray, events: list[dict], meta: dict) -> dict:
    """frame_labels with the corrections stored in a session's meta.json."""
    return frame_labels(
        t,
        events,
        meta["task"],
        meta.get("latency_s", 0.0),
        meta.get("shifts"),
        meta.get("dropped"),
        meta.get("release_pad_s"),
        meta.get("invalid"),
        meta.get("finger"),
    )
