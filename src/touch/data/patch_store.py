"""Window dataset over the cut fingertip patches (scripts/extract_patches.py).

All patch JPEGs of one source are packed once into a memory-mapped file
(data/patches/<source>_store.bin) so DataLoader workers (spawned on macOS) can read them without
copying gigabytes. A sample is (track k, frame t, finger f) and returns the causal window
t-T+1..t: only past frames.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from touch.data.catalog import Sequence
from touch.data.patches import landmark_features, primary_hand
from touch.data.projection import hand_geometry, random_window_projection

PATCH_ROOT = Path("data/patches")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Track:
    """One (sequence, view): per-frame patch locations, landmark features and labels."""

    seq: Sequence
    view: str
    offset: np.ndarray  # [N, 5] int64 into the store, -1 = no patch
    length: np.ndarray  # [N, 5]
    present: np.ndarray  # [N] bool
    lm_feat: np.ndarray  # [N, F]
    L_px: np.ndarray  # [N, 5]
    contact: np.ndarray  # [N, 5] bool
    valid: np.ndarray  # [N] bool, finger labels known

    def __len__(self) -> int:
        return len(self.present)


def build_store(source: str, seqs: list[Sequence], root: Path = PATCH_ROOT) -> None:
    """Pack the <view>.npz files of `seqs` that still hold `jpeg` into <source>_store.bin +
    <source>_index.npz, appending to an existing store.

    Entries already in the store are kept as they are (their per-sequence `jpeg` arrays were
    dropped by scripts/extract_patches.py after packing); a sequence that has `jpeg` again
    (re-extracted) replaces its old entry. The old bytes stay in the file (no repacking). The
    store is written to a temporary copy and swapped in at the end, so a crash leaves it intact.
    """
    bin_path, idx_path = root / f"{source}_store.bin", root / f"{source}_index.npz"
    entries: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if idx_path.exists():
        with np.load(idx_path) as z:
            starts = np.concatenate([[0], np.cumsum(z["sizes"])])
            for i, k in enumerate(z["keys"]):
                a, b = starts[i], starts[i + 1]
                entries[str(k)] = (z["offset"][a:b], z["length"][a:b])
    tmp = bin_path.with_suffix(".tmp")
    if bin_path.exists():
        shutil.copyfile(bin_path, tmp)
    else:
        tmp.write_bytes(b"")
    pos = tmp.stat().st_size
    with open(tmp, "ab") as fh:
        for s in seqs:
            for v in s.views:
                p = root / s.source / s.participant / s.name / f"{v}.npz"
                if not p.exists():
                    continue
                with np.load(p) as z:
                    if "jpeg" not in z.files:
                        continue  # already packed
                    off = z["offset"].astype(np.int64)
                    off[off >= 0] += pos
                    z["jpeg"].tofile(fh)
                    pos += z["jpeg"].size
                    entries[f"{s.participant}/{s.name}/{v}"] = (off, z["length"])
    assert entries, f"no patches to pack for {source}"
    keys = list(entries)
    idx_tmp = idx_path.with_name(idx_path.stem + ".tmp.npz")
    np.savez(
        idx_tmp,
        keys=np.array(keys),
        sizes=np.array([len(entries[k][0]) for k in keys]),
        offset=np.concatenate([entries[k][0] for k in keys]),
        length=np.concatenate([entries[k][1] for k in keys]),
    )
    tmp.replace(bin_path)
    idx_tmp.replace(idx_path)


def load_tracks(
    seqs: list[Sequence], views: list[str], root: Path = PATCH_ROOT, lm_feats: str = "v1"
) -> list[Track]:
    """Tracks for the given sequences/views. Offsets refer to <source>_store.bin.

    lm_feats "v1" uses the features stored at extraction; any other version is recomputed from
    the sequence's MediaPipe landmarks (same primary hand), so no re-extraction is needed.
    """
    idx = {}
    for source in {s.source for s in seqs}:
        with np.load(root / f"{source}_index.npz") as z:
            starts = np.concatenate([[0], np.cumsum(z["sizes"])])
            for i, k in enumerate(z["keys"]):
                a, b = starts[i], starts[i + 1]
                idx[(source, str(k))] = (z["offset"][a:b], z["length"][a:b])
    tracks = []
    for s in seqs:
        F = s.fingers
        for v in views:
            key = (s.source, f"{s.participant}/{s.name}/{v}")
            if v not in s.views or key not in idx:
                continue
            with np.load(root / s.source / s.participant / s.name / f"{v}.npz") as z:
                present, lm_feat, L_px = z["present"], z["lm_feat"], z["L_px"]
            if lm_feats != "v1":
                lm_feat, _ = landmark_features(primary_hand(s.landmarks(v)), lm_feats)
                assert len(lm_feat) == len(present), key
            off, ln = idx[key]
            tracks.append(
                Track(s, v, off, ln, present, lm_feat, L_px, F["contact"], F["valid"].astype(bool))
            )
    return tracks


def with_lm_feats(tracks: list[Track], lm_feats: str) -> list[Track]:
    """Tracks loaded with v1 features -> the same tracks with `lm_feats` features (for scripts
    that share one track list between models with different feature versions)."""
    if lm_feats == "v1":
        return tracks
    out = []
    for tr in tracks:
        feats, _ = landmark_features(primary_hand(tr.seq.landmarks(tr.view)), lm_feats)
        out.append(replace(tr, lm_feat=feats))
    return out


class PatchReader:
    """Lazy memmap reader (opened inside each worker)."""

    def __init__(self, root: Path = PATCH_ROOT):
        self.root = root
        self._mm: dict[str, np.memmap] = {}

    def get(self, source: str, offset: int, length: int) -> np.ndarray:
        if source not in self._mm:
            self._mm[source] = np.memmap(self.root / f"{source}_store.bin", np.uint8, "r")
        buf = self._mm[source][offset : offset + length]
        img = cv2.imdecode(np.asarray(buf), cv2.IMREAD_COLOR)
        return img[:, :, ::-1]  # BGR -> RGB


def to_tensor(rgb: np.ndarray) -> np.ndarray:
    """[..., H, W, 3] uint8 -> [..., 3, H, W] float32, ImageNet-normalized."""
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.moveaxis(x, -1, -3)


def represent(x: np.ndarray, prev: np.ndarray, input_repr: str) -> np.ndarray:
    """x, prev [..., H, W, 3] in [0, 1] (prev = the frame before x) -> [..., C, H, W] float32.

    rgb: ImageNet-normalized. ndiff: DeepPhys normalized difference (x - prev) / (x + prev),
    scaled x4 to roughly unit range.
    """
    rgb = (x - IMAGENET_MEAN) / IMAGENET_STD
    if input_repr == "rgb":
        out = rgb
    else:
        nd = (x - prev) / (x + prev + 1e-3) * 4.0
        if input_repr == "ndiff":
            out = nd
        elif input_repr == "rgb+ndiff":
            out = np.concatenate([rgb, nd], -1)
        else:
            raise ValueError(input_repr)
    return np.moveaxis(out, -1, -3).astype(np.float32)


IN_CHANS = {"rgb": 3, "ndiff": 3, "rgb+ndiff": 6}


class WindowDataset(Dataset):
    """Samples (track, t, finger) -> causal window of T frames.

    Returns dict: img [T, C, H, W] float32, lm [T, F], present [T] bool, finger int,
    label [T] float32, label_mask [T] bool.
    """

    def __init__(
        self,
        tracks: list[Track],
        T: int,
        index: np.ndarray,
        input_repr: str = "rgb",
        augment: bool = False,
        size: int = 112,
        seed: int = 0,
        load_images: bool = True,
        proj_aug: float = 0.0,
    ):
        self.tracks, self.T, self.index = tracks, T, index
        self.load_images = load_images
        self.proj_aug = proj_aug  # probability of simulated projector light (touch.data.projection)
        self._geo: dict[int, tuple[np.ndarray, tuple[int, int]]] = {}
        self.input_repr, self.augment, self.size = input_repr, augment, size
        self.reader = PatchReader()
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.index)

    def _patch(self, tr: Track, i: int, f: int) -> np.ndarray | None:
        if i < 0 or tr.offset[i, f] < 0:
            return None
        return self.reader.get(tr.seq.source, int(tr.offset[i, f]), int(tr.length[i, f]))

    def __getitem__(self, j: int) -> dict:
        k, t, f = (int(v) for v in self.index[j])
        tr = self.tracks[k]
        T, S = self.T, self.size
        raw = np.zeros((T + 1, S, S, 3), np.uint8)  # row 0 = frame t-T (only for ndiff)
        present = np.zeros(T, bool)
        lm = np.zeros((T, tr.lm_feat.shape[1]), np.float32)
        label = np.zeros(T, np.float32)
        mask = np.zeros(T, bool)
        src = np.full(T + 1, -1)  # frame each row's patch came from (-1 = none)
        last, last_i = None, -1
        for i in range(t - T, max(-1, t - 2 * T - 1), -1):  # newest patch at or before t-T
            if i >= 0 and tr.offset[i, f] >= 0:
                last = self._patch(tr, i, f) if self.load_images else None
                last_i = i
                break
        if last is not None and self.load_images:
            raw[0], src[0] = last, last_i
        for w in range(T):
            i = t - T + 1 + w
            p = self._patch(tr, i, f) if self.load_images else None
            if p is not None or (not self.load_images and i >= 0 and tr.offset[i, f] >= 0):
                present[w], lm[w] = True, tr.lm_feat[i]
                if p is not None:
                    raw[w + 1], src[w + 1] = p, i
                last, last_i = p, i
            elif last is not None:
                raw[w + 1], src[w + 1] = last, last_i  # hold the last seen patch (causal)
            if i >= 0 and tr.valid[i]:
                label[w], mask[w] = float(tr.contact[i, f]), present[w]
        x = raw.astype(np.float32) / 255.0
        if self.augment and self.load_images and self.rng.random() < self.proj_aug:
            if k not in self._geo:
                self._geo[k] = hand_geometry(tr.seq, tr.view)
            geo, hw = self._geo[k]
            x = random_window_projection(self.rng, x, src, geo, hw, f)
        if self.augment:
            x = self._color_jitter(x)
            if self.rng.random() < 0.5:
                x = x[:, :, ::-1]
        img = represent(x[1:], x[:-1], self.input_repr)
        return {
            "img": torch.from_numpy(np.ascontiguousarray(img)),
            "lm": torch.from_numpy(lm),
            "present": torch.from_numpy(present),
            "finger": f,
            "label": torch.from_numpy(label),
            "label_mask": torch.from_numpy(mask),
        }

    def _color_jitter(self, x: np.ndarray) -> np.ndarray:
        # same jitter for the whole window, so temporal colour change is preserved
        b = self.rng.uniform(0.75, 1.25)
        c = self.rng.uniform(0.75, 1.25)
        gains = self.rng.uniform(0.92, 1.08, size=3).astype(np.float32)  # white balance
        m = x.mean()
        x = ((x - m) * c + m) * b * gains
        return np.clip(x, 0, 1)


def all_index(tracks: list[Track], fingers=range(5)) -> np.ndarray:
    """Every (track, t, finger) with a patch and a known label at t. [M, 3] int64."""
    rows = []
    for k, tr in enumerate(tracks):
        ts = np.flatnonzero(tr.present & tr.valid)
        for f in fingers:
            rows.append(np.stack([np.full_like(ts, k), ts, np.full_like(ts, f)], 1))
    return np.concatenate(rows).astype(np.int64)


def sample_weights(
    tracks: list[Track], index: np.ndarray, near: int = 3, hover_weight: float = 1.0
) -> np.ndarray:
    """Down-weight fingers that never touch in an ordinary press sequence (easy negatives);
    up-weight frames within +-`near` frames of a contact change (where timing is decided);
    multiply hover-only (no-contact) sequences by `hover_weight` (hard negatives)."""
    w = np.ones(len(index), np.float64)
    for k, tr in enumerate(tracks):
        sel = index[:, 0] == k
        c = tr.contact & tr.valid[:, None]
        near_change = np.zeros_like(c)
        d = np.diff(c.astype(np.int8), axis=0) != 0
        for s in range(-near, near + 1):
            sh = np.roll(np.vstack([d, np.zeros((1, 5), bool)]), s, axis=0)
            near_change |= sh
        never = ~c.any(axis=0)
        no_contact_seq = tr.seq.meta.get("variant") == "no-contact"
        ts, fs = index[sel, 1], index[sel, 2]
        wk = np.ones(len(ts))
        if not no_contact_seq:
            wk[never[fs]] = 0.25
        else:
            wk *= hover_weight
        wk[near_change[ts, fs]] *= 3.0
        w[sel] = wk
    return w
