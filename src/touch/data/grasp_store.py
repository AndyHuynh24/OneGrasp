"""Window dataset for the grasp model over scripts/prepare_dexycb.py output.

A track is one (sequence, camera). A sample is (track k, frame t): the causal window t-T+1..t
of 5 fingertip patches + 1 hand crop per frame (past frames only), with labels:
finger contact [T, 5], hand state [T] (0 none, 1 touch, 2 hold), per-vertex contact [T, 778]
(-1 = ignore) and the frame-t mesh (root-relative MANO vertices / joints, right-hand canonical).
Big per-frame arrays (vertices, vertex labels) live in memory-mapped caches so DataLoader
workers do not each copy them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from touch.data import dexycb as D
from touch.data.patch_store import IMAGENET_MEAN, IMAGENET_STD

PROC = Path("data/dexycb_proc")
PATCH_ROOT = Path("data/patches")
N_SLOTS = 6  # 5 fingertip patches + hand crop


LABEL_FILES = ("table", "hold")  # extra per-sequence label files next to the track npz files


@dataclass
class GraspTrack:
    subject: str
    seq: str
    cam: str
    split: str
    offset: np.ndarray  # [N, 6] into dexycb_store.bin, -1 = none
    length: np.ndarray  # [N, 6]
    present: np.ndarray  # [N] bool, MediaPipe hand used
    lm_feat: np.ndarray  # [N, F]
    finger: np.ndarray  # [N, 5] int8, 1 / 0 / -1 ignore
    state: np.ndarray  # [N] int8, 0 / 1 / 2 / -1 ignore
    valid: np.ndarray  # [N] bool, DexYCB hand pose annotated (mesh target exists)
    start: int  # first row of this track in the mesh / vertex-label caches
    flip: bool
    mp_right: np.ndarray  # [N] int8
    store: str = "dexycb"  # data/patches/<store>_store.bin; start < 0 = no mesh / vertex labels
    table: np.ndarray | None = None  # [N] int8 hand touches the table 1 / 0 / -1 ignore

    def table_or_unknown(self) -> np.ndarray:
        return self.table if self.table is not None else np.full(len(self), -1, np.int8)

    def __len__(self) -> int:
        return len(self.present)


def _cache_paths(root: Path) -> tuple[Path, Path, Path]:
    return root / "dexycb_mesh.npy", root / "dexycb_vlab.npy", root / "dexycb_cache_keys.txt"


def load_grasp_tracks(
    split: str, proc: Path = PROC, root: Path = PATCH_ROOT, hold_rule: str = "thumb"
) -> tuple[list[GraspTrack], np.memmap, np.memmap]:
    """Tracks of one DexYCB s1 split + memory-mapped caches:
    mesh [R, 799, 3] float16 (778 vertices then 21 joints, metres) and vlab [R, 778] int8.
    The caches cover every prepared (sequence, camera) and are rebuilt if new ones appear.
    hold_rule: "thumb" = the stored state label; "any2" = recomputed from the stored distances
    (dexycb.labels_from_dist)."""
    files = sorted(p for p in proc.glob("*/*/*.npz")
                   if p.name != "contact.npz" and not p.name.startswith(LABEL_FILES))  # fmt: skip
    files = [p for p in files if (p.parent / "done").exists()]
    keys = [str(p.relative_to(proc)) for p in files]
    mesh_p, vlab_p, keys_p = _cache_paths(root)
    if not keys_p.exists() or keys_p.read_text().split("\n") != keys:
        _build_cache(files, mesh_p, vlab_p)
        keys_p.write_text("\n".join(keys))
    mesh = np.load(mesh_p, mmap_mode="r")
    vlab = np.load(vlab_p, mmap_mode="r")
    tracks, start = [], 0
    for p in files:
        subject, seq = p.parent.parent.name, p.parent.name
        with np.load(p) as z:
            n = len(z["present"])
            assert "jpeg" not in z.files, f"{p} not packed: run scripts/prepare_dexycb.py"
            if D.split_of(subject) == split:
                with np.load(p.parent / "contact.npz") as c:
                    finger, state, valid = c["finger"], c["state"], c["valid"]
                    if hold_rule != "thumb":
                        state = D.labels_from_dist(c["part_dist"], c["obj_disp"], valid,
                                                   hold_rule=hold_rule)["state"]  # fmt: skip
                tp = p.parent / "table.npz"  # scripts/make_table_labels.py
                table = np.load(tp)["table"] if tp.exists() else None
                assert table is None or len(table) == n, tp
                tracks.append(
                    GraspTrack(subject, seq, p.stem, split, z["offset"], z["length"],
                               z["present"], z["lm_feat"], finger, state, valid, start,
                               bool(z["flip"]), z["mp_right"], table=table)
                )  # fmt: skip
        start += n
    return tracks, mesh, vlab


def _build_cache(files: list[Path], mesh_p: Path, vlab_p: Path) -> None:
    sizes = []
    for p in files:
        with np.load(p) as z:
            sizes.append(len(z["present"]))
    R = int(sum(sizes))
    mesh = np.lib.format.open_memmap(mesh_p.with_suffix(".tmp.npy"), "w+", np.float16, (R, 799, 3))
    vlab = np.lib.format.open_memmap(vlab_p.with_suffix(".tmp.npy"), "w+", np.int8, (R, 778))
    r = 0
    for p, n in zip(files, sizes, strict=True):
        with np.load(p) as z, np.load(p.parent / "contact.npz") as c:
            mesh[r : r + n, :778] = z["verts"]
            mesh[r : r + n, 778:] = z["joints"]
            vlab[r : r + n] = c["vert"]
        r += n
    mesh.flush()
    vlab.flush()
    del mesh, vlab
    mesh_p.with_suffix(".tmp.npy").replace(mesh_p)
    vlab_p.with_suffix(".tmp.npy").replace(vlab_p)


def grasp_index(tracks: list[GraspTrack]) -> np.ndarray:
    """Every (track, t) with a hand and some known label at t -> [M, 2] int64."""
    rows = []
    for k, tr in enumerate(tracks):
        known = (tr.state >= 0) | (tr.finger >= 0).any(1)
        ts = np.flatnonzero(tr.present & known)
        rows.append(np.stack([np.full_like(ts, k), ts], 1))
    return np.concatenate(rows).astype(np.int64)


def grasp_weights(tracks: list[GraspTrack], index: np.ndarray, near: int = 3) -> np.ndarray:
    """Balance hand states (inverse frequency over the index) and up-weight frames within
    +-`near` frames of a state or finger-contact change (where timing is decided) x3."""
    st = np.array([tracks[k].state[t] for k, t in index])
    freq = {s: max((st == s).mean(), 1e-6) for s in (-1, 0, 1, 2)}
    w = np.array([1.0 / freq[s] for s in st])
    for k, tr in enumerate(tracks):
        lab = np.concatenate([tr.state[:, None], tr.finger], 1).astype(np.int16)
        ch = np.zeros(len(tr), bool)
        ch[1:] = (np.diff(lab, axis=0) != 0).any(1)
        near_ch = np.convolve(ch, np.ones(2 * near + 1), "same") > 0
        sel = index[:, 0] == k
        w[sel] *= np.where(near_ch[index[sel, 1]], 3.0, 1.0)
    return w / w.mean()


EGOP_PROC = Path("data/egop_grasp")
PVDB_PROC = Path("data/pvdb_grasp")
PVDB_VAL_PROC = Path("data/pvdb_grasp_val")
HOCAP_PROC = Path("data/hocap_proc")
OWN_PROC = Path("data/own_grasp")  # scripts/prepare_own_grasp.py: our recordings, state labels


def load_hocap_tracks(
    split: str, proc: Path = HOCAP_PROC, root: Path = PATCH_ROOT, hold_rule: str = "any2"
) -> tuple[list[GraspTrack], np.memmap, np.memmap]:
    """HO-Cap train (subjects 1-4, 6, 7) / val (9) tracks (scripts/prepare_hocap_train.py) +
    their own mesh / vertex-label caches (hocap_mesh.npy, hocap_vlab.npy; tr.start indexes them,
    store "hocap_grasp"). Subjects 5 and 8 (HO-Cap test) are never prepared.
    hold_rule "any2" = the stored state; "air" = hold only while the object is off the table
    (hold_<side>.npz, scripts/make_hold_air_labels.py)."""
    files = sorted(
        p
        for p in proc.glob("*/*/*.npz")
        if (p.parent / "done").exists() and not p.name.startswith(LABEL_FILES)
    )
    keys = [str(p.relative_to(proc)) for p in files]
    mesh_p, vlab_p, keys_p = (
        root / "hocap_mesh.npy",
        root / "hocap_vlab.npy",
        root / "hocap_keys.txt",
    )
    if not keys_p.exists() or keys_p.read_text().split("\n") != keys:
        sizes = []
        for p in files:
            with np.load(p) as z:
                assert "jpeg" not in z.files, f"{p} not packed"
                sizes.append(len(z["present"]))
        R = int(sum(sizes))
        mesh = np.lib.format.open_memmap(mesh_p.with_suffix(".tmp.npy"), "w+", np.float16,
                                         (R, 799, 3))  # fmt: skip
        vlab = np.lib.format.open_memmap(vlab_p.with_suffix(".tmp.npy"), "w+", np.int8, (R, 778))
        r = 0
        for p, n in zip(files, sizes, strict=True):
            with np.load(p) as z:
                mesh[r : r + n, :778], mesh[r : r + n, 778:] = z["verts"], z["joints"]
                vlab[r : r + n] = z["vert"]
            r += n
        mesh.flush()
        vlab.flush()
        del mesh, vlab
        mesh_p.with_suffix(".tmp.npy").replace(mesh_p)
        vlab_p.with_suffix(".tmp.npy").replace(vlab_p)
        keys_p.write_text("\n".join(keys))
    mesh, vlab = np.load(mesh_p, mmap_mode="r"), np.load(vlab_p, mmap_mode="r")
    tracks, start = [], 0
    for p in files:
        with np.load(p) as z:
            n = len(z["present"])
            if str(z["split"]) == split:
                assert p.parent.parent.name not in ("subject_5", "subject_8")
                side = p.stem.split("_")[-1]
                tp = p.parent / f"table_{side}.npz"  # make_table_labels.py
                table = np.load(tp)["table"] if tp.exists() else None
                assert table is None or len(table) == n, tp
                state = z["state"]
                if hold_rule == "air":
                    state = np.load(p.parent / f"hold_{side}.npz")["state"]
                    assert len(state) == n
                tracks.append(
                    GraspTrack(p.parent.parent.name, p.parent.name, p.stem, split, z["offset"],
                               z["length"], z["present"], z["lm_feat"], z["finger"], state,
                               z["valid"], start, bool(z["flip"]), z["mp_right"],
                               store="hocap_grasp", table=table)
                )  # fmt: skip
        start += n
    return tracks, mesh, vlab


def load_egop_tracks(proc: Path = EGOP_PROC, store: str = "egop_grasp",
                     split: str = "train") -> list[GraspTrack]:  # fmt: skip
    """Train-split "surface, no object" tracks (scripts/prepare_egop_grasp.py; EgoPressure or
    PVDB): state 0 (none) wherever a hand is tracked; vertex / mesh labels ignored; finger labels
    from the file when present (PVDB pressure-pad contact), else ignored."""
    tracks = []
    for p in sorted(proc.glob("*/*/*.npz")):
        with np.load(p) as z:
            assert str(z["split"]) == split, p
            n = len(z["present"])
            state = np.where(z["present"], 0, -1).astype(np.int8)
            finger = z["finger"] if "finger" in z.files else np.full((n, 5), -1, np.int8)
            assert finger.shape == (n, 5)
            # pressure pad on the table (PVDB): any finger on it = touching the table
            table = np.where((finger >= 0).all(1), (finger == 1).any(1), -1).astype(np.int8)
            tracks.append(
                GraspTrack(p.parent.parent.name, p.parent.name, p.stem, split, z["offset"],
                           z["length"], z["present"], z["lm_feat"], finger,
                           state, np.zeros(n, bool), -1, bool(z["flip"]), z["mp_right"],
                           store=store, table=table)
            )  # fmt: skip
    return tracks


def load_own_tracks(split: str, proc: Path = OWN_PROC) -> list[GraspTrack]:
    """Our own key-labelled grasp recordings (scripts/prepare_own_grasp.py): hand state only;
    finger / vertex / mesh / table labels unknown."""
    tracks = []
    for p in sorted((proc / split).glob("*/track.npz")):
        with np.load(p) as z:
            assert str(z["split"]) == split, p
            n = len(z["present"])
            tracks.append(
                GraspTrack("own", p.parent.name, "cam", split, z["offset"], z["length"],
                           z["present"], z["lm_feat"], z["finger"], z["state"],
                           np.zeros(n, bool), -1, bool(z["flip"]), z["mp_right"],
                           store="own_grasp")
            )  # fmt: skip
    return tracks


class StoreReader:
    def __init__(self, root: Path = PATCH_ROOT):
        self.root = root
        self._mm: dict[str, np.memmap] = {}

    def get(self, offset: int, length: int, store: str = "dexycb") -> np.ndarray:
        if store not in self._mm:
            self._mm[store] = np.memmap(self.root / f"{store}_store.bin", np.uint8, "r")
        mm = self._mm[store]
        img = cv2.imdecode(np.asarray(mm[offset : offset + length]), cv2.IMREAD_COLOR)
        return img[:, :, ::-1]  # BGR -> RGB


def to_input(x: torch.Tensor) -> torch.Tensor:
    """uint8 [..., H, W, 3] (on any device) -> ImageNet-normalized float32 [..., 3, H, W].
    Done on the training device: workers hand over uint8 (4x less to copy between processes)."""
    mean = torch.as_tensor(IMAGENET_MEAN, device=x.device)
    std = torch.as_tensor(IMAGENET_STD, device=x.device)
    return ((x.float() / 255.0 - mean) / std).movedim(-1, -3).contiguous()


class GraspWindowDataset(Dataset):
    """(track, t) -> dict:
    tips [T, 5, 112, 112, 3], hand [T, S, S, 3] uint8 RGB (to_input() normalizes on device);
    lm [T, F]; present [T] bool;
    finger [T, 5] float32 + finger_mask [T, 5]; state [T] long (-1 ignore);
    vlab [T, 778] float32 + vlab_mask [T, 778]; verts [778, 3], joints [21, 3] float32 (frame t,
    metres) + mesh_mask bool. A frame without a hand repeats the last seen images (causal).
    `target` = position of frame t in the window: T-1 (default, causal: frames t-T+1..t) or e.g.
    T//2 for the offline bidirectional model (frames after the track end count as no hand)."""

    def __init__(self, tracks, mesh, vlab, T: int, index: np.ndarray, augment=False, seed=0,
                 tip_size: int = 112, hand_size: int = 160, root: Path = PATCH_ROOT,
                 target: int | None = None, caches: dict | None = None,
                 aug: str = "mild"):  # fmt: skip
        assert aug in ("mild", "webcam")
        self.aug = aug  # "webcam": + white balance / gamma / blur / low-res / noise / JPEG
        self.tracks, self.mesh, self.vlab, self.T, self.index = tracks, mesh, vlab, T, index
        self.caches = caches or {}  # store -> (mesh, vlab) for tracks not in the DexYCB caches
        self.target = T - 1 if target is None else target
        assert 0 <= self.target < T
        self.augment, self.tip, self.hs = augment, tip_size, hand_size
        self.reader = StoreReader(root)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.index)

    def _imgs(self, tr: GraspTrack, i: int) -> list[np.ndarray] | None:
        if i < 0 or i >= len(tr) or not tr.present[i]:
            return None
        return [self.reader.get(int(tr.offset[i, s]), int(tr.length[i, s]), tr.store)
                for s in range(6)]  # fmt: skip

    def __getitem__(self, j: int) -> dict:
        k, t = (int(v) for v in self.index[j])
        tr, T = self.tracks[k], self.T
        s0 = t - self.target  # track frame at window position 0
        tips = np.zeros((T, 5, self.tip, self.tip, 3), np.uint8)
        hand = np.zeros((T, self.hs, self.hs, 3), np.uint8)
        present = np.zeros(T, bool)
        lm = np.zeros((T, tr.lm_feat.shape[1]), np.float32)
        finger = np.zeros((T, 5), np.float32)
        fmask = np.zeros((T, 5), bool)
        state = np.full(T, -1, np.int64)
        table = np.zeros(T, np.float32)
        tmask = np.zeros(T, bool)
        tab = tr.table_or_unknown()
        vl = np.zeros((T, 778), np.float32)
        vmask = np.zeros((T, 778), bool)
        mesh_c, vlab_c = self.caches.get(tr.store, (self.mesh, self.vlab))
        last = None
        for i in range(s0 - 1, max(-1, s0 - T - 1), -1):  # newest images before the window
            last = self._imgs(tr, i)
            if last is not None:
                break
        for w in range(T):
            i = s0 + w
            cur = self._imgs(tr, i)
            if cur is not None:
                present[w], lm[w], last = True, tr.lm_feat[i], cur
            if last is not None:
                tips[w] = np.stack(last[:5])
                hand[w] = last[5]
            if 0 <= i < len(tr) and present[w]:
                f = tr.finger[i]
                finger[w], fmask[w] = f == 1, f >= 0
                state[w] = tr.state[i]
                table[w], tmask[w] = tab[i] == 1, tab[i] >= 0
                if tr.start >= 0:
                    v = np.asarray(vlab_c[tr.start + i])
                    vl[w], vmask[w] = v == 1, v >= 0
        if tr.start >= 0:
            m = np.asarray(mesh_c[tr.start + t], np.float32)
        else:
            m = np.zeros((799, 3), np.float32)
        mesh_ok = bool(present[self.target] and tr.valid[t] and np.isfinite(m).all())
        if self.augment:
            tips, hand = self._jitter(tips, hand)
            if self.aug == "webcam":
                tips, hand = self._webcam(tips, hand)
        return {
            "tips": torch.from_numpy(tips),
            "hand": torch.from_numpy(hand),
            "lm": torch.from_numpy(lm),
            "present": torch.from_numpy(present),
            "finger": torch.from_numpy(finger),
            "finger_mask": torch.from_numpy(fmask),
            "state": torch.from_numpy(state),
            "table": torch.from_numpy(table),
            "table_mask": torch.from_numpy(tmask),
            "vlab": torch.from_numpy(vl),
            "vlab_mask": torch.from_numpy(vmask),
            "verts": torch.from_numpy(m[:778]),
            "joints": torch.from_numpy(m[778:]),
            "mesh_mask": torch.tensor(mesh_ok),
        }

    def _webcam(self, tips: np.ndarray, hand: np.ndarray, p: float = 0.8):
        """Webcam-like degradation of the crops, tips [T, 5, s, s, 3] and hand [T, h, h, 3] uint8.
        Camera effects (white balance, gamma, blur, resolution, JPEG quality) are drawn once per
        window and shared by all crops; sensor noise is drawn per image."""
        r = self.rng
        if r.random() >= p:
            return tips, hand
        gains = r.uniform(0.8, 1.2, 3).astype(np.float32)
        gamma = float(np.exp(r.uniform(np.log(0.7), np.log(1.4))))
        blur = r.uniform(0.3, 1.5) if r.random() < 0.5 else 0.0
        scale = r.uniform(0.35, 0.8) if r.random() < 0.5 else 1.0
        noise = r.uniform(2.0, 10.0) if r.random() < 0.5 else 0.0
        quality = int(r.integers(30, 81)) if r.random() < 0.5 else 0
        lut = np.clip(255.0 * (np.arange(256) / 255.0) ** gamma, 0, 255).astype(np.uint8)

        def one(x: np.ndarray) -> np.ndarray:
            x = cv2.LUT(x, lut)
            y = x.astype(np.float32) * gains
            if blur:
                y = cv2.GaussianBlur(y, (0, 0), blur)
            if scale < 1.0:
                h, w = y.shape[:2]
                small = cv2.resize(y, (max(8, int(w * scale)), max(8, int(h * scale))),
                                   interpolation=cv2.INTER_AREA)  # fmt: skip
                y = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            if noise:
                y = y + r.normal(0.0, noise, y.shape).astype(np.float32)
            x = np.clip(y, 0, 255).astype(np.uint8)
            if quality:
                x = cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, quality])[1],
                                 cv2.IMREAD_UNCHANGED)  # fmt: skip
            return x

        tips = np.stack([np.stack([one(im) for im in fr]) for fr in tips])
        hand = np.stack([one(im) for im in hand])
        return tips, hand

    def _jitter(self, *xs):
        # one jitter per window (keeps temporal colour change), same for tips and hand crop
        b, c = self.rng.uniform(0.75, 1.25, 2)
        gains = self.rng.uniform(0.92, 1.08, size=3).astype(np.float32)
        out = []
        for x in xs:
            x = x.astype(np.float32)
            m = x.mean()
            out.append(np.clip(((x - m) * c + m) * b * gains, 0, 255).astype(np.uint8))
        return out
