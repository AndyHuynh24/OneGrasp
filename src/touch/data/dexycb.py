"""DexYCB (CC BY-NC 4.0, Chao et al. CVPR 2021) -> hand-object contact / holding labels.

Camera only: labels come from DexYCB's fitted 3D MANO hand and YCB object poses, no sensors.
Per sequence (one camera's labels; 3D distances do not depend on the camera):
  vert_dist [N, 778]  distance (m) of every MANO vertex to the grasped object's surface
  part_dist [N, 6]    min distance per hand part: palm, thumb, index, middle, ring, pinky
  tip_dist  [N, 5]    min distance of each finger's distal segment (thumb..pinky)
  obj_disp  [N]       grasped object's displacement (m) from its first-frame position
  valid     [N]       hand annotated this frame
Contact / holding thresholds are applied later (labels_from_dist), so they can be chosen on
validation without recomputing distances.

Layout as downloaded by scripts/download_dexycb.sh: data/dexycb/<subject>/<sequence>/{meta.yml,
<serial>/{color_XXXXXX.jpg, labels_XXXXXX.npz}}, data/dexycb/{models, calibration}.
Pose convention verified against DexYCB's joint_3d (0.0 mm): hand pose = 3 global axis-angle +
45 PCA coefficients; full finger pose = coeffs @ hands_components + hands_mean (smplx's own PCA
path did not match), translation = pose_m[48:51].
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from touch.mano_path import MANO_DIR  # chumpy-free MANO pickles

ROOT = Path("data/dexycb")

SUBJECTS = [
    "20200709-subject-01", "20200813-subject-02", "20200820-subject-03", "20200903-subject-04",
    "20200908-subject-05", "20200918-subject-06", "20200928-subject-07", "20201002-subject-08",
    "20201015-subject-09", "20201022-subject-10",
]  # fmt: skip
SERIALS = ["836212060125", "839512060362", "840412060917", "841412060263",
           "932122060857", "932122060861", "932122061900", "932122062010"]  # fmt: skip
CAMS = [SERIALS[i] for i in (0, 2, 4, 6)]  # the cameras kept by download_dexycb.sh
# official "s1" setup (unseen subjects), dex-ycb-toolkit dex_ycb.py
SPLIT_S1 = {"train": [0, 1, 2, 3, 4, 5, 9], "val": [6], "test": [7, 8]}
# MANO fingertip vertices, thumb..pinky, as in DexYCB joint_3d (verified 0.0 mm; left middle = 445)
TIPS = {"right": [745, 317, 444, 556, 673], "left": [745, 317, 445, 556, 673]}
# MANO joint -> part: 0 palm/wrist; 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb
_JOINT_PART = np.array([0, 2, 2, 2, 3, 3, 3, 5, 5, 5, 4, 4, 4, 1, 1, 1])
_DISTAL_JOINTS = {1: 15, 2: 3, 3: 6, 4: 12, 5: 9}  # part -> MANO distal joint
# MANO joints (16) + fingertip vertices -> 21 joints in MediaPipe / DexYCB order
_ORDER21 = [0, 13, 14, 15, -1, 1, 2, 3, -1, 4, 5, 6, -1, 10, 11, 12, -1, 7, 8, 9, -1]


def split_of(subject: str) -> str:
    i = SUBJECTS.index(subject)
    return next(k for k, v in SPLIT_S1.items() if i in v)


@dataclass
class Seq:
    subject: str
    name: str
    root: Path
    meta: dict

    @property
    def split(self) -> str:
        return split_of(self.subject)

    @property
    def side(self) -> str:
        return self.meta["mano_sides"][0]

    @property
    def n_frames(self) -> int:
        return int(self.meta["num_frames"])

    @property
    def grasp_object(self) -> str:
        ycb = sorted(p.name for p in (self.root.parents[1] / "models").iterdir() if p.is_dir())
        return ycb[self.meta["ycb_ids"][self.meta["ycb_grasp_ind"]] - 1]

    def labels(self, cam: str, i: int) -> dict:
        with np.load(self.root / cam / f"labels_{i:06d}.npz") as z:
            return {k: z[k] for k in z.files}

    def image(self, cam: str, i: int) -> Path:
        return self.root / cam / f"color_{i:06d}.jpg"


def list_sequences(root: Path = ROOT, split: str | None = None) -> list[Seq]:
    out = []
    for subj in SUBJECTS:
        d = root / subj
        if not d.is_dir():
            continue
        for s in sorted(p for p in d.iterdir() if (p / "meta.yml").exists()):
            seq = Seq(subj, s.name, s, yaml.safe_load((s / "meta.yml").read_text()))
            if split is None or seq.split == split:
                out.append(seq)
    return out


@lru_cache(maxsize=2)
def mano_model(side: str):
    import pickle

    with open(MANO_DIR / f"MANO_{side.upper()}.pkl", "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=2)
def vertex_parts(side: str = "right") -> np.ndarray:
    """[778] part index per MANO vertex (0 palm, 1 thumb .. 5 pinky) by skinning-weight argmax."""
    return _JOINT_PART[np.asarray(mano_model(side)["weights"]).argmax(1)]


@lru_cache(maxsize=2)
def distal_mask(side: str = "right") -> np.ndarray:
    """[5, 778] bool: vertices of each finger's distal segment (thumb..pinky)."""
    j = np.asarray(mano_model(side)["weights"]).argmax(1)
    return np.stack([j == _DISTAL_JOINTS[p] for p in range(1, 6)])


def mano_forward(pose_m: np.ndarray, betas: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """pose_m [N, 51] (DexYCB), betas [10] -> vertices [N, 778, 3] and joints [N, 21, 3] (m)."""
    import torch

    mp = mano_model(side)
    p = np.asarray(pose_m, np.float32).reshape(-1, 51)
    full = p[:, 3:48] @ np.asarray(mp["hands_components"]) + np.asarray(mp["hands_mean"])
    m = _smplx_mano(side)
    with torch.no_grad():
        o = m(
            global_orient=torch.from_numpy(p[:, :3]),
            hand_pose=torch.from_numpy(full.astype(np.float32)),
            betas=torch.from_numpy(np.tile(np.asarray(betas, np.float32), (len(p), 1))),
            transl=torch.from_numpy(p[:, 48:51]),
        )
    v, j16 = o.vertices.numpy(), o.joints.numpy()
    j21 = np.stack(
        [v[:, TIPS[side][i // 4 - 1]] if k < 0 else j16[:, k] for i, k in enumerate(_ORDER21)], 1
    )
    assert v.shape == (len(p), 778, 3) and j21.shape == (len(p), 21, 3)
    return v, j21


@lru_cache(maxsize=2)
def _smplx_mano(side: str):
    import smplx

    return smplx.MANO(model_path=str(MANO_DIR), is_rhand=side == "right", use_pca=False,
                      flat_hand_mean=True, batch_size=1)  # fmt: skip


def mano_betas(seq: Seq, root: Path = ROOT) -> np.ndarray:
    f = root / "calibration" / f"mano_{seq.meta['mano_calib'][0]}" / "mano.yml"
    return np.asarray(yaml.safe_load(f.read_text())["betas"], np.float32)


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    v, f = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            f.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return np.asarray(v, np.float64), np.asarray(f, np.int64)


@lru_cache(maxsize=32)
def object_points(name: str, n: int = 30000, root: Path = ROOT) -> np.ndarray:
    """Area-weighted surface samples [n + V, 3] (m) of a YCB model (+ its vertices); seeded."""
    v, f = load_obj(root / "models" / name / "textured_simple.obj")
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(0)
    k = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = rng.random((2, n))
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * a[k] + (s * (1 - r2))[:, None] * b[k] + (s * r2)[:, None] * c[k]
    return np.concatenate([pts, v]).astype(np.float32)


def sequence_distances(seq: Seq, cam: str | None = None) -> dict[str, np.ndarray]:
    """Per-frame hand-to-grasped-object distances of one sequence (see module docstring)."""
    from scipy.spatial import cKDTree

    cam = cam or next(c for c in CAMS if (seq.root / c).is_dir())
    n = seq.n_frames
    lab = [seq.labels(cam, i) for i in range(n)]
    pose_m = np.stack([x["pose_m"][0] for x in lab])
    valid = np.array([x["joint_3d"][0, 0, 0] != -1 for x in lab]) & np.abs(pose_m).any(1)
    gi = seq.meta["ycb_grasp_ind"]
    obj_rt = np.stack([x["pose_y"][gi] for x in lab])  # [N, 3, 4] object -> camera
    pts = object_points(seq.grasp_object)
    verts, _ = mano_forward(np.where(valid[:, None], pose_m, 0), mano_betas(seq), seq.side)
    tree = cKDTree(pts)
    vd = np.full((n, 778), np.inf, np.float32)
    for i in np.flatnonzero(valid):
        r, t = obj_rt[i, :, :3], obj_rt[i, :, 3]
        local = (verts[i] - t) @ r  # hand vertices in the object's frame (R orthonormal)
        vd[i] = tree.query(local)[0]
    parts, dmask = vertex_parts(seq.side), distal_mask(seq.side)
    part_dist = np.stack([vd[:, parts == p].min(1) for p in range(6)], 1)
    tip_dist = np.stack([vd[:, m].min(1) for m in dmask], 1)
    pos = obj_rt[:, :, 3]
    disp = np.linalg.norm(pos - pos[0], axis=1)
    return dict(vert_dist=vd, part_dist=part_dist, tip_dist=tip_dist, obj_disp=disp,
                valid=valid, pose_m=pose_m, cam=np.array(cam))  # fmt: skip


def labels_from_dist(
    part_dist: np.ndarray,
    obj_disp: np.ndarray,
    valid: np.ndarray,
    contact_mm: float = 3.0,
    ignore_mm: float = 10.0,
    lift_m: float = 0.01,
    hold_rule: str = "thumb",
) -> dict[str, np.ndarray]:
    """Distances -> training labels.

    finger [N, 5]: 1 contact (<= contact_mm), 0 clear (> ignore_mm), -1 ignore (in between, or
    no annotation): the near-but-not-touching band is left out of training, because pose error
    of a few mm cannot tell hover from touch there.
    state [N]: 0 none, 1 touch (some part in contact), 2 hold (thumb + another finger in contact
    and the object moved > lift_m), -1 ignore (no annotation, or every part in the ignore band).
    hold_rule "thumb" (default, the original label) or "any2": hold = any 2 of the 6 parts (palm,
    thumb..pinky) in contact and the object moved > lift_m. "any2" fixes held objects labelled
    touch when the thumb is a few mm off (wrap / handle grips; 28% of DexYCB lifted-in-hand
    frames under "thumb"; docs/DECISIONS.md).
    """
    d = np.asarray(part_dist) * 1000.0
    c, far = d <= contact_mm, d > ignore_mm
    finger = np.where(c[:, 1:], 1, np.where(far[:, 1:], 0, -1)).astype(np.int8)
    finger[~valid] = -1
    if hold_rule == "thumb":
        grasp = c[:, 1] & c[:, 2:].any(1)
    elif hold_rule == "any2":
        grasp = c.sum(1) >= 2
    else:
        raise ValueError(hold_rule)
    hold = grasp & (np.asarray(obj_disp) > lift_m)
    state = np.where(hold, 2, np.where(c.any(1), 1, 0)).astype(np.int8)
    unsure = ~c.any(1) & ~far.all(1)  # nothing in contact, something in the ignore band
    state[unsure | ~valid] = -1
    return {"finger": finger, "state": state}


def intrinsics(serial: str, root: Path = ROOT) -> np.ndarray:
    """[3, 3] color camera matrix (640x480); the yml has a python/tuple tag, so parse by hand."""
    import re

    t = (root / "calibration" / "intrinsics" / f"{serial}_640x480.yml").read_text()
    color = t.split("color:")[1].split("depth:")[0]
    v = {
        k: float(re.search(rf"{k}:\s*([-\d.eE]+)", color).group(1))
        for k in ("fx", "fy", "ppx", "ppy")
    }
    return np.array([[v["fx"], 0, v["ppx"]], [0, v["fy"], v["ppy"]], [0, 0, 1]], np.float64)


def project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame points [..., 3] -> pixels [..., 2]."""
    p = np.asarray(points, np.float64)
    return np.stack(
        [K[0, 0] * p[..., 0] / p[..., 2] + K[0, 2], K[1, 1] * p[..., 1] / p[..., 2] + K[1, 2]], -1
    )
