"""HO-Cap (Wang et al., NeurIPS 2025 D&B; CC BY 4.0) loader: an out-of-distribution TEST set.

Never used for training or tuning. 9 subjects, 64 sequences of ~700 frames (30 fps), 8 RealSense
cameras (640x480) + HoloLens, 1-2 hands, 4 everyday objects per sequence (meshes in metres),
tasks: pick-and-place, handover, affordance use. Downloaded with plain curl from the Box links in
github.com/IRVLUTD/HO-Cap config/hocap_recordings.yaml (poses + models + calibration for all
sequences; videos only for the subjects we fetched).

Frames (from the HO-Cap toolkit, hocap_toolkit/loaders/sequence_loader.py):
  world = AprilTag "tag_1" on the table (z up; verified: resting objects sit just above z = 0);
  cam -> world = inv(tag_1) @ cam -> master; 3x4 row-major matrices.
  poses_o.npy [n_obj, N, 7] = quaternion xyzw (scipy order) + translation, object -> world.
  poses_m.npy [2, N, 51] = MANO (index 0 right, 1 left), 3 global + 45 PCA + 3 translation, the
  DexYCB convention (flat_hand_mean False, 45 comps), world frame; -1 everywhere = no hand.

Contact labels mirror touch.data.dexycb (3 mm contact, 3-10 mm ignored band) with one change for
multi-grasp clips: "hold" = thumb + another part in contact AND the touched object moved > 1 cm
since that contact episode began (DexYCB measures from frame 0, which breaks after a put-down).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

from touch.data import dexycb as D

ROOT = Path("data/hocap/ds")
SIDES = ("right", "left")
FPS = 30.0


@dataclass(frozen=True)
class Seq:
    subject: str
    name: str
    root: Path

    @cached_property
    def meta(self) -> dict:
        return yaml.safe_load((self.root / "meta.yaml").read_text())

    @property
    def has_video(self) -> bool:
        return (self.root / "meta.yaml").exists()

    @property
    def serials(self) -> list[str]:
        return [str(s) for s in self.meta["realsense"]["serials"]]

    @property
    def objects(self) -> list[str]:
        return list(self.meta["object_ids"])

    @property
    def sides(self) -> list[str]:
        return list(self.meta["mano_sides"])

    @property
    def n_frames(self) -> int:
        return int(np.load(self.root / "poses_o.npy", mmap_mode="r").shape[1])

    def image(self, cam: str, i: int) -> Path:
        return self.root / cam / f"color_{i:06d}.jpg"


TEST_SUBJECTS = ("subject_5", "subject_8")  # the HO-Cap test set: never trained or tuned on
VAL_SUBJECTS = ("subject_9",)


def split_of(subject: str) -> str:
    return "test" if subject in TEST_SUBJECTS else ("val" if subject in VAL_SUBJECTS else "train")


def list_sequences(
    root: Path = ROOT, video_only: bool = False, split: str | None = None
) -> list[Seq]:
    """split: None = all, "test" = subjects 5 + 8, "val" = 9, "train" = the rest."""
    out = [Seq(p.parent.name, p.name, p) for p in sorted(root.glob("subject_*/*")) if p.is_dir()]
    if split is not None:
        out = [s for s in out if split_of(s.subject) == split]
    return [s for s in out if s.has_video] if video_only else out


def _mat(v) -> np.ndarray:
    return np.asarray(v, np.float64).reshape(3, 4)


def cam_to_world(serial: str, root: Path = ROOT, file: str = "extrinsics_20231014.yaml"):
    """3x4 camera -> world (table tag) transform."""
    ex = yaml.safe_load((root / "calibration" / "extrinsics" / file).read_text())["extrinsics"]
    tag = np.vstack([_mat(ex["tag_1"]), [0, 0, 0, 1]])
    c2m = np.vstack([_mat(ex[serial]), [0, 0, 0, 1]])
    return (np.linalg.inv(tag) @ c2m)[:3]


def intrinsics(serial: str, root: Path = ROOT) -> np.ndarray:
    c = yaml.safe_load((root / "calibration" / "intrinsics" / f"{serial}.yaml").read_text())[
        "color"
    ]
    return np.array([[c["fx"], 0, c["ppx"]], [0, c["fy"], c["ppy"]], [0, 0, 1]], np.float64)


def betas(subject: str, root: Path = ROOT) -> np.ndarray:
    f = root / "calibration" / "mano" / f"{subject}.yaml"
    return np.asarray(yaml.safe_load(f.read_text())["betas"], np.float32)


def object_poses(seq: Seq) -> tuple[np.ndarray, np.ndarray]:
    """-> R [n_obj, N, 3, 3], t [n_obj, N, 3]: object -> world."""
    po = np.load(seq.root / "poses_o.npy").astype(np.float64)
    R = Rot.from_quat(po[..., :4].reshape(-1, 4)).as_matrix().reshape(*po.shape[:2], 3, 3)
    return R, po[..., 4:]


def hand_mesh(seq: Seq, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> verts [N, 778, 3], joints [N, 21, 3] (world, metres; our joint order), valid [N]."""
    pm = np.load(seq.root / "poses_m.npy")[SIDES.index(side)].astype(np.float32)
    valid = ~np.all(pm == -1, axis=1) & np.isfinite(pm).all(1)
    v, j = D.mano_forward(np.where(valid[:, None], pm, 0), betas(seq.subject), side)
    return v, j, valid


@lru_cache(maxsize=128)
def object_points(obj: str, n: int = 20000, root: Path = ROOT) -> np.ndarray:
    """Area-weighted surface samples + vertices [n + V, 3] (m, object frame); seeded."""
    v, f = D.load_obj(root / "models" / obj / "cleaned_mesh_10000.obj")
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(0)
    k = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = rng.random((2, n))
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * a[k] + (s * (1 - r2))[:, None] * b[k] + (s * r2)[:, None] * c[k]
    return np.concatenate([pts, v]).astype(np.float32)


def hand_distances(seq: Seq, side: str) -> dict[str, np.ndarray]:
    """Per-frame distances from one hand to the nearest object surface.
    vert_dist [N, 778], part_dist [N, 6] (palm, thumb..pinky; min over objects),
    part_obj [N, 6] (index of the nearest object per part), valid [N], verts / joints (world)."""
    from scipy.spatial import cKDTree

    verts, joints, valid = hand_mesh(seq, side)
    R, t = object_poses(seq)
    trees = [cKDTree(object_points(o)) for o in seq.objects]
    n = len(valid)
    vd_all = np.full((len(trees), n, 778), np.inf, np.float32)
    for i in np.flatnonzero(valid):
        for k, tree in enumerate(trees):
            local = (verts[i] - t[k, i]) @ R[k, i]  # world -> object frame (R orthonormal)
            vd_all[k, i] = tree.query(local)[0]
    vd = vd_all.min(0)
    parts = D.vertex_parts(side)
    pd_all = np.stack([vd_all[:, :, parts == p].min(2) for p in range(6)], 2)  # [K, N, 6]
    return dict(vert_dist=vd, part_dist=pd_all.min(0), part_obj=pd_all.argmin(0),
                valid=valid, verts=verts, joints=joints)  # fmt: skip


def episode_displacement(part_dist, part_obj, obj_t, contact_mm: float = 3.0) -> np.ndarray:
    """[N] metres the touched object moved since the current contact episode began (0 outside
    contact). An episode = consecutive frames in contact with the same object."""
    c = (part_dist * 1000.0) <= contact_mm
    n = len(c)
    out = np.zeros(n, np.float32)
    start, cur = None, -1
    for i in range(n):
        if c[i].any():
            k = int(part_obj[i][c[i]][np.argmin(part_dist[i][c[i]])])
            if k != cur:
                start, cur = i, k
            out[i] = np.linalg.norm(obj_t[k, i] - obj_t[k, start])
        else:
            start, cur = None, -1
    return out


def labels(seq: Seq, side: str) -> dict[str, np.ndarray]:
    """Distances + labels (finger [N,5], state [N]; touch.data.dexycb.labels_from_dist rules,
    episode-relative object motion) + touched object per frame (-1 none)."""
    r = hand_distances(seq, side)
    _, t = object_poses(seq)
    disp = episode_displacement(r["part_dist"], r["part_obj"], t)
    L = D.labels_from_dist(r["part_dist"], disp, r["valid"])
    c = r["part_dist"] * 1000.0 <= 3.0
    touched = np.where(
        c.any(1),
        r["part_obj"][np.arange(len(c)), np.where(c, r["part_dist"], np.inf).argmin(1)],
        -1,
    )
    return {**r, **L, "obj_disp": disp, "touched_obj": touched.astype(np.int8)}


def object_low_z(seq: Seq, n_pts: int = 1500) -> np.ndarray:
    """[n_obj, N] height (m) of each object's lowest surface point above the table (z = 0)."""
    R, t = object_poses(seq)
    out = []
    for k, o in enumerate(seq.objects):
        p = object_points(o, n_pts)[:n_pts].astype(np.float64)
        out.append(np.einsum("nj,pj->np", R[k, :, 2, :], p).min(1) + t[k, :, 2])
    return np.stack(out)


def hold_air(state: np.ndarray, touched_obj: np.ndarray, low_z: np.ndarray,
             clear_m: float = 0.01) -> np.ndarray:  # fmt: skip
    """Hold = held AND off the table: a "hold" frame whose touched object rests on the table
    (lowest point <= clear_m above it) becomes touch (docs/DECISIONS.md 2026-11-12: the object
    was set down, the hand is still on it). Other labels unchanged."""
    out = np.asarray(state).copy()
    t = np.flatnonzero(out == 2)
    k = np.asarray(touched_obj)[t]
    assert (k >= 0).all()
    out[t[low_z[k, t] <= clear_m]] = 1
    return out
