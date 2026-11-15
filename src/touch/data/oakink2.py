"""OakInk2 (Zhan et al., CVPR 2024; Hugging Face kelvin34501/OakInk-v2, card licence CC BY-SA 4.0 --
confirm on the project page before commercial use). EXTRA VALIDATION ONLY: never trained / tuned on.

Layout written by scripts/download_oakink2.py:
    data/oakink2/anno_preview/<seq>.pkl     official annotation (loaded with a restricted unpickler)
    data/oakink2/images/<seq>/<cam>/<frame>.jpg   allocentric_{top,left,right}, annotated frames
    data/oakink2/object_raw/, object_repair/   object meshes (official tarballs)
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

ROOT = Path("data/oakink2")
HF = "https://huggingface.co/datasets/kelvin34501/OakInk-v2/resolve/main"
CAMS = ("allocentric_top", "allocentric_left", "allocentric_right")

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
}


class _SafeUnpickler(pickle.Unpickler):
    """Only numpy arrays, torch tensors (nested bytes via torch.load(weights_only=True)) and
    OrderedDicts: the annotation pickles reference nothing else (checked with pickletools)."""

    def find_class(self, module: str, name: str):
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            import torch

            return lambda b: torch.load(io.BytesIO(b), weights_only=True)
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_anno(path: Path) -> dict:
    with open(path, "rb") as fh:
        return _SafeUnpickler(fh).load()


def anno_path(seq: str, root: Path = ROOT) -> Path:
    return root / "anno_preview" / f"{seq}.pkl"


def image_path(seq: str, cam: str, frame: int, root: Path = ROOT) -> Path:
    return root / "images" / seq / cam / f"{frame:06d}.jpg"


def list_sequences(root: Path = ROOT) -> list[str]:
    return sorted(p.name for p in (root / "images").iterdir() if (p / "done").exists())


def quat_to_rotmat(q):
    """[..., 4] (w, x, y, z; manotorch's order, oakink2 README) -> [..., 3, 3]."""
    import numpy as np

    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def gt_hand(anno: dict, frames, side: str):
    """GT MANO hand in the OakInk2 world frame (metres, from the measured coordinate spread) at
    mocap frame ids `frames`. Same layer as the official viewer (manotorch, rot_mode quat,
    use_pca False, flat_hand_mean True, center_idx 0, then + tsl): built here with smplx's lbs,
    our DexYCB / HO-Cap convention. -> verts [N, 778, 3], joints [N, 21, 3] (ORDER21)."""
    import numpy as np
    import torch
    from smplx.lbs import lbs

    from touch.data.dexycb import _smplx_mano
    from touch.models.graspnet import ORDER21, TIPS_RIGHT

    h = "rh" if side == "right" else "lh"
    m = _smplx_mano(side)
    raw = [anno["raw_mano"][int(f)] for f in frames]
    R = torch.from_numpy(quat_to_rotmat(np.stack([r[f"{h}__pose_coeffs"][0].numpy() for r in raw])))
    B = torch.stack([r[f"{h}__betas"][0] for r in raw]).double()
    tsl = np.stack([r[f"{h}__tsl"][0].numpy() for r in raw]).astype(np.float64)
    with torch.no_grad():
        v, j16 = lbs(
            B,
            R,
            m.v_template.double(),
            m.shapedirs.double(),
            m.posedirs.double(),
            m.J_regressor.double(),
            m.parents,
            m.lbs_weights.double(),
            pose2rot=False,
        )
    v, j16 = v.numpy(), j16.numpy()
    j21 = np.stack([v[:, TIPS_RIGHT[i // 4 - 1]] if k < 0 else j16[:, k]
                    for i, k in enumerate(ORDER21)], 1)  # fmt: skip
    root = j21[:, :1]
    return v - root + tsl[:, None], j21 - root + tsl[:, None]


def read_ply(path: Path):
    """Binary little-endian PLY (OakInk2 scan.ply: double xyz + uchar rgba, int faces)."""
    import numpy as np

    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode().splitlines()
    assert "format binary_little_endian 1.0" in header, path
    nv = nf = 0
    vprops: list[tuple[str, str]] = []
    cur = None
    for line in header:
        t = line.split()
        if t[:2] == ["element", "vertex"]:
            nv, cur = int(t[2]), "v"
        elif t[:2] == ["element", "face"]:
            nf, cur = int(t[2]), "f"
        elif t[0] == "property" and cur == "v":
            vprops.append((t[2], {"double": "<f8", "float": "<f4", "uchar": "u1"}[t[1]]))
        elif t[0] == "property" and cur == "f":
            assert t[1:4] == ["list", "uchar", "int"], line
    vt = np.dtype(vprops)
    verts = np.frombuffer(data, vt, nv, end)
    fdt = np.dtype([("n", "u1"), ("i", "<i4", 3)])
    faces = np.frombuffer(data, fdt, nf, end + nv * vt.itemsize)
    assert (faces["n"] == 3).all(), path
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], 1).astype(np.float64)
    return xyz, faces["i"].astype(np.int64)


def mesh_path(obj_id: str, root: Path = ROOT) -> Path:
    d = root / "object_raw" / "align_ds" / obj_id
    c = [p for p in d.iterdir() if p.suffix in (".obj", ".ply")]
    assert len(c) == 1, (d, c)
    return c[0]


def object_points(obj_id: str, n: int = 4000, seed: int = 0, root: Path = ROOT):
    """Area-weighted surface samples [n + V, 3] of an object mesh (object frame); seeded."""
    import numpy as np

    path = mesh_path(obj_id, root)
    if path.suffix == ".ply":
        v, f = read_ply(path)
    else:  # a few objects ship as Meshlab OBJ
        from touch.data.dexycb import load_obj

        v, f = load_obj(path)
    tri = v[f]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    k = rng.choice(len(f), n, p=area / area.sum())
    a, b = rng.random((2, n))
    s = np.sqrt(a)
    p = (
        (1 - s)[:, None] * tri[k, 0]
        + (s * (1 - b))[:, None] * tri[k, 1]
        + (s * b)[:, None] * tri[k, 2]
    )
    return np.concatenate([p, v])


# OakInk2's world frame is y-up (checked: resting objects' lowest points agree along +y to < 1 mm
# in single-level scenes; the top camera sits at y = +0.87 m looking down). Rotation to our z-up
# table convention (x, y, z) -> (x, -z, y), right-handed.
Y_UP_TO_Z_UP = ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))


def to_table_frame(x, table_y: float):
    """World points [..., 3] -> z-up frame with the table at z = 0 (m)."""
    import numpy as np

    out = np.asarray(x, np.float64) @ np.asarray(Y_UP_TO_Z_UP).T
    out[..., 2] -= table_y
    return out
