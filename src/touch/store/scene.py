"""Robot-free MuJoCo scene (MJCF) of a recorded table-frame scene: table plane at z = 0, the
objects (free bodies), the calibrated camera. A robot (e.g. a Panda) can be added with
<include> or MjSpec.attach. Stored in scenes.mjcf."""

from __future__ import annotations

import numpy as np


def camera_mjcf(K, cam_to_world, image_size, name: str = "rec") -> str:
    """MuJoCo camera at the calibrated pose (OpenCV axes -> MuJoCo: y and z flipped)."""
    K, C = np.asarray(K, float), np.asarray(cam_to_world, float)
    R, p = C[:, :3], C[:, 3]
    x, y = R[:, 0], -R[:, 1]
    fovy = np.degrees(2 * np.arctan(image_size[1] / 2 / K[1, 1]))
    return (f'<camera name="{name}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
            f'xyaxes="{x[0]:.6f} {x[1]:.6f} {x[2]:.6f} {y[0]:.6f} {y[1]:.6f} {y[2]:.6f}" '
            f'fovy="{fovy:.4f}"/>')  # fmt: skip


def scene_mjcf(camera: dict | None, objects: list[dict], table_half_m: float = 1.5) -> str:
    """objects: [{name, shape: cylinder | box | sphere | mesh, size_m, mesh_uri, pos, quat_wxyz}]
    (cylinder size_m = [radius, half_height]; box = half sizes; sphere = [radius])."""
    assets, bodies = [], []
    for i, o in enumerate(objects):
        pos = " ".join(f"{v:.6f}" for v in o.get("pos", [0, 0, 0]))
        quat = " ".join(f"{v:.6f}" for v in o.get("quat_wxyz", [1, 0, 0, 0]))
        shape = o.get("shape", "mesh" if o.get("mesh_uri") else "box")
        if shape == "mesh":
            assets.append(f'<mesh name="mesh{i}" file="{o["mesh_uri"]}"/>')
            geom = f'type="mesh" mesh="mesh{i}"'
        else:
            size = " ".join(f"{v:.6f}" for v in o["size_m"])
            geom = f'type="{shape}" size="{size}"'
        bodies.append(f'    <body name="{o.get("name", f"obj{i}")}" pos="{pos}" quat="{quat}">'
                      f'<freejoint/><geom {geom} density="300" friction="1.0 0.02 0.001" '
                      f'condim="4"/></body>')  # fmt: skip
    cam = ""
    if camera and "K" in camera:
        cam = "    " + camera_mjcf(camera["K"], camera["cam_to_world"], camera["image_size"])
    asset = "  <asset>\n    " + "\n    ".join(assets) + "\n  </asset>\n" if assets else ""
    return (
        f'<mujoco model="recorded_scene">\n  <compiler angle="radian"/>\n{asset}'
        f'  <worldbody>\n    <light pos="0 0 2" dir="0 0 -1"/>\n'
        f'    <geom name="table" type="plane" size="{table_half_m} {table_half_m} 0.01" '
        f'rgba="0.8 0.8 0.8 1"/>\n{cam}\n' + "\n".join(bodies) + "\n  </worldbody>\n</mujoco>\n"
    )


def resolve_uri(uri: str, hocap_root) -> str:
    """Store uri -> local path: hocap://<rel> under hocap_root (the HO-Cap download layout:
    <subject>/<seq>/<cam>/..., models/<obj>/...); other uris unchanged."""
    from pathlib import Path

    if uri.startswith("hocap://"):
        return str(Path(hocap_root) / uri.removeprefix("hocap://"))
    if "hocap/ds/models/" in uri:  # rows stored before mesh uris became hocap://
        return str(Path(hocap_root) / "models" / uri.split("hocap/ds/models/", 1)[1])
    return uri


def localize_mjcf(mjcf: str, objects: list[dict], hocap_root, xml_dir=None) -> str:
    """Point the scene's mesh files at local copies (relative to xml_dir if given)."""
    import os

    for o in objects:
        if o.get("mesh_uri"):
            local = resolve_uri(o["mesh_uri"], hocap_root)
            if xml_dir is not None:
                local = os.path.relpath(local, xml_dir)
            mjcf = mjcf.replace(f'file="{o["mesh_uri"]}"', f'file="{local}"')
    return mjcf
