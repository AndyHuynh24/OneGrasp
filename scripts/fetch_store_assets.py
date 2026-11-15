"""Download the videos and 3D-scene files that the hand-data store (Tiger Data) references, so a
local copy matches the database (docs/TIGER_DATA.md). The database holds only references:

  hocap://<subject>/<seq>/<cam>/color_%06d.jpg   HO-Cap image sequences (public, CC BY 4.0)
  hocap://models/<object>/<file>.obj             HO-Cap object meshes
  https://... , hf://datasets/<repo>/<path>      recordings uploaded by Andy (checked by SHA-256)
  a plain path                                   a recording only on the recording laptop so far

    python scripts/fetch_store_assets.py [--store URL] [--source hocap_val] [--session ID]
        [--out data/store_assets] [--no-video]
Writes (re-running skips what is already there):
  out/hocap/...                   HO-Cap images + meshes, in the HO-Cap download layout
  out/videos/<session>.<ext>      uploaded recordings
  out/scenes/<scene_id>.xml       each scene's MuJoCo file with mesh paths pointing at out/
  out/manifest.json               per session: video uri -> local path, scene xml, status
Then: python scripts/export_episodes.py --hocap-root <out>/hocap ...
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from remote_zip import CTX, HttpRangeFile  # noqa: E402

from touch.store import GraspStore  # noqa: E402
from touch.store.scene import localize_mjcf, resolve_uri  # noqa: E402

HOCAP_CFG = "https://raw.githubusercontent.com/IRVLUTD/HO-Cap/main/config/hocap_recordings.yaml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class HoCap:
    """Range-reads single files out of the official HO-Cap zips (no full download)."""

    def __init__(self, root: Path):
        self.root = root
        with urllib.request.urlopen(HOCAP_CFG, timeout=60, context=CTX) as r:
            self.cfg = yaml.safe_load(r.read())
        self._zips: dict[str, zipfile.ZipFile] = {}

    def _zip(self, key: str) -> zipfile.ZipFile:
        if key not in self._zips:
            f = io.BufferedReader(HttpRangeFile(self.cfg[key]), buffer_size=16 << 20)
            self._zips[key] = zipfile.ZipFile(f)
        return self._zips[key]

    def fetch(self, prefix: str, key: str) -> int:
        """All files of zip `key` whose name starts with prefix -> root/<name>. Returns count."""
        z = self._zip(key)
        infos = [i for i in z.infolist() if not i.is_dir() and i.filename.startswith(prefix)]
        infos.sort(key=lambda i: i.header_offset)  # zip order: sequential range reads
        for i in infos:
            dst = self.root / i.filename
            if dst.exists() and dst.stat().st_size == i.file_size:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            tmp.write_bytes(z.read(i))
            tmp.replace(dst)
        return len(infos)


def fetch_video(v: dict, out: Path, hocap: HoCap | None) -> tuple[str | None, str]:
    """-> (local path or pattern, status)."""
    uri = v["uri"]
    if uri.startswith("hocap://"):
        rel = uri.removeprefix("hocap://")
        folder = rel.rsplit("/", 1)[0]  # <subject>/<seq>/<cam>
        n = hocap.fetch(folder + "/color_", folder.split("/")[0])
        ok = n == v["n_frames"] if v["n_frames"] else n > 0
        return resolve_uri(uri, out / "hocap"), f"{n} frames" + ("" if ok else " (MISMATCH)")
    dst = out / "videos" / f"{v['session_id']}{Path(uri).suffix or '.mp4'}"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if uri.startswith("hf://"):
            from huggingface_hub import hf_hub_download

            kind, owner, name, path = uri.removeprefix("hf://").split("/", 3)
            assert kind == "datasets", uri
            src = hf_hub_download(f"{owner}/{name}", path, repo_type="dataset")
            shutil.copyfile(src, dst)
        elif uri.startswith(("https://", "http://")):
            with urllib.request.urlopen(uri, timeout=120, context=CTX) as r, dst.open("wb") as f:
                shutil.copyfileobj(r, f)
        elif Path(uri).is_file():  # on the recording laptop itself
            return uri, "local file (recording machine)"
        else:
            return None, "NOT UPLOADED yet (only on the recording laptop)"
    if v["sha256"] and sha256(dst) != v["sha256"]:
        return str(dst), "SHA-256 MISMATCH"
    return str(dst), "ok" + (" (sha256 verified)" if v["sha256"] else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=None, help="default $TIGER_DATABASE_URL")
    ap.add_argument("--source", default=None, help="only sessions of this source")
    ap.add_argument("--session", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("data/store_assets"))
    ap.add_argument("--no-video", action="store_true", help="scenes / meshes only")
    args = ap.parse_args()
    st = GraspStore(args.store)
    sessions = [s for s in st.sessions()
                if (args.source is None or s["source"] == args.source)
                and (not args.session or s["session_id"] in args.session)]  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    hocap = None
    manifest = {}
    for s in sessions:
        sid = s["session_id"]
        entry = {"source": s["source"], "videos": [], "scene_xml": None}
        sc = st.scene(s["scene_id"]) if s.get("scene_id") else None
        uris = [o["mesh_uri"] for o in (sc or {}).get("objects") or [] if o.get("mesh_uri")]
        vids = [] if args.no_video else st.videos(sid)
        if hocap is None and (any("hocap" in u for u in uris)
                              or any(v["uri"].startswith("hocap://") for v in vids)):  # fmt: skip
            hocap = HoCap(args.out / "hocap")
        for u in uris:  # meshes (+ textures next to them)
            local = Path(resolve_uri(u, args.out / "hocap"))
            if not local.exists():
                obj_dir = local.relative_to(args.out / "hocap").parent  # models/<obj>
                hocap.fetch(f"{obj_dir.as_posix()}/", "models")
        if sc and sc.get("mjcf"):
            xml_dir = args.out / "scenes"
            xml_dir.mkdir(exist_ok=True)
            xml = localize_mjcf(sc["mjcf"], sc["objects"] or [], args.out / "hocap", xml_dir)
            p = xml_dir / f"{sc['scene_id']}.xml"
            p.write_text(xml)
            entry["scene_xml"] = str(p)
        for v in vids:
            local, status = fetch_video(v, args.out, hocap)
            entry["videos"].append({"uri": v["uri"], "local": local, "status": status})
        manifest[sid] = entry
        print(f"{sid} ({s['source']}): scene {'ok' if entry['scene_xml'] else '-'}; "
              + "; ".join(f"video {e['status']}" for e in entry["videos"]), flush=True)  # fmt: skip
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(sessions)} sessions -> {args.out} (manifest.json)")
    st.close()


if __name__ == "__main__":
    main()
