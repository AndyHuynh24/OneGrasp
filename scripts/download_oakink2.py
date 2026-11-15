"""Fetch a few OakInk2 sequences for EXTRA VALIDATION (never training): the official annotation,
object meshes, and only the 3 fixed third-person cameras at the annotated frames, streamed out of
each sequence tarball (nothing else is written to disk) and stored as JPEG (quality 92).

    python scripts/download_oakink2.py [--n 8] [--seed 0] [--min-gb 1.0] [--max-gb 2.5]
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touch.data import oakink2 as O  # noqa: E402

API = "https://huggingface.co/api/datasets/kelvin34501/OakInk-v2/tree/main/data"


def _https_with_certifi() -> None:
    """python.org Python builds do not see the macOS keychain: use certifi's CA bundle."""
    import certifi

    ctx = ssl.create_default_context(cafile=certifi.where())
    urllib.request.install_opener(
        urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    )


def fetch(url: str, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 22):
            fh.write(chunk)
    tmp.rename(dst)


def pick(n: int, seed: int, lo: float, hi: float) -> list[str]:
    with urllib.request.urlopen(API, timeout=60) as r:
        files = json.load(r)
    groups = defaultdict(list)
    for f in files:
        name = f["path"].split("/")[1][: -len(".tar")]
        if lo <= f["size"] / 1e9 <= hi:
            groups[name.split("++")[0]].append(name)
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    keys = [keys[i] for i in rng.permutation(len(keys))[:n]]  # one sequence per scene / actor
    return sorted(groups[k][int(rng.integers(len(groups[k])))] for k in keys)


def stream_images(seq: str, anno: dict) -> int:
    serial = {s: nm for s, nm in anno["cam_def"].items() if nm in O.CAMS}
    keep = set(int(f) for f in anno["frame_id_list"])
    n = 0
    url = f"{O.HF}/data/{seq}.tar"
    with urllib.request.urlopen(url, timeout=120) as r, tarfile.open(fileobj=r, mode="r|") as tf:
        for m in tf:
            parts = Path(m.name).parts
            if not m.isfile() or len(parts) < 3 or parts[-2] not in serial:
                continue
            stem = Path(parts[-1]).stem
            if not stem.isdigit() or int(stem) not in keep:
                continue
            buf = np.frombuffer(tf.extractfile(m).read(), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
            dst = O.image_path(seq, serial[parts[-2]], int(stem))
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-gb", type=float, default=1.0)
    ap.add_argument("--max-gb", type=float, default=2.5)
    args = ap.parse_args()
    _https_with_certifi()
    for t in ("object_raw", "object_repair", "program"):  # small official tarballs
        if (O.ROOT / t / "done").exists():
            continue
        dst = O.ROOT / f"{t}.tar"
        fetch(f"{O.HF}/{t}.tar", dst)
        with tarfile.open(dst) as tf:
            tf.extractall(O.ROOT, filter="data")
        (O.ROOT / t / "done").write_text("")
        dst.unlink()  # extracted: the tarball is not needed again
    seqs = pick(args.n, args.seed, args.min_gb, args.max_gb)
    print("sequences:", *seqs, sep="\n  ", flush=True)
    t0 = time.time()
    for i, seq in enumerate(seqs):
        if (O.ROOT / "images" / seq / "done").exists():
            continue
        ap_ = O.anno_path(seq)
        fetch(f"{O.HF}/anno_preview/{seq}.pkl", ap_)
        anno = O.load_anno(ap_)
        n = stream_images(seq, anno)
        (O.ROOT / "images" / seq / "done").write_text(str(n))
        print(f"{i + 1}/{len(seqs)} {seq}: {n} images | {(time.time() - t0) / 60:.1f} min",
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
