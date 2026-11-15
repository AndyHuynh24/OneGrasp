"""Fetch only some cameras' colour frames of HO-Cap subjects (CC BY 4.0) from the official box.com
zips via HTTP range requests (scripts/remote_zip.py), into data/hocap/ds/<subject>/<seq>/<cam>/.

--subjects: TRAINING / VALIDATION subjects only (5 and 8, the HO-Cap test set, are refused).
--clips: named sequences of ANY subject (incl. test) + their poses, for evaluation / rendering
only: training never reads test subjects (grasp_store.load_hocap_tracks asserts it).

    python scripts/download_hocap_cams.py --subjects 1 2 3 4 6 7 9 \\
        --cams 105322251564 043422252387 108222250342
    python scripts/download_hocap_cams.py --clips subject_5/20231027_112303 --cams 105322251225
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_zip import HttpRangeFile  # noqa: E402

ROOT = Path("data/hocap/ds")
URLS = Path("data/hocap/HO-Cap/config/hocap_recordings.yaml")
TEST_SUBJECTS = {5, 8}


def fetch(job) -> str:
    subject, cams = job
    assert subject not in TEST_SUBJECTS
    url = yaml.safe_load(URLS.read_text())[f"subject_{subject}"]
    z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=16 << 20))
    names = [
        i
        for i in z.infolist()
        if not i.is_dir()
        and (
            ("/color_" in i.filename and i.filename.split("/")[-2] in cams)
            or (i.filename.count("/") == 2 and i.filename.endswith(".yaml"))
        )
    ]  # + <seq>/meta.yaml
    names.sort(key=lambda i: i.header_offset)  # zip order: sequential range reads
    t0, n = time.time(), 0
    for i in names:
        dst = ROOT / i.filename
        if dst.exists() and dst.stat().st_size == i.file_size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        tmp.write_bytes(z.read(i))
        tmp.replace(dst)
        n += 1
        if n % 2000 == 0:
            print(f"subject_{subject}: {n}/{len(names)} | {(time.time() - t0) / 60:.1f} min",
                  flush=True)  # fmt: skip
    return f"subject_{subject}: {len(names)} frames ({n} new) in {(time.time() - t0) / 60:.1f} min"


def fetch_clip(clip: str, cams: set[str]) -> str:
    """One sequence (subject_N/<seq>, or subject_N/* = all of them): colour frames of `cams`,
    meta.yaml, poses_*.npy."""
    subject, seq = clip.split("/")
    pre = f"{subject}/" if seq == "*" else f"{subject}/{seq}/"
    cfg = yaml.safe_load(URLS.read_text())
    n = 0
    for url, keep in (
        (cfg[subject], lambda f: f.endswith("/meta.yaml")
         or ("/color_" in f and f.split("/")[-2] in cams)),
        (cfg["poses"], lambda f: "/poses_" in f),
    ):  # fmt: skip
        z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=16 << 20))
        names = [
            i
            for i in z.infolist()
            if not i.is_dir() and i.filename.startswith(pre) and keep(i.filename)
        ]
        names.sort(key=lambda i: i.header_offset)  # fmt: skip
        for i in names:
            dst = ROOT / i.filename
            if dst.exists() and dst.stat().st_size == i.file_size:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(z.read(i))
            tmp.replace(dst)
            n += 1
    return f"{clip}: {n} files"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=[])
    ap.add_argument("--clips", nargs="+", default=[], help="subject_N/<seq>, evaluation only")
    ap.add_argument("--cams", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    assert not set(args.subjects) & TEST_SUBJECTS, "subjects 5 and 8 are the HO-Cap test set"
    for clip in args.clips:
        print(fetch_clip(clip, set(args.cams)), flush=True)
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(fetch, [(s, set(args.cams)) for s in args.subjects]):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
