"""Download a subset of PressureVisionDB (full 15 FPS release) without downloading whole zips.

Each participant is one ~27 GB zip on Dropbox. Dropbox serves HTTP range requests, so we read the
zip's central directory remotely, pick only the members we want (some sequences, some cameras,
force + metadata), fetch them in a few large range requests and check every member's CRC32.

Output mirrors PressureVision's own layout, so their SequenceReader can read it:
    data/pvdb/raw/<split>/<participant>/<sequence>/
        camera_<c>/#####.jpg, force/#####.pkl, metadata.json

Usage:
    python scripts/download_pvdb.py                       # default selection, all 36 participants
    python scripts/download_pvdb.py --participants 9008_beryl_cabric --cameras 1
    python scripts/download_pvdb.py --dry-run             # print what would be downloaded

Re-run safe: sequences with a .done file are skipped.
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from pathlib import Path

# From PressureVision recording/downloader.py (download_full_zips), MIT license.
FULL_ZIPS = {
    "test/9010_cray_crane": "https://www.dropbox.com/s/mqmlypbhh17cxmv/9010_cray_crane.zip?dl=1",
    "test/9011_pug_rahl": "https://www.dropbox.com/s/zkki9adz4xj06nh/9011_pug_rahl.zip?dl=1",
    "test/9014_jean_anthony": "https://www.dropbox.com/s/r5ypnry8jwzrqiu/9014_jean_anthony.zip?dl=1",
    "test/9019_eden_lawrence": "https://www.dropbox.com/s/wo5ezuym4hqz7hg/9019_eden_lawrence.zip?dl=1",
    "test/9025_patricia_petty": "https://www.dropbox.com/s/80wrfg6xktgfzrs/9025_patricia_petty.zip?dl=1",
    "test/9036_dellie_roy": "https://www.dropbox.com/s/3rskhp0cblsiqn4/9036_dellie_roy.zip?dl=1",
    "train_fold_1/9002_gwydion_black": "https://www.dropbox.com/s/ant48d0d81fps1g/9002_gwydion_black.zip?dl=1",
    "train_fold_1/9009_murlynd_rune": "https://www.dropbox.com/s/an672rw160gdrac/9009_murlynd_rune.zip?dl=1",
    "train_fold_1/9017_charles_gibson": "https://www.dropbox.com/s/m9uvuorec14aoa0/9017_charles_gibson.zip?dl=1",
    "train_fold_1/9023_alec_ben": "https://www.dropbox.com/s/euqtxf3yhl24spa/9023_alec_ben.zip?dl=1",
    "train_fold_1/9029_cassie_fitzgerald": "https://www.dropbox.com/s/28tj4j3n11uo59w/9029_cassie_fitzgerald.zip?dl=1",
    "train_fold_1/9034_evard_dresden": "https://www.dropbox.com/s/iq3t74vtnel4owt/9034_evard_dresden.zip?dl=1",
    "train_fold_2/9003_julian_case": "https://www.dropbox.com/s/4tb5j878y7ktoet/9003_julian_case.zip?dl=1",
    "train_fold_2/9012_archie_mccullough": "https://www.dropbox.com/s/h6hsggac1fvpjs9/9012_archie_mccullough.zip?dl=1",
    "train_fold_2/9018_winifer_dee": "https://www.dropbox.com/s/oqfbhk0pbds7ap5/9018_winifer_dee.zip?dl=1",
    "train_fold_2/9024_leanardo_mcdaniel": "https://www.dropbox.com/s/o0xa40qk6d1da1x/9024_leanardo_mcdaniel.zip?dl=1",
    "train_fold_2/9030_haley_rosmerta": "https://www.dropbox.com/s/lhsh69k7pjuw7gy/9030_haley_rosmerta.zip?dl=1",
    "train_fold_2/9035_griselda_hollow": "https://www.dropbox.com/s/9o7z5m4dlfl433y/9035_griselda_hollow.zip?dl=1",
    "train_fold_3/9006_jane_fox": "https://www.dropbox.com/s/qwvn8r7yv38czq3/9006_jane_fox.zip?dl=1",
    "train_fold_3/9013_john_archer": "https://www.dropbox.com/s/wu1hnsggu298t15/9013_john_archer.zip?dl=1",
    "train_fold_3/9020_nystul_baxter": "https://www.dropbox.com/s/3dyheg2qja3fsnx/9020_nystul_baxter.zip?dl=1",
    "train_fold_3/9026_beth_kane": "https://www.dropbox.com/s/ld6fc8gg2dgd80x/9026_beth_kane.zip?dl=1",
    "train_fold_3/9031_rhys_gibbs": "https://www.dropbox.com/s/m1owakjedxj6udh/9031_rhys_gibbs.zip?dl=1",
    "train_fold_3/9037_jasmine_meyers": "https://www.dropbox.com/s/1wryf03fwp45tku/9037_jasmine_meyers.zip?dl=1",
    "train_fold_4/9007_mustrum_buchanan": "https://www.dropbox.com/s/z55rkrl19e25yqu/9007_mustrum_buchanan.zip?dl=1",
    "train_fold_4/9015_kovertol_potter": "https://www.dropbox.com/s/jy99rd1d1eua8b1/9015_kovertol_potter.zip?dl=1",
    "train_fold_4/9021_owen_osullivan": "https://www.dropbox.com/s/pkrsszh6lo0xc7f/9021_owen_osullivan.zip?dl=1",
    "train_fold_4/9027_mason_beck": "https://www.dropbox.com/s/lsl9mniugmlcc90/9027_mason_beck.zip?dl=1",
    "train_fold_4/9032_emily_wallace": "https://www.dropbox.com/s/wg5x4sjx86om6ba/9032_emily_wallace.zip?dl=1",
    "train_fold_4/9038_lottie_shaw": "https://www.dropbox.com/s/ii6eashs3wj82t9/9038_lottie_shaw.zip?dl=1",
    "val_fold_5/9008_beryl_cabric": "https://www.dropbox.com/s/3asuynuo9uf5swm/9008_beryl_cabric.zip?dl=1",
    "val_fold_5/9016_frankie_brady": "https://www.dropbox.com/s/gf1r3cx0wyffu1c/9016_frankie_brady.zip?dl=1",
    "val_fold_5/9022_herbert_stokes": "https://www.dropbox.com/s/hdpi9alko5v8amx/9022_herbert_stokes.zip?dl=1",
    "val_fold_5/9028_bailey_larson": "https://www.dropbox.com/s/qlavuo6372ktu99/9028_bailey_larson.zip?dl=1",
    "val_fold_5/9033_rowan_burch": "https://www.dropbox.com/s/ugkicn91uw8ekq0/9033_rowan_burch.zip?dl=1",
    "val_fold_5/9039_tiana_gordon": "https://www.dropbox.com/s/nm52v65xccd8e31/9039_tiana_gordon.zip?dl=1",
}

# Same gesture names as EgoPressure (EgoPressure reused the PressureVision protocol).
DEFAULT_SEQ_REGEX = (
    r"^(index_press_(high|low|no-contact|pull|push|left|right)_x5"
    r"|press_fingers_(high|low|no-contact)_5x)_(left|right)$"
)
RUN_MAX_BYTES = 64 << 20  # one range request fetches at most this much
RUN_MAX_GAP = 1 << 20  # merge members separated by less than this


def curl(*args: str, retries: int = 5) -> bytes:
    for attempt in range(retries):
        r = subprocess.run(["curl", "-sfL", "--retry", "3", *args], capture_output=True)
        if r.returncode == 0:
            return r.stdout
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"curl failed ({r.returncode}): {args[-1][:80]}")


class RemoteFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (via curl)."""

    def __init__(self, share_url: str):
        self.share_url = share_url
        self.resolve()
        self.pos = 0

    def resolve(self) -> None:
        # Dropbox redirects to a temporary download URL; re-resolve when it expires.
        out = curl(
            "-r", "0-0", "-D", "-", "-o", "/dev/null", "-w", "\n%{url_effective}", self.share_url
        ).decode()
        self.url = out.strip().splitlines()[-1]
        self.size = int(re.search(r"content-range: bytes 0-0/(\d+)", out, re.I).group(1))

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = [off, self.pos + off, self.size + off][whence]
        return self.pos

    def read_range(self, start: int, end: int) -> bytes:
        """Bytes [start, end] inclusive."""
        for _ in range(3):
            try:
                data = curl("-r", f"{start}-{end}", self.url)
                if len(data) == end - start + 1:
                    return data
            except RuntimeError:
                pass
            self.resolve()
        raise RuntimeError(f"range {start}-{end} failed")

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        data = self.read_range(self.pos, end)
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def member_end(zi: zipfile.ZipInfo) -> int:
    # Upper bound: local header + name + (central) extra + data + data descriptor.
    return zi.header_offset + 30 + len(zi.filename.encode()) + len(zi.extra) + zi.compress_size + 24


def make_runs(members: list[zipfile.ZipInfo]) -> list[list[zipfile.ZipInfo]]:
    members = sorted(members, key=lambda z: z.header_offset)
    runs: list[list[zipfile.ZipInfo]] = []
    for zi in members:
        if runs:
            run = runs[-1]
            gap = zi.header_offset - member_end(run[-1])
            if gap < RUN_MAX_GAP and member_end(zi) - run[0].header_offset < RUN_MAX_BYTES:
                run.append(zi)
                continue
        runs.append([zi])
    return runs


def extract_member(buf: bytes, base: int, zi: zipfile.ZipInfo) -> bytes:
    off = zi.header_offset - base
    sig, _, _, method, _, _, _, _, _, nlen, xlen = struct.unpack(
        "<IHHHHHIIIHH", buf[off : off + 30]
    )
    if sig != 0x04034B50:
        raise ValueError(f"bad local header for {zi.filename}")
    start = off + 30 + nlen + xlen
    raw = buf[start : start + zi.compress_size]
    if method == zipfile.ZIP_STORED:
        data = raw
    elif method == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(raw, -15)
    else:
        raise ValueError(f"unsupported compression {method} for {zi.filename}")
    if zlib.crc32(data) != zi.CRC or len(data) != zi.file_size:
        raise ValueError(f"CRC/size mismatch for {zi.filename}")
    return data


def download_participant(
    key: str,
    url: str,
    out_root: Path,
    seq_re: re.Pattern,
    cameras: list[str],
    min_free_gb: float,
    dry_run: bool,
) -> tuple[int, int]:
    rf = RemoteFile(url)
    z = zipfile.ZipFile(io.BufferedReader(rf, 1 << 20))
    by_seq: dict[str, list[zipfile.ZipInfo]] = {}
    keep_dirs = tuple(f"camera_{c}/" for c in cameras) + ("force/",)
    for zi in z.infolist():
        if zi.is_dir():
            continue
        parts = zi.filename.split("/")
        if len(parts) < 2 or not seq_re.match(parts[0]):
            continue
        rest = "/".join(parts[1:])
        if rest == "metadata.json" or rest.startswith(keep_dirs):
            by_seq.setdefault(parts[0], []).append(zi)

    n_done = n_fail = 0
    for seq in sorted(by_seq):
        seq_dir = out_root / key / seq
        if (seq_dir / ".done").exists():
            continue
        members = by_seq[seq]
        nbytes = sum(m.compress_size for m in members)
        if dry_run:
            print(f"  would get {key}/{seq}: {len(members)} files, {nbytes / 1e6:.0f} MB")
            continue
        if shutil.disk_usage(out_root).free / 1e9 < min_free_gb:
            print(f"STOP: less than {min_free_gb} GB free", file=sys.stderr)
            sys.exit(2)
        t0 = time.time()
        try:
            for run in make_runs(members):
                start = run[0].header_offset
                end = min(member_end(run[-1]) + 1024, rf.size) - 1
                buf = rf.read_range(start, end)
                for zi in run:
                    dest = out_root / key / zi.filename
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(extract_member(buf, start, zi))
        except (RuntimeError, ValueError) as e:
            n_fail += 1
            print(f"FAIL  {key}/{seq}: {e}", file=sys.stderr)
            with open(out_root / "failed.txt", "a") as f:
                f.write(f"{time.strftime('%F %T')}  {key}/{seq}  {e}\n")
            continue
        dt = time.time() - t0
        n_frames = sum(1 for m in members if m.filename.split("/")[1] == "force")
        mb = nbytes / 1e6
        line = (
            f"{key}/{seq}  files={len(members)}  frames={n_frames}  {mb:.0f} MB  {dt:.0f}s  "
            f"{mb / max(dt, 1e-3):.1f} MB/s"
        )
        (seq_dir / ".done").write_text(line + "\n")
        with open(out_root / "downloaded.txt", "a") as f:
            f.write(line + "\n")
        print("done ", line, flush=True)
        n_done += 1
    return n_done, n_fail


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="data/pvdb/raw")
    ap.add_argument("--participants", nargs="*", help="e.g. 9008_beryl_cabric (default: all 36)")
    ap.add_argument(
        "--cameras", nargs="+", default=["1"], help="camera_1 = overhead (1.6 deg from pad normal)"
    )
    ap.add_argument("--seq-regex", default=DEFAULT_SEQ_REGEX)
    ap.add_argument("--min-free-gb", type=float, default=40)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    seq_re = re.compile(args.seq_regex)
    keys = [k for k in FULL_ZIPS if not args.participants or k.split("/")[1] in args.participants]
    tot_done = tot_fail = 0
    for key in keys:
        print(f"== {key}", flush=True)
        try:
            d, f = download_participant(
                key, FULL_ZIPS[key], out_root, seq_re, args.cameras, args.min_free_gb, args.dry_run
            )
        except (RuntimeError, zipfile.BadZipFile) as e:
            print(f"FAIL  {key}: {e}", file=sys.stderr)
            with open(out_root / "failed.txt", "a") as fh:
                fh.write(f"{time.strftime('%F %T')}  {key}  {e}\n")
            tot_fail += 1
            continue
        tot_done += d
        tot_fail += f
    print(f"finished: {tot_done} sequences downloaded, {tot_fail} failures")


if __name__ == "__main__":
    main()
