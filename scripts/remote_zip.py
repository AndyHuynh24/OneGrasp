"""List / extract single members of a large remote zip via HTTP range requests (no full download).

python scripts/remote_zip.py URL                       # list members with sizes
python scripts/remote_zip.py URL --get NAME [NAME..] --out DIR
"""

from __future__ import annotations

import argparse
import io
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

try:  # python.org builds ship without CA certificates
    import certifi

    CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = ssl.create_default_context()


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        # size from a 1-byte ranged GET (some hosts, e.g. box.com, answer HEAD with 404)
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, context=CTX) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        )
        with urllib.request.urlopen(req, context=CTX) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--get", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("."))
    args = ap.parse_args()
    f = io.BufferedReader(HttpRangeFile(args.url), buffer_size=8 << 20)
    z = zipfile.ZipFile(f)
    if not args.get:
        for i in z.infolist():
            print(f"{i.file_size / 1e6:10.1f} MB  {i.filename}")
        return
    for name in args.get:
        dst = args.out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        with z.open(name) as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out, 8 << 20)
        print("wrote", dst, dst.stat().st_size)


if __name__ == "__main__":
    main()
