"""Read selected members of a remote zip (Azure Blob SAS URL) with HTTP range requests (amendment A1).

Only the zip's central directory and the requested members are transferred. The URL carries a temporary
access token: pass it through the environment variable CHEXPERT_ZIP_URL, never on the command line or in a
file under version control.

  python3 tools/v3_remote_zip.py csv  OUT_DIR              # extract train.csv (and valid.csv)
  python3 tools/v3_remote_zip.py pairs OUT_DIR             # extract the images listed in results/V3A/chexpert_pairs.json
Writes OUT_DIR/<member path> and results/V3A/remote_files.json (member, size, CRC-32, SHA-256).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "V3A"


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file; reads become Range requests (with a small block cache)."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        self.s = requests.Session()
        r = self.s.head(url, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _get(self, a, b):
        for k in range(5):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=120)
                r.raise_for_status()
                assert len(r.content) == b - a
                return r.content
            except Exception:
                if k == 4:
                    raise
                time.sleep(2 ** k)

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > 4 * self.block:                                   # large member: one direct request
            data = self._get(self.pos, self.pos + n)
        else:
            out = []
            a, end = self.pos, self.pos + n
            while a < end:
                bi = a // self.block
                if bi not in self.cache:
                    if len(self.cache) > 64:
                        self.cache.clear()
                    lo = bi * self.block
                    self.cache[bi] = self._get(lo, min(lo + self.block, self.size))
                blk = self.cache[bi]
                o = a - bi * self.block
                take = min(end - a, len(blk) - o)
                out.append(blk[o:o + take])
                a += take
            data = b"".join(out)
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def open_zip():
    url = os.environ["CHEXPERT_ZIP_URL"]
    return zipfile.ZipFile(HTTPRangeFile(url))


def extract(z: zipfile.ZipFile, names, out: Path, workers=1):
    rec = []
    for nm in names:
        dst = out / nm
        if dst.is_file():
            data = dst.read_bytes()
        else:
            data = z.read(nm)                                    # zipfile checks the CRC-32
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        info = z.getinfo(nm)
        rec.append({"member": nm, "size": info.file_size, "crc32": f"{info.CRC:08x}",
                    "sha256": hashlib.sha256(data).hexdigest()})
    return rec


def main():
    mode, out = sys.argv[1], Path(sys.argv[2])
    z = open_zip()
    names = z.namelist()
    print(f"remote zip: {len(names)} members", flush=True)
    if mode == "csv":
        want = [n for n in names if n.endswith("/train.csv") or n.endswith("/valid.csv")]
        prefix = want[0].rsplit("/", 1)[0]
        print("csv members:", want, "prefix:", prefix)
    else:
        pairs = json.load(open(D / "chexpert_pairs.json"))["pairs"]
        root = json.load(open(D / "chexpert_pairs.json")).get("zip_prefix", "")
        want = []
        for p in pairs:
            for k in ("reference", "query"):
                nm = p[k] if p[k] in set(names) else f"{root}/{p[k].split('/', 1)[1]}"
                want.append(nm)
        missing = [n for n in want if n not in set(names)]
        assert not missing, f"{len(missing)} members not in zip, e.g. {missing[:3]}"
    rec = extract(z, want, out)
    f = D / "remote_files.json"
    old = json.load(open(f)) if f.is_file() else []
    seen = {r["member"] for r in rec}
    json.dump([r for r in old if r["member"] not in seen] + rec, open(f, "w"), indent=1)
    print(f"extracted {len(rec)} members to {out}")


if __name__ == "__main__":
    main()
