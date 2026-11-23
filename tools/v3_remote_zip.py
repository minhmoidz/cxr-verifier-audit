"""Read selected members of a remote zip (Azure Blob SAS URL) with HTTP range requests (amendment A1).

Only the zips' central directories and the requested members are transferred. The container URL carries
a temporary access token: pass it through the environment variable CHEXPERT_CONTAINER (read/list SAS),
never on the command line or in a file under version control. CheXpert is split into batch zips
(amendment A3): batch 1 holds train.csv; batches 2-4 hold the training images.

  python3 tools/v3_remote_zip.py csv  OUT_DIR    # extract train.csv and valid.csv from batch 1
  python3 tools/v3_remote_zip.py pairs OUT_DIR   # extract the images of results/V3A/chexpert_pairs.json
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


BATCHES = ["CheXpert-v1.0 batch 1 (validate & csv).zip", "CheXpert-v1.0 batch 2 (train 1).zip",
           "CheXpert-v1.0 batch 3 (train 2).zip", "CheXpert-v1.0 batch 4 (train 3).zip"]


def open_zip(blob):
    from urllib.parse import quote
    c = os.environ["CHEXPERT_CONTAINER"]
    base, sas = c.split("?", 1)
    return zipfile.ZipFile(HTTPRangeFile(f"{base}/{quote(blob)}?{sas}"))


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


def preprocess_256(data: bytes):
    """Pre-specified V3-A preprocessing: grayscale, center square crop, bilinear resize to 256 x 256."""
    from PIL import Image
    from torchvision import transforms
    with Image.open(io.BytesIO(data)) as im:
        g = im.convert("L")
        w, h = g.size
        s = min(w, h)
        g = g.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return transforms.Resize((256, 256))(g), (w, h)


def fetch_256(z: zipfile.ZipFile, names, out: Path):
    """Read each member (CRC-checked), record its SHA-256, and store only the preprocessed 256 x 256 PNG at
    OUT/<patient>/<study>/<view>.png (the full-resolution file is not kept)."""
    rec = []
    for nm in names:
        dst = out / (tail_key(nm)[:-4] + ".png")
        meta = dst.with_suffix(".json")
        if dst.is_file() and meta.is_file():
            rec.append(json.loads(meta.read_text())); continue
        data = z.read(nm)
        img, (w, h) = preprocess_256(data)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst)
        r = {"member": nm, "size": len(data), "crc32": f"{z.getinfo(nm).CRC:08x}",
             "sha256": hashlib.sha256(data).hexdigest(), "width": w, "height": h}
        meta.write_text(json.dumps(r))
        rec.append(r)
    return rec


def tail_key(name):
    """patientNNNNN/studyN/viewN_frontal.jpg: the part of a member path that identifies an image."""
    m = __import__("re").search(r"(patient\d+/study\d+/view\d+_\w+\.jpg)$", name)
    return m.group(1) if m else None


def main():
    mode, out = sys.argv[1], Path(sys.argv[2])
    f = D / "remote_files.json"
    old = json.load(open(f)) if f.is_file() else []
    rec = []
    if mode == "csv":
        z = open_zip(BATCHES[0])
        names = z.namelist()
        want = [n for n in names if n.endswith("train.csv") or n.endswith("valid.csv")]
        print(f"batch 1: {len(names)} members; csv members: {want}", flush=True)
        rec = [dict(r, batch=BATCHES[0]) for r in extract(z, want, out)]
    else:
        pairs = json.load(open(D / "chexpert_pairs.json"))["pairs"]
        need = {tail_key(p[k]) for p in pairs for k in ("reference", "query")}
        found = {}
        for blob in BATCHES[1:]:
            names = open_zip(blob).namelist()
            hit = {tail_key(n): n for n in names if tail_key(n) in need}
            print(f"{blob}: {len(names)} members, {len(hit)} needed", flush=True)
            found.update({k: (blob, v) for k, v in hit.items()})
            todo = sorted(hit.values())
            chunks = [todo[i::8] for i in range(8)]                # 8 threads, one remote handle each
            with ThreadPoolExecutor(8) as ex:
                for part in ex.map(lambda c: fetch_256(open_zip(blob), c, out) if c else [], chunks):
                    rec += [dict(r, batch=blob) for r in part]
        missing = need - set(found)
        assert not missing, f"{len(missing)} images not found, e.g. {sorted(missing)[:3]}"
        json.dump({k: v[1] for k, v in found.items()}, open(D / "chexpert_member_map.json", "w"), indent=0)
    seen = {r["member"] for r in rec}
    json.dump([r for r in old if r["member"] not in seen] + rec, open(f, "w"), indent=1)
    print(f"extracted {len(rec)} members to {out}")


if __name__ == "__main__":
    main()
