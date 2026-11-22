"""Fetch the NIH ChestX-ray14 train/val images that are not on disk, stored at 256 x 256 (E5).

E5 trains a verifier on the full official train/val list (28,008 patients, 86,524 images). Only
the images of the released training and validation lists are in images/. This script streams
the twelve NIH tarballs one at a time (nothing is written but the selected images), and for each
member:
  * if it is a missing train/val image: decode, convert to 8-bit grayscale and resize to
    256 x 256 with PIL bilinear resampling -- the exact operation of build_image_cache.py and of
    the RAMImageStore fallback in scripts/prl_lib.py -- and save it losslessly as PNG in
    images_256_extra/. Loading such a file through the pipeline gives the same pixels as loading
    the original, because a 256 x 256 image is not resized again.
  * if the same file already exists in images/: compare SHA-256 with the local copy. Any mismatch
    aborts, so the download is checked against the images used for every published result.

Download links are the NIH Clinical Center's static Box links (batch_download_zips.py).
Usage: python3 tools/fetch_nih_trainval_256.py [--parts 1,2,...] [--out images_256_extra]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LINKS = [
    "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
    "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
    "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
    "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
    "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
    "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
    "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
    "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
    "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
    "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
    "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
    "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=",".join(str(i) for i in range(1, 13)))
    ap.add_argument("--out", default="images_256_extra")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    local = ROOT / "images"
    trval = [l.strip() for l in open(ROOT / "image_pairs" / "train_val_list.txt") if l.strip()]
    on_disk = set(os.listdir(local))
    needed = sorted(set(trval) - on_disk)
    log_dir = ROOT / "results" / "E5" / "fetch"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "needed_images.txt").write_text("\n".join(needed) + "\n")
    needed_set = set(needed)
    print(f"train/val images: {len(trval)}; on disk: {len(set(trval) & on_disk)}; to fetch: {len(needed)}", flush=True)

    for part in (int(p) for p in args.parts.split(",")):
        done_marker = log_dir / f"part{part:02d}.json"
        if done_marker.is_file():
            print(f"part {part}: already done", flush=True)
            continue
        t0 = time.time()
        stats = {"part": part, "url": LINKS[part - 1], "members": 0, "saved": 0, "skipped_existing": 0,
                 "verified_identical": 0, "mismatch": [], "bytes": 0}
        req = urllib.request.Request(LINKS[part - 1], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            with tarfile.open(fileobj=resp, mode="r|gz") as tar:
                for m in tar:
                    if not m.isfile():
                        continue
                    name = os.path.basename(m.name)
                    stats["members"] += 1
                    stats["bytes"] += m.size
                    if name in needed_set:
                        dst = out / name
                        if dst.is_file():
                            stats["skipped_existing"] += 1
                            continue
                        data = tar.extractfile(m).read()
                        with Image.open(io.BytesIO(data)) as im:
                            g = im.convert("L").resize((256, 256), Image.BILINEAR)
                        tmp = dst.with_suffix(".tmp.png")
                        g.save(tmp, format="PNG")
                        os.replace(tmp, dst)
                        stats["saved"] += 1
                    elif name in on_disk:
                        data = tar.extractfile(m).read()
                        if sha256(data) == sha256((local / name).read_bytes()):
                            stats["verified_identical"] += 1
                        else:
                            stats["mismatch"].append(name)
                    if stats["members"] % 2000 == 0:
                        mb = stats["bytes"] / 1e6
                        print(f"  part {part}: {stats['members']} members, {mb:.0f} MB, "
                              f"{mb / (time.time() - t0):.1f} MB/s, saved {stats['saved']}", flush=True)
        stats["minutes"] = round((time.time() - t0) / 60, 1)
        if stats["mismatch"]:
            raise SystemExit(f"part {part}: {len(stats['mismatch'])} members differ from local images: "
                             f"{stats['mismatch'][:5]}")
        done_marker.write_text(json.dumps(stats, indent=1))
        print(f"part {part} done: {json.dumps({k: v for k, v in stats.items() if k != 'mismatch'})}", flush=True)

    have = set(os.listdir(out))
    missing = sorted(needed_set - have)
    (log_dir / "still_missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    print(f"finished: {len(needed_set) - len(missing)} of {len(needed_set)} fetched; missing {len(missing)}", flush=True)


if __name__ == "__main__":
    main()
