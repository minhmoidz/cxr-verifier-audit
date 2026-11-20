"""Decode every image referenced by the pair lists once into a uint8 memmap.

The datasets re-decode ~34k PNGs on every run (~11 min). Screening experiments need to
run in seconds, not minutes, so each unique image is decoded once at 256x256 and stored
in a flat array addressed by a filename -> row index map.
"""

import os
import json
import numpy as np
from PIL import Image
from multiprocessing import Pool

CACHE_DIR = './cache/'
SIZE = 256
PAIR_FILES = [
    './image_pairs/image_pairs_training_10000.txt',
    './image_pairs/image_pairs_validation_2000.txt',
    './image_pairs/image_pairs_testing_5000.txt',
]


def unique_filenames():
    names = set()
    for f in PAIR_FILES:
        pairs = np.loadtxt(f, dtype=str)
        names.update(pairs[:, 0])
        names.update(pairs[:, 1])
    return sorted(names)


def decode(args):
    idx, name, image_path = args
    with open(image_path + name, 'rb') as f:
        img = Image.open(f).convert('L').resize((SIZE, SIZE), Image.BILINEAR)
    return idx, np.asarray(img, dtype=np.uint8)


def main(image_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images') + '/', workers=8):
    os.makedirs(CACHE_DIR, exist_ok=True)
    names = unique_filenames()
    print(f'{len(names)} unique images -> {len(names) * SIZE * SIZE / 1e9:.2f} GB')

    index = {name: i for i, name in enumerate(names)}
    with open(CACHE_DIR + 'index.json', 'w') as f:
        json.dump(index, f)

    arr = np.lib.format.open_memmap(CACHE_DIR + f'images_{SIZE}.npy', mode='w+',
                                    dtype=np.uint8, shape=(len(names), SIZE, SIZE))

    with Pool(workers) as pool:
        work = ((i, n, image_path) for i, n in enumerate(names))
        for done, (idx, pixels) in enumerate(pool.imap_unordered(decode, work, chunksize=64)):
            arr[idx] = pixels
            if done % 2000 == 0:
                print(f'  {done}/{len(names)}', flush=True)

    arr.flush()
    print('cache written to', CACHE_DIR)


if __name__ == '__main__':
    main()
