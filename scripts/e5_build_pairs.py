"""E5: build the pretraining pair list of the independent strong verifier.

Pool: patients of the official NIH train/val list (28,008 patients, the list supplied with the
reference verifier), minus the patients of the released validation list (1,742), which are used
for checkpoint selection. Test and calibration patients are in the official test list and so are
not in the pool; the overlap is checked and written to the manifest, together with the overlap
with every other split of the study.

Pairs follow scripts/e1_build_clean_splits.py (same patient, different follow-up number, for
positives; image-weighted cross-patient sampling for negatives; balanced; random order within a
pair), except that positives are capped at 8 per patient: with every image of the pool available,
all same-patient combinations (397,799) would be dominated by patients with many follow-ups.

Output: results/E5/splits/e5_pretrain_pairs.txt and results/E5/splits/manifest.json
Usage:  python3 scripts/e5_build_pairs.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.e1_build_clean_splits import sample_negatives, write_pair_file  # noqa: E402

SEED = 20261122
CAP = 8
OUT = ROOT / "results" / "E5" / "splits"


def pts_of(path):
    return {int(x.split("_")[0]) for l in open(path) for x in l.split()[:2]}


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(ROOT / "Data_Entry_2017_v2020.csv")[["Image Index", "Patient ID", "Follow-up #"]]
    tv = set((ROOT / "image_pairs" / "train_val_list.txt").read_text().split())
    test_list = set((ROOT / "image_pairs" / "test_list.txt").read_text().split())
    v2k = pts_of(ROOT / "image_pairs" / "image_pairs_validation_2000.txt")
    d = df[df["Image Index"].isin(tv) & ~df["Patient ID"].isin(v2k)]
    avail = set(os.listdir(ROOT / "images")) | set(os.listdir(ROOT / "images_256_extra"))
    missing = sorted(set(d["Image Index"]) - avail)
    assert not missing, f"{len(missing)} pool images not on disk, e.g. {missing[:3]}"

    by_pt = defaultdict(list)
    for img, pid, fu in d.itertuples(index=False):
        by_pt[int(pid)].append((img, int(fu)))
    pos = []
    for pid in sorted(by_pt):
        cand = [(a, b) for (a, fa), (b, fb) in combinations(sorted(by_pt[pid]), 2) if fa != fb]
        if not cand:
            continue
        take = rng.choice(len(cand), size=min(CAP, len(cand)), replace=False)
        for i in sorted(take):
            a, b = cand[i]
            pos.append((a, b, 1) if rng.random() < 0.5 else (b, a, 1))
    p_to_imgs = {pid: [img for img, _ in v] for pid, v in by_pt.items()}
    neg = sample_negatives(p_to_imgs, len(pos), rng, {tuple(sorted((a, b))) for a, b, _ in pos})
    pairs = pos + neg
    rng.shuffle(pairs)
    OUT.mkdir(parents=True, exist_ok=True)
    sha = write_pair_file(OUT / "e5_pretrain_pairs.txt", pairs)

    e5 = {int(x.split("_")[0]) for a, b, _ in pairs for x in (a, b)}
    test_pts = {int(f.split("_")[0]) for f in test_list}
    splits = {
        "released test (testing_5000)": ROOT / "image_pairs" / "image_pairs_testing_5000.txt",
        "calibration (disjoint_700)": ROOT / "image_pairs" / "image_pairs_validation_disjoint_700.txt",
        "released validation (validation_2000)": ROOT / "image_pairs" / "image_pairs_validation_2000.txt",
        "verifier training (d_att_train_clean)": ROOT / "results" / "clean_splits" / "d_att_train_clean.txt",
        "local-verifier pretraining (d_pretrain_clean)": ROOT / "results" / "clean_splits" / "d_pretrain_clean.txt",
        "factorial training (training_10000)": ROOT / "image_pairs" / "image_pairs_training_10000.txt",
    }
    overlap = {k: len(e5 & pts_of(p)) for k, p in splits.items()}
    overlap["official test list (all 2,797 patients, incl. E6 candidates)"] = len(e5 & test_pts)
    manifest = {"seed": SEED, "cap_positives_per_patient": CAP, "n_pairs": len(pairs), "n_pos": len(pos),
                "n_neg": len(neg), "n_patients": len(e5), "n_images": len({x for a, b, _ in pairs for x in (a, b)}),
                "pool_patients": len(by_pt), "pool_images": len(d), "sha256": sha,
                "patient_overlap": overlap,
                "n_reference_list_patients": len({int(f.split("_")[0]) for f in tv})}
    assert overlap["released test (testing_5000)"] == 0 and overlap["calibration (disjoint_700)"] == 0
    assert overlap["released validation (validation_2000)"] == 0
    assert overlap["official test list (all 2,797 patients, incl. E6 candidates)"] == 0
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
