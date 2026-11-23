"""V3-A (docs/PREREGISTRATION_v3.md): the CheXpert pair set, built from train.csv only.

1. Frontal images of the training split; a patient's studies ordered by study number.
2. Eligible patients have frontal images in at least two studies. Reference: first frontal image (by file
   name) of the first such study; query: first frontal image of the second.
3. 2,000 eligible patients drawn without replacement with numpy.random.default_rng(20261124) from the
   sorted eligible patient IDs.
Positives: the 2,000 same-patient query-reference pairs; negatives: all cross-patient combinations.

Output: results/V3A/chexpert_pairs.json (patients, reference and query paths, counts, SHA-256 of train.csv)
Usage:  python3 scripts/v3a_build_chexpert_pairs.py /path/to/CheXpert-v1.0-small
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED, N = 20261124, 2000


def main():
    base = Path(sys.argv[1])
    csv = base / "train.csv"
    df = pd.read_csv(csv)
    df = df[df["Frontal/Lateral"] == "Frontal"].copy()
    pat = df["Path"].str.extract(r"train/(patient\d+)/study(\d+)/(view\d+_\w+\.jpg)$")
    assert pat.notna().all().all(), "unexpected Path format"
    df["patient"], df["study"], df["file"] = pat[0], pat[1].astype(int), pat[2]
    first = {}
    for (p, s), g in df.groupby(["patient", "study"]):
        first.setdefault(p, []).append((s, sorted(g["Path"])[0]))
    eligible = sorted(p for p, v in first.items() if len(v) >= 2)
    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(np.array(eligible), size=N, replace=False).tolist())
    rows = []
    for p in chosen:
        v = sorted(first[p])
        rows.append({"patient": p, "reference": v[0][1], "query": v[1][1],
                     "reference_study": int(v[0][0]), "query_study": int(v[1][0])})
    out = {"source": "CheXpert-v1.0 batch 1 (validate & csv)/train.csv (AIMI container, amendment A3)", "train_csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
           "seed": SEED, "n_frontal_images": int(len(df)), "n_patients_frontal": int(df["patient"].nunique()),
           "n_eligible": len(eligible), "n_chosen": N, "n_negatives": N * (N - 1), "pairs": rows}
    d = ROOT / "results" / "V3A"
    d.mkdir(parents=True, exist_ok=True)
    (d / "chexpert_pairs.json").write_text(json.dumps(out, indent=1))
    print(f"eligible {len(eligible)} of {out['n_patients_frontal']} patients; chose {N}; negatives {N * (N - 1)}")


if __name__ == "__main__":
    main()
