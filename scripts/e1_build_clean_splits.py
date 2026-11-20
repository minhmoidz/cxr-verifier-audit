#!/usr/bin/env python3
"""Build clean, patient-disjoint, deduplicated ChestX-ray14 splits for confirmatory PRL experiments (E1-E4).

All outputs live under results/clean_splits/ and NEVER overwrite legacy image_pairs/*.txt files.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "clean_splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20261113

FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


def load_pair_file(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) >= 3:
                rows.append((toks[0], toks[1], int(float(toks[2]))))
    return rows


def write_pair_file(path: Path, pairs: list[tuple[str, str, int]]) -> str:
    with path.open("w") as f:
        for a, b, y in pairs:
            f.write(f"{a} {b} {y}\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_negatives(
    patient_to_imgs: dict[int, list[str]],
    n_neg: int,
    rng: np.random.Generator,
    forbidden: set[tuple[str, str]],
) -> list[tuple[str, str, int]]:
    pids = sorted(patient_to_imgs.keys())
    probs = np.array([len(patient_to_imgs[p]) for p in pids], dtype=np.float64)
    probs /= probs.sum()
    negs = []
    seen_unordered = set(forbidden)
    attempts = 0
    while len(negs) < n_neg:
        attempts += 1
        if attempts > n_neg * 50:
            raise RuntimeError("Exceeded max attempts sampling unique negatives")
        p1, p2 = rng.choice(pids, size=2, replace=False, p=probs)
        i1 = rng.choice(patient_to_imgs[int(p1)])
        i2 = rng.choice(patient_to_imgs[int(p2)])
        key = tuple(sorted((i1, i2)))
        if key in seen_unordered:
            continue
        seen_unordered.add(key)
        if rng.random() < 0.5:
            negs.append((i1, i2, 0))
        else:
            negs.append((i2, i1, 0))
    return negs


def build_clean_pairs_from_patients(
    df_pool: pd.DataFrame,
    patient_ids: set[int],
    rng: np.random.Generator,
    max_pos: int | None = None,
) -> list[tuple[str, str, int]]:
    sub = df_pool[df_pool["_1"].isin(patient_ids)].copy()
    p_to_rows: dict[int, list[tuple[str, int]]] = defaultdict(list)
    p_to_imgs: dict[int, list[str]] = defaultdict(list)
    for row in sub.itertuples(index=False):
        pid = int(row._1)  # Patient ID
        img = str(row._0)  # Image Index
        fu = int(row._2)   # Follow-up #
        p_to_rows[pid].append((img, fu))
        p_to_imgs[pid].append(img)

    all_pos: list[tuple[str, str, int]] = []
    for pid, items in sorted(p_to_rows.items()):
        for (i1, f1), (i2, f2) in combinations(items, 2):
            if f1 != f2:
                if rng.random() < 0.5:
                    all_pos.append((i1, i2, 1))
                else:
                    all_pos.append((i2, i1, 1))

    rng.shuffle(all_pos)
    if max_pos is not None and max_pos < len(all_pos):
        pos = all_pos[:max_pos]
    else:
        pos = all_pos

    forbidden = {tuple(sorted((a, b))) for a, b, _ in pos}
    neg = sample_negatives(p_to_imgs, len(pos), rng, forbidden)
    combined = pos + neg
    rng.shuffle(combined)
    return combined


def audit_pair_list(pairs: list[tuple[str, str, int]], img_to_pt: dict[str, int], img_to_fu: dict[str, int]):
    pts = set()
    unordered = set()
    self_cnt = 0
    same_pt_in_neg = 0
    diff_pt_in_pos = 0
    same_fu_in_pos = 0
    gap1 = 0
    gap5 = 0
    for a, b, y in pairs:
        if a == b:
            self_cnt += 1
        unordered.add(tuple(sorted((a, b))))
        pa, pb = img_to_pt[a], img_to_pt[b]
        pts.add(pa); pts.add(pb)
        if y == 1:
            if pa != pb:
                diff_pt_in_pos += 1
            d = abs(img_to_fu[a] - img_to_fu[b])
            if d == 0:
                same_fu_in_pos += 1
            elif d == 1:
                gap1 += 1
            if d >= 5:
                gap5 += 1
        else:
            if pa == pb:
                same_pt_in_neg += 1
    return {
        "n_pairs": len(pairs),
        "n_pos": sum(y for _, _, y in pairs),
        "n_neg": sum(1 - y for _, _, y in pairs),
        "n_unique_unordered": len(unordered),
        "n_duplicates": len(pairs) - len(unordered),
        "n_unique_patients": len(pts),
        "self_pairs": self_cnt,
        "label_errors": diff_pt_in_pos + same_pt_in_neg,
        "pos_same_followup": same_fu_in_pos,
        "pos_abs_fu_1": gap1,
        "pos_abs_fu_ge5": gap5,
    }


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(ROOT / "Data_Entry_2017_v2020.csv")
    # Reorder columns cleanly for positional access
    df_small = df[["Image Index", "Patient ID", "Follow-up #"]].rename(
        columns={"Image Index": "_0", "Patient ID": "_1", "Follow-up #": "_2"}
    )
    img_to_pt = dict(zip(df["Image Index"], df["Patient ID"]))
    img_to_fu = dict(zip(df["Image Index"], df["Follow-up #"]))

    on_disk = set(os.listdir(ROOT / "images"))
    tv_set = set((ROOT / "image_pairs" / "train_val_list.txt").read_text().split())
    te_set = set((ROOT / "image_pairs" / "test_list.txt").read_text().split())

    tr10 = load_pair_file(ROOT / "image_pairs" / "image_pairs_training_10000.txt")
    v2k = load_pair_file(ROOT / "image_pairs" / "image_pairs_validation_2000.txt")
    d700 = load_pair_file(ROOT / "image_pairs" / "image_pairs_validation_disjoint_700.txt")
    te5k = load_pair_file(ROOT / "image_pairs" / "image_pairs_testing_5000.txt")

    p_tr10 = {img_to_pt[a] for a, b, _ in tr10} | {img_to_pt[b] for a, b, _ in tr10}
    p_v2k = {img_to_pt[a] for a, b, _ in v2k} | {img_to_pt[b] for a, b, _ in v2k}
    p_d700 = {img_to_pt[a] for a, b, _ in d700} | {img_to_pt[b] for a, b, _ in d700}
    p_te5k = {img_to_pt[a] for a, b, _ in te5k} | {img_to_pt[b] for a, b, _ in te5k}

    # Verify all p_tr10 images exist on disk and are in train_val_list
    df_tr10_pool = df_small[df_small["_1"].isin(p_tr10) & df_small["_0"].isin(on_disk & tv_set)].copy()
    multi_pts = sorted(
        p for p, cnt in df_tr10_pool.groupby("_1")["_0"].count().items() if cnt >= 2
    )
    single_pts = sorted(set(p_tr10) - set(multi_pts))
    multi_arr = np.array(multi_pts); rng.shuffle(multi_arr)
    single_arr = np.array(single_pts); rng.shuffle(single_arr)

    # Stratified 2:1 patient split -> D_pretrain (~2/3) vs D_att_train (~1/3)
    n_m_pre = (2 * len(multi_arr)) // 3
    n_s_pre = (2 * len(single_arr)) // 3
    pts_pretrain = set(map(int, multi_arr[:n_m_pre])) | set(map(int, single_arr[:n_s_pre]))
    pts_att_train = set(map(int, multi_arr[n_m_pre:])) | set(map(int, single_arr[n_s_pre:]))
    assert len(pts_pretrain & pts_att_train) == 0

    pairs_pretrain = build_clean_pairs_from_patients(df_tr10_pool, pts_pretrain, rng)
    pairs_att_clean = build_clean_pairs_from_patients(df_tr10_pool, pts_att_train, rng)
    pairs_all_tv = build_clean_pairs_from_patients(df_tr10_pool, p_tr10, rng)

    # Equal-size 700-pair validation subsample from validation_2000 (350 pos + 350 neg)
    v2k_pos = [x for x in v2k if x[2] == 1]
    v2k_neg = [x for x in v2k if x[2] == 0]
    pos_idx = rng.choice(len(v2k_pos), size=350, replace=False)
    neg_idx = rng.choice(len(v2k_neg), size=350, replace=False)
    pairs_v2k_700 = [v2k_pos[i] for i in pos_idx] + [v2k_neg[i] for i in neg_idx]
    rng.shuffle(pairs_v2k_700)

    # Same-source 1000-pair diagnostic stress test from testing_5000 unique images (500 same-image + 500 cross-pt neg)
    te5k_imgs = sorted({a for a, b, _ in te5k} | {b for a, b, _ in te5k})
    te_pt_to_imgs = defaultdict(list)
    for im in te5k_imgs:
        te_pt_to_imgs[img_to_pt[im]].append(im)
    samesrc_pos_imgs = list(rng.choice(te5k_imgs, size=500, replace=False))
    pairs_samesrc_pos = [(im, im, 1) for im in samesrc_pos_imgs]
    pairs_samesrc_neg = sample_negatives(te_pt_to_imgs, 500, rng, forbidden={(im, im) for im in samesrc_pos_imgs})
    pairs_samesrc_1000 = pairs_samesrc_pos + pairs_samesrc_neg
    rng.shuffle(pairs_samesrc_1000)

    outputs = {
        "d_pretrain_clean.txt": pairs_pretrain,
        "d_att_train_clean.txt": pairs_att_clean,
        "d_all_trainval_clean.txt": pairs_all_tv,
        "d_val_2000_sub700.txt": pairs_v2k_700,
        "d_test_samesrc_1000.txt": pairs_samesrc_1000,
    }

    manifest = {
        "seed": SEED,
        "splits": {},
        "overlap_matrix": {},
    }

    split_pts = {
        "legacy_train_10000": p_tr10,
        "validation_2000": p_v2k,
        "validation_disjoint_700": p_d700,
        "testing_5000": p_te5k,
    }

    for fname, pairs in outputs.items():
        fpath = OUT_DIR / fname
        sha = write_pair_file(fpath, pairs)
        info = audit_pair_list(pairs, img_to_pt, img_to_fu)
        info["sha256"] = sha
        info["path"] = str(fpath.relative_to(ROOT))
        # Assert 0 duplicates & 0 label errors
        assert info["n_duplicates"] == 0, f"Duplicate pairs in {fname}"
        assert info["label_errors"] == 0, f"Label errors in {fname}"
        if fname != "d_test_samesrc_1000.txt":
            assert info["self_pairs"] == 0 and info["pos_same_followup"] == 0
        manifest["splits"][fname] = info
        split_pts[fname] = {img_to_pt[a] for a, b, _ in pairs} | {img_to_pt[b] for a, b, _ in pairs}

    # Check overlaps
    names = list(split_pts.keys())
    for i, n1 in enumerate(names):
        for n2 in names[i + 1 :]:
            ov = len(split_pts[n1] & split_pts[n2])
            manifest["overlap_matrix"][f"{n1}__vs__{n2}"] = ov

    # Strict disjointness assertions
    assert manifest["overlap_matrix"]["d_pretrain_clean.txt__vs__d_att_train_clean.txt"] == 0
    for s_clean in ("d_pretrain_clean.txt", "d_att_train_clean.txt", "d_all_trainval_clean.txt"):
        for s_eval in ("validation_2000", "validation_disjoint_700", "testing_5000"):
            assert manifest["overlap_matrix"][f"{s_eval}__vs__{s_clean}"] == 0 or manifest["overlap_matrix"].get(f"{s_clean}__vs__{s_eval}", 0) == 0

    # Check E4 Use-Case-2 image pools (nih_labels.csv)
    nih = pd.read_csv(ROOT / "chexnet" / "nih_labels.csv")
    nih_disk = nih[nih["Image Index"].isin(on_disk)].copy()
    e4_pools = {}
    for split_name, pt_set in [("e4_train_tr10", p_tr10), ("e4_val_v2k", p_v2k), ("e4_test_official", set(nih[nih["fold"] == "test"]["Patient ID"]))]:
        if split_name == "e4_test_official":
            sub_u = nih_disk[nih_disk["fold"] == "test"]
        else:
            sub_u = nih_disk[(nih_disk["fold"] != "test") & nih_disk["Patient ID"].isin(pt_set)]
        e4_pools[split_name] = {
            "n_images": int(len(sub_u)),
            "n_patients": int(sub_u["Patient ID"].nunique()),
            "pos_counts": {f: int(sub_u[f].sum()) for f in FINDINGS},
        }
    manifest["e4_classifier_pools"] = e4_pools

    out_json = OUT_DIR / "clean_splits_manifest.json"
    out_json.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["splits"], indent=2))
    print("=== Crucial Overlaps ===")
    for k, v in manifest["overlap_matrix"].items():
        if v > 0 or ("clean" in k and ("pretrain" in k or "att_train" in k)):
            print(f"  {k}: {v}")
    print("=== E4 image pools ===")
    for k, v in e4_pools.items():
        print(f"  {k}: n_images={v['n_images']}, n_patients={v['n_patients']}, min_finding_pos={min(v['pos_counts'].values())}")


if __name__ == "__main__":
    main()
