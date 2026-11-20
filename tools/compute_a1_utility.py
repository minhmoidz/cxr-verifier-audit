#!/usr/bin/env python3
"""Patient-clustered uncertainty for frozen-classifier utility.

The prediction files contain one evaluation of one frozen classifier per sanitizer. They
support sampling uncertainty over test patients, but not variation over classifier training
seeds. A replicate samples test patients with replacement and gives every image from a sampled
patient the same multiplicity. The identical patient draw is used for all sanitizers.

The 14-finding estimand is fixed before resampling. If a draw has no positive or no negative
image for any finding, the whole draw is rejected and redrawn; a missing AUC is never replaced
by 0.5 and the finding set never changes. Rejection counts are recorded in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from numba import njit, prange
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "reviewer_gaps"

CONFIGS = [
    ("real_data", "cls_none", "None (real radiographs)"),
    ("prichexy_mu0.001", "cls_ff_mu0.001", "PriCheXy-Net mu=0.001"),
    ("prichexy_mu0.005", "cls_ff_mu0.005", "PriCheXy-Net mu=0.005"),
    ("prichexy_mu0.01", "cls_ff_mu0.01", "PriCheXy-Net mu=0.01"),
    ("blur_s2", "cls_blur_s2", "Gaussian blur sigma=2"),
    ("elastic_m004", "cls_elastic_m004", "Random elastic mu=0.04"),
    ("privacy_net", "cls_privacy_net", "Privacy-Net"),
    ("dppix_b8", "cls_dppix_b8", "DP-Pix b=8"),
    ("dppix_b4", "cls_dppix_b4", "DP-Pix b=4"),
    ("dppix_b2", "cls_dppix_b2", "DP-Pix b=2"),
    ("dppix_b1", "cls_dppix_b1", "DP-Pix b=1"),
]

FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

COMPARISONS = [
    ("elastic_minus_learned", "elastic_m004", "prichexy_mu0.01"),
    ("learned_minus_blur", "prichexy_mu0.01", "blur_s2"),
    ("elastic_minus_blur", "elastic_m004", "blur_s2"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_inputs():
    label_path = ROOT / "chexnet" / "nih_labels.csv"
    labels = pd.read_csv(label_path)
    labels = labels.loc[labels["fold"] == "test"].copy()
    if labels["Image Index"].duplicated().any():
        raise ValueError("duplicate Image Index values in test labels")
    labels = labels.set_index("Image Index", drop=False)
    image_ids = labels.index.to_numpy()
    y = labels[FINDINGS].to_numpy(dtype=np.uint8)

    patient_ids = labels["Patient ID"].astype(str).to_numpy()
    patients, patient_code = np.unique(patient_ids, return_inverse=True)
    prediction = []
    manifest = {str(label_path.relative_to(ROOT)): sha256(label_path)}
    expected = set(image_ids)
    for _name, tag, _display in CONFIGS:
        path = ROOT / "chexnet" / "results" / tag / "preds.csv"
        df = pd.read_csv(path)
        if "Image Index" not in df or df["Image Index"].duplicated().any():
            raise ValueError(f"{path}: absent or duplicate Image Index")
        observed = set(df["Image Index"])
        if observed != expected:
            raise ValueError(
                f"{path}: prediction/label identity mismatch "
                f"(missing={len(expected-observed)}, extra={len(observed-expected)})")
        df = df.set_index("Image Index").loc[image_ids]
        cols = ["prob_" + finding for finding in FINDINGS]
        if not np.isfinite(df[cols].to_numpy()).all():
            raise ValueError(f"{path}: non-finite prediction")
        prediction.append(df[cols].to_numpy(dtype=np.float64).T)
        manifest[str(path.relative_to(ROOT))] = sha256(path)

    return (labels, y.T, np.stack(prediction), patients,
            patient_code.astype(np.int32), manifest)


def draw_valid_patient_multiplicities(y, patient_code, n_patients, n_boot, rng):
    """Draw multinomial patient multiplicities, rejecting any globally invalid draw."""
    pos = np.zeros((y.shape[0], n_patients), dtype=np.int32)
    neg = np.zeros_like(pos)
    for finding in range(y.shape[0]):
        pos[finding] = np.bincount(patient_code, weights=y[finding], minlength=n_patients)
        neg[finding] = np.bincount(patient_code, weights=1 - y[finding], minlength=n_patients)

    accepted = []
    accepted_count = 0
    rejected = 0
    p = np.full(n_patients, 1.0 / n_patients)
    while accepted_count < n_boot:
        need = n_boot - accepted_count
        batch = rng.multinomial(n_patients, p, size=max(need, 64)).astype(np.int16)
        valid = ((batch @ pos.T) > 0).all(axis=1) & ((batch @ neg.T) > 0).all(axis=1)
        rejected += int((~valid).sum())
        take = batch[valid][:need]
        accepted.append(take)
        accepted_count += len(take)
    return np.concatenate(accepted, axis=0)[:n_boot], rejected


def prepare_sorted(y, scores, patient_code):
    n_cfg, n_find, n_img = scores.shape
    sorted_scores = np.empty_like(scores)
    sorted_y = np.empty((n_cfg, n_find, n_img), dtype=np.uint8)
    sorted_patient = np.empty((n_cfg, n_find, n_img), dtype=np.int32)
    for config in range(n_cfg):
        for finding in range(n_find):
            order = np.argsort(scores[config, finding], kind="mergesort")
            sorted_scores[config, finding] = scores[config, finding, order]
            sorted_y[config, finding] = y[finding, order]
            sorted_patient[config, finding] = patient_code[order]
    return sorted_scores, sorted_y, sorted_patient


@njit(parallel=True, cache=True)
def bootstrap_weighted_auc(sorted_scores, sorted_y, sorted_patient, multiplicities):
    """Exact frequency-weighted AUC, with half credit for score ties."""
    n_boot = multiplicities.shape[0]
    n_cfg, n_find, n_img = sorted_scores.shape
    out = np.empty((n_boot, n_cfg, n_find), dtype=np.float64)
    for replicate in prange(n_boot):
        mult = multiplicities[replicate]
        for config in range(n_cfg):
            for finding in range(n_find):
                cumulative_negative = 0.0
                numerator = 0.0
                total_positive = 0.0
                i = 0
                while i < n_img:
                    score = sorted_scores[config, finding, i]
                    group_positive = 0.0
                    group_negative = 0.0
                    j = i
                    while j < n_img and sorted_scores[config, finding, j] == score:
                        weight = float(mult[sorted_patient[config, finding, j]])
                        if sorted_y[config, finding, j] == 1:
                            group_positive += weight
                        else:
                            group_negative += weight
                        j += 1
                    numerator += group_positive * (cumulative_negative + 0.5 * group_negative)
                    cumulative_negative += group_negative
                    total_positive += group_positive
                    i = j
                out[replicate, config, finding] = numerator / (total_positive * cumulative_negative)
    return out


def rounded_summary(values):
    return {
        "bootstrap_mean": round(float(np.mean(values)), 4),
        "bootstrap_se": round(float(np.std(values, ddof=1)), 4),
        "ci_95": [round(float(x), 4) for x in np.percentile(values, [2.5, 97.5])],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _labels, y, scores, patients, patient_code, inputs = load_inputs()
    points_by_finding = np.empty(scores.shape[:2], dtype=np.float64)
    for config in range(scores.shape[0]):
        for finding in range(scores.shape[1]):
            points_by_finding[config, finding] = roc_auc_score(
                y[finding], scores[config, finding]) * 100
    point_macro = points_by_finding.mean(axis=1)

    rng = np.random.default_rng(args.seed)
    multiplicities, rejected = draw_valid_patient_multiplicities(
        y, patient_code, len(patients), args.n_bootstrap, rng)
    boot_findings = bootstrap_weighted_auc(
        *prepare_sorted(y, scores, patient_code), multiplicities) * 100
    boot_macro = boot_findings.mean(axis=2)

    per_config = []
    per_finding = []
    for config, (name, _tag, display_name) in enumerate(CONFIGS):
        per_config.append({
            "config": name,
            "display_name": display_name,
            "utility_point": round(float(point_macro[config]), 4),
            **rounded_summary(boot_macro[:, config]),
            "across_training_seed_sd": None,
            "n_classifier_training_seeds": 1,
        })
        for finding, finding_name in enumerate(FINDINGS):
            values = boot_findings[:, config, finding]
            positive = y[finding] == 1
            per_finding.append({
                "config": name,
                "finding": finding_name,
                "n_positive_images": int(positive.sum()),
                "n_negative_images": int((~positive).sum()),
                "n_positive_patients": int(np.unique(patient_code[positive]).size),
                "n_negative_patients": int(np.unique(patient_code[~positive]).size),
                "auc_point": round(float(points_by_finding[config, finding]), 4),
                **rounded_summary(values),
            })

    config_index = {name: i for i, (name, _tag, _display) in enumerate(CONFIGS)}
    comparisons = {}
    for comparison, left, right in COMPARISONS:
        li, ri = config_index[left], config_index[right]
        diff = boot_macro[:, li] - boot_macro[:, ri]
        comparisons[comparison] = {
            "left": left,
            "right": right,
            "point_diff": round(float(point_macro[li] - point_macro[ri]), 4),
            **rounded_summary(diff),
            "ci_contains_zero": bool(np.percentile(diff, 2.5) <= 0 <= np.percentile(diff, 97.5)),
            "equivalence_margin": None,
            "equivalence_conclusion": "not tested",
        }

    method = {
        "estimand": "unweighted mean of the fixed 14 finding AUCs",
        "point_estimate": "computed once on the original, unresampled test data",
        "resampling_unit": "test patient",
        "within_patient_structure": "all images retained with the sampled patient's multiplicity",
        "pairing": "one common patient-resample matrix used for every sanitizer",
        "interval": "percentile bootstrap, 2.5th and 97.5th percentiles",
        "bootstrap_se": "sample SD across patient-bootstrap replicates",
        "training_seed_uncertainty": "unavailable: each sanitizer has one frozen-classifier prediction file",
        "fixed_finding_grid": FINDINGS,
        "missing_class_rule": "reject and redraw the whole replicate; never assign AUC=0.5 or alter findings",
        "equivalence_rule": "a CI containing zero is not evidence of equivalence; no margin was specified",
        "n_bootstrap": args.n_bootstrap,
        "rng_seed": args.seed,
        "rejected_missing_class_draws": rejected,
        "n_test_images": int(y.shape[1]),
        "n_test_patients": int(len(patients)),
    }
    output = {
        "schema_version": 2,
        "method": method,
        "inputs_sha256": inputs,
        "software": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "per_config_table": per_config,
        "comparisons": comparisons,
    }
    json_path = OUT_DIR / "a1_utility_uncertainty.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    pd.DataFrame(per_config).to_csv(OUT_DIR / "a1_utility_uncertainty.csv", index=False)
    pd.DataFrame(per_finding).to_csv(OUT_DIR / "a1_utility_per_finding.csv", index=False)
    print(json.dumps(method, indent=2))
    print(pd.DataFrame(per_config).to_string(index=False))
    print(json.dumps(comparisons, indent=2))
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
