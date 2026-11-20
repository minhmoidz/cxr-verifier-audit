"""Exploratory paired bootstrap for one rank change in Table 2: MIMIC full recipe versus ImageNet monotone head.

Quantities (TPR at 1% FPR on the test ROC, percentage points, direction MIMIC minus ImageNet):
  d_raw  = TPR(MIMIC full, unsanitized)        - TPR(ImageNet monotone head, unsanitized)
  d_san  = TPR(MIMIC full, learned mu=0.01)    - TPR(ImageNet monotone head, learned mu=0.01)
  change = d_san - d_raw   (how much the MIMIC advantage moves once the images are sanitized)

Both conditions are scored on the same released test list (image_pairs_testing_5000.txt), so one
patient resample can be applied to all four arms (2 configurations x 2 image conditions) in each
replicate. Patient resampling uses the dyadic weights of phase1_bootstrap: a positive pair gets m_p,
a negative pair m_a * m_b. In every replicate the ROC, and therefore the 1%-FPR threshold, is
recomputed from the weighted scores of each seed.

Seeds (three per arm, 0-2). Each arm is its own training run, so seeds are not matched across arms.
  patient       seeds averaged within each arm; only patients are resampled. This treats the three
                trained models as fixed and covers test-patient sampling only.
  patient+seed  in addition, the three seeds of each arm are resampled with replacement,
                independently per arm, before averaging. With three seeds this is a coarse
                approximation of training variability, not a full account of it.

This comparison was chosen after the rank changes in Table 2 had been seen. It is exploratory and
is not a confirmatory test of a preregistered hypothesis.

Output: results/prl_final/rank_reversal_ci.json
Usage:  python3 tools/prl_rank_reversal_ci.py [--n_boot 2000] [--seed 20261120]
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "artifacts", "paper_upgrade", "phase1_patient_bootstrap"))
import phase1_bootstrap as B  # noqa: E402

ARMS = {  # name: (init, arch, sanitizer)
    "mimic_raw": ("chexgenbench", "resnet50_metric_l1", "none"),
    "inmono_raw": ("imagenet", "resnet50_monohead", "none"),
    "mimic_san": ("chexgenbench", "resnet50_metric_l1", "ff_mu0.01"),
    "inmono_san": ("imagenet", "resnet50_monohead", "ff_mu0.01"),
}
OUT = os.path.join(ROOT, "results", "prl_final", "rank_reversal_ci.json")
TARGET = 0.01


def load_arm(init, arch, san):
    rows = []
    for pat in ("results/e1/runs/*_summary.json", "results/prl_final/runs/*_summary.json"):
        for f in glob.glob(os.path.join(ROOT, pat)):
            r = json.load(open(f))
            if (r.get("train_path") and "att_train_clean" in os.path.basename(r["train_path"])
                    and (r["init_name"], r["arch"], r["sanitizer"]) == (init, arch, san) and float(r["lr"]) == 1e-4):
                assert os.path.realpath(r["test_path"]) == os.path.realpath(B.TEST_PAIRS), f
                p = f.replace("_summary.json", "_preds.npz")
                if not os.path.exists(p):
                    raise SystemExit(f"MISSING per-pair predictions: {p}")
                d = np.load(p)
                rows.append({"seed": r["seed"], "y": d["test_y"].astype(float), "p": d["test_p"].astype(np.float64),
                             "logged_tpr1": r["test_metrics"]["tpr_at_1pct"], "file": os.path.relpath(p, ROOT),
                             "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
    rows.sort(key=lambda t: t["seed"])
    if [r["seed"] for r in rows] != [0, 1, 2]:
        raise SystemExit(f"MISSING seeds for {(init, arch, san)}: found {[r['seed'] for r in rows]}")
    return rows


def tpr(y, s, w):
    v, _ = B.weighted_tpr_at(y, s, w, TARGET)
    return None if v is None else 100.0 * v


def summarize(x):
    x = np.asarray(x)
    return {"ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))], "median": float(np.median(x)),
            "n_boot": int(len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261120)
    a = ap.parse_args()

    _, _, y, pid_a, pid_b = B.load_pairs(B.TEST_PAIRS)
    ca, cb, is_pos, npat = B._encode_pids(pid_a, pid_b)
    arms = {k: load_arm(*v) for k, v in ARMS.items()}
    for k, rows in arms.items():
        for r in rows:
            assert np.array_equal(r["y"], y), f"label order of {r['file']} differs from the test list"

    # Point estimates on the unweighted test set; check against the logged per-run TPR (2 decimals).
    ones = np.ones_like(y)
    per_seed = {k: [tpr(y, r["p"], ones) for r in rows] for k, rows in arms.items()}
    for k, rows in arms.items():
        for v, r in zip(per_seed[k], rows):
            assert abs(v - r["logged_tpr1"]) < 0.011, (k, r["seed"], v, r["logged_tpr1"])
    mean = {k: float(np.mean(v)) for k, v in per_seed.items()}
    point = {"d_raw": mean["mimic_raw"] - mean["inmono_raw"], "d_san": mean["mimic_san"] - mean["inmono_san"]}
    point["change"] = point["d_san"] - point["d_raw"]

    rng = np.random.default_rng(a.seed)
    boot = {"patient": {"d_raw": [], "d_san": [], "change": []}, "patient+seed": {"d_raw": [], "d_san": [], "change": []}}
    dropped = {"patient": 0, "patient+seed": 0}
    arm_boot = {k: [] for k in arms}                                   # per-arm seed-mean TPR, patient mode
    for _ in range(a.n_boot):
        w = B.dyadic_weights_fast(ca, cb, is_pos, npat, rng)         # one patient resample for all arms
        vals = {k: [tpr(y, r["p"], w) for r in rows] for k, rows in arms.items()}
        pick = {k: rng.integers(0, 3, 3) for k in arms}               # independent seed resample per arm
        for mode in boot:
            if mode == "patient":
                m = {k: v for k, v in vals.items()}
            else:
                m = {k: [vals[k][i] for i in pick[k]] for k in arms}
            if any(x is None for v in m.values() for x in v):
                dropped[mode] += 1
                continue
            m = {k: float(np.mean(v)) for k, v in m.items()}
            if mode == "patient":
                for k in arms:
                    arm_boot[k].append(m[k])
            dr, ds = m["mimic_raw"] - m["inmono_raw"], m["mimic_san"] - m["inmono_san"]
            boot[mode]["d_raw"].append(dr); boot[mode]["d_san"].append(ds); boot[mode]["change"].append(ds - dr)

    res = {
        "status": "exploratory, chosen after Table 2 rank changes were observed; not preregistered",
        "metric": "TPR at 1% FPR on the test ROC (percentage points), threshold recomputed in every replicate",
        "direction": "MIMIC full recipe minus ImageNet monotone head; change = d_san - d_raw",
        "test_pairs": os.path.relpath(B.TEST_PAIRS, ROOT), "n_pairs": int(len(y)),
        "n_pos": int(y.sum()), "n_patients": int(npat),
        "resampling": "patients with replacement, dyadic weights (positive m_p, negative m_a*m_b), "
                      "same resample for all four arms in each replicate",
        "seed_handling": {"patient": "three seeds averaged within each arm; models treated as fixed",
                          "patient+seed": "three seeds resampled with replacement independently per arm, then averaged"},
        "n_boot": a.n_boot, "rng_seed": a.seed, "dropped_replicates": dropped,
        "per_seed_tpr1": per_seed, "arm_mean_tpr1": mean, "point": point,
        "arm_ci_patient": {k: summarize(v) for k, v in arm_boot.items()},
        "ci": {mode: {q: summarize(v) for q, v in qs.items()} for mode, qs in boot.items()},
        "share_of_replicates": {mode: {"d_raw_gt_0": float(np.mean(np.array(qs["d_raw"]) > 0)),
                                       "d_san_lt_0": float(np.mean(np.array(qs["d_san"]) < 0)),
                                       "both": float(np.mean((np.array(qs["d_raw"]) > 0) & (np.array(qs["d_san"]) < 0)))}
                                for mode, qs in boot.items()},
        "inputs": {k: [{"seed": r["seed"], "file": r["file"], "sha256": r["sha256"]} for r in rows] for k, rows in arms.items()},
    }
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_pairs", "n_pos", "n_patients", "arm_mean_tpr1", "arm_ci_patient", "point", "ci",
                                           "share_of_replicates", "dropped_replicates")}, indent=1))


if __name__ == "__main__":
    main()
