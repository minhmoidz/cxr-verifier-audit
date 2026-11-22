"""E4 and T2: patient-clustered intervals for the secondary verifier and for relative leakage.

Same scheme as Table S5 (tools/prl_paired_ci.py): 2,000 replicates; in each, test patients are
resampled with replacement and every pair is weighted by patient multiplicity (positive m_p,
negative m_a * m_b); the same patient draw is used for every arm and both verifiers.
  patient          seeds averaged within each arm
  patient+seed     additionally, seeds resampled with replacement independently per arm
Each replicate records every arm's seed-mean TPR at 1% FPR (test ROC) and AUC, so differences,
ratios and between-verifier contrasts all come from the same replicates.

Arms (published runs, clean protocol, seeds 0-2, test pairs testing_5000):
  reference      pretrained_ver / released recipe: none, ff_mu0.01, blur_s2, elastic 0.045, elastic 0.04
  imagenet_full  imagenet / full recipe (resnet50_metric_l1, lr 1e-4): none, ff_mu0.01, blur_s2, elastic 0.04

Outputs: results/E4/secondary_ci.json
Usage:   python3 tools/v2_secondary_ci.py [--n_boot 2000] [--seed 20261122]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "artifacts", "paper_upgrade", "phase1_patient_bootstrap"))
import phase1_bootstrap as B  # noqa: E402

OUT = os.path.join(ROOT, "results", "E4", "secondary_ci.json")
ARMS = {
    "reference": {"key": ("pretrained_ver", "resnet50_bce"),
                  "sans": ["none", "ff_mu0.01", "blur_s2", "elastic_m0.045_keyed", "elastic_m0.04_keyed"]},
    "imagenet_full": {"key": ("imagenet", "resnet50_metric_l1"),
                      "sans": ["none", "ff_mu0.01", "blur_s2", "elastic_m0.04_keyed"]},
}


def seed_scores(init, arch, san, y):
    rows = []
    for pat in ("results/e1/runs/*_summary.json", "results/prl_final/runs/*_summary.json"):
        for f in glob.glob(os.path.join(ROOT, pat)):
            r = json.load(open(f))
            if (r.get("train_path") and "att_train_clean" in r["train_path"] and r["init_name"] == init
                    and r["arch"] == arch and r["sanitizer"] == san and float(r["lr"]) == 1e-4):
                d = np.load(f.replace("_summary.json", "_preds.npz"))
                assert np.array_equal(d["test_y"].astype(float), y), f
                rows.append((int(r["seed"]), d["test_p"].astype(np.float64), r["run_id"]))
    rows.sort()
    assert [s for s, _, _ in rows] == [0, 1, 2], (init, arch, san, [s for s, _, _ in rows])
    return rows


def metrics(y, s, w):
    t, _ = B.weighted_tpr_at(y, s, w, 0.01)
    a = B.weighted_auc(y, s, w)
    return (None if t is None else 100 * t), (None if a is None else 100 * a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261122)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    _, _, y, pid_a, pid_b = B.load_pairs(B.TEST_PAIRS)
    arms, run_ids = {}, {}
    for ver, spec in ARMS.items():
        for san in spec["sans"]:
            rows = seed_scores(*spec["key"], san, y)
            arms[(ver, san)] = [s for _, s, _ in rows]
            run_ids[f"{ver}/{san}"] = [rid for _, _, rid in rows]
    names = list(arms)
    ones = np.ones_like(y)
    point = {}
    for k in names:
        vals = [metrics(y, s, ones) for s in arms[k]]
        point[k] = {"tpr1": float(np.mean([v[0] for v in vals])), "auc": float(np.mean([v[1] for v in vals]))}

    reps = {"patient": [], "patient+seed": []}
    dropped = {"patient": 0, "patient+seed": 0}
    t0 = time.time()
    for b in range(a.n_boot):
        w = B.dyadic_weights(pid_a, pid_b, rng)
        seed_idx = {k: rng.integers(0, 3, 3) for k in names}
        per_seed = {k: [metrics(y, s, w) for s in arms[k]] for k in names}
        for mode in reps:
            row, bad = {}, False
            for k in names:
                vals = per_seed[k] if mode == "patient" else [per_seed[k][i] for i in seed_idx[k]]
                if any(v[0] is None or v[1] is None for v in vals):
                    bad = True
                    break
                row[k] = (np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals]))
            if bad:
                dropped[mode] += 1
            else:
                reps[mode].append(row)
        if b % 200 == 0:
            print(f"  replicate {b}/{a.n_boot} ({time.time() - t0:.0f}s)", flush=True)

    # Quantities: (name, function of a row {arm: (tpr1, auc)} -> float)
    T = lambda r, v, s: r[(v, s)][0]
    A = lambda r, v, s: r[(v, s)][1]
    Q = {}
    for v, el in (("reference", "elastic_m0.045_keyed"), ("reference", "elastic_m0.04_keyed"),
                  ("imagenet_full", "elastic_m0.04_keyed")):
        tag = {"elastic_m0.045_keyed": "e045", "elastic_m0.04_keyed": "e04"}[el]
        Q[f"{v}/field_minus_{tag}/tpr1"] = lambda r, v=v, el=el: T(r, v, "ff_mu0.01") - T(r, v, el)
        Q[f"{v}/field_minus_{tag}/auc"] = lambda r, v=v, el=el: A(r, v, "ff_mu0.01") - A(r, v, el)
        Q[f"{v}/field_over_{tag}/tpr1"] = lambda r, v=v, el=el: T(r, v, "ff_mu0.01") / T(r, v, el)
        Q[f"{v}/{tag}_over_raw/tpr1"] = lambda r, v=v, el=el: T(r, v, el) / T(r, v, "none")
    for v in ARMS:
        Q[f"{v}/field_minus_b2/tpr1"] = lambda r, v=v: T(r, v, "ff_mu0.01") - T(r, v, "blur_s2")
        Q[f"{v}/field_minus_b2/auc"] = lambda r, v=v: A(r, v, "ff_mu0.01") - A(r, v, "blur_s2")
        Q[f"{v}/field_over_b2/tpr1"] = lambda r, v=v: T(r, v, "ff_mu0.01") / T(r, v, "blur_s2")
        Q[f"{v}/field_over_raw/tpr1"] = lambda r, v=v: T(r, v, "ff_mu0.01") / T(r, v, "none")
        Q[f"{v}/b2_over_raw/tpr1"] = lambda r, v=v: T(r, v, "blur_s2") / T(r, v, "none")
        Q[f"{v}/field_minus_b2_over_raw/tpr1"] = lambda r, v=v: (T(r, v, "ff_mu0.01") - T(r, v, "blur_s2")) / T(r, v, "none")
    # Between verifiers (same patient draw): reference minus imagenet_full.
    Q["between/field_over_b2/tpr1"] = lambda r: T(r, "reference", "ff_mu0.01") / T(r, "reference", "blur_s2") \
        - T(r, "imagenet_full", "ff_mu0.01") / T(r, "imagenet_full", "blur_s2")
    Q["between/field_minus_b2/tpr1"] = lambda r: (T(r, "reference", "ff_mu0.01") - T(r, "reference", "blur_s2")) \
        - (T(r, "imagenet_full", "ff_mu0.01") - T(r, "imagenet_full", "blur_s2"))
    Q["between/field_minus_b2_over_raw/tpr1"] = lambda r: \
        (T(r, "reference", "ff_mu0.01") - T(r, "reference", "blur_s2")) / T(r, "reference", "none") \
        - (T(r, "imagenet_full", "ff_mu0.01") - T(r, "imagenet_full", "blur_s2")) / T(r, "imagenet_full", "none")

    prow = {k: (point[k]["tpr1"], point[k]["auc"]) for k in names}
    out = {"n_boot": a.n_boot, "seed": a.seed, "dropped_replicates": dropped,
           "scheme": "patient-clustered dyadic weights (positive m_p, negative m_a*m_b), same draw for all arms",
           "run_ids": run_ids, "arms_point": {f"{v}/{s}": point[(v, s)] for v, s in names}, "quantities": {}}
    for q, f in Q.items():
        entry = {"point": float(f(prow))}
        for mode, rows in reps.items():
            vals = np.array([f(r) for r in rows], float)
            entry[mode] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        out["quantities"][q] = entry
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    for q, e in out["quantities"].items():
        print(f"{q:45s} {e['point']:8.3f}  patient {e['patient'][0]:8.3f} {e['patient'][1]:8.3f}  "
              f"patient+seed {e['patient+seed'][0]:8.3f} {e['patient+seed'][1]:8.3f}")
    print("dropped:", dropped)


if __name__ == "__main__":
    main()
