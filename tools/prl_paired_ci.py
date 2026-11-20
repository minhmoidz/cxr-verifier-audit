"""Paired, patient-clustered 95% CIs for the directly observed privacy-utility comparisons.

Learned field (mu=0.01) versus Gaussian blur (sigma=2) and random elastic deformation
(strength 0.045), the two control settings whose frozen utility lies closest below the field's.
Attacker: reference-verifier initialization, released recipe, clean protocol, seeds 0-2.

Leakage (test pairs, testing_5000): differences in TPR at 1% FPR (test ROC) and AUC, field minus
control. The same patient resample is used for both arms in each replicate: positive pairs get
weight m_p, negative pairs m_a * m_b (the dyadic scheme validated in
artifacts/paper_upgrade/phase1_patient_bootstrap/). Two intervals:
  patient     seeds averaged within each arm, patients resampled
  hierarchical patients resampled and, independently per arm, seeds resampled with replacement
Utility (frozen classifier, 25,596 official test images, 14 findings): difference in macro AUC,
field minus control, with patients resampled. The classifier is fixed, so no seed component.

Output: results/prl_final/paired_ci.json, read by tools/prl_tables.py.
Usage: python3 tools/prl_paired_ci.py [--n_boot 2000] [--seed 20261120]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "artifacts", "paper_upgrade", "phase1_patient_bootstrap"))
import phase1_bootstrap as B  # noqa: E402
from compute_a1_utility import bootstrap_weighted_auc, draw_valid_patient_multiplicities, prepare_sorted  # noqa: E402

LEARNED = "ff_mu0.01"
CONTROLS = ("blur_s2", "elastic_m0.045_keyed")
OUT = os.path.join(ROOT, "results", "prl_final", "paired_ci.json")


def seed_preds(san):
    """Per-seed test scores of the reference-verifier attacker (clean protocol) for one sanitizer."""
    rows = []
    for pat in ("results/e1/runs/*_summary.json", "results/prl_final/runs/*_summary.json"):
        for f in glob.glob(os.path.join(ROOT, pat)):
            r = json.load(open(f))
            if (r.get("train_path") and "att_train_clean" in r["train_path"] and r["init_name"] == "pretrained_ver"
                    and r["arch"] == "resnet50_bce" and r["sanitizer"] == san and float(r["lr"]) == 1e-4):
                d = np.load(f.replace("_summary.json", "_preds.npz"))
                rows.append((r["seed"], d["test_y"], d["test_p"].astype(np.float64)))
    rows.sort(key=lambda t: t[0])
    assert [s for s, _, _ in rows] == [0, 1, 2], (san, [s for s, _, _ in rows])
    return rows


def leakage(n_boot, rng):
    _, _, y, pid_a, pid_b = B.load_pairs(B.TEST_PAIRS)
    L = seed_preds(LEARNED)
    for _, yy, _ in L:
        assert np.array_equal(yy.astype(float), y), "prediction order differs from testing_5000"
    Ls = [p for _, _, p in L]
    out = {}
    for ctrl in CONTROLS:
        C = seed_preds(ctrl)
        for _, yy, _ in C:
            assert np.array_equal(yy.astype(float), y)
        Cs = [p for _, _, p in C]
        res = {}
        for metric, tf in (("tpr1", 0.01), ("auc", None)):
            def point(S):
                vals = []
                for s in S:
                    w = np.ones_like(y)
                    v = B._paired_metric(y, s, w, None, tf)
                    vals.append(v)
                return float(np.mean(vals))
            pt = point(Ls) - point(Cs)
            lo, hi, n = B.paired_gap_ci(Ls, Cs, y, pid_a, pid_b, n_boot, rng, hierarchical=False, target_fpr=tf)
            hlo, hhi, hn = B.paired_gap_ci(Ls, Cs, y, pid_a, pid_b, n_boot, rng, hierarchical=True, target_fpr=tf)
            res[metric] = {"point": pt, "patient_ci": [lo, hi], "hier_ci": [hlo, hhi], "n_boot": [n, hn]}
        out[ctrl] = res
    return out


def utility(n_boot, rng):
    from scripts.run_e4 import load_e4_splits  # same image order as the saved frozen predictions
    _, te_y, te_pts = load_e4_splits()["test"]
    y = te_y.T.astype(np.uint8)                                   # (14, N)
    pats, code = np.unique(te_pts, return_inverse=True)
    names = (LEARNED,) + CONTROLS
    probs = []
    for s in names:
        p = np.load(os.path.join(ROOT, "results", "e4", "frozen", f"u_frozen_{s}_test_preds.npz"))["te_probs"]
        assert p.shape == (y.shape[1], 14), (s, p.shape)
        probs.append(p.T.astype(np.float64))
    scores = np.stack(probs)                                      # (cfg, 14, N)
    mult, rejected = draw_valid_patient_multiplicities(y, code.astype(np.int32), len(pats), n_boot, rng)
    ss, sy, sp = prepare_sorted(y, scores, code.astype(np.int32))
    point = bootstrap_weighted_auc(ss, sy, sp, np.ones((1, len(pats)), dtype=np.int16))[0].mean(axis=1) * 100
    boot = bootstrap_weighted_auc(ss, sy, sp, mult).mean(axis=2) * 100      # (n_boot, cfg) macro AUC
    out = {"macro_auc": {s: float(point[i]) for i, s in enumerate(names)}, "rejected_draws": int(rejected)}
    for i, ctrl in enumerate(CONTROLS, start=1):
        d = boot[:, 0] - boot[:, i]
        out[ctrl] = {"point": float(point[0] - point[i]),
                     "patient_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "n_boot": int(len(d))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261120)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    res = {"n_boot": a.n_boot, "seed": a.seed, "attacker": "pretrained_ver/resnet50_bce, clean protocol, seeds 0-2",
           "learned": LEARNED, "controls": list(CONTROLS),
           "leakage": leakage(a.n_boot, rng), "utility_frozen": utility(a.n_boot, np.random.default_rng(a.seed + 1))}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
