"""E3: the released evaluation protocol inside this study, and the pairing effect under both initializations.

ImageNet initialization, released recipe, released 10,000/2,000 pair lists, SR test pairs, learned field
mu=0.01, 2x2 train x validation pairing (SS/SS = released protocol), seeds 0-4
(results/E3/cells/*/..._labels_predictions.txt). The reference-initialization factorial cells
(archive/p1*_ff_mu0.01_*_pretrained_ver_seed*, original validation list) give the same 2x2 under the
reference initialization.

Bootstrap as tools/analyze_factorial.py (Table 1, Table S4): 2,000 replicates; test patients resampled
with dyadic weights; within an initialization, seeds resampled with replacement jointly across the four
cells (paired by index); the two initializations resample seeds independently and share the patient draw.
Reported per initialization: cell means (AUC, TPR at 1% FPR), training effect (SR minus SS training,
averaged over validation pairings), validation effect, interaction; and the difference in each effect
between ImageNet and reference initialization.

Output: results/E3/e3_ci.json.  Usage: python3 tools/v2_e3_ci.py [--n_boot 2000]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "artifacts", "paper_upgrade", "phase1_patient_bootstrap"))
sys.path.insert(0, os.path.join(ROOT, "artifacts", "paper_upgrade", "phase5_pairing_audit"))
import phase1_bootstrap as B  # noqa: E402
import pairing_resolver as Rz  # noqa: E402

SS, SR = "sanitized_sanitized", "sanitized_raw"
CELLS = [(SS, SS), (SS, SR), (SR, SS), (SR, SR)]
TAG = {SS: "SS", SR: "SR"}


def load(init, tp, vp, seed):
    cid = Rz.cell_id("ff_mu0.01", init, tp, seed, vp, "orig")
    base = os.path.join(ROOT, "results", "E3", "cells") if init == "imagenet" else os.path.join(ROOT, "archive")
    f = os.path.join(base, cid, f"{cid}_labels_predictions.txt")
    return np.loadtxt(f, skiprows=1)[:, 1]


def effects(m):
    """m: dict cell -> value. Training, validation effects and interaction."""
    return {"train_effect": np.mean([m[(SR, vp)] - m[(SS, vp)] for vp in (SS, SR)]),
            "val_effect": np.mean([m[(tp, SR)] - m[(tp, SS)] for tp in (SS, SR)]),
            "interaction": (m[(SR, SR)] - m[(SR, SS)]) - (m[(SS, SR)] - m[(SS, SS)])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_boot", type=int, default=2000)
    n_boot = ap.parse_args().n_boot
    _, _, y, pa, pb = B.load_pairs(B.TEST_PAIRS)
    ca, cb, is_pos, npat = B._encode_pids(pa, pb)
    S = {init: {c: np.stack([load(init, *c, s) for s in range(5)]) for c in CELLS} for init in ("imagenet", "pretrained_ver")}
    for init in S:
        for c in CELLS:
            assert S[init][c].shape == (5, len(y)), (init, c, S[init][c].shape)

    def metric(sc, w, kind):
        return np.array([100 * B.weighted_auc(y, s, w) if kind == "auc" else 100 * B.weighted_tpr_at(y, s, w)[0] for s in sc])

    out = {"n_boot": n_boot, "rng_seed": 20261124, "n_seeds": 5}
    rng = np.random.default_rng(20261124)
    ones = np.ones(len(y))
    for kind in ("auc", "tpr"):
        point = {init: {c: float(metric(S[init][c], ones, kind).mean()) for c in CELLS} for init in S}
        per_seed = {init: {c: [float(v) for v in metric(S[init][c], ones, kind)] for c in CELLS} for init in S}
        reps = []
        for _ in range(n_boot):
            w = B.dyadic_weights_fast(ca, cb, is_pos, npat, rng)
            row = {}
            for init in S:
                idx = rng.integers(0, 5, 5)
                row[init] = {c: float(metric(S[init][c][idx], w, kind).mean()) for c in CELLS}
            reps.append(row)
        res = {}
        for init in S:
            e_pt = effects(point[init])
            e_rep = [effects(r[init]) for r in reps]
            res[init] = {"cells": {f"{TAG[c[0]]}{TAG[c[1]]}": {"point": point[init][c], "per_seed": per_seed[init][c],
                                                               "ci": [float(np.percentile([r[init][c] for r in reps], q)) for q in (2.5, 97.5)]}
                                   for c in CELLS},
                         **{k: {"point": float(e_pt[k]), "ci": [float(np.percentile([e[k] for e in e_rep], q)) for q in (2.5, 97.5)]}
                            for k in e_pt}}
        diff = {}
        for k in ("train_effect", "val_effect", "interaction"):
            pt = effects(point["imagenet"])[k] - effects(point["pretrained_ver"])[k]
            d = [effects(r["imagenet"])[k] - effects(r["pretrained_ver"])[k] for r in reps]
            diff[k] = {"point": float(pt), "ci": [float(np.percentile(d, q)) for q in (2.5, 97.5)]}
        # Released protocol (ImageNet SS/SS) minus reference-init SS/SS, and minus ImageNet SR/SR.
        diff["imagenet_SSSS_minus_reference_SSSS"] = {
            "point": point["imagenet"][(SS, SS)] - point["pretrained_ver"][(SS, SS)],
            "ci": [float(np.percentile([r["imagenet"][(SS, SS)] - r["pretrained_ver"][(SS, SS)] for r in reps], q)) for q in (2.5, 97.5)]}
        res["imagenet_minus_reference"] = diff
        out[kind] = res
    os.makedirs(os.path.join(ROOT, "results", "E3"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "E3", "e3_ci.json"), "w"), indent=1)
    for kind in ("auc", "tpr"):
        for init in ("imagenet", "pretrained_ver"):
            r = out[kind][init]
            print(kind, init, {c: round(v["point"], 2) for c, v in r["cells"].items()},
                  "train", round(r["train_effect"]["point"], 2), [round(x, 2) for x in r["train_effect"]["ci"]])
        print(kind, "diff", {k: (round(v["point"], 2), [round(x, 2) for x in v["ci"]]) for k, v in out[kind]["imagenet_minus_reference"].items()})


if __name__ == "__main__":
    main()
