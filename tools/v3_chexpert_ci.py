"""V3-A analysis (docs/PREREGISTRATION_v3.md): CheXpert pair set, patient-clustered bootstrap.

Per model (results/V3A/scores/<model_id>.npz from tools/v3_chexpert_score.py): positives are the 2,000
same-patient query-reference pairs, negatives all 3,998,000 cross-patient pairs; SR uses the raw reference,
SS the sanitized reference. Metrics: AUC, TPR at 1% and 0.1% FPR (tools/v2_extended_ci.SortedScores).

Bootstrap: 2,000 replicates, RNG seed 20261124; patients resampled with replacement (multinomial counts m);
positive weight m_p, negative weight m_a m_b; one patient draw shared by all models. Patient+seed intervals
also resample seeds within each arm (independently across arms; jointly across the four cells of the
learned-field factorial, as Table S4).

Hypotheses (decision = 95% patient+seed interval), for verifier v in {reference, e5} (imagenet_full secondary):
  H1  SR: field - blur < 0 (also field - elastic 0.045 / 0.04, secondary)
  H2  [field SS - field SR] - [blur SS - blur SR] > 0 (also with elastic 0.045, secondary)
  H3  factorial (reference init., field): mean over validation pairings of (SR - SS training) AUC, SR test > 0
  H4  (V3-C) SS attack: ffv3c_SS - ffv3c_SR < 0, verifiers v3c_reference and v3c_e5; H5 the same under SR

Output: results/V3A/chexpert_ci.json
Usage:  python3 tools/v3_chexpert_ci.py [--n_boot 2000] [--workers 16] [--seeds 0,1,2]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.v2_extended_ci import METRICS, SortedScores  # noqa: E402

D = ROOT / "results" / "V3A"
_G = {}


def model_sets(model_id):
    d = np.load(D / "scores" / f"{model_id}.npz")
    w, b = d["head_w"].astype(np.float64), float(d["head_b"][0])
    q = d["q_san"].astype(np.float64)
    n = len(q)
    ii, jj = np.where(~np.eye(n, dtype=bool))
    out = {}
    for pairing, key in (("SR", "r_raw"), ("SS", "r_san")):
        r = d[key].astype(np.float64)
        full = np.empty((n, n))
        for s in range(0, n, 64):
            full[s:s + 64] = np.abs(q[s:s + 64, None, :] - r[None, :, :]) @ w + b
        out[pairing] = SortedScores(np.diag(full).copy(), np.arange(n), full[ii, jj], ii, jj)
    return out


def _init(mults):
    _G["mults"] = mults


def run_model(model_id):
    res = {}
    for pairing, ss in model_sets(model_id).items():
        point = ss.metrics(None)
        arr = np.full((len(_G["mults"]), len(METRICS)), np.nan)
        for r, m in enumerate(_G["mults"]):
            v = ss.metrics(m)
            if v is not None:
                arr[r] = [v[k] for k in METRICS]
        res[pairing] = (point, arr)
    return model_id, res


def config_of(m):
    if m["family"] == "factorial_ref":
        return f"factorial_ref/{m['sanitizer']}/{m['train_pairing']}{m['val_pairing']}"
    return f"{m['family']}/{m['sanitizer']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds", default="", help="restrict non-factorial arms to these verifier seeds, e.g. 0,1,2")
    a = ap.parse_args()
    keep_seeds = {int(x) for x in a.seeds.split(",")} if a.seeds else None
    models = [json.loads((D / "scores" / f).read_text()) for f in sorted(p.name for p in (D / "scores").glob("*.json"))]
    if keep_seeds is not None:
        models = [m for m in models if m["family"] == "factorial_ref" or int(m["seed"]) in keep_seeds]
    n = len(json.load(open(D / "chexpert_pairs.json"))["pairs"])
    rng = np.random.default_rng(a.seed)
    mults = [np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64) for _ in range(a.n_boot)]
    with Pool(a.workers, initializer=_init, initargs=(mults,)) as pool:
        per_model = dict(pool.map(run_model, [m["model_id"] for m in models]))
    by_cfg = defaultdict(list)
    for m in models:
        by_cfg[config_of(m)].append(m)
    arms = {}       # (cfg, pairing) -> dict(point, boot_patient, boot_ps)
    fact_draws = {}
    for cfg, ms in by_cfg.items():
        ms = sorted(ms, key=lambda m: int(m["seed"]))
        key = cfg.rsplit("/", 1)[0] if cfg.startswith("factorial_ref") else cfg
        if key not in fact_draws:
            g = np.random.default_rng([a.seed, sum(map(ord, key))])
            fact_draws[key] = g.integers(0, len(ms), (a.n_boot, len(ms)))
        draws = fact_draws[key]
        for pairing in ("SR", "SS"):
            pts = np.array([[per_model[m["model_id"]][pairing][0][k] for k in METRICS] for m in ms])
            boots = np.stack([per_model[m["model_id"]][pairing][1] for m in ms])        # seeds x boot x metric
            arms[(cfg, pairing)] = {"point": pts.mean(0), "seeds": [int(m["seed"]) for m in ms],
                                    "per_seed": pts, "bp": np.nanmean(boots, axis=0),
                                    "bps": np.stack([np.nanmean(boots[draws[r], r, :], axis=0) for r in range(a.n_boot)])}

    def ci(x):
        x = x[np.isfinite(x)]
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    def lincomb(terms):
        pt = sum(c * arms[k]["point"] for k, c in terms)
        bp = sum(c * arms[k]["bp"] for k, c in terms)
        bps = sum(c * arms[k]["bps"] for k, c in terms)
        return {met: {"point": float(pt[i]), "patient": ci(bp[:, i]), "patient+seed": ci(bps[:, i])}
                for i, met in enumerate(METRICS)}

    out = {"n_boot": a.n_boot, "seed": a.seed, "n_patients": n, "n_negatives": n * (n - 1),
           "seeds_filter": a.seeds or "all",
           "arms": {f"{c}/{p}": {"seeds": v["seeds"], **{met: float(v["point"][i]) for i, met in enumerate(METRICS)},
                                 **{f"{met}_ci_ps": ci(v["bps"][:, i]) for i, met in enumerate(METRICS)}}
                    for (c, p), v in arms.items()},
           "contrasts": {}}
    C = out["contrasts"]
    for v in ("reference", "e5", "imagenet_full"):
        f = f"{v}/ff_mu0.01"
        for c in ("blur_s2", "elastic_m0.045_keyed", "elastic_m0.04_keyed"):
            k = f"{v}/{c}"
            if (k, "SR") not in arms or (f, "SR") not in arms:
                continue
            for p in ("SR", "SS"):
                C[f"H1/{p}/{v}/ff_mu0.01_minus_{c}"] = lincomb([((f, p), 1), ((k, p), -1)])
            C[f"H2/{v}/ff_mu0.01_minus_{c}"] = lincomb([((f, "SS"), 1), ((f, "SR"), -1), ((k, "SS"), -1), ((k, "SR"), 1)])
        for c in ("none", "ff_mu0.01", "blur_s2", "elastic_m0.045_keyed", "elastic_m0.04_keyed"):
            if (f"{v}/{c}", "SR") in arms and c != "none":
                C[f"SSminusSR/{v}/{c}"] = lincomb([((f"{v}/{c}", "SS"), 1), ((f"{v}/{c}", "SR"), -1)])
    fr = "factorial_ref/ff_mu0.01"
    if all((f"{fr}/{x}", "SR") in arms for x in ("SSSS", "SSSR", "SRSS", "SRSR")):
        C["H3/factorial_ref/ff_mu0.01/train_effect"] = lincomb(
            [((f"{fr}/SRSS", "SR"), .5), ((f"{fr}/SRSR", "SR"), .5), ((f"{fr}/SSSS", "SR"), -.5), ((f"{fr}/SSSR", "SR"), -.5)])
    for v in ("reference", "e5"):
        a_, b_ = f"v3c_{v}/ffv3c_SS_s42", f"v3c_{v}/ffv3c_SR_s42"
        if (a_, "SR") in arms and (b_, "SR") in arms:
            C[f"H4/SS/{v}/SSadv_minus_SRadv"] = lincomb([((a_, "SS"), 1), ((b_, "SS"), -1)])
            C[f"H5/SR/{v}/SSadv_minus_SRadv"] = lincomb([((a_, "SR"), 1), ((b_, "SR"), -1)])
            if (f"{v}/ff_mu0.01", "SR") in arms:      # replication check (descriptive): SR-adv minus released generator
                for pr in ("SR", "SS"):
                    C[f"repl/{pr}/{v}/SRadv_minus_released"] = lincomb([((b_, pr), 1), ((f"{v}/ff_mu0.01", pr), -1)])
        # Exploratory (not in the plan): final-epoch generators (sanitizers *_last).
        for a2, b2, tag in (("ffv3c_SS_s42_last", "ffv3c_SR_s42_last", "SSlast_minus_SRlast"),
                            ("ffv3c_SS_s42_last", "ffv3c_SS_s42", "SSlast_minus_SS"),
                            ("ffv3c_SR_s42_last", "ffv3c_SR_s42", "SRlast_minus_SR")):
            A_, B_ = f"v3c_{v}/{a2}", f"v3c_{v}/{b2}"
            if (A_, "SR") in arms and (B_, "SR") in arms:
                for pr in ("SR", "SS"):
                    C[f"V3Cx/{pr}/{v}/{tag}"] = lincomb([((A_, pr), 1), ((B_, pr), -1)])
    # V3-D: Privacy-Net at the utility-matched point (pnv3d); matched controls from results/V3D/selection.json.
    sel_p = ROOT / "results" / "V3D" / "selection.json"
    if sel_p.is_file():
        sel = json.load(open(sel_p))
        for v in ("reference", "e5"):
            pn = f"v3d_{v}/pnv3d"
            if (pn, "SR") not in arms:
                continue
            for fam_ in ("blur", "elastic"):
                k = f"{v}/{sel['matched'][fam_]}"
                if (k, "SR") not in arms:
                    continue
                for pr in ("SR", "SS"):
                    C[f"V3D/{pr}/{v}/pn_minus_{fam_}"] = lincomb([((pn, pr), 1), ((k, pr), -1)])
                C[f"H7/{v}/pn_minus_{fam_}"] = lincomb([((pn, "SS"), 1), ((pn, "SR"), -1), ((k, "SS"), -1), ((k, "SR"), 1)])
                if fam_ == "blur":
                    C[f"H6/SR/{v}/pn_minus_blur"] = lincomb([((pn, "SR"), 1), ((k, "SR"), -1)])
        a_, b_ = "v3d_reference/pnv3d", "v3d_referenceSS/pnv3d"
        if (a_, "SR") in arms and (b_, "SR") in arms:
            for pr in ("SR", "SS"):
                C[f"H8/{pr}/reference/SRtrain_minus_SStrain"] = lincomb([((a_, pr), 1), ((b_, pr), -1)])
    name = "chexpert_ci.json" if not a.seeds else f"chexpert_ci_seeds{a.seeds.replace(',', '')}.json"
    (D / name).write_text(json.dumps(out, indent=1))
    print(f"wrote {D / name}: {len(arms)} arms, {len(C)} contrasts")


if __name__ == "__main__":
    main()
