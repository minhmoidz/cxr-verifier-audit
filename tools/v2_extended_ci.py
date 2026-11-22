"""E1/E2 analysis: extended negative set and SS test pairs, patient-clustered bootstrap.

Reads results/E1E2/scores/<model_id>.npz (tools/v2_embed_score.py) and computes, per model and per
test pairing (SR = reference raw; SS = reference also sanitized):
  ext   2,500 released positive pairs vs 1,073 x 1,072 = 1,150,256 cross-patient gallery pairs
  t5k   the 5,000 released test pairs
metrics AUC, TPR at 1% FPR and TPR at 0.1% FPR (percent; TPR by linear interpolation of the ROC at
the target FPR, the convention of prl_lib.compute_binary_metrics).

Bootstrap (2,000 replicates, as Table S5): test patients (1,307) resampled with replacement; a
positive pair gets weight m_p, a negative pair m_a * m_b. The patient draws are generated once and
shared by every model, so contrasts between configurations are paired. Per configuration:
  patient        seed scores averaged within each arm
  patient+seed   seeds also resampled with replacement within each arm (independently per arm)

Outputs: results/E1E2/extended_ci.json (points, intervals, contrasts) and
         results/E1E2/boot/<config>.npz (per-replicate arm values, for contrasts).
Usage:   python3 tools/v2_extended_ci.py [--n_boot 2000] [--seed 20261123] [--workers 24]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts" / "paper_upgrade" / "phase1_patient_bootstrap"))
import phase1_bootstrap as B  # noqa: E402

D = ROOT / "results" / "E1E2"
TARGETS = {"tpr1": 0.01, "tpr01": 0.001}
METRICS = ("auc", "tpr1", "tpr01")
SETS = ("ext", "t5k")
PAIRINGS = ("SR", "SS")

_G = {}


class SortedScores:
    """Positives and negatives with per-pair patient codes, pre-sorted for fast weighted metrics."""

    def __init__(self, pos_s, pos_pa, neg_s, neg_pa, neg_pb):
        o = np.argsort(-neg_s, kind="stable")
        self.neg_desc = neg_s[o]
        self.na, self.nb = neg_pa[o], neg_pb[o]
        self.pos = pos_s
        self.pp = pos_pa

    def metrics(self, m):
        wn = (m[self.na] * m[self.nb]).astype(np.float64) if m is not None else np.ones(len(self.neg_desc))
        wp = m[self.pp].astype(np.float64) if m is not None else np.ones(len(self.pos))
        Wn, Wp = wn.sum(), wp.sum()
        if Wn == 0 or Wp == 0:
            return None
        cfp = np.cumsum(wn)
        out = {}
        for k, f in TARGETS.items():
            j = int(np.searchsorted(cfp, f * Wn, side="right"))
            thr = self.neg_desc[j] if j < len(self.neg_desc) else -np.inf
            out[k] = 100.0 * wp[self.pos > thr].sum() / Wp
        # AUC: weight of negatives strictly below each positive, ties counted half.
        asc = self.neg_desc[::-1]
        cw = np.concatenate([[0.0], np.cumsum(wn[::-1])])
        lo = np.searchsorted(asc, self.pos, side="left")
        hi = np.searchsorted(asc, self.pos, side="right")
        out["auc"] = 100.0 * float((wp * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))).sum() / (Wp * Wn))
        return out


def model_sets(npz_path, pid_codes, y):
    d = np.load(npz_path)
    gp = [str(p) for p in d["gal_patients"]]
    gcode = np.array([pid_codes[p] for p in gp])
    w, b = d["head_w"].astype(np.float64), float(d["head_b"][0])
    q = d["gal_q_san"].astype(np.float64)
    n = len(gp)
    ii, jj = np.where(~np.eye(n, dtype=bool))
    out = {}
    for pairing, refkey, tkey in (("SR", "gal_r_raw", "test_SR_logit"), ("SS", "gal_r_san", "test_SS_logit")):
        r = d[refkey].astype(np.float64)
        neg = np.empty(len(ii))
        step = 64
        for s in range(0, n, step):                          # logits for query rows s..s+step
            blk = np.abs(q[s:s + step, None, :] - r[None, :, :]) @ w + b
            mask = (ii >= s) & (ii < s + step)
            neg[mask] = blk[ii[mask] - s, jj[mask]]
        t = d[tkey].astype(np.float64)
        pos_mask = y == 1
        out[("ext", pairing)] = (t[pos_mask], neg, gcode[ii], gcode[jj])
        out[("t5k", pairing)] = (t, None, None, None)
    return out


def _init(pid_a_codes, pid_b_codes, y, pid_codes, mults):
    _G.update(pa=pid_a_codes, pb=pid_b_codes, y=y, pid_codes=pid_codes, mults=mults)


def run_model(model_id):
    pa, pb, y, mults = _G["pa"], _G["pb"], _G["y"], _G["mults"]
    sets = model_sets(D / "scores" / f"{model_id}.npz", _G["pid_codes"], y)
    res = {}
    for (st, pairing), (ps, neg, na, nb) in sets.items():
        if st == "ext":
            ss = SortedScores(ps, pa[y == 1], neg, na, nb)
        else:
            ss = SortedScores(ps[y == 1], pa[y == 1], ps[y == 0], pa[y == 0], pb[y == 0])
        point = ss.metrics(None)
        arr = np.full((len(mults), len(METRICS)), np.nan)
        for r, m in enumerate(mults):
            v = ss.metrics(m)
            if v is not None:
                arr[r] = [v[k] for k in METRICS]
        res[(st, pairing)] = (point, arr)
    return model_id, res


def config_of(m):
    fam = m["family"]
    if fam in ("factorial_ref", "factorial_imagenet"):
        return f"{fam}/{m['sanitizer']}/{m['train_pairing']}{m['val_pairing']}"
    return f"{fam}/{m['sanitizer']}"


def seed_draws(cfg, n_seeds, n_boot, base):
    h = int(hashlib.sha256(f"{base}:{cfg}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(h).integers(0, n_seeds, (n_boot, n_seeds))


def summarize(vals):
    v = vals[np.isfinite(vals)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], int(len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    models = [m for m in json.load(open(D / "models.json")) if (D / "scores" / f"{m['model_id']}.npz").exists()]
    _, _, y, pid_a, pid_b = B.load_pairs(B.TEST_PAIRS)
    pats = sorted(set(pid_a) | set(pid_b))
    pid_codes = {p: i for i, p in enumerate(pats)}
    pa = np.array([pid_codes[p] for p in pid_a]); pb = np.array([pid_codes[p] for p in pid_b])
    rng = np.random.default_rng(a.seed)
    mults = [np.bincount(rng.integers(0, len(pats), len(pats)), minlength=len(pats)).astype(np.int64)
             for _ in range(a.n_boot)]
    with Pool(a.workers, initializer=_init, initargs=(pa, pb, y, pid_codes, mults)) as pool:
        per_model = dict(pool.imap_unordered(run_model, [m["model_id"] for m in models]))

    by_cfg = defaultdict(list)
    for m in models:
        by_cfg[config_of(m)].append(m)
    (D / "boot").mkdir(exist_ok=True)
    out = {"n_boot": a.n_boot, "seed": a.seed, "n_test_patients": len(pats),
           "n_ext_negatives": None, "configs": {}}
    cfg_arrays = {}
    for cfg, ms in sorted(by_cfg.items()):
        ms = sorted(ms, key=lambda m: m["seed"])
        ent = {"seeds": [m["seed"] for m in ms], "models": [m["model_id"] for m in ms],
               "sources": sorted({m.get("source", "published") for m in ms})}
        draws = seed_draws(cfg, len(ms), a.n_boot, a.seed)
        arrays = {}
        for st in SETS:
            for pairing in PAIRINGS:
                pts = np.array([[per_model[m["model_id"]][(st, pairing)][0][k] for k in METRICS] for m in ms])
                arr = np.stack([per_model[m["model_id"]][(st, pairing)][1] for m in ms])   # (seeds, boot, metric)
                pat = arr.mean(axis=0)
                hier = np.stack([arr[draws[r], r, :].mean(axis=0) for r in range(a.n_boot)])
                arrays[f"{st}_{pairing}_patient"] = pat
                arrays[f"{st}_{pairing}_hier"] = hier
                for i, k in enumerate(METRICS):
                    ci_p, n_p = summarize(pat[:, i]); ci_h, n_h = summarize(hier[:, i])
                    ent[f"{st}/{pairing}/{k}"] = {"point": float(pts[:, i].mean()),
                                                  "per_seed": [float(x) for x in pts[:, i]],
                                                  "patient": ci_p, "patient+seed": ci_h, "n_valid": [n_p, n_h]}
        cfg_arrays[cfg] = arrays
        np.savez_compressed(D / "boot" / f"{cfg.replace('/', '__')}.npz", **arrays)
        out["configs"][cfg] = ent

    # Contrasts (difference of seed means, paired patient draws; arms resampled independently for seeds).
    def contrast(c1, c2, st, p1, p2=None):
        p2 = p2 or p1
        if c1 not in cfg_arrays or c2 not in cfg_arrays:
            return None
        e = {}
        for i, k in enumerate(METRICS):
            pt = out["configs"][c1][f"{st}/{p1}/{k}"]["point"] - out["configs"][c2][f"{st}/{p2}/{k}"]["point"]
            dp = cfg_arrays[c1][f"{st}_{p1}_patient"][:, i] - cfg_arrays[c2][f"{st}_{p2}_patient"][:, i]
            dh = cfg_arrays[c1][f"{st}_{p1}_hier"][:, i] - cfg_arrays[c2][f"{st}_{p2}_hier"][:, i]
            e[k] = {"point": pt, "patient": summarize(dp)[0], "patient+seed": summarize(dh)[0]}
        return e

    def lincomb(terms, st, pairing):
        """sum(coef * arm) for terms [(cfg, coef)], same test pairing for every arm."""
        if any(c not in cfg_arrays for c, _ in terms):
            return None
        e = {}
        for i, k in enumerate(METRICS):
            pt = sum(coef * out["configs"][c][f"{st}/{pairing}/{k}"]["point"] for c, coef in terms)
            dp = sum(coef * cfg_arrays[c][f"{st}_{pairing}_patient"][:, i] for c, coef in terms)
            dh = sum(coef * cfg_arrays[c][f"{st}_{pairing}_hier"][:, i] for c, coef in terms)
            e[k] = {"point": pt, "patient": summarize(dp)[0], "patient+seed": summarize(dh)[0]}
        return e

    C = {}
    for st in SETS:
        for pairing in PAIRINGS:
            for fam in ("factorial_ref", "factorial_imagenet"):
                for san in ("ff_mu0.01", "blur_s2", "elastic_m0.04_keyed"):
                    f = lambda tv: f"{fam}/{san}/{tv}"
                    C[f"{st}/{pairing}/{fam}/{san}/train_effect"] = lincomb(
                        [(f("SRSS"), .5), (f("SRSR"), .5), (f("SSSS"), -.5), (f("SSSR"), -.5)], st, pairing)
                    C[f"{st}/{pairing}/{fam}/{san}/val_effect"] = lincomb(
                        [(f("SSSR"), .5), (f("SRSR"), .5), (f("SSSS"), -.5), (f("SRSS"), -.5)], st, pairing)
                    C[f"{st}/{pairing}/{fam}/{san}/interaction"] = lincomb(
                        [(f("SRSR"), 1), (f("SRSS"), -1), (f("SSSR"), -1), (f("SSSS"), 1)], st, pairing)
        for pairing in PAIRINGS:
            for ver, ctrls in (("reference", ("blur_s2", "elastic_m0.045_keyed", "elastic_m0.04_keyed")),
                               ("imagenet_full", ("blur_s2", "elastic_m0.04_keyed"))):
                for c in ctrls:
                    C[f"{st}/{pairing}/{ver}/ff_mu0.01_minus_{c}"] = contrast(f"{ver}/ff_mu0.01", f"{ver}/{c}", st, pairing)
            for fam in ("factorial_ref", "factorial_imagenet"):
                for san in ("ff_mu0.01", "blur_s2", "elastic_m0.04_keyed"):
                    for vp in ("SS", "SR"):
                        C[f"{st}/{pairing}/{fam}/{san}/train_SR_minus_SS_at_val{vp}"] = contrast(
                            f"{fam}/{san}/SR{vp}", f"{fam}/{san}/SS{vp}", st, pairing)
        # E2: SS test minus SR test, same models.
        for cfg in cfg_arrays:
            C[f"{st}/SS_minus_SR/{cfg}"] = contrast(cfg, cfg, st, "SS", "SR")
    out["contrasts"] = {k: v for k, v in C.items() if v is not None}
    out["n_ext_negatives"] = 1073 * 1072
    json.dump(out, open(D / "extended_ci.json", "w"), indent=1)
    print(f"wrote {D / 'extended_ci.json'}: {len(out['configs'])} configurations, {len(out['contrasts'])} contrasts")


if __name__ == "__main__":
    main()
