"""Phase 1: Patient-clustered bootstrap for the 11-configuration leakage benchmark.

Computes, for each sanitizer configuration and each attacker arm (frozen / ImageNet-init
"naive" / verifier-init "adapted"):

  - ROC AUC on the test split
  - TPR at 1% FPR read from the test ROC (descriptive operating point)
  - the single threshold selected on VALIDATION at 1% FPR, applied to the test split:
    realised test FPR and TPR
  - 95% patient-clustered bootstrap intervals, computed with a DYADIC / multiplicity
    weighting so that negative pairs (two patients) are treated correctly
  - across-seed SD, patient-sampling CI, and a hierarchical seed+patient interval

Paired comparisons (same bootstrap patient sample for both arms):
  - audit gap = verifier-init AUC - ImageNet-init AUC
  - paired differences of learned mu=0.01 vs random elastic and vs Gaussian blur at
    matched utility (verifier-init arm)

Usage:
  python3 phase1_bootstrap.py [--n_boot 10000] [--seed 0] [--out <dir>]
  python3 phase1_bootstrap.py --sim          # run the simulation self-test only

Everything needed is on disk (per-seed test predictions in archive/, per-seed validation
thresholds in scores/val_thresholds.json, frozen scores in scores/).
"""

import argparse
import csv
import hashlib
import subprocess
import time
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# ROOT should be the repo root. Script lives at artifacts/paper_upgrade/phase1_patient_bootstrap/
if not os.path.isdir(os.path.join(ROOT, 'image_pairs')):
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.isdir(os.path.join(ROOT, 'image_pairs')):
    raise RuntimeError('cannot locate repo root')

TEST_PAIRS = os.path.join(ROOT, 'image_pairs/image_pairs_testing_5000.txt')
VAL_PAIRS = os.path.join(ROOT, 'image_pairs/image_pairs_validation_2000.txt')
SCORE_DIR = os.path.join(ROOT, 'artifacts/paper_upgrade/phase1_patient_bootstrap/scores')

# 11 configurations: config key -> (naive tag, adapted tag)
CONFIGS = [
    ('none', 'none', 'e11_none'),
    ('ff_mu0.001', 'ff_mu0.001', 'e11_ff_mu0.001'),
    ('ff_mu0.005', 'ff_mu0.005', 'e11_ff_mu0.005'),
    ('ff_mu0.01', 'ff_mu0.01', 'e11_ff_mu0.01'),
    ('blur_s2', 'blur_s2', 'e11_blur_s2'),
    ('elastic_m004', 'elastic_m004', 'e11_elastic_m004'),
    ('privacy_net', 'privacy_net', 'e11_privacy_net'),
    ('dppix_b1', 'dppix_b1', 'e11_dppix_b1'),
    ('dppix_b2', 'dppix_b2', 'e11_dppix_b2'),
    ('dppix_b4', 'dppix_b4', 'e11_dppix_b4'),
    ('dppix_b8', 'dppix_b8', 'e11_dppix_b8'),
]


# --------------------------------------------------------------------------- data loading

def load_pairs(path):
    rows = np.loadtxt(path, dtype=str)
    names_a, names_b = rows[:, 0], rows[:, 1]
    y = rows[:, 2].astype(float)
    # patient id of each endpoint; positive pairs share one, negative pairs span two
    pid_a = np.array([n.split('_')[0] for n in names_a])
    pid_b = np.array([n.split('_')[0] for n in names_b])
    return names_a, names_b, y, pid_a, pid_b


def per_seed_test_scores(tag, n_seeds=10):
    """ImageNet-init / verifier-init per-seed test scores (ordered by seed 0..n-1)."""
    rows = []
    for sd in range(n_seeds):
        f = glob.glob(os.path.join(ROOT, f'archive/ver_{tag}_seed{sd}/*_labels_predictions.txt'))
        if not f:
            return None
        a = np.loadtxt(f[0], skiprows=1)
        rows.append(a[:, 1])
    return np.stack(rows)  # (n_seeds, n_pairs)


def load_frozen_scores(tag, phase):
    s = np.load(os.path.join(SCORE_DIR, f'frozen_{tag}_{phase}_scores.npy'))
    y = np.load(os.path.join(SCORE_DIR, f'frozen_{tag}_{phase}_labels.npy'))
    return y, s


def load_val_thresholds():
    p = os.path.join(SCORE_DIR, 'val_thresholds.json')
    if os.path.isfile(p):
        return json.load(open(p))
    return {}


# --------------------------------------------------------------------------- metrics (weighted)

def weighted_roc(y, s, w):
    """Weighted ROC: returns ascending-FPR thresholds, weighted FPR, weighted TPR.

    Convention matches sklearn: a pair is "positive" when score >= threshold. For a given
    threshold value we need the weight of positives/negatives at or above it, so we sort
    by score descending and accumulate, then reverse to get ascending FPR.
    """
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    W_pos = w[y == 1].sum()
    W_neg = w[y == 0].sum()
    if W_pos == 0 or W_neg == 0:
        return None, None, None
    idx = np.argsort(-s, kind='stable')          # descending score
    s_d = s[idx]; y_d = y[idx]; w_d = w[idx]
    cp = np.cumsum(w_d * (y_d == 1))              # weight of positives with score >= current
    cn = np.cumsum(w_d * (y_d == 0))              # weight of negatives with score >= current
    # Distinct thresholds, walking the scores from high to low.  For a block of TIED
    # scores the ROC point must use the cumulative weight at the END of the block (all
    # pairs with score >= that value), so take the LAST index of each tied run.
    # The previous implementation used np.unique(..., return_index=True), which returns
    # the FIRST occurrence and therefore undercounts ties; it shifted the AUC by up to
    # 1.3e-4 points on the DP-Pix configs, the only ones with tied scores.
    # Walking high->low makes cn (and hence FPR) non-decreasing, so the arrays are
    # already in ascending-FPR order and must NOT be reversed.
    last_idx = np.r_[np.nonzero(np.diff(s_d))[0], len(s_d) - 1]
    thresh = s_d[last_idx]            # descending score, paired with ascending FPR
    fpr = cn[last_idx] / W_neg
    tpr = cp[last_idx] / W_pos
    return thresh, fpr, tpr


def weighted_auc(y, s, w):
    thr, fpr, tpr = weighted_roc(y, s, w)
    if thr is None:
        return None
    # trapezoidal integration of TPR over FPR (0..1)
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    return float(np.trapezoid(tpr, fpr))


def weighted_tpr_at(y, s, w, target_fpr=0.01):
    thr, fpr, tpr = weighted_roc(y, s, w)
    if thr is None:
        return None, None
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    val = float(np.interp(target_fpr, fpr, tpr))
    return val, None


def weighted_operating_point(y_test, s_test, w_test, threshold):
    """Realised weighted FPR/TPR on test given a fixed threshold (per cent)."""
    pos = y_test == 1
    neg = y_test == 0
    w_pos = w_test[pos].sum()
    w_neg = w_test[neg].sum()
    fpr = 100.0 * w_test[neg][s_test[neg] >= threshold].sum() / w_neg if w_neg > 0 else float('nan')
    tpr = 100.0 * w_test[pos][s_test[pos] >= threshold].sum() / w_pos if w_pos > 0 else float('nan')
    return fpr, tpr


def threshold_at_fpr(y_val, s_val, w_val, target_fpr=0.01):
    """Single threshold achieving target weighted FPR on validation."""
    thr, fpr, tpr = weighted_roc(y_val, s_val, w_val)
    if thr is None:
        return None
    fpr = np.concatenate([[0.0], fpr])
    thr = np.concatenate([[np.max(s_val) + 1.0], thr])
    return float(np.interp(target_fpr, fpr, thr))


# --------------------------------------------------------------------------- dyadic weights

def _encode_pids(pid_a, pid_b):
    """Map patient ids to 0..K-1 without an object-identity cache.

    The previous cache used ``id(pid_a)`` as its key. CPython may reuse an object id after
    garbage collection, allowing a different pair list to receive stale patient codes. The
    arrays are small enough that encoding at each bootstrap entry point is negligible.
    """
    pats, inverse = np.unique(np.concatenate([pid_a, pid_b]), return_inverse=True)
    codes_a = inverse[:len(pid_a)]
    codes_b = inverse[len(pid_a):]
    return codes_a, codes_b, codes_a == codes_b, len(pats)


def dyadic_weights_fast(codes_a, codes_b, is_pos, n_patients, rng):
    """Vectorised version of dyadic_weights using integer-coded patient ids."""
    draws = rng.integers(0, n_patients, n_patients)
    m = np.bincount(draws, minlength=n_patients)
    w = m[codes_a].astype(float)
    neg = ~is_pos
    w[neg] = w[neg] * m[codes_b][neg]
    return w


def dyadic_weights(pid_a, pid_b, rng):
    """Bootstrap multiplicities per pair from a patient resample (vectorised).

    Resample the set of patients with replacement (as many draws as there are patients).

    For a positive pair (both images of the same patient p): the pair is a single
    observation belonging to ONE patient cluster.  Under the cluster bootstrap, each
    draw of patient p replicates the pair once, so weight = m_p (the number of times p
    is drawn).  This is the standard one-way cluster bootstrap weight for within-cluster
    observations.

    For a negative pair (patients p_i, p_j): the observation bridges two patients and
    is treated as a dyad.  Under the two-way (dyadic) cluster bootstrap, the pair is
    replicated once for every combination of the m_ai copies of p_i and the m_bj copies
    of p_j, so weight = m_ai * m_bj.

    The hybrid scheme (positive one-way, negative two-way) is validated by a coverage
    simulation (see audit_sim_coverage.py): ~94.7% coverage on 150 simulated cohorts
    (nominal 95%), while the pure product scheme (positive = m_p^2) gives 100% coverage
    (too conservative) and the naive pair bootstrap gives 56% (anti-conservative).
    """
    ca, cb, is_pos, npats = _encode_pids(pid_a, pid_b)
    return dyadic_weights_fast(ca, cb, is_pos, npats, rng)


# --------------------------------------------------------------------------- seed handling

def mean_over_seeds(metric_fn, seeds_scores, y, w):
    vals = []
    for s in seeds_scores:
        v = metric_fn(y, s, w)
        if v is None or (isinstance(v, tuple) and v[0] is None):
            return None
        vals.append(v if not isinstance(v, tuple) else v[0])
    return float(np.mean(vals))


def mean_tpr_at(seeds_scores, y, w, target=0.01):
    vals = []
    for s in seeds_scores:
        v, _ = weighted_tpr_at(y, s, w, target)
        if v is None:
            return None
        vals.append(v)
    return float(np.mean(vals))


def seed_std_across(metric_fn, seeds_scores, y):
    """Across-training-seed SD of a metric on the full test set (NOT a CI)."""
    vals = []
    w_full = np.ones(len(y))
    for s in seeds_scores:
        v = metric_fn(y, s, w_full)
        if v is None:
            return float('nan')
        vals.append(v)
    if len(vals) < 2:
        return 0.0   # single deterministic arm (frozen)
    return float(np.std(vals, ddof=1))


def seed_std_tpr(seeds_scores, y):
    """Across-training-seed SD of TPR@1%FPR, expressed in percentage points."""
    vals = []
    w_full = np.ones(len(y))
    for s in seeds_scores:
        v, _ = weighted_tpr_at(y, s, w_full, 0.01)
        vals.append(v * 100.0)
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1))


# --------------------------------------------------------------------------- bootstrap

def operating_point_intervals(seeds_scores, y, pid_a, pid_b, thresholds, n_boot, rng):
    """Patient-clustered intervals for the realised test operating point.

    THRESHOLD POLICY: one validation-selected threshold PER SEED, held FIXED.  The
    validation split is never resampled inside the bootstrap; only test patients are
    resampled (plus training seeds in the hierarchical variant).  This matches the point
    estimand, which is the mean over seeds of the per-seed metric at that seed's own
    threshold.

    Returns dict with three explicitly separated estimands for each of TPR and FPR:
      'A_perseed_avg'  patient-only; per-seed 2.5/97.5 percentiles, endpoints averaged
                       across seeds.  LEGACY: an average of per-seed conditional
                       intervals, NOT an interval for the mean over seeds.  Retained so
                       older tables remain reproducible.
      'B_mean_seeds'   patient-only; ONE patient resample per replicate applied to every
                       seed at its own threshold, averaged over seeds, then percentiled.
                       PRIMARY: this is the interval for the reported point estimate.
      'C_hier'         hierarchical: as B, additionally resampling the training seeds
                       with replacement inside each replicate (patient + seed).

    A defect fixed here: the interval was previously computed with `thh=tvals[0]`, i.e.
    seed 0's threshold applied to all seeds, while the point estimate used per-seed
    thresholds.  Point and interval targeted different quantities.
    """
    S = np.asarray(seeds_scores)
    ths = np.asarray(thresholds, dtype=float)
    if len(ths) != len(S):
        raise ValueError(f'{len(ths)} thresholds for {len(S)} seeds - refusing to zip-truncate')
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    # indicators do not depend on the bootstrap weights: precompute once
    Ipos = np.stack([(S[k][pos] >= ths[k]).astype(np.float64) for k in range(len(S))])
    Ineg = np.stack([(S[k][neg] >= ths[k]).astype(np.float64) for k in range(len(S))])

    ca, cb, is_pos, npat = _encode_pids(pid_a, pid_b)
    T = np.empty((n_boot, len(S)))
    F = np.empty((n_boot, len(S)))
    for r in range(n_boot):
        w = dyadic_weights_fast(ca, cb, is_pos, npat, rng)
        wp, wn = w[pos], w[neg]
        sp, sn = wp.sum(), wn.sum()
        T[r] = 100.0 * (Ipos @ wp) / sp if sp > 0 else np.nan
        F[r] = 100.0 * (Ineg @ wn) / sn if sn > 0 else np.nan

    out = {}
    ns = len(S)
    for lab, D in (('tpr', T), ('fpr', F)):
        a_lo = float(np.nanmean(np.nanpercentile(D, 2.5, axis=0)))
        a_hi = float(np.nanmean(np.nanpercentile(D, 97.5, axis=0)))
        mb = np.nanmean(D, axis=1)
        b_lo, b_hi = (float(x) for x in np.nanpercentile(mb, [2.5, 97.5]))
        if ns > 1:
            sel = rng.integers(0, ns, size=(n_boot, ns))
            hb = np.nanmean(np.take_along_axis(D, sel, axis=1), axis=1)
            c_lo, c_hi = (float(x) for x in np.nanpercentile(hb, [2.5, 97.5]))
        else:
            hb, c_lo, c_hi = mb, b_lo, b_hi
        out[lab] = {
            'A_perseed_avg': [round(a_lo, 4), round(a_hi, 4)],
            'B_mean_seeds': [round(b_lo, 4), round(b_hi, 4)],
            'C_hier': [round(c_lo, 4), round(c_hi, 4)],
            'boot_mean': round(float(np.nanmean(mb)), 4),
            'boot_median': round(float(np.nanmedian(mb)), 4),
            'quantiles': {q: round(float(np.nanpercentile(mb, q)), 4)
                          for q in (2.5, 25, 50, 75, 97.5)},
            'n_valid': int(np.isfinite(mb).sum()),
            'degenerate': bool(np.nanstd(mb) == 0.0),
        }
    return out


def patient_ci_test(seeds_scores, y, pid_a, pid_b, metric_fn, n_boot, rng, target_fpr=None):
    """Patient-clustered bootstrap on test pairs, averaged over seeds.

    For each seed, run n_boot patient resamples; pool the per-seed percentile intervals by
    averaging their endpoints. Returns (lo, hi, n_valid_replicates).
    """
    los, his, nvalid = [], [], []
    for s in seeds_scores:
        lo, hi, nv = [], [], 0
        for _ in range(n_boot):
            w = dyadic_weights(pid_a, pid_b, rng)
            if metric_fn is None:
                v, _ = weighted_tpr_at(y, s, w, target_fpr)
            else:
                v = metric_fn(y, s, w)
            if v is None:
                continue
            lo.append(v); hi.append(v); nv += 1
        if lo:
            los.append(np.percentile(lo, 2.5)); his.append(np.percentile(hi, 97.5))
        nvalid.append(nv)
    if not los:
        return None, None, 0
    return float(np.mean(los)), float(np.mean(his)), int(min(nvalid))


def hierarchical_ci(seeds_scores, y, pid_a, pid_b, metric_fn, n_boot, rng,
                    n_seed_draw=10, target_fpr=None):
    """Two-level bootstrap: resample seeds with replacement, then resample patients.

    Captures BOTH training-seed variation and patient-population variation in one interval.
    """
    vals = []
    for _ in range(n_boot):
        drawn = rng.integers(0, len(seeds_scores), n_seed_draw)
        w = dyadic_weights(pid_a, pid_b, rng)
        row = []
        for i in drawn:
            if metric_fn is None:
                v, _ = weighted_tpr_at(y, seeds_scores[i], w, target_fpr)
            else:
                v = metric_fn(y, seeds_scores[i], w)
            if v is not None:
                row.append(v)
        if row:
            vals.append(np.mean(row))
    if not vals:
        return None, None, 0
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)


def _paired_metric(y, s, w, metric_fn, target_fpr):
    """One replicate's metric value for one score array, or None on a degenerate weighting.

    metric_fn(y, s, w) -> float takes priority when given. Otherwise target_fpr selects
    TPR-at-that-FPR (percentage points, via weighted_tpr_at) instead of the default AUC
    (percentage points, via weighted_auc) -- the same metric_fn=None-means-TPR convention
    hierarchical_ci already uses, so the two functions read the same way.
    """
    if metric_fn is not None:
        return metric_fn(y, s, w)
    if target_fpr is not None:
        v, _ = weighted_tpr_at(y, s, w, target_fpr)
        return None if v is None else v * 100
    v = weighted_auc(y, s, w)
    return None if v is None else v * 100


def paired_gap_ci(adapted_scores, naive_scores, y, pid_a, pid_b, n_boot, rng, hierarchical=False,
                   metric_fn=None, target_fpr=None):
    """Paired gap (adapted - naive) patient-clustered CI, in the metric's own units.

    Default (metric_fn=None, target_fpr=None) is the original AUC-in-percentage-points gap,
    numerically unchanged: the RNG is drawn in the same order regardless of which metric is
    computed from a given replicate's weights, so existing AUC callers see bit-identical output.
    Pass target_fpr=0.01 for a paired TPR@1%FPR gap instead (or metric_fn for anything else).

    The SAME patient resample is used for both arms in every replicate. If hierarchical,
    seeds are also resampled with replacement per arm (independent arms, as the two arms'
    seeds are not assumed matched); otherwise all seeds are averaged. A replicate is dropped
    (not silently zero-filled) if ANY drawn seed's metric is undefined for that weighting.
    """
    gaps = []
    for _ in range(n_boot):
        w = dyadic_weights(pid_a, pid_b, rng)
        if hierarchical:
            ia = rng.integers(0, len(adapted_scores), len(adapted_scores))
            ina = rng.integers(0, len(naive_scores), len(naive_scores))
            ad_vals = [_paired_metric(y, adapted_scores[i], w, metric_fn, target_fpr) for i in ia]
            na_vals = [_paired_metric(y, naive_scores[i], w, metric_fn, target_fpr) for i in ina]
        else:
            ad_vals = [_paired_metric(y, s, w, metric_fn, target_fpr) for s in adapted_scores]
            na_vals = [_paired_metric(y, s, w, metric_fn, target_fpr) for s in naive_scores]
        if any(v is None for v in ad_vals) or any(v is None for v in na_vals):
            continue
        gaps.append(float(np.mean(ad_vals)) - float(np.mean(na_vals)))
    if not gaps:
        return None, None, 0
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5)), len(gaps)


# --------------------------------------------------------------------------- main analysis

def analyze_config(cfg, n_boot, rng, val_thresholds):
    name, naive_tag, adapted_tag = cfg
    print(f'\n=== {name} ({naive_tag} / {adapted_tag}) ===', flush=True)

    names_a, names_b, y, pid_a, pid_b = load_pairs(TEST_PAIRS)
    out = {'config': name}
    arms = {'frozen': None, 'imagenet': None, 'verifier': None}

    # frozen (deterministic)
    yf, sf = load_frozen_scores(naive_tag, 'testing')
    assert (yf == y).all()
    arms['frozen'] = sf[None, :]  # single "seed"
    out['frozen'] = {'n_seeds': 1}

    # ImageNet-init and verifier-init (10 seeds)
    S_n = per_seed_test_scores(naive_tag)
    S_a = per_seed_test_scores(adapted_tag)
    if S_n is not None and S_a is not None:
        assert (np.loadtxt(glob.glob(os.path.join(ROOT, f'archive/ver_{naive_tag}_seed0/*_labels_predictions.txt'))[0], skiprows=1)[:, 0] == y).all()
        arms['imagenet'] = S_n
        arms['verifier'] = S_a
        out['imagenet'] = {'n_seeds': len(S_n)}
        out['verifier'] = {'n_seeds': len(S_a)}

    # load validation pairs + thresholds
    yv, sv_frozen = load_frozen_scores(naive_tag, 'validation')
    vt = val_thresholds.get(naive_tag, {})
    vta = val_thresholds.get(adapted_tag, {})
    # frozen arm: deterministic, its validation threshold comes from the frozen
    # validation scores themselves (single threshold, no seeds)
    fz_thresh = threshold_at_fpr(yv, sv_frozen, np.ones(len(yv)), 0.01)

    for arm in ('frozen', 'imagenet', 'verifier'):
        S = arms[arm]
        if S is None:
            out[arm] = {'missing': True}
            continue
        rec = out[arm]
        w_full = np.ones(len(y))

        # point estimates on full test set
        auc_pt = np.mean([roc_auc_score(y, s) for s in S]) * 100
        tpr_pt = np.mean([float(np.interp(0.01, roc_curve(y, s)[0], roc_curve(y, s)[1])) * 100
                          for s in S])
        rec['auc_point'] = round(auc_pt, 2)
        rec['tpr_at_1pct_point'] = round(tpr_pt, 2)

        # across-seed SD
        rec['auc_across_seed_sd'] = round(seed_std_across(
            lambda yy, ss, ww: roc_auc_score(yy, ss) * 100, S, y), 2)
        rec['tpr_at_1pct_across_seed_sd'] = round(seed_std_tpr(S, y), 2)

        # patient-clustered bootstrap (per-seed, averaged endpoints)
        auc_lambda = lambda yy, ss, ww: weighted_auc(yy, ss, ww) * 100.0
        tpr_lambda = lambda yy, ss, ww: weighted_tpr_at(yy, ss, ww, 0.01)[0] * 100.0
        la, ha, nva = patient_ci_test(S, y, pid_a, pid_b, auc_lambda, n_boot, rng)
        if la is None:
            la, ha = float('nan'), float('nan')
        rec['auc_ci'] = [round(la, 2), round(ha, 2)]
        rec['auc_ci_n_replicates'] = nva
        lt, ht, nvt = patient_ci_test(S, y, pid_a, pid_b, tpr_lambda, n_boot, rng)
        rec['tpr_at_1pct_ci'] = [round(lt, 2), round(ht, 2)]
        rec['tpr_at_1pct_ci_n_replicates'] = nvt

        # hierarchical seed+patient interval
        lh, hh, _ = hierarchical_ci(S, y, pid_a, pid_b, auc_lambda, n_boot, rng)
        rec['auc_hier_ci'] = [round(lh, 2), round(hh, 2)] if lh is not None else [None, None]
        lth, hth, _ = hierarchical_ci(S, y, pid_a, pid_b, tpr_lambda, n_boot, rng)
        rec['tpr_at_1pct_hier_ci'] = [round(lth, 2), round(hth, 2)] if lth is not None else [None, None]

        # validation-selected threshold -> realised test operating point (per seed)
        if arm == 'frozen':
            # frozen: deterministic, single threshold from frozen validation scores
            if fz_thresh is not None:
                ft, fprs, tprs = [], [], []
                for s in S:
                    f, t = weighted_operating_point(y, s, np.ones(len(y)), fz_thresh)
                    fprs.append(f); tprs.append(t)
                rec['val_thresh_tpr'] = round(float(np.mean(tprs)), 2)
                rec['val_thresh_tpr_sd'] = 0.0
                rec['realised_test_fpr'] = round(float(np.mean(fprs)), 2)
                rec['realised_test_fpr_sd'] = 0.0
                op = operating_point_intervals(
                    S, y, pid_a, pid_b, [fz_thresh] * len(S), n_boot, rng)
                rec['op'] = op
                rec['val_thresh_tpr_ci'] = op['tpr']['B_mean_seeds']
                rec['realised_fpr_ci'] = op['fpr']['B_mean_seeds']
            else:
                rec['val_thresh_tpr'] = None
                rec['realised_test_fpr'] = None
        else:
            thresh_src = vta if (arm == 'verifier') else vt
            tvals = thresh_src.get('thresholds', [])
            if tvals:
                fprs, tprs = [], []
                for s, th in zip(S, tvals):
                    f, t = weighted_operating_point(y, s, np.ones(len(y)), th)
                    fprs.append(f); tprs.append(t)
                rec['val_thresh_tpr'] = round(float(np.mean(tprs)), 2)
                rec['val_thresh_tpr_sd'] = round(float(np.std(tprs, ddof=1)), 2)
                rec['realised_test_fpr'] = round(float(np.mean(fprs)), 2)
                rec['realised_test_fpr_sd'] = round(float(np.std(fprs, ddof=1)), 2)
                # per-seed validation thresholds, held fixed; only test patients
                # are resampled.  See operating_point_intervals for the estimands.
                op = operating_point_intervals(S, y, pid_a, pid_b, tvals, n_boot, rng)
                rec['op'] = op
                rec['val_thresh_tpr_ci'] = op['tpr']['B_mean_seeds']
                rec['realised_fpr_ci'] = op['fpr']['B_mean_seeds']
            else:
                rec['val_thresh_tpr'] = None
                rec['realised_test_fpr'] = None
                print(f'  WARNING: no validation thresholds for {arm}')

        print(f'  {arm:10s} AUC {auc_pt:6.2f} CI {rec["auc_ci"]} '
              f'TPR {tpr_pt:6.2f} CI {rec["tpr_at_1pct_ci"]}', flush=True)

    # paired audit gap (verifier - imagenet)
    if arms['verifier'] is not None and arms['imagenet'] is not None:
        g_pt = out['verifier']['auc_point'] - out['imagenet']['auc_point']
        gl, gh, _ = paired_gap_ci(arms['verifier'], arms['imagenet'], y, pid_a, pid_b,
                                  n_boot, rng)
        glh, ghh, _ = paired_gap_ci(arms['verifier'], arms['imagenet'], y, pid_a, pid_b,
                                    n_boot, rng, hierarchical=True)
        out['audit_gap'] = {
            'point': round(g_pt, 2),
            'patient_ci': [round(gl, 2), round(gh, 2)] if gl is not None else [None, None],
            'hier_ci': [round(glh, 2), round(ghh, 2)] if glh is not None else [None, None],
        }
        print(f'  gap {g_pt:+.2f} patientCI {out["audit_gap"]["patient_ci"]}', flush=True)
    return out


def paired_utility_diff(cfg_learned, cfg_control, n_boot, rng):
    """verifier-init AUC(control) - AUC(learned mu=0.01), same patient sample."""
    _, naive_l, adapted_l = cfg_learned
    _, naive_c, adapted_c = cfg_control
    names_a, names_b, y, pid_a, pid_b = load_pairs(TEST_PAIRS)
    Sl = per_seed_test_scores(adapted_l)
    Sc = per_seed_test_scores(adapted_c)
    diffs, diffs_hier = [], []
    for _ in range(n_boot):
        w = dyadic_weights(pid_a, pid_b, rng)
        ml = np.mean([weighted_auc(y, s, w) * 100 for s in Sl])
        mc = np.mean([weighted_auc(y, s, w) * 100 for s in Sc])
        if ml is None or mc is None:
            continue
        diffs.append(mc - ml)
        # hierarchical variant
        ia = rng.integers(0, len(Sl), len(Sl)); ic = rng.integers(0, len(Sc), len(Sc))
        hl = np.mean([weighted_auc(y, Sl[i], w) * 100 for i in ia])
        hc = np.mean([weighted_auc(y, Sc[i], w) * 100 for i in ic])
        diffs_hier.append(hc - hl)
    if not diffs:
        return None
    pt = np.mean([roc_auc_score(y, s) * 100 for s in Sc]) - np.mean([roc_auc_score(y, s) * 100 for s in Sl])
    return {
        'point': round(float(pt), 2),
        'patient_ci': [round(float(np.percentile(diffs, 2.5)), 2),
                       round(float(np.percentile(diffs, 97.5)), 2)],
        'hier_ci': [round(float(np.percentile(diffs_hier, 2.5)), 2),
                    round(float(np.percentile(diffs_hier, 97.5)), 2)],
        'n_boot': len(diffs),
    }


# --------------------------------------------------------------------------- outputs

def write_results_long(all_results, outdir):
    path = os.path.join(outdir, 'results_long.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'arm', 'metric', 'point', 'across_seed_sd',
                    'patient_ci_lo', 'patient_ci_hi', 'hier_ci_lo', 'hier_ci_hi'])
        for r in all_results:
            for arm in ('frozen', 'imagenet', 'verifier'):
                rec = r.get(arm)
                if not rec or rec.get('missing'):
                    continue
                w.writerow([r['config'], arm, 'auc', rec['auc_point'],
                            rec['auc_across_seed_sd'],
                            rec['auc_ci'][0], rec['auc_ci'][1],
                            rec.get('auc_hier_ci', [None, None])[0],
                            rec.get('auc_hier_ci', [None, None])[1]])
                w.writerow([r['config'], arm, 'tpr_at_1pct', rec['tpr_at_1pct_point'],
                            rec['tpr_at_1pct_across_seed_sd'],
                            rec['tpr_at_1pct_ci'][0], rec['tpr_at_1pct_ci'][1],
                            rec.get('tpr_at_1pct_hier_ci', [None, None])[0],
                            rec.get('tpr_at_1pct_hier_ci', [None, None])[1]])
                if rec.get('val_thresh_tpr') is not None:
                    w.writerow([r['config'], arm, 'tpr_at_val_thresh', rec['val_thresh_tpr'],
                                rec['val_thresh_tpr_sd'],
                                rec.get('val_thresh_tpr_ci', [None, None])[0],
                                rec.get('val_thresh_tpr_ci', [None, None])[1], '', ''])
                    w.writerow([r['config'], arm, 'realised_test_fpr', rec['realised_test_fpr'],
                                rec['realised_test_fpr_sd'],
                                rec.get('realised_fpr_ci', [None, None])[0],
                                rec.get('realised_fpr_ci', [None, None])[1], '', ''])
            if r.get('audit_gap'):
                g = r['audit_gap']
                w.writerow([r['config'], 'verifier_minus_imagenet', 'audit_gap_auc', g['point'],
                            '', g['patient_ci'][0], g['patient_ci'][1],
                            g.get('hier_ci', [None, None])[0], g.get('hier_ci', [None, None])[1]])
    print(f'wrote {path}')


def write_ci_summary(all_results, outdir):
    path = os.path.join(outdir, 'ci_summary.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'arm', 'auc_point', 'auc_ci', 'tpr_point', 'tpr_ci'])
        for r in all_results:
            for arm in ('frozen', 'imagenet', 'verifier'):
                rec = r.get(arm)
                if not rec or rec.get('missing'):
                    continue
                w.writerow([r['config'], arm, rec['auc_point'],
                            f"{rec['auc_ci'][0]:.2f}-{rec['auc_ci'][1]:.2f}",
                            rec['tpr_at_1pct_point'],
                            f"{rec['tpr_at_1pct_ci'][0]:.2f}-{rec['tpr_at_1pct_ci'][1]:.2f}"])
    print(f'wrote {path}')


# --------------------------------------------------------------------------- simulation self-test

def run_simulation(n_boot=2000, seed=7):
    """Small synthetic dyadic dataset where the true AUC is known.

    Construct patients with a latent "identifiability" so pairs of the same patient score
    higher; the true population AUC can be computed exactly. Then verify the cluster
    bootstrap interval contains the truth with approximately nominal coverage and that
    positive/negative dyadic weighting behaves (negatives never collapse to a single
    patient). Also verifies weighted_auc == unweighted AUC when all weights are equal.
    """

    def P_tpr_point(yy, ss):
        fpr, tpr, _ = roc_curve(yy, ss)
        return float(np.interp(0.01, fpr, tpr))
    rng = np.random.default_rng(seed)
    n_patients = 300
    latent = rng.normal(size=n_patients)
    pos_pairs, neg_pairs = [], []
    for _ in range(400):
        p = rng.integers(0, n_patients)
        pos_pairs.append(p)
    for _ in range(400):
        a = rng.integers(0, n_patients)
        b = rng.integers(0, n_patients)
        while b == a:
            b = rng.integers(0, n_patients)
        neg_pairs.append((a, b))
    pid_a = np.array([f'p{p:03d}' for p in [x if not isinstance(x, tuple) else x[0] for x in pos_pairs]] +
                     [f'p{a:03d}' for a, b in neg_pairs])
    pid_b = np.array([f'p{p:03d}' for p in [x if not isinstance(x, tuple) else x[0] for x in pos_pairs]] +
                     [f'p{b:03d}' for a, b in neg_pairs])
    y = np.array([1.0] * len(pos_pairs) + [0.0] * len(neg_pairs))
    # scores: same-patient high, different-patient low, noise around patient latent
    n = len(y)
    s = np.zeros(n)
    for i, pa in enumerate(pid_a):
        la = latent[int(pa[1:])]
        if y[i] == 1:
            s[i] = la + rng.normal(1.0, 0.3)
        else:
            lb = latent[int(pid_b[i][1:])]
            s[i] = 0.3 * (la + lb) + rng.normal(-1.0, 0.3)
    # ---- sanity: equal weights => weighted AUC == sklearn AUC
    w1 = np.ones(n)
    assert abs(weighted_auc(y, s, w1) - roc_auc_score(y, s)) < 1e-9, 'weighted AUC sanity failed'
    # ---- dyadic weights never put all mass on one class
    for _ in range(50):
        w = dyadic_weights(pid_a, pid_b, np.random.default_rng(_))
        assert (w[y == 1].sum() > 0) and (w[y == 0].sum() > 0)
    # ---- bootstrap coverage: resample only pairs, must stay near the point estimate
    point = roc_auc_score(y, s)
    point_tpr = P_tpr_point(y, s)
    los, his, los_t, his_t = [], [], [], []
    rng2 = np.random.default_rng(seed + 1)
    for _ in range(20):
        lo, hi, _ = patient_ci_test(s[None, :], y, pid_a, pid_b,
                                    lambda yy, ss, ww: weighted_auc(yy, ss, ww),
                                    n_boot, rng2)
        lt, ht, _ = patient_ci_test(s[None, :], y, pid_a, pid_b, None, n_boot, rng2,
                                    target_fpr=0.01)
        los.append(lo); his.append(hi); los_t.append(lt); his_t.append(ht)
    print('  simulation: point AUC', round(point, 4), ' TPR@1%', round(point_tpr, 4))
    print('  simulation: mean AUC bootstrap CI', round(np.mean(los), 4), '-',
          round(np.mean(his), 4))
    print('  simulation: mean TPR bootstrap CI', round(np.mean(los_t), 4), '-',
          round(np.mean(his_t), 4))
    ok_auc = np.mean(los) < point < np.mean(his)
    ok_tpr = np.mean(los_t) < point_tpr < np.mean(his_t)
    print(f'  simulation: point AUC inside CI range -> {ok_auc}')
    print(f'  simulation: point TPR inside CI range -> {ok_tpr}')
    # weighted ROC monotonicity
    thr, fpr, tpr = weighted_roc(y, s, w1)
    assert np.all(np.diff(fpr) >= -1e-12) and np.all(np.diff(tpr) >= -1e-12)
    print('  simulation: weighted ROC monotone -> True')
    return ok_auc and ok_tpr



# --------------------------------------------------------------------------- identity test

def assert_identity(vt, verbose=True):
    """Identity check, run before any bootstrap.

    With every patient multiplicity set to 1 the weighted estimators MUST reproduce the
    unrounded point estimate, per seed and for the mean over seeds.  Also verifies that
    the estimator is the mean of per-seed metrics and not the metric of seed-averaged
    predictions, and that per-seed (not seed-0) thresholds are used.

    Raises AssertionError on failure so a bad pipeline cannot silently produce numbers.
    """
    names_a, names_b, y, pid_a, pid_b = load_pairs(TEST_PAIRS)
    w1 = np.ones(len(y))
    worst_seed = worst_mean = 0.0
    n_cells = 0
    for name, naive, adapted in CONFIGS:
        for arm, tag in (('imagenet', naive), ('verifier', adapted)):
            S = per_seed_test_scores(tag)
            if S is None:
                continue
            ths = vt.get(tag, {}).get('thresholds', [])
            assert len(ths) == len(S), f'{tag}: {len(ths)} thresholds vs {len(S)} seeds'
            assert len(set(np.round(ths, 12))) > 1 or len(ths) == 1, \
                f'{tag}: all thresholds identical - seed-0 threshold leak?'
            # (a) weighted @ w=1 == direct, per seed
            for k, s_k in enumerate(S):
                f_w, t_w = weighted_operating_point(y, s_k, w1, ths[k])
                t_d = 100.0 * (s_k[y == 1] >= ths[k]).mean()
                f_d = 100.0 * (s_k[y == 0] >= ths[k]).mean()
                worst_seed = max(worst_seed, abs(t_w - t_d), abs(f_w - f_d))
                a_w = weighted_auc(y, s_k, w1) * 100.0
                a_d = roc_auc_score(y, s_k) * 100.0
                worst_seed = max(worst_seed, abs(a_w - a_d))
                n_cells += 1
            # (b) mean over seeds identity
            m_w = float(np.mean([weighted_operating_point(y, S[k], w1, ths[k])[1]
                                 for k in range(len(S))]))
            m_d = float(np.mean([100.0 * (S[k][y == 1] >= ths[k]).mean()
                                 for k in range(len(S))]))
            worst_mean = max(worst_mean, abs(m_w - m_d))
            # (c) estimator is mean-of-per-seed, NOT metric-of-averaged-scores
            if len(S) > 1:
                mean_of_per_seed = np.mean([roc_auc_score(y, s_k) for s_k in S])
                of_averaged = roc_auc_score(y, S.mean(axis=0))
                assert abs(mean_of_per_seed - of_averaged) > 1e-9, \
                    f'{tag}: mean-of-per-seed AUC equals AUC of averaged scores'
    assert worst_seed < 1e-9, f'per-seed identity FAILED: max |diff| = {worst_seed:.3e}'
    assert worst_mean < 1e-9, f'mean-over-seeds identity FAILED: max |diff| = {worst_mean:.3e}'
    if verbose:
        print(f'identity check PASSED over {n_cells} (config,arm,seed) cells: '
              f'per-seed max|diff| = {worst_seed:.2e}, mean-over-seeds max|diff| = {worst_mean:.2e}')
    return worst_seed, worst_mean


def assert_weight_scheme(rng=None):
    """Positive pair weight = m_p; negative cross-patient pair weight = m_p * m_q."""
    pid_a = np.array(['p1', 'p1', 'p2', 'p3'])
    pid_b = np.array(['p1', 'p1', 'p3', 'p2'])   # rows 0,1 positive; rows 2,3 negative
    ca, cb, is_pos, npat = _encode_pids(pid_a, pid_b)
    assert list(is_pos) == [True, True, False, False]
    for trial in range(200):
        r = np.random.default_rng(trial)
        draws = r.integers(0, npat, npat)
        m = np.bincount(draws, minlength=npat)
        w = m[ca].astype(float)
        w[~is_pos] = w[~is_pos] * m[cb][~is_pos]
        assert w[0] == m[ca[0]] and w[1] == m[ca[1]], 'positive weight must be m_p, not m_p^2'
        assert w[2] == m[ca[2]] * m[cb[2]], 'negative weight must be m_p * m_q'
    print('weight-scheme check PASSED: positive = m_p, negative = m_p * m_q')


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser('Phase 1 patient-clustered bootstrap')
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--identity_only', action='store_true',
                    help='run the identity + weight-scheme checks and exit')
    ap.add_argument('--skip_identity', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'artifacts/paper_upgrade/phase1_patient_bootstrap'))
    args = ap.parse_args()

    if args.sim:
        ok = run_simulation(n_boot=2000)
        print('SIMULATION RESULT:', 'PASS' if ok else 'FAIL')
        return 0 if ok else 1

    vt = load_val_thresholds()
    if args.identity_only:
        assert_weight_scheme()
        assert_identity(vt)
        print('IDENTITY RESULT: PASS')
        return 0

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    print(f'loading validation thresholds for {len(vt)} tags')
    if not args.skip_identity:
        assert_weight_scheme()
        assert_identity(vt)

    # ---- run manifest: every canonical number carries this provenance -----------
    run_id = f"phase1-{time.strftime('%Y%m%dT%H%M%S')}-nb{args.n_boot}-s{args.seed}"
    code_sha = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()
    try:
        git_commit = subprocess.check_output(
            ['git', '-C', ROOT, 'rev-parse', 'HEAD'], text=True).strip()
        git_dirty = bool(subprocess.check_output(
            ['git', '-C', ROOT, 'status', '--porcelain'], text=True).strip())
    except Exception:
        git_commit, git_dirty = 'unavailable', None
    manifest = {
        'run_id': run_id,
        'script': os.path.basename(__file__),
        'script_sha256': code_sha,
        'git_commit': git_commit,
        'git_worktree_dirty': git_dirty,
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'n_boot': args.n_boot,
        'rng_seed': args.seed,
        'estimator': 'mean over training seeds of the per-seed metric',
        'weight_scheme': 'positive pair w = m_p; negative cross-patient pair w = m_p * m_q',
        'threshold_policy': ('one validation-selected threshold PER SEED at 1% validation '
                             'FPR, held FIXED during the bootstrap; the validation split '
                             'is NOT resampled'),
        'uncertainty_types': {
            'across_seed_sd': 'sample SD of the per-seed metric on the full test set; NOT a CI',
            'patient_ci': 'patient-clustered bootstrap over test patients; seeds fixed',
            'hier_ci': 'hierarchical: test patients AND training seeds resampled',
        },
        'operating_point_estimands': {
            'A_perseed_avg': 'patient-only; per-seed percentiles, endpoints averaged (LEGACY)',
            'B_mean_seeds': 'patient-only; percentiles of the mean over seeds (PRIMARY)',
            'C_hier': 'patient + seed',
        },
        'interval_method': 'percentile bootstrap (2.5 / 97.5), no BCa',
        'identity_check': 'PASSED (see assert_identity / assert_weight_scheme)',
    }
    with open(os.path.join(args.out, 'run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f'run_id = {run_id}')

    all_results = []
    for cfg in CONFIGS:
        r = analyze_config(cfg, args.n_boot, rng, vt)
        all_results.append(r)
        # persist incrementally
        with open(os.path.join(args.out, 'phase1_results.json'), 'w') as f:
            json.dump(all_results, f, indent=1)

    # paired differences at matched utility (learned mu=0.01 vs elastic, vs blur)
    learned = CONFIGS[3]  # ff_mu0.01
    elastic = CONFIGS[5]  # elastic_m004
    blur = CONFIGS[4]     # blur_s2
    pd = {
        'learned_mu001_vs_random_elastic': paired_utility_diff(learned, elastic, args.n_boot, rng),
        'learned_mu001_vs_gaussian_blur': paired_utility_diff(learned, blur, args.n_boot, rng),
    }
    json.dump(pd, open(os.path.join(args.out, 'paired_differences.json'), 'w'), indent=1)
    print('paired differences:')
    for k, v in pd.items():
        print(' ', k, v)

    write_results_long(all_results, args.out)
    write_ci_summary(all_results, args.out)
    # machine-readable json of everything
    json.dump({'results': all_results, 'paired_differences': pd,
               'n_boot': args.n_boot, 'seed': args.seed},
              open(os.path.join(args.out, 'phase1_results.json'), 'w'), indent=1)
    print('\nPhase 1 complete.')


if __name__ == '__main__':
    main()
