"""Generate every table and figure-data file of the PRL manuscript from runs on disk.

Inputs (never typed by hand):
  results/e1/runs/*_summary.json          clean-split attacker runs (E1)
  results/prl_final/runs/*_summary.json   PRL-DEC-009 queue
  results/e4/e4_frozen_summary.csv        frozen-classifier utility
  results/e4/retrain/*_summary.json       retrained-classifier utility
  artifacts/factorial_results.json        legacy train x validation pairing factorial

Outputs (outputs/generated/): tab_construction.tex, tab_factorial.tex, tab_utility.tex,
frontier.csv, prl_results.json. A cell with no finished run prints \\TODO{...}.
"""
import csv
import glob
import re
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "outputs", "generated")
BEST_RAW_INIT = ("pretrained_ver", "resnet50_bce")


def neg(s):
    """Typeset ASCII minus signs of numbers as $-$ (the hand edit applied to every submitted table)."""
    return re.sub(r"(?<![\w$])-(?=\d)", "$-$", s)


def load_runs():
    runs = defaultdict(list)
    for pattern in ("results/e1/runs/*_summary.json", "results/prl_final/runs/*_summary.json"):
        for f in glob.glob(os.path.join(ROOT, pattern)):
            r = json.load(open(f))
            if r.get("train_path") is None:          # zero-shot rows are not attackers here
                continue
            if "att_train_clean" not in os.path.basename(r["train_path"]):
                continue
            key = (r["init_name"], r["arch"], r["sanitizer"], float(r["lr"]))
            tm = r["test_metrics"]
            op = r["calibration"]["disjoint700_SR"]["operating_points"]["fpr_1pct"]
            runs[key].append({"seed": r["seed"], "auc": tm["auc"], "tpr1": tm["tpr_at_1pct"],
                              "rank1": r["sr_gallery"]["rank1"], "sel_ep": r["selected_epoch"],
                              "cal_tpr1": op["tpr_pct"], "cal_fpr1": op["fpr_pct"],
                              "n_id": r["sr_gallery"]["n_identities"]})
    return runs


def stat(rows, k):
    v = np.array([r[k] for r in rows], float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0), len(v)


def cell(rows, k, digits=2, sd=True, want_n=3):
    if not rows:
        return "\\TODO{run}"
    m, s, n = stat(rows, k)
    txt = f"{m:.{digits}f}" + (f"$\\pm${s:.{digits}f}" if sd and n > 1 else "")
    return txt + ("" if n >= want_n else f"$^{{n={n}}}$")


def pick(runs, init, arch, san, lr=None):
    if lr is not None:
        return runs.get((init, arch, san, lr), [])
    hits = [v for (i, a, s, l), v in runs.items() if (i, a, s) == (init, arch, san)]
    return hits[0] if len(hits) == 1 else []


CONSTRUCTIONS = [  # (label, init, arch, lr); None starts a new block
    ("ImageNet, released recipe", "imagenet", "resnet50_bce", 1e-4),
    ("ImageNet, monotone head only", "imagenet", "resnet50_monohead", 1e-4),
    ("ImageNet, AdamW + head lr only", "imagenet", "resnet50_adamw_headlr", 1e-4),
    ("ImageNet, full recipe", "imagenet", "resnet50_metric_l1", 1e-4),
    None,
    ("Local verifier, released recipe", "vstar_clean", "resnet50_bce", 1e-4),
    ("Local verifier, full recipe", "vstar_clean", "resnet50_metric_l1", 1e-4),
    ("MIMIC verifier, released recipe", "chexgenbench", "resnet50_bce", 1e-4),
    ("MIMIC verifier, full recipe", "chexgenbench", "resnet50_metric_l1", 1e-4),
    ("Reference verifier, released recipe", "pretrained_ver", "resnet50_bce", 1e-4),
    None,
    ("ConvNeXt-T, ImageNet, released", "imagenet", "convnext_tiny_bce", 1e-4),
]


def construction_table(runs, out):
    best = runs.get(BEST_RAW_INIT + ("none", 1e-4), [])
    best_tpr = stat(best, "tpr1")[0] if best else None
    specs = [c for c in CONSTRUCTIONS if c is not None]
    # Rank of each configuration by mean TPR@1% on raw and on sanitized images (1 = highest).
    from scipy.stats import rankdata
    rt = [stat(runs[(i, a, "none", l)], "tpr1")[0] for _, i, a, l in specs]
    st = [stat(runs[(i, a, "ff_mu0.01", l)], "tpr1")[0] for _, i, a, l in specs]
    rank = {c[0]: (int(x), int(y)) for c, x, y in zip(specs, rankdata([-v for v in rt]), rankdata([-v for v in st]))}
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Unsanitized} & \multicolumn{3}{c}{Learned field, $\mu{=}0.01$} & TPR rank \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Verifier configuration & AUC & TPR@1\% & $\rho$ & AUC & TPR@1\% & Rank-1 & raw$\to$san. \\", r"\midrule"]
    rows_json = []
    for spec in CONSTRUCTIONS:
        if spec is None:
            lines.append(r"\midrule")
            continue
        label, init, arch, lr = spec
        raw = runs.get((init, arch, "none", lr), [])
        ff = runs.get((init, arch, "ff_mu0.01", lr), [])
        adeq = (f"{stat(raw, 'tpr1')[0] / best_tpr:.2f}" if raw and best_tpr else "\\TODO{run}")
        rr, rs = rank[label]
        lines.append(f"{label} & {cell(raw, 'auc')} & {cell(raw, 'tpr1')} & {adeq} & "
                     f"{cell(ff, 'auc')} & {cell(ff, 'tpr1')} & {cell(ff, 'rank1')} & {rr}$\\to${rs} \\\\")
        rows_json.append({"label": label, "init": init, "arch": arch, "lr": lr,
                          "raw": raw, "ff_mu0.01": ff, "rank_raw": rr, "rank_ff": rs})
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(out, "w").write("% GENERATED by tools/prl_tables.py -- do not edit\n" + "\n".join(lines) + "\n")
    return rows_json


def factorial_table(out):
    p = os.path.join(ROOT, "artifacts", "factorial_results.json")
    src = os.path.join(GEN, "factorial_tables.tex")
    if not os.path.isfile(src):  # rebuild from the released JSON without rerunning the bootstrap
        import subprocess, sys
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "analyze_factorial.py"), "--tables-only"], check=True)
    # The factorial tables were generated by tools/analyze_factorial.py; reuse its two blocks.
    txt = open(src).read()
    blocks = txt.split("\\begin{tabular}")
    body = ("\\begin{tabular}" + blocks[2].split("% disjoint")[0].strip()
            .replace("Elastic $\\mu{=}0.04$", "Elastic")
            .replace("Learned $\\mu{=}0.01$", "Learned field")
            .replace("Blur $\\sigma{=}2$", "Blur")
            .replace("Train effect & Val effect", "Training & Validation"))
    open(out, "w").write("% GENERATED from outputs/generated/factorial_tables.tex "
                         "(tools/analyze_factorial.py, artifacts/factorial_results.json)\n" + neg(body) + "\n")
    return json.load(open(p))


def utility(out):
    frozen = {r["sanitizer"]: r for r in csv.DictReader(open(os.path.join(ROOT, "results", "e4", "e4_frozen_summary.csv")))}
    retr = defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, "results", "e4", "retrain", "*_summary.json")):
        r = json.load(open(f))
        retr[r["sanitizer"]].append(r["test_raw_macro_auc"])
    names = [("none", "Raw baseline"), ("ff_mu0.01", "Learned field $\\mu{=}0.01$"),
             ("blur_s2", "Blur $\\sigma{=}2$"), ("elastic_m0.04_keyed", "Elastic 0.04")]
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"Release & Frozen classifier & Retrained classifier \\", r"\midrule"]
    # Bold marks the highest AUC among sanitized releases, separately for each column.
    san = [k for k, _ in names if k != "none"]
    best_fz = max(san, key=lambda k: float(frozen[k]["test_macro_auc"]) if k in frozen else -1)
    best_rt = max(san, key=lambda k: np.mean(retr[k]) if retr.get(k) else -1)
    for k, lab in names:
        fz = f"{float(frozen[k]['test_macro_auc']):.2f}" if k in frozen else "\\TODO{run}"
        if k == best_fz:
            fz = f"\\textbf{{{fz}}}"
        v = retr.get(k, [])
        m = f"{np.mean(v):.2f}" if v else ""
        rt = ("\\TODO{run}" if not v else (f"\\textbf{{{m}}}" if k == best_rt else m) +
              (f"$\\pm${np.std(v, ddof=1):.2f}" if len(v) > 1 else "") + ("" if len(v) >= 3 else f"$^{{n={len(v)}}}$"))
        lines.append(f"{lab} & {fz} & {rt} \\\\")
        if k == "none":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(out, "w").write("% GENERATED by tools/prl_tables.py -- do not edit\n" + "\n".join(lines) + "\n")
    return {"frozen": frozen, "retrain": dict(retr)}


def frontier(runs, out):
    frozen = {r["sanitizer"]: float(r["test_macro_auc"]) for r in
              csv.DictReader(open(os.path.join(ROOT, "results", "e4", "e4_frozen_summary.csv")))}
    sans = ["none", "ff_mu0.001", "ff_mu0.005", "ff_mu0.01", "blur_s1.5", "blur_s2", "blur_s2.5",
            "elastic_m0.035_keyed", "elastic_m0.04_keyed", "elastic_m0.045_keyed"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sanitizer", "family", "utility_frozen", "attacker", "n", "auc", "auc_sd", "tpr1", "tpr1_sd", "rank1", "rank1_sd"])
        for s in sans:
            fam = s.split("_")[0]
            for att, init, arch, lr in (("reference", "pretrained_ver", "resnet50_bce", 1e-4),
                                        ("imagenet_full_recipe", "imagenet", "resnet50_metric_l1", 1e-4)):
                rows = runs.get((init, arch, s, lr), [])
                if rows:
                    a, asd, n = stat(rows, "auc"); t, tsd, _ = stat(rows, "tpr1"); r1, r1sd, _ = stat(rows, "rank1")
                    w.writerow([s, fam, frozen.get(s, ""), att, n, f"{a:.2f}", f"{asd:.2f}", f"{t:.2f}", f"{tsd:.2f}", f"{r1:.2f}", f"{r1sd:.2f}"])
                else:
                    w.writerow([s, fam, frozen.get(s, ""), att, 0, "", "", "", "", "", ""])


# ----------------------------------------------------------------------------- prose numbers
# Every number the manuscript prose uses is defined here and read with \N{key} (two decimals),
# \N{key@1} or \N{key@0}. An undefined key prints a red TODO, which `build.sh final` refuses.
ATT = {"inr": ("imagenet", "resnet50_bce", 1e-4), "inm": ("imagenet", "resnet50_monohead", 1e-4),
       "ina": ("imagenet", "resnet50_adamw_headlr", 1e-4), "inf": ("imagenet", "resnet50_metric_l1", 1e-4),
       "vsr": ("vstar_clean", "resnet50_bce", 1e-4), "vsf": ("vstar_clean", "resnet50_metric_l1", 1e-4),
       "vs3": ("vstar_clean", "resnet50_metric_l1", 3e-5),
       "mir": ("chexgenbench", "resnet50_bce", 1e-4), "mif": ("chexgenbench", "resnet50_metric_l1", 1e-4),
       "mi3": ("chexgenbench", "resnet50_metric_l1", 3e-5),
       "ref": ("pretrained_ver", "resnet50_bce", 1e-4), "cnx": ("imagenet", "convnext_tiny_bce", 1e-4)}
SAN = {"raw": "none", "ff001": "ff_mu0.001", "ff005": "ff_mu0.005", "ff": "ff_mu0.01",
       "b15": "blur_s1.5", "b2": "blur_s2", "b25": "blur_s2.5", "e035": "elastic_m0.035_keyed",
       "e04": "elastic_m0.04_keyed", "e045": "elastic_m0.045_keyed", "pnet": "privacy_net",
       "dppix": "dppix_b8_fixed"}


def _interp(xs, ys, x):
    pts = sorted(zip(xs, ys))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def numbers(runs, fact, out):
    N = {}
    best = stat(runs[BEST_RAW_INIT + ("none", 1e-4)], "tpr1")[0]
    for a, (init, arch, lr) in ATT.items():
        for sk, san in SAN.items():
            rows = runs.get((init, arch, san, lr), [])
            if not rows:
                continue
            for m, k in (("auc", "auc"), ("tpr", "tpr1"), ("r1", "rank1"), ("ctpr", "cal_tpr1"), ("cfpr", "cal_fpr1")):
                mu, sd, n = stat(rows, k)
                N[f"{a}-{sk}-{m}"] = mu
                N[f"{a}-{sk}-{m}-sd"] = sd
            N[f"{a}-{sk}-n"] = len(rows)
            N[f"{a}-{sk}-ep1"] = sum(r["sel_ep"] == 1 for r in rows)
            if sk == "raw":
                N[f"{a}-adeq"] = stat(rows, "tpr1")[0] / best
    frozen = {r["sanitizer"]: float(r["test_macro_auc"]) for r in
              csv.DictReader(open(os.path.join(ROOT, "results", "e4", "e4_frozen_summary.csv")))}
    retr = defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, "results", "e4", "retrain", "*_summary.json")):
        r = json.load(open(f)); retr[r["sanitizer"]].append(r["test_raw_macro_auc"])
    for sk, san in SAN.items():
        if san in frozen:
            N[f"u-{sk}"] = frozen[san]
        if san in retr:
            N[f"ur-{sk}"] = float(np.mean(retr[san])); N[f"ur-{sk}-sd"] = float(np.std(retr[san], ddof=1))
    for sk in SAN:
        if f"u-{sk}" in N and "u-raw" in N:
            N[f"du-{sk}"] = round(N["u-raw"], 2) - round(N[f"u-{sk}"], 2)
        if f"ur-{sk}" in N and "ur-raw" in N:
            N[f"dur-{sk}"] = round(N["ur-raw"], 2) - round(N[f"ur-{sk}"], 2)
    # Control leakage interpolated at the learned field's frozen utility (reference attacker).
    for fam, keys in (("blur", ("b15", "b2", "b25")), ("el", ("e035", "e04", "e045"))):
        for m in ("auc", "tpr"):
            v = _interp([N[f"u-{k}"] for k in keys], [N[f"ref-{k}-{m}"] for k in keys], N["u-ff"])
            if v is not None:
                N[f"ref-{fam}-at-ff-{m}"] = v
    # Selected deltas used in the prose. Differences are taken between the two-decimal values the
    # tables print, so a reader subtracting table entries gets the same number as the text.
    d = lambda a, b: round(N[a], 2) - round(N[b], 2)
    N["d-recipe-raw-auc"] = d("inf-raw-auc", "inr-raw-auc")
    N["d-recipe-ff-auc"] = d("inf-ff-auc", "inr-ff-auc")
    N["d-mono-ff-auc"] = d("inm-ff-auc", "inr-ff-auc")
    N["d-vsr-ff-auc"] = d("vsr-ff-auc", "inr-ff-auc")
    N["d-mir-ff-auc"] = d("mir-ff-auc", "inr-ff-auc")
    N["d-span-ff-auc"] = d("ref-ff-auc", "inr-ff-auc")
    N["d-span-ff-tpr"] = d("ref-ff-tpr", "inr-ff-tpr")
    N["ref-ff-tpr-ratio"] = N["ref-ff-tpr"] / N["inr-ff-tpr"]
    # Ranking of the ten Table 1 constructions by raw-image and by sanitized-image TPR@1%.
    from scipy.stats import rankdata, spearmanr
    t1 = ["inr", "inm", "ina", "inf", "vsr", "vsf", "mir", "mif", "ref", "cnx"]
    raw_t = [N[f"{a}-raw-tpr"] for a in t1]; san_t = [N[f"{a}-ff-tpr"] for a in t1]
    for a, rr, rs in zip(t1, rankdata([-x for x in raw_t]), rankdata([-x for x in san_t])):
        N[f"{a}-rank-raw"] = int(rr); N[f"{a}-rank-ff"] = int(rs)
    N["spearman-tpr"] = float(spearmanr(raw_t, san_t).statistic); N["n-t1"] = len(t1)
    N["d-mif-inm-raw-tpr"] = d("mif-raw-tpr", "inm-raw-tpr")          # MIMIC full minus ImageNet monotone head
    N["d-inm-mif-ff-tpr"] = d("inm-ff-tpr", "mif-ff-tpr")
    # Sensitivity of the adequacy threshold (relative index; the top configuration has rho = 1).
    best_all = max(raw_t)
    N["n-adeq-050"] = sum(x / best_all >= 0.5 for x in raw_t)
    N["n-adeq-045"] = sum(x / best_all >= 0.45 for x in raw_t)
    raw9 = [x for a, x in zip(t1, raw_t) if a != "ref"]
    N["n-t1-exref"] = len(raw9); N["n-adeq-050-exref"] = sum(x / max(raw9) >= 0.5 for x in raw9)
    N["denom-exref"] = max(raw9)
    # Directly observed comparisons at nearby frozen utilities (reference attacker).
    for k in ("b2", "e045"):
        N[f"d-obs-{k}-tpr"] = d(f"ref-{k}-tpr", "ref-ff-tpr")             # control minus field (positive)
        N[f"du-obs-{k}"] = d("u-ff", f"u-{k}")                             # field minus control
    # Paired patient-clustered CIs (tools/prl_paired_ci.py), if computed.
    pci = os.path.join(ROOT, "results", "prl_final", "paired_ci.json")
    if os.path.isfile(pci):
        P = json.load(open(pci))
        tag = {"blur_s2": "b2", "elastic_m0.045_keyed": "e045"}
        for ctrl, k in tag.items():
            for m in ("tpr1", "auc"):
                e = P["leakage"][ctrl][m]
                N[f"ci-{k}-{m}"] = e["point"]
                N[f"ci-{k}-{m}-plo"], N[f"ci-{k}-{m}-phi"] = e["patient_ci"]
                N[f"ci-{k}-{m}-hlo"], N[f"ci-{k}-{m}-hhi"] = e["hier_ci"]
                # Same interval expressed as "control minus field" (a positive reduction).
                N[f"ci-{k}-{m}-red-lo"], N[f"ci-{k}-{m}-red-hi"] = -e["hier_ci"][1], -e["hier_ci"][0]
            u = P["utility_frozen"][ctrl]
            N[f"ci-{k}-u"] = u["point"]; N[f"ci-{k}-u-lo"], N[f"ci-{k}-u-hi"] = u["patient_ci"]
            # observed differences from unrounded seed means (the point estimates of Table S5), not from rounded macros
            N[f"d-obs-{k}-tpr"] = -P["leakage"][ctrl]["tpr1"]["point"]
            N[f"du-obs-{k}"] = u["point"]
        N["ci-nboot"] = P["n_boot"]
    N["n-cells"] = len(runs)
    assert all(len(v) == 3 for v in runs.values()), "every clean-protocol cell should have three seeds"
    vf = [r for k in (("vstar_clean", "resnet50_metric_l1", "ff_mu0.01", 1e-4), ("chexgenbench", "resnet50_metric_l1", "ff_mu0.01", 1e-4),
                      ("vstar_clean", "resnet50_metric_l1", "ff_mu0.01", 3e-5), ("chexgenbench", "resnet50_metric_l1", "ff_mu0.01", 3e-5)) for r in runs.get(k, [])]
    N["verfull-ff-n"] = len(vf); N["verfull-ff-ep1"] = sum(r["sel_ep"] == 1 for r in vf)
    n_id = {r["n_id"] for v in runs.values() for r in v}
    assert len(n_id) == 1, n_id
    N["n-id"] = n_id.pop(); N["chance-r1"] = 100.0 / N["n-id"]
    N["ref-ff-r1-x"] = N["ref-ff-r1"] / N["chance-r1"]
    N["e04-over-ff-tpr"] = N["ref-e04-tpr"] / N["ref-ff-tpr"]
    # Train x validation pairing factorial (reference-verifier init, released training list).
    lab = {"ff_mu0.01": "fff", "blur_s2": "fb", "elastic_m004": "fe"}
    for cfg, v in fact["results"].items():
        p = lab[cfg]
        N[f"{p}-nseed"] = v["n_seeds"]
        for met in ("auc", "tpr"):
            for name, key in (("train", "train_effect"), ("val", "val_effect"), ("int", "interaction"),
                              ("disss", "disjoint_minus_orig_SS"), ("dissr", "disjoint_minus_orig_SR"),
                              ("ssss", "sanitized_sanitized/sanitized_sanitized/orig"),
                              ("srsr", "sanitized_raw/sanitized_raw/orig")):
                e = v[met][key]
                N[f"{p}-{met}-{name}"] = e["point"]; N[f"{p}-{met}-{name}-lo"] = e["ci"][0]; N[f"{p}-{met}-{name}-hi"] = e["ci"][1]
    N["fac-nboot"] = fact["n_boot"]
    # Split sizes, read from the manifests rather than typed.
    man = json.load(open(os.path.join(ROOT, "results", "clean_splits", "clean_splits_manifest.json")))
    att = man["splits"]["d_att_train_clean.txt"]
    N["att-pairs"] = att["n_pairs"]; N["att-pts"] = att["n_unique_patients"]
    N["vs-pts"] = man["splits"]["d_pretrain_clean.txt"]["n_unique_patients"]
    N["vs-pairs"] = man["splits"]["d_pretrain_clean.txt"]["n_pairs"]
    pools = man["e4_classifier_pools"]
    N["u-train-img"] = pools["e4_train_tr10"]["n_images"]; N["u-val-img"] = pools["e4_val_v2k"]["n_images"]
    N["u-test-img"] = pools["e4_test_official"]["n_images"]; N["u-test-pts"] = pools["e4_test_official"]["n_patients"]
    sm = {r["file"].split("/")[-1]: r for r in csv.DictReader(open(os.path.join(ROOT, "results", "split_manifest.csv")))}
    for k, f in (("test", "image_pairs_testing_5000.txt"), ("val", "image_pairs_validation_2000.txt"),
                 ("cal", "image_pairs_validation_disjoint_700.txt"), ("rel", "image_pairs_training_10000.txt")):
        N[f"{k}-pairs"] = int(sm[f]["n_pairs"]); N[f"{k}-pts"] = int(sm[f]["unique_patients"])
    N["n-att-runs"] = sum(len(v) for v in runs.values())

    ints = {k for k, v in N.items() if isinstance(v, int) and not isinstance(v, bool)}
    lines = ["% GENERATED by tools/prl_tables.py -- do not edit. Read with \\N{key}, \\N{key@1}, \\N{key@0}."]
    for k in sorted(N):
        v = N[k]
        if k in ints:
            txt = f"{v:,}".replace(",", "{,}")
            vs = {"": txt, "@1": txt, "@0": txt}
        else:
            vs = {"": f"{v:.2f}", "@1": f"{v:.1f}", "@0": f"{v:.0f}"}
        for suf, t in vs.items():
            t = t.replace("-", "$-$") if t.startswith("-") else t
            lines.append(f"\\expandafter\\def\\csname n@{k}{suf}\\endcsname{{{t}}}")
    open(out, "w").write("\n".join(lines) + "\n")
    return N


def main():
    os.makedirs(GEN, exist_ok=True)
    runs = load_runs()
    out = {"construction": construction_table(runs, os.path.join(GEN, "tab_construction.tex")),
           "factorial": factorial_table(os.path.join(GEN, "tab_factorial.tex")),
           "utility": utility(os.path.join(GEN, "tab_utility.tex"))}
    frontier(runs, os.path.join(GEN, "frontier.csv"))
    out["numbers"] = numbers(runs, out["factorial"], os.path.join(GEN, "numbers.tex"))
    json.dump(out, open(os.path.join(GEN, "prl_results.json"), "w"), indent=1, default=str)
    todo = sum(open(os.path.join(GEN, f)).read().count("\\TODO") for f in os.listdir(GEN) if f.endswith(".tex"))
    print(f"wrote {GEN}; cells still waiting for runs: {todo}")


if __name__ == "__main__":
    main()
