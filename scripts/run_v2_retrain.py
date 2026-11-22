"""Retrain the Fig. 2 verifier runs whose checkpoints were not saved (revision round v2).

The 111 published verifier runs kept only 8 checkpoints. E1 (extended negative set) and E2
(attacker that also sanitizes the reference) need the trained networks, so this script replays
each published run with exactly the arguments recorded in its *_summary.json (initialization,
recipe, sanitizer, seed, learning rate, pair lists, pairings, batch size, epochs, patience) and
saves the best-validation checkpoint. Training is seeded but not bit-deterministic, so these are
replicates of the published runs; analysis scripts compare their 5,000-pair test metrics with
the published ones.

Besides the published evaluations, every run also scores the released test pairs with the
reference sanitized too ("test5k_SS", E2).

Runs: reference verifier (pretrained_ver, released recipe) at the ten Fig. 2 settings, and the
ImageNet full-recipe verifier (resnet50_metric_l1, lr 1e-4) at its five settings, three seeds
each, minus the runs whose published checkpoint exists.

Outputs: results/R_retrain/runs/<run_id>_{summary.json,preds.npz,best.pth}
Usage:   python3 -u scripts/run_v2_retrain.py [--only SUBSTRING] [--list]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import prl_lib as P  # noqa: E402

OUT = ROOT / "results" / "R_retrain" / "runs"
CLEAN = ROOT / "results" / "clean_splits"
VAL_2K = ROOT / "image_pairs" / "image_pairs_validation_2000.txt"
CAL_700 = ROOT / "image_pairs" / "image_pairs_validation_disjoint_700.txt"
VAL_SUB700 = CLEAN / "d_val_2000_sub700.txt"
TEST_5K = ROOT / "image_pairs" / "image_pairs_testing_5000.txt"

REF_SETTINGS = ["none", "ff_mu0.001", "ff_mu0.005", "ff_mu0.01", "blur_s1.5", "blur_s2", "blur_s2.5",
                "elastic_m0.035_keyed", "elastic_m0.04_keyed", "elastic_m0.045_keyed"]
IMNET_SETTINGS = ["none", "ff_mu0.005", "ff_mu0.01", "blur_s2", "elastic_m0.04_keyed"]


def published_runs():
    """Published summaries of the Fig. 2 runs, keyed by (verifier, sanitizer, seed)."""
    found = {}
    for pat in ("results/e1/runs/*_summary.json", "results/prl_final/runs/*_summary.json"):
        for f in sorted(glob.glob(str(ROOT / pat))):
            r = json.load(open(f))
            if r.get("train_path") is None or "att_train_clean" not in os.path.basename(r["train_path"]):
                continue
            if (r["init_name"], r["arch"]) == ("pretrained_ver", "resnet50_bce") and r["sanitizer"] in REF_SETTINGS:
                ver = "reference"
            elif (r["init_name"], r["arch"], float(r["lr"])) == ("imagenet", "resnet50_metric_l1", 1e-4) \
                    and r["sanitizer"] in IMNET_SETTINGS:
                ver = "imagenet_full"
            else:
                continue
            key = (ver, r["sanitizer"], int(r["seed"]))
            assert key not in found, f"duplicate published run for {key}"
            found[key] = (Path(f), r)
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring filter on run_id")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    pub = published_runs()
    assert len(pub) == 3 * (len(REF_SETTINGS) + len(IMNET_SETTINGS)), f"expected 45 published runs, found {len(pub)}"
    plan = []
    for key, (path, r) in sorted(pub.items()):
        ckpt = path.with_name(path.name.replace("_summary.json", "_best.pth"))
        if ckpt.is_file():
            print(f"published checkpoint kept: {key} -> {ckpt.relative_to(ROOT)}")
            continue
        run_id = "retrain__" + r["run_id"]
        if args.only and args.only not in run_id:
            continue
        plan.append((key, r, run_id))
    # Settings used in the main-text comparisons first, so the key results arrive early.
    key_first = {("reference", "ff_mu0.01"), ("reference", "blur_s2"), ("reference", "elastic_m0.045_keyed"),
                 ("reference", "elastic_m0.04_keyed"), ("reference", "none"), ("imagenet_full", "ff_mu0.01"),
                 ("imagenet_full", "blur_s2"), ("imagenet_full", "elastic_m0.04_keyed"), ("imagenet_full", "none")}
    plan.sort(key=lambda t: (t[0][:2] not in key_first, t[0][0], t[0][1], t[0][2]))
    print(f"{len(plan)} runs to train")
    if args.list:
        for key, _, run_id in plan:
            print(" ", run_id)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    store = P.RAMImageStore(use_cache=True)
    meta_lut = P.load_nih_metadata_lut()
    for i, (key, r, run_id) in enumerate(plan, 1):
        if (OUT / f"{run_id}_summary.json").is_file():
            print(f"[{i}/{len(plan)}] CACHED {run_id}", flush=True)
            continue
        print(f"[{i}/{len(plan)}] RUN {run_id}", flush=True)
        assert r["train_pairing"] == r["val_pairing"] == r["test_pairing"] == "SR"
        assert set(r["calibration"]) == {"disjoint700_SR", "val2k_sub700_SR"}
        assert r["fixed_epochs"] is None and int(r["mc_k"]) == 1
        rec = P.train_and_evaluate_pair_run(
            run_id=run_id, out_dir=OUT, store=store, meta_lut=meta_lut,
            spec=P.AttackerSpec(r["arch"], r["init_name"], int(r["input_size"]), bool(r["use_imagenet_norm"])),
            sanitizer_name=r["sanitizer"],
            train_path=ROOT / r["train_path"], val_path=ROOT / r["val_path"], test_path=ROOT / r["test_path"],
            cal_paths={"disjoint700_SR": (CAL_700, "SR"), "val2k_sub700_SR": (VAL_SUB700, "SR")},
            train_pairing="SR", val_pairing="SR", test_pairing="SR",
            seed=int(r["seed"]), lr=float(r["lr"]), batch_size=int(r["batch_size"]),
            max_epochs=int(r["max_epochs"]), patience=int(r["patience"]),
            save_ckpt=True,
            disjoint_check_splits={"att_train": ROOT / r["train_path"], "val_2k": VAL_2K,
                                   "cal_700": CAL_700, "test_5k": TEST_5K},
            extra_evals={"test5k_SS": (TEST_5K, "SS")},
        )
        rec_pub = {"published_run_id": r["run_id"], "published_test_auc": r["test_metrics"]["auc"],
                   "published_test_tpr1": r["test_metrics"]["tpr_at_1pct"],
                   "published_selected_epoch": r["selected_epoch"]}
        (OUT / f"{run_id}_published.json").write_text(json.dumps(rec_pub, indent=1))
        tm = rec["test_metrics"]
        print(f"  -> ep={rec['selected_epoch']}/{rec['epochs_ran']} ({rec['runtime_sec']}s) "
              f"AUC={tm['auc']:.2f}% (published {rec_pub['published_test_auc']:.2f}%) "
              f"TPR@1%={tm['tpr_at_1pct']:.2f}% (published {rec_pub['published_test_tpr1']:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
