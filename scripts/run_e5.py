"""E5: independent strong verifier (reference-verifier data scale, no PriCheXy-Net weights).

Step 1 (pretraining): ImageNet initialization, full recipe (resnet50_metric_l1: monotone head,
AdamW, larger head learning rate), raw-raw pairs from results/E5/splits/e5_pretrain_pairs.txt
(26,266 patients of the official train/val list), checkpoint selection on the released validation
list (RR), test on the released test list (RR, zero-shot raw linkage). lr 1e-4, batch 32, at most
100 epochs, patience 5, seed 0.
Step 2 (sanitized evaluation): exactly the protocol of the reference-verifier rows of Table 2 and
Fig. 2 -- released recipe (resnet50_bce), SR training on the clean verifier-training list, SR
validation on the released validation list, SR test, calibration on the disjoint list, three seeds
-- but initialized from the E5 checkpoint instead of the PriCheXy-Net reference verifier, at the
ten Fig. 2 settings. Every run also scores the SS test pairs (E2) and saves its checkpoint.

Outputs: results/E5/pretrain/, results/E5/runs/<run_id>_{summary.json,preds.npz,best.pth}
Usage:   python3 -u scripts/run_e5.py [--step 1|2|all] [--only SUBSTRING]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prl_lib as P  # noqa: E402

E5 = ROOT / "results" / "E5"
PAIRS = E5 / "splits" / "e5_pretrain_pairs.txt"
CKPT = E5 / "e5_verifier_best.pth"
CLEAN = ROOT / "results" / "clean_splits"
ATT = CLEAN / "d_att_train_clean.txt"
VAL_2K = ROOT / "image_pairs" / "image_pairs_validation_2000.txt"
CAL_700 = ROOT / "image_pairs" / "image_pairs_validation_disjoint_700.txt"
VAL_SUB700 = CLEAN / "d_val_2000_sub700.txt"
TEST_5K = ROOT / "image_pairs" / "image_pairs_testing_5000.txt"
SETTINGS = ["none", "ff_mu0.01", "blur_s2", "elastic_m0.045_keyed", "elastic_m0.04_keyed",
            "ff_mu0.001", "ff_mu0.005", "blur_s1.5", "blur_s2.5", "elastic_m0.035_keyed"]


def step1(store, meta_lut):
    if CKPT.is_file():
        print(f"[SKIP] E5 checkpoint exists: {CKPT.relative_to(ROOT)}", flush=True)
        return
    rec = P.train_and_evaluate_pair_run(
        run_id="e5_pretrain", out_dir=E5 / "pretrain", store=store, meta_lut=meta_lut,
        spec=P.AttackerSpec("resnet50_metric_l1", "imagenet", input_size=256, use_imagenet_norm=True),
        sanitizer_name="none", train_path=PAIRS, val_path=VAL_2K, test_path=TEST_5K,
        cal_paths={"disjoint700_RR": (CAL_700, "RR")},
        train_pairing="RR", val_pairing="RR", test_pairing="RR",
        seed=0, lr=1e-4, batch_size=32, max_epochs=100, patience=5, save_ckpt=True,
        disjoint_check_splits={"pretrain_e5": PAIRS, "val_2k": VAL_2K, "cal_700": CAL_700, "test_5k": TEST_5K},
    )
    CKPT.write_bytes(Path(rec["ckpt_saved_path"]).read_bytes())
    print(f"E5 pretraining done in {rec['runtime_sec']}s: selected epoch {rec['selected_epoch']}/{rec['epochs_ran']}, "
          f"raw test AUC {rec['test_metrics']['auc']:.2f}%, TPR@1% {rec['test_metrics']['tpr_at_1pct']:.2f}%", flush=True)


def step2(store, meta_lut, only):
    assert CKPT.is_file(), "run step 1 first"
    out = E5 / "runs"
    out.mkdir(parents=True, exist_ok=True)
    plan = [(san, sd) for san in SETTINGS for sd in (0, 1, 2)]
    for i, (san, sd) in enumerate(plan, 1):
        run_id = f"e5__resnet50_bce__e5_init__{san}__att_train_clean__pat5__lr0.0001__s{sd}"
        if only and only not in run_id:
            continue
        if (out / f"{run_id}_summary.json").is_file():
            print(f"[{i}/{len(plan)}] CACHED {run_id}", flush=True)
            continue
        print(f"[{i}/{len(plan)}] RUN {run_id}", flush=True)
        rec = P.train_and_evaluate_pair_run(
            run_id=run_id, out_dir=out, store=store, meta_lut=meta_lut,
            spec=P.AttackerSpec("resnet50_bce", f"custom:{CKPT}", 256, True),
            sanitizer_name=san, train_path=ATT, val_path=VAL_2K, test_path=TEST_5K,
            cal_paths={"disjoint700_SR": (CAL_700, "SR"), "val2k_sub700_SR": (VAL_SUB700, "SR")},
            train_pairing="SR", val_pairing="SR", test_pairing="SR",
            seed=sd, lr=1e-4, batch_size=32, max_epochs=100, patience=5, save_ckpt=True,
            disjoint_check_splits={"att_train": ATT, "val_2k": VAL_2K, "cal_700": CAL_700, "test_5k": TEST_5K},
            extra_evals={"test5k_SS": (TEST_5K, "SS")},
        )
        tm = rec["test_metrics"]
        print(f"  -> ep={rec['selected_epoch']}/{rec['epochs_ran']} ({rec['runtime_sec']}s) AUC={tm['auc']:.2f}% "
              f"TPR@1%={tm['tpr_at_1pct']:.2f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    store = P.RAMImageStore(use_cache=True)
    meta_lut = P.load_nih_metadata_lut()
    if a.step in ("1", "all"):
        step1(store, meta_lut)
    if a.step in ("2", "all"):
        step2(store, meta_lut, a.only)


if __name__ == "__main__":
    main()
