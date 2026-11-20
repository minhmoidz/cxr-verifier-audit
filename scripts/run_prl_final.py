"""Final PRL experiment queue (PRL-DEC-009, locked 2026-11-19 before any run below).

Same protocol as E1 `init_clean10k`: train on results/clean_splits/d_att_train_clean.txt
(9,796 pairs, 2,811 patients), SR training / SR validation (image_pairs_validation_2000) /
SR test (image_pairs_testing_5000), batch 32, max 100 epochs, patience 5, minimum validation
loss checkpoint, calibration on disjoint_700 and val2k_sub700. Outputs go to
results/prl_final/runs/ and finished runs are skipped on restart.

Groups, in execution order:
  recipe    ImageNet init, which part of the metric_l1 recipe matters:
            resnet50_monohead / resnet50_adamw_headlr x {none, ff_mu0.01} x 3 seeds   (12)
  frontier_ver     strongest attacker (reference-verifier init, released recipe) on the
            utility sweep: ff_mu0.001/0.005, blur s1.5/2/2.5, elastic 0.035/0.04/0.045 (keyed)
            x 3 seeds                                                                (24)
  frontier_metric  no identity pretraining (ImageNet init, metric_l1 recipe):
            ff_mu0.005, blur_s2, elastic_m0.04_keyed x 3 seeds                         (9)
  destructive_ver  reference-verifier init on privacy_net, dppix_b8_fixed x 3 seeds   (6)
  arch_convnext    ConvNeXt-T BCE, ImageNet init, {none, ff_mu0.01} x 3 seeds         (6)
  lr_match         addendum: full recipe at lr 1e-4 for V*_clean and the MIMIC verifier
            (E1 ran them at 3e-5), {none, ff_mu0.01} x 3 seeds                          (12)

Usage: python3 -u scripts/run_prl_final.py [--groups recipe,frontier_ver,...] [--seeds 0,1,2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import prl_lib as P  # noqa: E402

CLEAN = ROOT / "results" / "clean_splits"
OUT = ROOT / "results" / "prl_final" / "runs"
ATT = CLEAN / "d_att_train_clean.txt"
VAL_2K = ROOT / "image_pairs" / "image_pairs_validation_2000.txt"
CAL_700 = ROOT / "image_pairs" / "image_pairs_validation_disjoint_700.txt"
VAL_SUB700 = CLEAN / "d_val_2000_sub700.txt"
TEST_5K = ROOT / "image_pairs" / "image_pairs_testing_5000.txt"

FRONTIER = ["ff_mu0.001", "ff_mu0.005", "blur_s1.5", "blur_s2", "blur_s2.5",
            "elastic_m0.035_keyed", "elastic_m0.04_keyed", "elastic_m0.045_keyed"]
ORDER = ["recipe", "frontier_ver", "frontier_metric", "destructive_ver", "arch_convnext", "lr_match"]


def build_specs(groups, seeds):
    specs = []

    def add(group, arch, init, san):
        cgb = init == "chexgenbench"          # MIMIC checkpoint: 224 px, [0,1] input, as in E1
        for sd in seeds:
            specs.append(dict(group=group, arch=arch, init=init, san=san, seed=sd,
                              input_size=224 if cgb else 256, use_norm=not cgb,
                              run_id=f"{group}__{arch}__{init}__{san}__att_train_clean__pat5__lr0.0001__s{sd}"))

    if "recipe" in groups:
        for arch in ("resnet50_monohead", "resnet50_adamw_headlr"):
            for san in ("none", "ff_mu0.01"):
                add("recipe", arch, "imagenet", san)
    if "frontier_ver" in groups:
        for san in FRONTIER:
            add("frontier_ver", "resnet50_bce", "pretrained_ver", san)
    if "frontier_metric" in groups:
        for san in ("ff_mu0.005", "blur_s2", "elastic_m0.04_keyed"):
            add("frontier_metric", "resnet50_metric_l1", "imagenet", san)
    if "destructive_ver" in groups:
        for san in ("privacy_net", "dppix_b8_fixed"):
            add("destructive_ver", "resnet50_bce", "pretrained_ver", san)
    if "arch_convnext" in groups:
        for san in ("none", "ff_mu0.01"):
            add("arch_convnext", "convnext_tiny_bce", "imagenet", san)
    if "lr_match" in groups:   # PRL-DEC-009 addendum: full recipe at the ImageNet lr (1e-4)
        for init in ("vstar_clean", "chexgenbench"):
            for san in ("none", "ff_mu0.01"):
                add("lr_match", "resnet50_metric_l1", init, san)
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=",".join(ORDER))
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    groups = [g for g in ORDER if g in args.groups.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    specs = build_specs(groups, seeds)
    OUT.mkdir(parents=True, exist_ok=True)
    store = P.RAMImageStore(use_cache=True)
    meta_lut = P.load_nih_metadata_lut()

    for i, s in enumerate(specs, 1):
        if (OUT / f"{s['run_id']}_summary.json").is_file():
            print(f"[{i}/{len(specs)}] CACHED {s['run_id']}", flush=True)
            continue
        print(f"[{i}/{len(specs)}] RUN {s['run_id']}", flush=True)
        rec = P.train_and_evaluate_pair_run(
            run_id=s["run_id"], out_dir=OUT, store=store, meta_lut=meta_lut,
            spec=P.AttackerSpec(s["arch"], s["init"], s["input_size"], s["use_norm"]),
            sanitizer_name=s["san"], train_path=ATT, val_path=VAL_2K, test_path=TEST_5K,
            cal_paths={"disjoint700_SR": (CAL_700, "SR"), "val2k_sub700_SR": (VAL_SUB700, "SR")},
            train_pairing="SR", val_pairing="SR", test_pairing="SR",
            seed=s["seed"], lr=1e-4, batch_size=32, max_epochs=100, patience=5,
            save_ckpt=False,
            disjoint_check_splits={"att_train": ATT, "val_2k": VAL_2K, "cal_700": CAL_700,
                                   "test_5k": TEST_5K},
        )
        tm = rec["test_metrics"]
        print(f"  -> ep={rec['selected_epoch']}/{rec['epochs_ran']} ({rec['runtime_sec']}s) "
              f"AUC={tm['auc']:.2f}% TPR@1%={tm['tpr_at_1pct']:.2f}% "
              f"Rank1={rec['sr_gallery']['rank1']:.2f}%", flush=True)
        rec["group"] = s["group"]
        (OUT / f"{s['run_id']}_group.json").write_text(json.dumps({"group": s["group"]}))


if __name__ == "__main__":
    main()
