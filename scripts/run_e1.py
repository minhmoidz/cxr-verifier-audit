"""Run Confirmatory Experiment E1: Clean Pretraining, Data/Compute Disentanglement & Cross-Dataset Init.

Writes outputs cleanly under results/e1/ and logs summary tables to results/e1/e1_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import prl_lib as P  # noqa: E402

CLEAN = ROOT / "results" / "clean_splits"
OUT = ROOT / "results" / "e1"

VAL_2K = ROOT / "image_pairs" / "image_pairs_validation_2000.txt"
CAL_700 = ROOT / "image_pairs" / "image_pairs_validation_disjoint_700.txt"
VAL_SUB700 = CLEAN / "d_val_2000_sub700.txt"
TEST_5K = ROOT / "image_pairs" / "image_pairs_testing_5000.txt"


def default_cals() -> Dict[str, tuple[Path, str]]:
    return {
        "disjoint700_SR": (CAL_700, "SR"),
        "val2k_sub700_SR": (VAL_SUB700, "SR"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-vstar-mixed", action="store_true", help="Skip legacy 40k V* runs if vstar_verifier.pth missing.")
    ap.add_argument("--pruned", action="store_true", help="Run PRL-DEC-008 pruned suite + metric_l1 head-calibration diagnostic.")
    args = ap.parse_args()
    if args.pruned:
        args.skip_vstar_mixed = True

    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    store = P.RAMImageStore(use_cache=True)
    meta_lut = P.load_nih_metadata_lut()

    # Step 1: Train V*_clean strictly on d_pretrain_clean (20,502 clean train_val pairs, 0 overlap with D_att/val/cal/test)
    vstar_clean_run = OUT / "vstar_clean"
    vstar_clean_ckpt = OUT / "vstar_clean_best.pth"
    if args.overwrite or not vstar_clean_ckpt.is_file():
        print("=== [E1 Step 1] Pretraining V*_clean on d_pretrain_clean (20,502 pairs, RR) ===", flush=True)
        rec = P.train_and_evaluate_pair_run(
            run_id="vstar_clean",
            out_dir=vstar_clean_run,
            store=store,
            meta_lut=meta_lut,
            spec=P.AttackerSpec("resnet50_bce", "imagenet", input_size=256, use_imagenet_norm=True),
            sanitizer_name="none",
            train_path=CLEAN / "d_pretrain_clean.txt",
            val_path=VAL_2K,
            test_path=TEST_5K,
            cal_paths={"disjoint700_RR": (CAL_700, "RR")},
            train_pairing="RR",
            val_pairing="RR",
            test_pairing="RR",
            seed=0,
            lr=1e-4,
            batch_size=32,
            max_epochs=100,
            patience=5,
            save_ckpt=True,
            disjoint_check_splits={
                "pretrain": CLEAN / "d_pretrain_clean.txt",
                "att_train": CLEAN / "d_att_train_clean.txt",
                "val_2k": VAL_2K,
                "cal_700": CAL_700,
                "test_5k": TEST_5K,
            },
        )
        # Copy/save canonical vstar_clean_best.pth at results/e1/vstar_clean_best.pth
        src = Path(rec["ckpt_saved_path"])
        vstar_clean_ckpt.write_bytes(src.read_bytes())
        print(f"V*_clean done in {rec['runtime_sec']}s: sel_ep={rec['selected_epoch']}, RR test AUC={rec['test_metrics']['auc']:.2f}%", flush=True)
    else:
        print(f"[SKIP] V*_clean checkpoint already exists at {vstar_clean_ckpt}", flush=True)

    runs_spec = []

    def _add(group: str, init_name: str, san: str, train_path: Path | None, seed: int, lr: float = 1e-4, patience: int | None = 5, fixed_epochs: int | None = None, input_size: int = 256, use_norm: bool = True, arch: str = "resnet50_bce"):
        budget_tag = f"ep{fixed_epochs}" if fixed_epochs is not None else f"pat{patience}"
        tr_tag = "zeroshot" if train_path is None else train_path.stem.replace("image_pairs_training_", "tr").replace("d_", "")
        run_id = f"{group}__{init_name}__{san}__{tr_tag}__{budget_tag}__lr{lr:g}__s{seed}"
        runs_spec.append(dict(
            group=group, run_id=run_id, arch=arch, init_name=init_name, san=san, train_path=train_path,
            seed=seed, lr=lr, patience=patience, fixed_epochs=fixed_epochs,
            input_size=input_size, use_norm=use_norm,
        ))

    # Step 2: Zero-shot frozen baselines across inits
    for san in ("none", "ff_mu0.01"):
        _add("zeroshot", "vstar_clean", san, None, 0)
        _add("zeroshot", "chexgenbench", san, None, 0, input_size=224, use_norm=False)
        if not args.skip_vstar_mixed and (ROOT / "networks" / "vstar_verifier.pth").is_file():
            _add("zeroshot", "vstar", san, None, 0)

    # Step 3: Clean Init Disentanglement on D_att_clean (9,796 pairs, 2,811 pts) SR/SR
    att_clean = CLEAN / "d_att_train_clean.txt"
    for init_name in ("imagenet", "vstar_clean", "pretrained_ver", "chexgenbench", "vstar"):
        if init_name == "vstar" and (args.skip_vstar_mixed or not (ROOT / "networks" / "vstar_verifier.pth").is_file()):
            continue
        is_cgb = init_name == "chexgenbench"
        for san in ("none", "ff_mu0.01"):
            for sd in seeds:
                _add("init_clean10k", init_name, san, att_clean, sd,
                     input_size=224 if is_cgb else 256, use_norm=not is_cgb)

    if args.pruned:
        # Targeted PRL-DEC-008 diagnostic: resolve Epoch-1 stop & head bias starvation via resnet50_metric_l1
        for init_name, diag_lr in (("vstar_clean", 3e-5), ("chexgenbench", 3e-5), ("imagenet", 1e-4)):
            is_cgb = init_name == "chexgenbench"
            for san in ("none", "ff_mu0.01"):
                for sd in seeds:
                    _add("metric_l1_diag", init_name, san, att_clean, sd, lr=diag_lr,
                         input_size=224 if is_cgb else 256, use_norm=not is_cgb, arch="resnet50_metric_l1")
    else:
        # Step 4: Data Scale / Horizon / LR Disentanglement for ImageNet-init
        # 4a. Legacy 10k (9,053 pts, same 10k pairs count) & All clean train_val (30,298 unique pairs, 8,471 pts)
        for tr_p in (ROOT / "image_pairs" / "image_pairs_training_10000.txt", CLEAN / "d_all_trainval_clean.txt"):
            for san in ("none", "ff_mu0.01"):
                for sd in seeds:
                    _add("scale_budget", "imagenet", san, tr_p, sd)

        # 4b. Fixed horizon 30 epochs (no early stop) vs patience=5 on att_clean
        for san in ("none", "ff_mu0.01"):
            for sd in seeds:
                _add("horizon_ep30", "imagenet", san, att_clean, sd, patience=None, fixed_epochs=30)

        # 4c. LR sweep on att_clean (ff_mu0.01 + none for lr=1e-5, 3e-4, 1e-3)
        for lr in (1e-5, 3e-4, 1e-3):
            for san in ("none", "ff_mu0.01"):
                for sd in seeds:
                    _add("lr_sweep", "imagenet", san, att_clean, sd, lr=lr)

    records: List[dict] = []
    total = len(runs_spec)
    for idx_r, cfg in enumerate(runs_spec, 1):
        rdir = OUT / "runs"
        sum_p = rdir / f"{cfg['run_id']}_summary.json"
        if not args.overwrite and sum_p.is_file():
            with sum_p.open("r") as f:
                rec = json.load(f)
            rec["group"] = cfg["group"]
            records.append(rec)
            print(f"[{idx_r}/{total}] CACHED {cfg['run_id']}: AUC={rec['test_metrics']['auc']:.2f}%", flush=True)
            continue

        print(f"[{idx_r}/{total}] RUN {cfg['run_id']} ...", flush=True)
        disjoint = None
        if cfg["train_path"] is not None:
            disjoint = {
                "att_train": cfg["train_path"],
                "val_2k": VAL_2K,
                "cal_700": CAL_700,
                "test_5k": TEST_5K,
            }
            if cfg["init_name"] == "vstar_clean":
                disjoint["pretrain_vstar_clean"] = CLEAN / "d_pretrain_clean.txt"

        rec = P.train_and_evaluate_pair_run(
            run_id=cfg["run_id"],
            out_dir=rdir,
            store=store,
            meta_lut=meta_lut,
            spec=P.AttackerSpec(cfg.get("arch", "resnet50_bce"), cfg["init_name"], cfg["input_size"], cfg["use_norm"]),
            sanitizer_name=cfg["san"],
            train_path=cfg["train_path"],
            val_path=VAL_2K,
            test_path=TEST_5K,
            cal_paths=default_cals(),
            train_pairing="SR",
            val_pairing="SR",
            test_pairing="SR",
            seed=cfg["seed"],
            lr=cfg["lr"],
            batch_size=32,
            max_epochs=100,
            patience=cfg["patience"],
            fixed_epochs=cfg["fixed_epochs"],
            save_ckpt=(cfg["seed"] == 0 and cfg["group"] in ("init_clean10k",)),
            disjoint_check_splits=disjoint,
        )
        rec["group"] = cfg["group"]
        records.append(rec)
        tm = rec["test_metrics"]
        dcal = rec["calibration"]["disjoint700_SR"]["operating_points"]["fpr_1pct"]
        print(
            f"  -> ep={rec['selected_epoch']}/{rec['epochs_ran']} ({rec['runtime_sec']}s) "
            f"AUC={tm['auc']:.2f}% TPR@1%={tm['tpr_at_1pct']:.2f}% "
            f"Cal700(FPR={dcal['fpr_pct']:.2f}%,TPR={dcal['tpr_pct']:.2f}%) "
            f"Rank1={rec['sr_gallery']['rank1']:.2f}%",
            flush=True,
        )

    # Write flat CSV & grouped summary
    rows = []
    for r in records:
        dcal = r["calibration"]["disjoint700_SR"]["operating_points"]["fpr_1pct"]
        vsub = r["calibration"]["val2k_sub700_SR"]["operating_points"]["fpr_1pct"]
        rows.append({
            "group": r.get("group", ""),
            "run_id": r["run_id"],
            "init": r["init_name"],
            "sanitizer": r["sanitizer"],
            "train_file": Path(r["train_path"]).name if r["train_path"] else "NONE",
            "seed": r["seed"],
            "lr": r["lr"],
            "selected_epoch": r["selected_epoch"],
            "epochs_ran": r["epochs_ran"],
            "auc": r["test_metrics"]["auc"],
            "tpr_at_1pct": r["test_metrics"]["tpr_at_1pct"],
            "tpr_at_0_1pct": r["test_metrics"]["tpr_at_0_1pct"],
            "pr_auc": r["test_metrics"]["pr_auc"],
            "cal700_fpr": round(dcal["fpr_pct"], 3),
            "cal700_tpr": round(dcal["tpr_pct"], 3),
            "valsub700_fpr": round(vsub["fpr_pct"], 3),
            "valsub700_tpr": round(vsub["tpr_pct"], 3),
            "gallery_rank1": r["sr_gallery"]["rank1"],
            "gallery_rank5": r["sr_gallery"]["rank5"],
            "fu1_4_auc": round(r["subgroups"].get("fu1_4", {}).get("auc", float("nan")), 2),
            "fu_ge5_auc": round(r["subgroups"].get("fu_ge5", {}).get("auc", float("nan")), 2),
            "same_sex_auc": round(r["subgroups"].get("same_sex", {}).get("auc", float("nan")), 2),
            "runtime_sec": r["runtime_sec"],
        })
    csv_path = OUT / "e1_runs.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Grouped mean +- sd summary
    grouped: Dict[tuple, List[dict]] = {}
    for row in rows:
        key = (row["group"], row["init"], row["sanitizer"], row["train_file"], row["lr"], row["selected_epoch"] if row["train_file"] == "NONE" else None)
        # Group across seeds only
        key = (row["group"], row["init"], row["sanitizer"], row["train_file"], row["lr"])
        grouped.setdefault(key, []).append(row)

    sum_rows = []
    for (grp, init_n, san, tr_f, lr), grp_rows in grouped.items():
        m = lambda k: round(float(np.mean([x[k] for x in grp_rows])), 2)
        s = lambda k: round(float(np.std([x[k] for x in grp_rows], ddof=1)) if len(grp_rows) > 1 else 0.0, 2)
        sum_rows.append({
            "group": grp, "init": init_n, "sanitizer": san, "train_file": tr_f, "lr": lr,
            "n_seeds": len(grp_rows),
            "mean_sel_ep": round(float(np.mean([x["selected_epoch"] for x in grp_rows])), 1),
            "auc_mean": m("auc"), "auc_sd": s("auc"),
            "tpr1_mean": m("tpr_at_1pct"), "tpr1_sd": s("tpr_at_1pct"),
            "cal700_fpr_mean": m("cal700_fpr"), "cal700_tpr_mean": m("cal700_tpr"),
            "rank1_mean": m("gallery_rank1"), "rank5_mean": m("gallery_rank5"),
            "same_sex_auc_mean": m("same_sex_auc"),
        })
    with (OUT / "e1_grouped_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
        w.writeheader()
        w.writerows(sum_rows)
    print(f"Wrote {csv_path} ({len(rows)} runs) and {OUT / 'e1_grouped_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
