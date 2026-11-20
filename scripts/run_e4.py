"""Run Confirmatory Experiment E4: Two-Regime Clinical Utility (U_frozen & U_retrain) + Iso-Utility.

Use Case 1 (U_frozen): Fixed NIH classifier (networks/pretrained_classifier.pth) evaluated
across sanitizers on both validation (3,484 imgs) and official test (25,596 imgs).

Use Case 2 (U_retrain): Clean DenseNet-121 trained from ImageNet weights on sanitized
e4_train_tr10 (18,089 imgs), validated on sanitized e4_val_v2k (3,484 imgs) for 15 epochs
(AdamW + CosineAnnealingLR, selecting best val macro-AUC), and tested on BOTH real raw
official test radiographs AND sanitized official test radiographs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import prl_lib as P  # noqa: E402
from tools.compute_a1_utility import (  # noqa: E402
    bootstrap_weighted_auc,
    draw_valid_patient_multiplicities,
    prepare_sorted,
)
from utils import utils  # noqa: E402

OUT = ROOT / "results" / "e4"

FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

FROZEN_SANITIZERS = [
    "none",
    "ff_mu0.001",
    "ff_mu0.005",
    "ff_mu0.01",
    "blur_s1.5",
    "blur_s2",
    "blur_s2.5",
    "elastic_m0.035_keyed",
    "elastic_m0.04_keyed",
    "elastic_m0.045_keyed",
    "privacy_net",
    "dppix_b8_fixed",
]

RETRAIN_SANITIZERS = [
    "none",
    "ff_mu0.005",
    "ff_mu0.01",
    "blur_s1.5",
    "blur_s2",
    "blur_s2.5",
    "elastic_m0.04_keyed",
    "privacy_net",
    "dppix_b8_fixed",
]


def load_e4_splits() -> Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]:
    """Return {split_name: (filenames, y_bool (N, 14), patient_ids (N,))} with strict patient disjointness."""
    have = {p.name for p in P.IMAGE_DIR.iterdir()}
    trval_set = {x.strip() for x in (ROOT / "image_pairs" / "train_val_list.txt").open() if x.strip() in have}
    test_set = {x.strip() for x in (ROOT / "image_pairs" / "test_list.txt").open() if x.strip() in have}
    v2k_pairs = P.load_pair_file(ROOT / "image_pairs" / "image_pairs_validation_2000.txt")
    v2k_pts = {P.patient_id_from_name(fn) for fn in v2k_pairs[0] + v2k_pairs[1]}

    df = pd.read_csv(ROOT / "chexnet" / "nih_labels.csv")
    df = df[df["Image Index"].isin(have)].copy()

    trval_df = df[df["Image Index"].isin(trval_set)].copy()
    trval_df["pt"] = trval_df["Image Index"].str.split("_").str[0]
    val_df = trval_df[trval_df["pt"].isin(v2k_pts)]
    train_df = trval_df[~trval_df["pt"].isin(v2k_pts)]
    test_df = df[df["Image Index"].isin(test_set)].copy()
    test_df["pt"] = test_df["Image Index"].str.split("_").str[0]

    p_tr = set(train_df["pt"])
    p_va = set(val_df["pt"])
    p_te = set(test_df["pt"])
    assert len(p_tr & p_va) == 0 and len(p_tr & p_te) == 0 and len(p_va & p_te) == 0

    def _pack(sub: pd.DataFrame) -> Tuple[List[str], np.ndarray, np.ndarray]:
        fns = sub["Image Index"].tolist()
        y = sub[FINDINGS].to_numpy(dtype=np.float32)
        pts = sub["pt"].to_numpy()
        return fns, y, pts

    return {"train": _pack(train_df), "val": _pack(val_df), "test": _pack(test_df)}


@torch.no_grad()
def eval_classifier(
    model: nn.Module,
    store: P.RAMImageStore,
    filenames: Sequence[str],
    sanitizer: P.BaseSanitizer,
    device: torch.device,
    is_logits: bool,
    san_cache: Dict[str, torch.Tensor] | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    mean_t = torch.tensor(P.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(P.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    preds = []
    for i in range(0, len(filenames), batch_size):
        fns = filenames[i : i + batch_size]
        if san_cache is not None and all(fn in san_cache for fn in fns):
            t = torch.stack([san_cache[fn] for fn in fns], dim=0).to(device, non_blocking=True)
        else:
            raw = store.get_batch_tensor(fns, device=device)
            t = sanitizer.sanitize_batch(raw, fns).to(device, non_blocking=True)
        rgb = P.preprocess_for_attacker(t, 224, True, mean_t, std_t)
        out = model(rgb)
        if is_logits:
            out = torch.sigmoid(out)
        preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_14_aucs(y: np.ndarray, probs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    per = {}
    for k, name in enumerate(FINDINGS):
        per[name] = float(roc_auc_score(y[:, k], probs[:, k])) * 100.0
    return float(np.mean(list(per.values()))), per


def bootstrap_utility_ci(
    y: np.ndarray, probs: np.ndarray, pts: np.ndarray, n_boot: int = 1000, seed: int = 20261113
) -> Dict[str, object]:
    uniq = np.unique(pts)
    lut = {p: i for i, p in enumerate(uniq)}
    p_code = np.asarray([lut[p] for p in pts], dtype=np.int32)
    y_t = y.T.astype(np.float64)        # (14, N)
    s_t = probs[None, ...].transpose(0, 2, 1).astype(np.float64)  # (1, 14, N)
    rng = np.random.default_rng(seed)
    mult, rej = draw_valid_patient_multiplicities(y_t, p_code, len(uniq), n_boot, rng)
    boot_f = bootstrap_weighted_auc(*prepare_sorted(y_t, s_t, p_code), mult) * 100.0
    macro = boot_f[:, 0, :].mean(axis=1)
    return {
        "macro_ci95": [round(float(np.percentile(macro, 2.5)), 2), round(float(np.percentile(macro, 97.5)), 2)],
        "rejected": int(rej),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--sanitizers", default="", help="Optional comma-separated subset for U_retrain")
    ap.add_argument("--only-frozen", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for E4")

    splits = load_e4_splits()
    tr_fns, tr_y, tr_pts = splits["train"]
    va_fns, va_y, va_pts = splits["val"]
    te_fns, te_y, te_pts = splits["test"]
    t5k_l, t5k_r, _ = P.load_pair_file(ROOT / "image_pairs" / "image_pairs_testing_5000.txt")
    t5k_img_set = set(t5k_l) | set(t5k_r)
    t5k_pt_set = {P.patient_id_from_name(fn) for fn in t5k_img_set}
    mask_t5k_imgs = np.array([fn in t5k_img_set for fn in te_fns], dtype=bool)
    mask_t5k_pts = np.array([pt in t5k_pt_set for pt in te_pts], dtype=bool)
    print(
        f"E4 pools: train={len(tr_fns)} ({len(np.unique(tr_pts))} pts), "
        f"val={len(va_fns)} ({len(np.unique(va_pts))} pts), "
        f"test={len(te_fns)} ({len(np.unique(te_pts))} pts; "
        f"t5k_imgs={int(mask_t5k_imgs.sum())}, t5k_pts_imgs={int(mask_t5k_pts.sum())})",
        flush=True,
    )

    store = P.RAMImageStore(use_cache=True)
    print("Preloading all 47,169 256x256 images into RAMImageStore...", flush=True)
    t_pre = time.time()
    store.preload(tr_fns + va_fns + te_fns, max_workers=8)
    print(f"Preload complete in {time.time() - t_pre:.1f}s", flush=True)

    # ===== Part 1: U_frozen =====
    frozen_dir = OUT / "frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_json = frozen_dir / "e4_frozen_summary.json"
    if args.overwrite or not frozen_json.is_file():
        print("\n=== [E4 Part 1] U_frozen Evaluation across 12 Sanitizer Configs ===", flush=True)
        clf_obj = torch.load(ROOT / "networks" / "pretrained_classifier.pth", map_location="cpu", weights_only=False)
        frozen_net = clf_obj["model"].to(device).eval()
        frozen_rows = []
        for san_name in FROZEN_SANITIZERS:
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats(device)
            san = P.build_sanitizer(san_name, device=device, seed=0)
            va_probs = eval_classifier(frozen_net, store, va_fns, san, device, is_logits=False, batch_size=args.batch_size)
            te_probs = eval_classifier(frozen_net, store, te_fns, san, device, is_logits=False, batch_size=args.batch_size)
            va_macro, _ = compute_14_aucs(va_y, va_probs)
            te_macro, te_per = compute_14_aucs(te_y, te_probs)
            t5k_img_macro, _ = compute_14_aucs(te_y[mask_t5k_imgs], te_probs[mask_t5k_imgs])
            t5k_pt_macro, _ = compute_14_aucs(te_y[mask_t5k_pts], te_probs[mask_t5k_pts])
            ci = bootstrap_utility_ci(te_y, te_probs, te_pts, n_boot=1000, seed=20261113)
            ci_t5k = bootstrap_utility_ci(
                te_y[mask_t5k_imgs], te_probs[mask_t5k_imgs], te_pts[mask_t5k_imgs], n_boot=1000, seed=20261113
            )
            peak_vram_mib = round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)
            np.savez_compressed(frozen_dir / f"u_frozen_{san_name}_test_preds.npz", te_probs=te_probs.astype(np.float32))
            row = {
                "sanitizer": san_name,
                "val_macro_auc": round(va_macro, 2),
                "test_macro_auc": round(te_macro, 2),
                "test_ci95_low": ci["macro_ci95"][0],
                "test_ci95_high": ci["macro_ci95"][1],
                "test_t5k_imgs_8062_auc": round(t5k_img_macro, 2),
                "test_t5k_imgs_ci95": ci_t5k["macro_ci95"],
                "test_t5k_pts_23142_auc": round(t5k_pt_macro, 2),
                "per_finding": {k: round(v, 2) for k, v in te_per.items()},
                "peak_vram_mib": peak_vram_mib,
                "runtime_sec": round(time.time() - t0, 1),
            }
            frozen_rows.append(row)
            print(
                f"  U_frozen {san_name:22s}: val={va_macro:.2f}% | "
                f"test25k={te_macro:.2f}% [{ci['macro_ci95'][0]:.2f}, {ci['macro_ci95'][1]:.2f}] | "
                f"t5k_8k={t5k_img_macro:.2f}% [{ci_t5k['macro_ci95'][0]:.2f}, {ci_t5k['macro_ci95'][1]:.2f}] "
                f"({row['runtime_sec']}s, {peak_vram_mib} MiB)",
                flush=True,
            )
            del san
            torch.cuda.empty_cache()

        with frozen_json.open("w") as f:
            json.dump(frozen_rows, f, indent=2)
        with (OUT / "e4_frozen_summary.csv").open("w", newline="") as f:
            cols = [
                "sanitizer", "val_macro_auc", "test_macro_auc", "test_ci95_low", "test_ci95_high",
                "test_t5k_imgs_8062_auc", "test_t5k_pts_23142_auc", "peak_vram_mib", "runtime_sec",
            ]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows([{k: r[k] for k in cols} for r in frozen_rows])
        del frozen_net
        torch.cuda.empty_cache()
    else:
        print(f"[SKIP] U_frozen already computed at {frozen_json}", flush=True)

    if args.only_frozen:
        return

    # ===== Part 2: U_retrain =====
    retrain_dir = OUT / "retrain"
    retrain_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    sel_sans = [s.strip() for s in args.sanitizers.split(",") if s.strip()] if args.sanitizers else list(RETRAIN_SANITIZERS)
    mean_t = torch.tensor(P.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(P.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    raw_none = P.IdentitySanitizer()

    for san_name in sel_sans:
        print(f"\n=== [E4 Part 2] U_retrain Sanitizer: {san_name} ===", flush=True)
        san = P.build_sanitizer(san_name, device=device, seed=0)
        # Pre-sanitize train/val/test images once for deterministic sanitizers -> fast 15 epochs!
        san_cache = P.precompute_sanitized_map(store, san, tr_fns + va_fns + te_fns, device=device, batch_size=args.batch_size)
        for sd in seeds:
            run_id = f"u_retrain__{san_name}__s{sd}"
            sum_p = retrain_dir / f"{run_id}_summary.json"
            if not args.overwrite and sum_p.is_file():
                with sum_p.open("r") as f:
                    rec = json.load(f)
                print(
                    f"  CACHED {run_id}: san_sel(raw={rec['test_raw_macro_auc']:.2f}% san={rec['test_san_macro_auc']:.2f}%)",
                    flush=True,
                )
                continue

            t0 = time.time()
            torch.cuda.reset_peak_memory_stats(device)
            utils.seed_all(sd)
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            model.classifier = nn.Linear(model.classifier.in_features, len(FINDINGS))
            model = model.to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            crit = nn.BCEWithLogitsLoss()

            rng = np.random.default_rng(sd)
            n_tr = len(tr_fns)
            bs = args.batch_size
            best_va_san, best_ep_san = -1.0, 0
            best_va_raw, best_ep_raw = -1.0, 0
            best_state_san = copy.deepcopy(model.state_dict())
            best_state_raw = copy.deepcopy(model.state_dict())
            val_san_curve, val_raw_curve = [], []

            for ep in range(1, args.epochs + 1):
                model.train()
                perm = rng.permutation(n_tr)
                for start in range(0, n_tr, bs):
                    idx = perm[start : start + bs]
                    bf = [tr_fns[k] for k in idx]
                    yb = torch.from_numpy(tr_y[idx]).to(device=device, dtype=torch.float32)
                    if san_cache is not None:
                        t = torch.stack([san_cache[fn] for fn in bf], dim=0).to(device, non_blocking=True)
                    else:
                        t = san.sanitize_batch(store.get_batch_tensor(bf, device=device), bf).to(device)
                    # Random horizontal flip on half the batch
                    flip_mask = torch.rand(t.shape[0], device=device) < 0.5
                    if flip_mask.any():
                        t = t.clone()
                        t[flip_mask] = torch.flip(t[flip_mask], dims=[3])
                    rgb = P.preprocess_for_attacker(t, 224, True, mean_t, std_t)
                    opt.zero_grad()
                    loss = crit(model(rgb), yb)
                    loss.backward()
                    opt.step()
                sched.step()

                va_san_probs = eval_classifier(model, store, va_fns, san, device, is_logits=True, san_cache=san_cache, batch_size=args.batch_size)
                va_raw_probs = eval_classifier(model, store, va_fns, raw_none, device, is_logits=True, san_cache=None, batch_size=args.batch_size)
                va_san_auc, _ = compute_14_aucs(va_y, va_san_probs)
                va_raw_auc, _ = compute_14_aucs(va_y, va_raw_probs)
                val_san_curve.append(round(va_san_auc, 4))
                val_raw_curve.append(round(va_raw_auc, 4))
                if va_san_auc > best_va_san:
                    best_va_san = va_san_auc
                    best_ep_san = ep
                    best_state_san = copy.deepcopy(model.state_dict())
                if va_raw_auc > best_va_raw:
                    best_va_raw = va_raw_auc
                    best_ep_raw = ep
                    best_state_raw = copy.deepcopy(model.state_dict())

            # Evaluate val_san-selected checkpoint
            model.load_state_dict(best_state_san)
            san_sel_raw_probs = eval_classifier(model, store, te_fns, raw_none, device, is_logits=True, san_cache=None, batch_size=args.batch_size)
            san_sel_san_probs = eval_classifier(model, store, te_fns, san, device, is_logits=True, san_cache=san_cache, batch_size=args.batch_size)

            # Evaluate val_raw-selected checkpoint
            model.load_state_dict(best_state_raw)
            raw_sel_raw_probs = eval_classifier(model, store, te_fns, raw_none, device, is_logits=True, san_cache=None, batch_size=args.batch_size)
            raw_sel_san_probs = eval_classifier(model, store, te_fns, san, device, is_logits=True, san_cache=san_cache, batch_size=args.batch_size)

            raw_macro, raw_per = compute_14_aucs(te_y, san_sel_raw_probs)
            san_macro, san_per = compute_14_aucs(te_y, san_sel_san_probs)
            raw_t5k_macro, _ = compute_14_aucs(te_y[mask_t5k_imgs], san_sel_raw_probs[mask_t5k_imgs])
            san_t5k_macro, _ = compute_14_aucs(te_y[mask_t5k_imgs], san_sel_san_probs[mask_t5k_imgs])

            rawsel_raw_macro, rawsel_raw_per = compute_14_aucs(te_y, raw_sel_raw_probs)
            rawsel_san_macro, _ = compute_14_aucs(te_y, raw_sel_san_probs)
            rawsel_raw_t5k, _ = compute_14_aucs(te_y[mask_t5k_imgs], raw_sel_raw_probs[mask_t5k_imgs])

            raw_ci = bootstrap_utility_ci(te_y, san_sel_raw_probs, te_pts, n_boot=1000, seed=sd)
            san_ci = bootstrap_utility_ci(te_y, san_sel_san_probs, te_pts, n_boot=1000, seed=sd)
            rawsel_raw_ci = bootstrap_utility_ci(te_y, raw_sel_raw_probs, te_pts, n_boot=1000, seed=sd)
            peak_vram_mib = round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)

            np.savez_compressed(
                retrain_dir / f"{run_id}_test_preds.npz",
                raw_probs=san_sel_raw_probs.astype(np.float32),
                san_probs=san_sel_san_probs.astype(np.float32),
                raw_sel_raw_probs=raw_sel_raw_probs.astype(np.float32),
                raw_sel_san_probs=raw_sel_san_probs.astype(np.float32),
            )
            rec = {
                "run_id": run_id,
                "sanitizer": san_name,
                "seed": sd,
                "epochs": args.epochs,
                "best_epoch": best_ep_san,
                "best_epoch_val_san": best_ep_san,
                "best_epoch_val_raw": best_ep_raw,
                "best_val_macro_auc": round(best_va_san, 2),
                "best_val_raw_macro_auc": round(best_va_raw, 2),
                "val_auc_curve": val_san_curve,
                "val_raw_auc_curve": val_raw_curve,
                "test_raw_macro_auc": round(raw_macro, 2),
                "test_raw_ci95": raw_ci["macro_ci95"],
                "test_san_macro_auc": round(san_macro, 2),
                "test_san_ci95": san_ci["macro_ci95"],
                "test_raw_t5k_8062_auc": round(raw_t5k_macro, 2),
                "test_san_t5k_8062_auc": round(san_t5k_macro, 2),
                "val_raw_sel_test_raw_macro_auc": round(rawsel_raw_macro, 2),
                "val_raw_sel_test_raw_ci95": rawsel_raw_ci["macro_ci95"],
                "val_raw_sel_test_san_macro_auc": round(rawsel_san_macro, 2),
                "val_raw_sel_test_raw_t5k_8062_auc": round(rawsel_raw_t5k, 2),
                "raw_per_finding": {k: round(v, 2) for k, v in raw_per.items()},
                "san_per_finding": {k: round(v, 2) for k, v in san_per.items()},
                "val_raw_sel_raw_per_finding": {k: round(v, 2) for k, v in rawsel_raw_per.items()},
                "peak_vram_mib": peak_vram_mib,
                "runtime_sec": round(time.time() - t0, 1),
            }
            with sum_p.open("w") as f:
                json.dump(rec, f, indent=2)
            print(
                f"  {run_id}: ep(san={best_ep_san},raw={best_ep_raw}) | "
                f"san_sel->raw={raw_macro:.2f}%(t5k={raw_t5k_macro:.2f}%) san={san_macro:.2f}% | "
                f"raw_sel->raw={rawsel_raw_macro:.2f}%(t5k={rawsel_raw_t5k:.2f}%) "
                f"({rec['runtime_sec']}s, {peak_vram_mib} MiB)",
                flush=True,
            )
            del model, opt
            torch.cuda.empty_cache()

        del san, san_cache
        torch.cuda.empty_cache()

    # Summaries across ALL completed retrain runs in results/e4/retrain/
    retrain_rows = [json.loads(p.read_text()) for p in sorted(retrain_dir.glob("u_retrain__*__s*_summary.json"))]
    if retrain_rows:
        with (OUT / "e4_retrain_runs.csv").open("w", newline="") as f:
            cols = [
                "run_id", "sanitizer", "seed", "best_epoch_val_san", "best_epoch_val_raw",
                "best_val_macro_auc", "best_val_raw_macro_auc",
                "test_raw_macro_auc", "test_san_macro_auc", "test_raw_t5k_8062_auc",
                "val_raw_sel_test_raw_macro_auc", "val_raw_sel_test_raw_t5k_8062_auc",
                "peak_vram_mib", "runtime_sec",
            ]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows([{k: r.get(k) for k in cols} for r in retrain_rows])

        grouped: Dict[str, List[dict]] = {}
        for r in retrain_rows:
            grouped.setdefault(r["sanitizer"], []).append(r)
        sum_rows = []
        for san_n, gr in grouped.items():
            m = lambda k: round(float(np.mean([x[k] for x in gr])), 2)
            s = lambda k: round(float(np.std([x[k] for x in gr], ddof=1)) if len(gr) > 1 else 0.0, 2)
            sum_rows.append({
                "sanitizer": san_n,
                "n_seeds": len(gr),
                "val_san_mean": m("best_val_macro_auc"),
                "val_raw_mean": m("best_val_raw_macro_auc"),
                "san_sel_test_raw_mean": m("test_raw_macro_auc"),
                "san_sel_test_raw_sd": s("test_raw_macro_auc"),
                "san_sel_test_san_mean": m("test_san_macro_auc"),
                "san_sel_test_raw_t5k_mean": m("test_raw_t5k_8062_auc"),
                "raw_sel_test_raw_mean": m("val_raw_sel_test_raw_macro_auc"),
                "raw_sel_test_raw_t5k_mean": m("val_raw_sel_test_raw_t5k_8062_auc"),
            })
        with (OUT / "e4_retrain_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
            w.writeheader()
            w.writerows(sum_rows)
    print(f"Wrote E4 summaries to {OUT}", flush=True)


if __name__ == "__main__":
    main()
