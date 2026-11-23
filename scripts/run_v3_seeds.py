"""V3-B (docs/PREREGISTRATION_v3.md): seeds 3 and 4 for the arms of the main NIH contrasts.

Reference verifier and ImageNet full recipe: the arguments recorded for the published seed-0 run of the same
setting (as scripts/run_v2_retrain.py replays them), with the seed changed. Independent verifier (E5): the
arguments of scripts/run_e5.py step 2. Every run also scores the released test pairs under the SS attack.

Outputs: results/V3B/runs/<run_id>_{summary.json,preds.npz,best.pth}
Usage:   python3 -u scripts/run_v3_seeds.py [--only SUBSTRING] [--list]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import prl_lib as P  # noqa: E402
from scripts.run_v2_retrain import CAL_700, TEST_5K, VAL_2K, VAL_SUB700, published_runs  # noqa: E402
from scripts.run_e5 import ATT, CKPT  # noqa: E402

OUT = ROOT / "results" / "V3B" / "runs"
SEEDS = (3, 4)
ARMS = {"reference": ["none", "ff_mu0.01", "blur_s2", "elastic_m0.045_keyed", "elastic_m0.04_keyed"],
        "e5": ["none", "ff_mu0.01", "blur_s2", "elastic_m0.045_keyed", "elastic_m0.04_keyed"],
        "imagenet_full": ["none", "ff_mu0.01", "blur_s2", "elastic_m0.04_keyed"]}


def plan():
    pub = published_runs()
    runs = []
    for sd in SEEDS:  # all seed-3 runs first, so a partial queue still adds one seed to every arm
        for ver, sans in ARMS.items():
            for san in sans:
                run_id = f"v3b__{ver}__{san}__s{sd}"
                if ver == "e5":
                    kw = dict(spec=P.AttackerSpec("resnet50_bce", f"custom:{CKPT}", 256, True), train_path=ATT,
                              lr=1e-4, batch_size=32, max_epochs=100, patience=5)
                else:
                    r = pub[(ver, san, 0)][1]
                    assert r["train_pairing"] == r["val_pairing"] == r["test_pairing"] == "SR"
                    kw = dict(spec=P.AttackerSpec(r["arch"], r["init_name"], int(r["input_size"]),
                                                  bool(r["use_imagenet_norm"])),
                              train_path=ROOT / r["train_path"], lr=float(r["lr"]), batch_size=int(r["batch_size"]),
                              max_epochs=int(r["max_epochs"]), patience=int(r["patience"]))
                runs.append((run_id, ver, san, sd, kw))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    runs = [x for x in plan() if a.only in x[0]]
    print(f"{len(runs)} runs")
    if a.list:
        for x in runs:
            print(" ", x[0], x[4]["spec"], x[4]["lr"], x[4]["batch_size"], x[4]["max_epochs"], x[4]["patience"])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    store = P.RAMImageStore(use_cache=True)
    meta_lut = P.load_nih_metadata_lut()
    for i, (run_id, ver, san, sd, kw) in enumerate(runs, 1):
        if (OUT / f"{run_id}_summary.json").is_file():
            print(f"[{i}/{len(runs)}] CACHED {run_id}", flush=True)
            continue
        print(f"[{i}/{len(runs)}] RUN {run_id}", flush=True)
        tp = kw.pop("train_path")
        rec = P.train_and_evaluate_pair_run(
            run_id=run_id, out_dir=OUT, store=store, meta_lut=meta_lut, sanitizer_name=san,
            train_path=tp, val_path=VAL_2K, test_path=TEST_5K,
            cal_paths={"disjoint700_SR": (CAL_700, "SR"), "val2k_sub700_SR": (VAL_SUB700, "SR")},
            train_pairing="SR", val_pairing="SR", test_pairing="SR", seed=sd, save_ckpt=True,
            disjoint_check_splits={"att_train": tp, "val_2k": VAL_2K, "cal_700": CAL_700, "test_5k": TEST_5K},
            extra_evals={"test5k_SS": (TEST_5K, "SS")}, **kw)
        tm = rec["test_metrics"]
        print(f"  -> ep={rec['selected_epoch']}/{rec['epochs_ran']} ({rec['runtime_sec']}s) AUC={tm['auc']:.2f}% "
              f"TPR@1%={tm['tpr_at_1pct']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
