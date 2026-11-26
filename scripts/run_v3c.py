"""V3-C (docs/PREREGISTRATION_v3.md): PriCheXy-Net retrained with an SR or an SS adversary, and its evaluation.

Steps (each skips finished work):
  gen SR|SS   train the generator with config_files/v3c_<v>_s42.json (released trainer, adversary_pairing option;
              batch 16 per amendment A4; resumes an interrupted run, amendment A5) and copy the lowest-validation-total-loss generator, as released, to
              results/V3C/gen/ffv3c_<v>_s42/generator_lowest_total_loss.pth (with its SHA-256 and loss curves)
  verifiers   for each retrained generator: reference and independent (E5) initialization, released recipe,
              SR pairs, seeds 0-2 (results/V3C/runs/v3c__<verifier>__<sanitizer>__s<seed>)
  utility     frozen-classifier macro AUC on the official test images for the released mu = 0.01 generator and
              both retrained generators (results/V3C/utility_frozen.json)
  explore     exploratory, not in the plan (decided after the SS-adversary run selected its epoch-1 checkpoint):
              the same verifiers and utility for the final-epoch generators (sanitizers ffv3c_<v>_s42_last)

Usage: python3 -u scripts/run_v3c.py gen SR | gen SS | verifiers | utility | explore
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "V3C"
SANS = ["ffv3c_SR_s42", "ffv3c_SS_s42"]
SANS_LAST = [s + "_last" for s in SANS]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gen(v):
    name = f"ffv3c_{v}_s42"
    dst = OUT / "gen" / name
    if (dst / "generator_lowest_total_loss.pth").is_file():
        print(f"{name}: done"); return
    exp = ROOT / "archive" / f"v3c_{v}_s42"
    cmd = [sys.executable, "-u", "train_architecture.py", "--config", f"v3c_{v}_s42.json"]
    if (exp / "resume_state.pth").is_file():  # interrupted (out of GPU memory, or paused): continue (amendment A5)
        print(f"resuming {exp}", flush=True)
        cmd.append("--resume")
    elif exp.exists():                        # interrupted before the first epoch ended: restart
        print(f"removing interrupted attempt {exp}", flush=True)
        shutil.rmtree(exp)
    subprocess.run(cmd, cwd=ROOT, check=True)
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("generator_lowest_total_loss.pth", "generator_latest.pth"):
        shutil.copy2(exp / f, dst / f)
    for f in exp.glob("*.json"):
        shutil.copy2(f, dst / f.name)
    for f in exp.glob("*.png"):
        shutil.copy2(f, dst / f.name)
    meta = {"name": name, "config": json.load(open(ROOT / "config_files" / f"v3c_{v}_s42.json")),
            "generator_sha256": sha(dst / "generator_lowest_total_loss.pth"),
            "wall_hours_incl_pauses": round((time.time() - (exp / f"v3c_{v}_s42.zip").stat().st_mtime) / 3600, 2)}
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{name}: trained, {meta['wall_hours_incl_pauses']} h wall time")


def verifiers(sans=SANS):
    from scripts import prl_lib as P
    from scripts.run_v2_retrain import CAL_700, TEST_5K, VAL_2K, VAL_SUB700, published_runs
    from scripts.run_e5 import ATT, CKPT
    r0 = published_runs()[("reference", "ff_mu0.01", 0)][1]
    specs = {"reference": dict(spec=P.AttackerSpec(r0["arch"], r0["init_name"], int(r0["input_size"]),
                                                    bool(r0["use_imagenet_norm"])), train_path=ROOT / r0["train_path"],
                               lr=float(r0["lr"]), batch_size=int(r0["batch_size"]), max_epochs=int(r0["max_epochs"]),
                               patience=int(r0["patience"])),
             "e5": dict(spec=P.AttackerSpec("resnet50_bce", f"custom:{CKPT}", 256, True), train_path=ATT, lr=1e-4,
                        batch_size=32, max_epochs=100, patience=5)}
    out = OUT / "runs"
    out.mkdir(parents=True, exist_ok=True)
    store = P.RAMImageStore(use_cache=True)
    meta_lut = P.load_nih_metadata_lut()
    plan = [(ver, san, sd) for sd in (0, 1, 2) for san in sans for ver in ("reference", "e5")]
    for i, (ver, san, sd) in enumerate(plan, 1):
        run_id = f"v3c__{ver}__{san}__s{sd}"
        if (out / f"{run_id}_summary.json").is_file():
            print(f"[{i}/{len(plan)}] CACHED {run_id}", flush=True); continue
        if not (OUT / "gen" / san.removesuffix("_last") / "generator_lowest_total_loss.pth").is_file():
            print(f"[{i}/{len(plan)}] SKIP {run_id}: generator not trained yet", flush=True); continue
        print(f"[{i}/{len(plan)}] RUN {run_id}", flush=True)
        kw = dict(specs[ver])
        tp = kw.pop("train_path")
        rec = P.train_and_evaluate_pair_run(
            run_id=run_id, out_dir=out, store=store, meta_lut=meta_lut, sanitizer_name=san,
            train_path=tp, val_path=VAL_2K, test_path=TEST_5K,
            cal_paths={"disjoint700_SR": (CAL_700, "SR"), "val2k_sub700_SR": (VAL_SUB700, "SR")},
            train_pairing="SR", val_pairing="SR", test_pairing="SR", seed=sd, save_ckpt=True,
            disjoint_check_splits={"att_train": tp, "val_2k": VAL_2K, "cal_700": CAL_700, "test_5k": TEST_5K},
            extra_evals={"test5k_SS": (TEST_5K, "SS")}, **kw)
        tm = rec["test_metrics"]
        print(f"  -> ep={rec['selected_epoch']}/{rec['epochs_ran']} AUC={tm['auc']:.2f}% TPR@1%={tm['tpr_at_1pct']:.2f}%", flush=True)


def utility(sans=SANS):
    import numpy as np
    import torch
    from scripts import prl_lib as P
    from scripts.run_e4 import bootstrap_utility_ci, compute_14_aucs, eval_classifier, load_e4_splits
    dst = OUT / "utility_frozen.json"
    rows = json.load(open(dst)) if dst.is_file() else []
    done = {r["sanitizer"] for r in rows}
    todo = [s for s in ["ff_mu0.01"] + sans if s not in done and (s == "ff_mu0.01" or (OUT / "gen" / s.removesuffix("_last")).is_dir())]
    if not todo:
        print("utility: nothing to do"); return
    device = torch.device("cuda")
    te_fns, te_y, te_pts = load_e4_splits()["test"]
    store = P.RAMImageStore(use_cache=True)
    store.preload(te_fns)
    net = torch.load(ROOT / "networks" / "pretrained_classifier.pth", map_location="cpu", weights_only=False)["model"].to(device).eval()
    for s in todo:
        san = P.build_sanitizer(s, device=device, seed=0)
        probs = eval_classifier(net, store, te_fns, san, device, is_logits=False, batch_size=32)
        macro, per = compute_14_aucs(te_y, probs)
        ci = bootstrap_utility_ci(te_y, probs, te_pts, n_boot=1000, seed=20261113)
        np.savez_compressed(OUT / f"u_frozen_{s}_test_preds.npz", te_probs=probs.astype(np.float32))
        rows.append({"sanitizer": s, "test_macro_auc": round(macro, 2), "test_ci95": ci["macro_ci95"],
                     "per_finding": {k: round(v, 2) for k, v in per.items()}})
        print(f"utility {s}: {macro:.2f} [{ci['macro_ci95'][0]:.2f}, {ci['macro_ci95'][1]:.2f}]", flush=True)
        json.dump(rows, open(dst, "w"), indent=1)


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "gen":
        gen(sys.argv[2])
    elif step == "verifiers":
        verifiers()
    elif step == "utility":
        utility()
    elif step == "explore":
        verifiers(SANS_LAST)
        utility(SANS_LAST)
    else:
        raise SystemExit(__doc__)
