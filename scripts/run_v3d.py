"""V3-D (docs/PREREGISTRATION_v3.md): a second learned sanitizer family, Privacy-Net, at a utility-matched point.

Steps (each skips finished work):
  target      frozen-classifier macro AUC on the validation images of the retrained-utility split for the learned
              field (mu = 0.01, the target T) and the blur and elastic controls (results/V3D/val_utility.json)
  gen         train Privacy-Net with the released trainer (generator_type privacy_net, SR adversary, batch 16,
              seed 42, 100 epochs, every epoch saved). Verification-loss weight w = 0.1; if the validation utility
              after epoch 1 is below T the attempt is discarded and w is divided by 10 (0.1, 0.01, 0.001).
              Attempts: results/V3D/w_attempts.json. Resumes an interrupted run (as amendment A5).
  select      validation utility of every epoch generator; among those with utility >= T, the lowest validation
              verification term -log(1 - p); copy to results/V3D/gen/pnv3d/generator_selected.pth; matched blur
              and elastic strengths by validation utility (results/V3D/selection.json)
  verifiers   pnv3d: reference and independent (E5) initialization, SR pairs, seeds 0-2, and reference
              initialization trained and selected on SS pairs, seeds 0-2; released Privacy-Net (privacy_net):
              reference and independent, SR pairs, seeds 0-2 (results/V3D/runs/v3d__<family>__<sanitizer>__s<seed>)
  utility     frozen-classifier macro AUC on the official test images for pnv3d and privacy_net
              (results/V3D/utility_frozen.json)

Usage: python3 -u scripts/run_v3d.py target | gen | select | verifiers | utility
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

OUT = ROOT / "results" / "V3D"
WEIGHTS = (0.1, 0.01, 0.001)
N_EPOCHS = 100
CONTROLS = {"blur": ["blur_s1.5", "blur_s2", "blur_s2.5"],
            "elastic": ["elastic_m0.035_keyed", "elastic_m0.04_keyed", "elastic_m0.045_keyed"]}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jload(p, default):
    return json.load(open(p)) if Path(p).is_file() else default


class _Val:
    """Frozen released classifier on the validation images of the retrained-utility split."""

    def __init__(self):
        import torch
        from scripts import prl_lib as P
        from scripts.run_e4 import load_e4_splits
        self.P, self.torch = P, torch
        self.device = torch.device("cuda")
        self.fns, self.y, _ = load_e4_splits()["val"]
        self.store = P.RAMImageStore(use_cache=True)
        self.store.preload(self.fns)
        self.net = torch.load(ROOT / "networks" / "pretrained_classifier.pth", map_location="cpu",
                              weights_only=False)["model"].to(self.device).eval()

    def auc(self, san):
        from scripts.run_e4 import compute_14_aucs, eval_classifier
        probs = eval_classifier(self.net, self.store, self.fns, san, self.device, is_logits=False, batch_size=32)
        return round(float(compute_14_aucs(self.y, probs)[0]), 4)

    def auc_name(self, name):
        return self.auc(self.P.build_sanitizer(name, device=self.device, seed=0))

    def auc_ckpt(self, ckpt):
        return self.auc(self.P.PrivacyNetSanitizer(ckpt, self.device))


def target():
    dst = OUT / "val_utility.json"
    vu = _jload(dst, {})
    todo = [s for s in ["ff_mu0.01", "none"] + CONTROLS["blur"] + CONTROLS["elastic"] if s not in vu]
    if not todo:
        print("target: done", vu); return vu
    OUT.mkdir(parents=True, exist_ok=True)
    V = _Val()
    for s in todo:
        vu[s] = V.auc_name(s)
        print(f"validation utility {s}: {vu[s]:.2f}", flush=True)
        json.dump(vu, open(dst, "w"), indent=1)
    return vu


def _config(w, max_epochs):
    name = f"v3d_pn_w{w:g}"
    cfg = {"experiment_description": name, "image_path": "./images/", "ac_loss_weight": 1, "ver_loss_weight": w,
           "generator_type": "privacy_net", "mu": 0.01, "image_size": 256, "batch_size": 16, "learning_rate": 0.0001,
           "max_epochs": max_epochs, "show_every_n_epochs": 10, "show_every_n_iterations": 100, "seed": 42,
           "adversary_pairing": "SR", "save_every_epoch": True}
    (ROOT / "config_files" / f"{name}.json").write_text(json.dumps(cfg, indent=1))
    return name


def _train(name):
    exp = ROOT / "archive" / name
    cmd = [sys.executable, "-u", "train_architecture.py", "--config", f"{name}.json"]
    if (exp / "resume_state.pth").is_file():
        print(f"resuming {exp}", flush=True); cmd.append("--resume")
    elif exp.exists():
        print(f"removing interrupted attempt {exp}", flush=True); shutil.rmtree(exp)
    subprocess.run(cmd, cwd=ROOT, check=True)


def gen():
    T = target()["ff_mu0.01"]
    att_p = OUT / "w_attempts.json"
    att = _jload(att_p, [])
    for w in WEIGHTS:
        rec = next((a for a in att if a["w"] == w), None)
        if rec and rec["status"] == "rejected":
            continue
        name = f"v3d_pn_w{w:g}"
        exp = ROOT / "archive" / name
        if rec is None:
            if not (exp / "generator_epoch001.pth").is_file():
                _config(w, 1)
                _train(name)
            u1 = _Val().auc_ckpt(exp / "generator_epoch001.pth")
            rec = {"w": w, "val_utility_epoch1": u1, "target": T, "status": "accepted" if u1 >= T else "rejected"}
            att.append(rec)
            json.dump(att, open(att_p, "w"), indent=1)
            print(f"w = {w:g}: validation utility after epoch 1 = {u1:.2f} (T = {T:.2f}) -> {rec['status']}", flush=True)
            if rec["status"] == "rejected":
                continue
        if not (exp / f"generator_epoch{N_EPOCHS:03d}.pth").is_file():
            _config(w, N_EPOCHS)
            _train(name)
        print(f"gen: w = {w:g} trained for {N_EPOCHS} epochs", flush=True)
        return w
    raise SystemExit("gen: every weight was rejected after epoch 1 (report as deviation)")


def select():
    att = _jload(OUT / "w_attempts.json", [])
    acc = [a for a in att if a["status"] == "accepted"]
    assert acc, "no accepted weight"
    w = acc[0]["w"]
    exp = ROOT / "archive" / f"v3d_pn_w{w:g}"
    assert (exp / f"generator_epoch{N_EPOCHS:03d}.pth").is_file(), "training not finished"
    vu = target()
    T = vu["ff_mu0.01"]
    ep_p = OUT / "epoch_val_utility.json"
    eu = _jload(ep_p, {})
    V = None
    for e in range(1, N_EPOCHS + 1):
        if str(e) in eu:
            continue
        V = V or _Val()
        eu[str(e)] = V.auc_ckpt(exp / f"generator_epoch{e:03d}.pth")
        print(f"epoch {e}: validation utility {eu[str(e)]:.2f}", flush=True)
        json.dump(eu, open(ep_p, "w"), indent=1)
    import pickle
    ld = pickle.load(open(exp / "loss_dict.pkl", "rb"))
    vll = ld["validation"]["log_likelihood_ver_loss"]    # -log(1 - p), validation pairs, per epoch
    ok = [e for e in range(1, N_EPOCHS + 1) if eu[str(e)] >= T]
    if ok:
        e_sel, rule = min(ok, key=lambda e: vll[e - 1]), "lowest validation -log(1-p) among epochs with utility >= T"
    else:
        e_sel, rule = max(range(1, N_EPOCHS + 1), key=lambda e: eu[str(e)]), "no epoch reached T: highest utility"
    dst = OUT / "gen" / "pnv3d"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exp / f"generator_epoch{e_sel:03d}.pth", dst / "generator_selected.pth")
    shutil.copy2(exp / "generator_latest.pth", dst / "generator_latest.pth")
    u_sel = eu[str(e_sel)]
    match = {fam: min(names, key=lambda s: abs(vu[s] - u_sel)) for fam, names in CONTROLS.items()}
    sel = {"w": w, "target_T": T, "selected_epoch": e_sel, "rule": rule, "val_utility_selected": u_sel,
           "n_epochs_at_or_above_T": len(ok), "matched": match, "val_utility_controls": vu,
           "val_loglik_ver": vll, "val_total_loss": ld["validation"]["total_loss"],
           "train_loglik_ver": ld["training"]["log_likelihood_ver_loss"], "epoch_val_utility": eu,
           "generator_sha256": sha(dst / "generator_selected.pth"),
           "config": json.load(open(ROOT / "config_files" / f"v3d_pn_w{w:g}.json"))}
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    print(f"select: epoch {e_sel} (validation utility {u_sel:.2f}, T {T:.2f}; {len(ok)} epochs >= T); matched {match}")


def verifiers():
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
    specs["referenceSS"] = specs["reference"]
    out = OUT / "runs"
    out.mkdir(parents=True, exist_ok=True)
    store = P.RAMImageStore(use_cache=True)
    meta_lut = P.load_nih_metadata_lut()
    plan = [(fam, san, sd) for sd in (0, 1, 2) for fam, san in
            (("reference", "pnv3d"), ("e5", "pnv3d"), ("referenceSS", "pnv3d"), ("reference", "privacy_net"),
             ("e5", "privacy_net"))]
    for i, (fam, san, sd) in enumerate(plan, 1):
        run_id = f"v3d__{fam}__{san}__s{sd}"
        if (out / f"{run_id}_summary.json").is_file():
            print(f"[{i}/{len(plan)}] CACHED {run_id}", flush=True); continue
        if san == "pnv3d" and not (OUT / "gen" / "pnv3d" / "generator_selected.pth").is_file():
            print(f"[{i}/{len(plan)}] SKIP {run_id}: no selected generator", flush=True); continue
        pr = "SS" if fam == "referenceSS" else "SR"
        print(f"[{i}/{len(plan)}] RUN {run_id}", flush=True)
        kw = dict(specs[fam])
        tp = kw.pop("train_path")
        rec = P.train_and_evaluate_pair_run(
            run_id=run_id, out_dir=out, store=store, meta_lut=meta_lut, sanitizer_name=san,
            train_path=tp, val_path=VAL_2K, test_path=TEST_5K,
            cal_paths={"disjoint700_SR": (CAL_700, "SR"), "val2k_sub700_SR": (VAL_SUB700, "SR")},
            train_pairing=pr, val_pairing=pr, test_pairing="SR", seed=sd, save_ckpt=True,
            disjoint_check_splits={"att_train": tp, "val_2k": VAL_2K, "cal_700": CAL_700, "test_5k": TEST_5K},
            extra_evals={"test5k_SS": (TEST_5K, "SS")}, **kw)
        tm = rec["test_metrics"]
        print(f"  -> ep={rec['selected_epoch']}/{rec['epochs_ran']} AUC={tm['auc']:.2f}% TPR@1%={tm['tpr_at_1pct']:.2f}%", flush=True)


def utility():
    import numpy as np
    import torch
    from scripts import prl_lib as P
    from scripts.run_e4 import bootstrap_utility_ci, compute_14_aucs, eval_classifier, load_e4_splits
    dst = OUT / "utility_frozen.json"
    rows = _jload(dst, [])
    done = {r["sanitizer"] for r in rows}
    todo = [s for s in ("pnv3d", "privacy_net") if s not in done and
            (s != "pnv3d" or (OUT / "gen" / "pnv3d" / "generator_selected.pth").is_file())]
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
    steps = {"target": target, "gen": gen, "select": select, "verifiers": verifiers, "utility": utility}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        raise SystemExit(__doc__)
    t0 = time.time()
    steps[sys.argv[1]]()
    print(f"{sys.argv[1]}: {round((time.time() - t0) / 60, 1)} min", flush=True)
