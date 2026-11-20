"""Real-training runner for the P1 2x2 attacker-pairing audit.

Trains one attacker per (config, snn_init, train_pairing, seed) cell using the SAME
AgentSiameseNetwork / train_snn / validate_snn / test_snn code path that produced every
existing Table 1 arm (agents/AgentSiameseNetwork.py, utils/utils.py), with real early stopping
(patience from the config, unmodified) and validation-based model selection. Test is evaluated
exactly once, on the best-validation checkpoint, after training stops -- never used for
selection.

Model selection is 100% validation-driven:
  - early stopping watches validation loss (agents/AgentSiameseNetwork.py, unchanged),
  - the saved checkpoint is the best-validation network,
  - test is read once, after stopping, from that checkpoint.
No test-set number ever influences which epoch, seed, or checkpoint is kept.

Because train_pairing changes what the training/validation DATA LOADERS contain (not just the
network), a fresh AgentSiameseNetwork is required for every (config, train_pairing) pair;
snn_init could be swapped on an already-built agent. pairing_resolver.all_cells() groups cells
that could share loaders, but this runner currently constructs one agent per cell. That costs
extra decode time without changing the cell's data or training procedure.

Usage:
  python3 run_pairing_audit.py --config ff_mu0.01 --seeds 0
  python3 run_pairing_audit.py --config ff_mu0.01,none --seeds 0,1,2
  python3 run_pairing_audit.py --config all --seeds 0,1,2   # the full planned design (locked
                                                              # before running; not pre-registered
                                                              # in any timestamped external registry)

For 'none' (no sanitizer configured), train_pairing is provably a no-op (see
pairing_resolver.resolve_run's train_pairing_equivalence field and
tests/test_dataset_train_pairing.py / test_train_snn_flow_field_pairing.py for the proof): a
'sanitized_raw' cell is ALIASED to its 'sanitized_sanitized' twin's already-completed output
rather than trained a second time, and is recorded with status='ALIASED', never as an
independent training run.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import shutil
import sys
import time

import numpy as np
import torch
import torch.optim as optim

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils.utils as utils  # noqa: E402  (also breaks the SiameseDataset<->utils circular import)
from utils.EarlyStopping import EarlyStopping  # noqa: E402
from networks.SiameseNetwork import SiameseNetwork  # noqa: E402
from agents.AgentSiameseNetwork import AgentSiameseNetwork  # noqa: E402
from sklearn import metrics as _sk_metrics  # noqa: E402

import pairing_resolver as R  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "tools"))
import recover_p1_validation as RPV  # noqa: E402  (reuse its select_threshold; do not reimplement)
sys.path.insert(0, os.path.join(ROOT, "artifacts/paper_upgrade/phase1_patient_bootstrap"))
import phase1_bootstrap as B  # noqa: E402  (reuse weighted_operating_point; do not reimplement)

OUT_ROOT = os.path.join(ROOT, "artifacts", "pairing_audit")
PRETRAINED_VER_MODEL = R.PRETRAINED_VER_MODEL

RUNTIME_CODE_FILES = (
    "artifacts/paper_upgrade/phase5_pairing_audit/run_pairing_audit.py",
    "artifacts/paper_upgrade/phase5_pairing_audit/pairing_resolver.py",
    "agents/AgentSiameseNetwork.py",
    "datasets/SiameseDataset.py",
    "utils/utils.py",
    "utils/EarlyStopping.py",
    "utils/GaussianSmoothing.py",
    "networks/SiameseNetwork.py",
    "networks/UNet_PriCheXyNet.py",
    "networks/UNet_PrivacyNet.py",
)


def hash_runtime_code() -> dict:
    """Hash dirty-worktree code that a Git commit alone cannot identify."""
    return {path: R.sha256_file(os.path.join(ROOT, path)) for path in RUNTIME_CODE_FILES}


def hash_run_outputs(out_dir: str) -> dict:
    """Hash every regular output file after a successful run."""
    return {
        os.path.relpath(os.path.join(out_dir, name), ROOT): R.sha256_file(os.path.join(out_dir, name))
        for name in sorted(os.listdir(out_dir))
        if os.path.isfile(os.path.join(out_dir, name))
    }


def saved_test_roc_metrics(results_path: str) -> dict:
    """Recompute descriptive test-ROC metrics from the saved, row-aligned scores."""
    import numpy as np

    predictions_path = results_path.replace("_results.txt", "_labels_predictions.txt")
    data = np.loadtxt(predictions_path, skiprows=1)
    y, scores = data[:, 0], data[:, 1]
    fpr, tpr, _ = _sk_metrics.roc_curve(y, scores)
    return {
        "n_pairs": int(len(y)),
        "auc": float(_sk_metrics.roc_auc_score(y, scores)),
        "tpr_at_test_roc_1pct_fpr": float(np.interp(0.01, fpr, tpr)),
        "semantics": "descriptive test ROC; not a validation-selected threshold",
    }


def completed_manifest_matches(manifest_path: str, results_path: str, spec: dict) -> tuple[bool, str]:
    """Only resume-skip an output whose completed manifest authenticates the result file AND
    whose recorded provenance still matches what would be resolved right now.

    Returns (ok, reason). `reason` explains a mismatch when ok is False, or names which
    provenance path (launch-time vs. post-hoc) authenticated a match when ok is True. This is
    deliberately stricter than checking the results file's hash alone: a config edit, a
    replaced pair list, or a replaced sanitizer/init checkpoint can all leave `results.txt`
    itself untouched (same run_id, same output path) while making the manifest's authentication
    of that file meaningless. None of those cases may be silently reused.
    """
    if not os.path.isfile(manifest_path):
        return False, f"no manifest at {manifest_path}"
    if not os.path.isfile(results_path):
        return False, f"no results file at {results_path}"
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"manifest unreadable: {exc}"

    if manifest.get("status") != "completed":
        return False, f"manifest status is {manifest.get('status')!r}, not 'completed'"

    # Launch-time provenance (new runner) vs. post-hoc provenance (four original pilot cells,
    # backfilled and explicitly labelled as such). Never silently treat one as the other.
    if "output_sha256" in manifest:
        provenance_path = "launch-time"
        output_hashes = manifest["output_sha256"]
        recorded_spec = manifest.get("resolved_run_spec", {})
    elif "posthoc_provenance_verification" in manifest:
        provenance_path = "post-hoc"
        pv = manifest["posthoc_provenance_verification"]
        output_hashes = pv.get("output_sha256", {})
        recorded_spec = manifest.get("resolved_run_spec", {})
    else:
        return False, "manifest has neither output_sha256 nor posthoc_provenance_verification"

    rel_results = os.path.relpath(results_path, ROOT)
    if rel_results not in output_hashes:
        return False, f"manifest ({provenance_path}) does not list {rel_results} among its outputs"
    if R.sha256_file(results_path) != output_hashes[rel_results]:
        return False, f"{rel_results} on disk does not match the hash recorded in the manifest"

    # Config/input drift: compare every hash the CURRENT resolve_run(...) call just computed
    # (which reads the config file, pair lists, and checkpoints off disk right now) against what
    # the manifest recorded when the run completed. A silent config edit, pair-list swap, or
    # checkpoint replacement that happens to leave results.txt's bytes alone is still a drift.
    for key in ("base_config_sha256",):
        if recorded_spec.get(key) != spec.get(key):
            return False, f"{key} drifted: manifest={recorded_spec.get(key)!r} current={spec.get(key)!r}"
    for key in ("pair_list_sha256", "checkpoint_sha256"):
        recorded, current = recorded_spec.get(key) or {}, spec.get(key) or {}
        if recorded != current:
            return False, f"{key} drifted: manifest={recorded!r} current={current!r}"
    for key in ("snn_init", "train_pairing", "seed", "config_name"):
        if recorded_spec.get(key) != spec.get(key):
            return False, f"{key} drifted: manifest={recorded_spec.get(key)!r} current={spec.get(key)!r}"

    # Selected checkpoint and predictions must still be the files the manifest authenticated,
    # not merely present -- a checkpoint or prediction file replaced out-of-band (same path,
    # different bytes) after the manifest was written must not be trusted implicitly.
    selected = manifest.get("selected_checkpoint") or (
        manifest.get("posthoc_provenance_verification", {}).get("selected_checkpoint"))
    if selected:
        ckpt_path = os.path.join(ROOT, selected["path"])
        if not os.path.isfile(ckpt_path):
            return False, f"selected checkpoint missing: {selected['path']}"
        if R.sha256_file(ckpt_path) != selected["sha256"]:
            return False, f"selected checkpoint hash drifted: {selected['path']}"
    predictions_path = results_path.replace("_results.txt", "_labels_predictions.txt")
    rel_pred = os.path.relpath(predictions_path, ROOT)
    if rel_pred in output_hashes:
        if not os.path.isfile(predictions_path):
            return False, f"predictions file missing: {rel_pred}"
        if R.sha256_file(predictions_path) != output_hashes[rel_pred]:
            return False, f"predictions file hash drifted: {rel_pred}"

    return True, f"authenticated via {provenance_path} provenance"


def reset_for_run(agent: AgentSiameseNetwork, snn_init: str, seed: int) -> None:
    """Reuse an already-built agent (same data loaders) for a new (snn_init, seed) run.
    Mirrors run_verification.py's reset_for_run exactly, so a P1 cell with
    snn_init='imagenet', train_pairing='sanitized_sanitized' is bit-for-bit the same procedure
    as the existing Table 1 ImageNet-init arm -- not a re-implementation that could subtly
    diverge from it.
    """
    utils.seed_all(seed)
    agent.net = SiameseNetwork().cuda()
    if snn_init in ("pretrained_ver", "vstar"):
        agent.net.load_state_dict(torch.load(PRETRAINED_VER_MODEL if snn_init == "pretrained_ver" else R.VSTAR_MODEL), strict=True)
    agent.best_net = copy.deepcopy(agent.net)
    agent.optimizer = optim.Adam(agent.net.parameters(), lr=agent.learning_rate)
    agent.start_epoch = 0
    agent.es = EarlyStopping(patience=agent.early_stopping)
    agent.best_loss = 100000
    agent.loss_dict = {"training": [], "validation": []}


def tpr_at(y, s, target):
    import numpy as np

    fpr, tpr, _ = _sk_metrics.roc_curve(y, s)
    return float(__import__("numpy").interp(target, fpr, tpr))


def score_validation_deployment_pairing(agent: AgentSiameseNetwork, results_path: str) -> dict:
    """Score the best checkpoint on the 2,000-pair validation list under the FIXED deployment
    pairing (sanitized query / raw reference) -- the pairing utils.test_snn always applies,
    regardless of this cell's train_pairing -- and select a validation threshold from it.

    This is deliberately NOT agent.validation_loader (built with this cell's own train_pairing,
    e.g. sanitized_sanitized, and used only for the training-time early-stopping LOSS). Reusing
    that loader here would conflate two different pairings under one name. Instead this builds a
    dedicated loader forced to train_pairing='sanitized_raw' and scores it with the same
    unmodified utils.test_snn used for the real test set, so no scoring logic is duplicated.

    Threshold rule and downstream realised-FPR/TPR computation are reused, not reimplemented,
    from tools/recover_p1_validation.py (RPV.select_threshold) and
    artifacts/paper_upgrade/phase1_patient_bootstrap/phase1_bootstrap.py (B.weighted_operating_point),
    both already verified against saved GPU predictions with a pre-registered numerical tolerance.
    """
    val_loader = utils.get_data_loader(
        phase="validation", experimental_step="retrainSNN",
        image_size=agent.image_size, n_channels=agent.n_channels,
        batch_size=agent.batch_size, shuffle=False,
        num_workers=agent.num_workers, pin_memory=agent.pin_memory,
        b=agent.b, m=agent.m, eps=agent.eps, image_path=agent.IMAGE_PATH,
        blur_sigma=agent.blur_sigma, elastic_mu=agent.elastic_mu,
        train_pairing="sanitized_raw",
    )
    y_val, s_val = utils.test_snn(agent.perturbation_type, agent.best_net, agent.perturbation_net,
                                   agent.grid_identity, agent.gauss_filter, agent.mu, val_loader)
    y_val, s_val = y_val.numpy(), s_val.numpy()
    threshold, val_fpr, val_tpr = RPV.select_threshold(y_val, s_val)

    predictions_path = results_path.replace("_results.txt", "_labels_predictions.txt")
    saved = np.loadtxt(predictions_path, skiprows=1)
    y_test, s_test = saved[:, 0], saved[:, 1]
    test_fpr, test_tpr = B.weighted_operating_point(y_test, s_test, np.ones(len(y_test)), threshold)

    val_pairs_path = os.environ.get("VAL_PAIRS_FILE") or os.path.join(
        ROOT, "image_pairs", "image_pairs_validation_2000.txt")
    val_pairs = np.loadtxt(val_pairs_path, dtype=str)
    if len(val_pairs) != len(y_val):
        raise RuntimeError(
            f"validation pair-list length ({len(val_pairs)}) != scored rows ({len(y_val)}); "
            "refusing to save misaligned pair IDs"
        )
    if not np.array_equal(val_pairs[:, 2].astype(float), y_val):
        raise RuntimeError("validation pair-list labels do not match test_snn's returned labels; "
                            "refusing to trust pair-ID alignment")

    npz_path = os.path.join(OUT_ROOT, f"{agent.config['experiment_description']}_validation.npz")
    os.makedirs(OUT_ROOT, exist_ok=True)
    np.savez_compressed(npz_path, pair_ids=val_pairs[:, :2], labels=y_val, scores=s_val)

    return {
        "validation_selected_threshold": float(threshold),
        "validation_fpr_percent": 100 * val_fpr,
        "validation_tpr_percent": 100 * val_tpr,
        "realized_test_fpr_percent": float(test_fpr),
        "realized_test_tpr_percent": float(test_tpr),
        "validation_scores_file": os.path.relpath(npz_path, ROOT),
        "validation_scores_sha256": R.sha256_file(npz_path),
        "semantics": (
            "lowest observed validation threshold with empirical FPR<=1%, score>=threshold, "
            "under fixed deployment pairing (sanitized query / raw reference); distinct from the "
            "training-time validation LOSS used for checkpoint/early-stopping selection, and "
            "distinct from the descriptive test-ROC TPR@1%FPR reported elsewhere"
        ),
    }


def alias_from_equivalent_cell(spec: dict) -> dict:
    """Handle a train_pairing_equivalence cell (currently only 'none' config, sanitized_raw):
    copy the proven-identical twin's outputs under THIS cell's own run_id, instead of training
    independently. Refuses (does not fall back to training) if the twin is not itself a
    verified-complete run -- an aliased cell must never silently become an independent one just
    because its twin happens to be missing.
    """
    equiv = spec["train_pairing_equivalence"]
    run_id = spec["run_id"]
    twin_run_id = equiv["equivalent_to_run_id"]

    twin_resolved = R.resolve_run(spec["config_name"], spec["snn_init"],
                                   equiv["equivalent_to_train_pairing"], spec["seed"])
    twin_out_dir = os.path.join(ROOT, twin_resolved["output_dir"])
    twin_results_path = os.path.join(ROOT, twin_resolved["results_file"])
    twin_manifest_path = os.path.join(OUT_ROOT, f"{twin_run_id}_manifest.json")

    ok, reason = completed_manifest_matches(twin_manifest_path, twin_results_path, twin_resolved)
    if not ok:
        raise RuntimeError(
            f"REFUSING TO ALIAS {run_id}: its equivalent twin {twin_run_id} is not a verified "
            f"complete run ({reason}). Run the twin first; this cell must not be trained "
            "independently merely because the twin is missing or stale."
        )

    out_dir = os.path.join(ROOT, spec["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    copied = {}
    for name in sorted(os.listdir(twin_out_dir)):
        src = os.path.join(twin_out_dir, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(out_dir, name.replace(twin_run_id, run_id))
        shutil.copy2(src, dst)
        copied[os.path.relpath(dst, ROOT)] = R.sha256_file(dst)

    manifest_path = os.path.join(OUT_ROOT, f"{run_id}_manifest.json")
    manifest = {
        "run_id": run_id,
        "status": "ALIASED",
        "alias_of": twin_run_id,
        "alias_proof": equiv["proof"],
        "resolved_run_spec": spec,
        "aliased_from_manifest": os.path.relpath(twin_manifest_path, ROOT),
        "output_sha256": copied,
        "note": (
            "This cell was NOT trained independently: proven equivalent to its twin's training "
            "run (see alias_proof), and every output file here is a byte-identical copy (hashes "
            "above match the twin's). Do not count this as an independent GPU training run in "
            "any tally, run_summary, or results.tsv row."
        ),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    return {"run_id": run_id, "status": "ALIASED", "alias_of": twin_run_id, "minutes": 0.0}


def run_one_cell(spec: dict, resume: bool = True) -> dict:
    """Train and test one resolved cell. Returns the result record; also writes it to disk."""
    if spec.get("train_pairing_equivalence"):
        return alias_from_equivalent_cell(spec)

    run_id = spec["run_id"]
    out_dir = os.path.join(ROOT, spec["output_dir"])
    results_path = os.path.join(ROOT, spec["results_file"])
    manifest_path = os.path.join(OUT_ROOT, f"{run_id}_manifest.json")

    if resume and os.path.isfile(results_path):
        ok, reason = completed_manifest_matches(manifest_path, results_path, spec)
        if ok:
            return {"run_id": run_id, "status": "skipped_verified_complete",
                    "results_file": spec["results_file"], "resume_reason": reason}
        raise RuntimeError(
            f"REFUSING TO REUSE unauthenticated/stale output for {run_id}: {reason}. "
            "The existing output is left in place (not deleted or overwritten); move it to "
            "quarantine or pass --no-resume to replace it explicitly."
        )

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(OUT_ROOT, exist_ok=True)

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()

    # Manifest is written with status='running' BEFORE anything that can fail (agent
    # construction, checkpoint loading, CUDA allocation) so a constructor-time crash still
    # leaves a manifest recording that the cell was attempted, rather than vanishing with no
    # artifact trace beyond (maybe) a run_summary.json entry.
    manifest = {
        "run_id": run_id,
        "command": "python3 " + shlex.join(sys.argv),
        "resolved_run_spec": spec,
        "git_provenance": R.git_provenance(),
        "runtime_code_sha256": hash_runtime_code(),
        "started_utc": started_utc,
        "selection_procedure": (
            "early stopping on validation loss (patience from config, unmodified); the saved "
            "checkpoint is the best-validation network; test is read once from that checkpoint "
            "after training stops and never influences selection"
        ),
        "status": "running",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1, default=str)

    try:
        manifest["software"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
        }
        # Pretraining provenance: ImageNet weights are torchvision's bundled resnet50 default
        # (only ever loaded when snn_init='imagenet'; 'pretrained_ver' below overwrites them
        # immediately via strict full-checkpoint load and never uses the ImageNet init). This is
        # PLANNED protocol, not a pre-registered protocol: no timestamped registry entry exists
        # for this design; do not describe it as pre-registered anywhere it is cited from.
        manifest["pretraining_provenance"] = {
            "imagenet_weights": (
                "torchvision.models.resnet50(pretrained=True), the framework-bundled default "
                "weights for this torchvision version; used only when snn_init='imagenet'"
                if spec["snn_init"] == "imagenet" else
                "not used: snn_init='pretrained_ver' strict-loads the verifier checkpoint over "
                "a freshly constructed (unused) ImageNet initialization before any training step"
            ),
            "verifier_checkpoint_sha256": spec["checkpoint_sha256"].get("init_checkpoint"),
            "sanitizer_checkpoint_sha256": spec["checkpoint_sha256"].get("sanitizer_checkpoint"),
            "torchvision": __import__("torchvision").__version__,
            "protocol_status": "planned (locked before this cell's run); not pre-registered "
                                "in any timestamped external registry",
        }
        manifest["preprocessing"] = {
            "input_resize": [spec["resolved_config"]["image_size"]] * 2,
            "sanitizer": spec["resolved_config"]["perturbation_type"],
            "train_and_checkpoint_validation_pairing": spec["train_pairing"],
            "checkpoint_validation_pairing": spec.get("val_pairing", spec["train_pairing"]),
            "validation_list": spec.get("val_list_file"),
            "test_pairing": spec["test_pairing"],
            "siamese_input": (
                "one-channel sanitizer output expanded to RGB then normalized with "
                "ImageNet mean/std; none uses the config's native RGB transform"
            ),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1, default=str)

        os.environ["PRICHEXY_SEED"] = str(spec["seed"])
        if spec.get("val_list", "orig") == "disjoint":
            os.environ["VAL_PAIRS_FILE"] = os.path.join(ROOT, spec["val_list_file"])
        else:
            os.environ.pop("VAL_PAIRS_FILE", None)
        config = dict(spec["resolved_config"])
        config["experiment_description"] = run_id

        utils.seed_all(spec["seed"])
        agent = AgentSiameseNetwork(config)
        reset_for_run(agent, spec["snn_init"], spec["seed"])

        torch.cuda.reset_peak_memory_stats()
        agent.run()  # == training_validation() then testing_evaluation(); see AgentSiameseNetwork.run
    except Exception as exc:  # noqa: BLE001
        manifest["status"] = "FAILED"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["failed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["minutes_before_failure"] = round((time.time() - t0) / 60, 2)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1, default=str)
        raise

    minutes = (time.time() - t0) / 60
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3

    result = {"run_id": run_id, "status": "completed", "minutes": round(minutes, 2),
              "peak_gpu_gib": round(peak_gib, 2), "epochs_trained": len(agent.loss_dict["training"])}
    if os.path.isfile(results_path):
        with open(results_path) as f:
            result["results_txt"] = f.read()

    checkpoint_path = os.path.join(out_dir, f"{run_id}_best_network.pth")
    best_epoch = int(np.argmin(agent.loss_dict["validation"])) + 1

    try:
        validation_metrics = score_validation_deployment_pairing(agent, results_path)
    except Exception as exc:  # noqa: BLE001
        # Training/testing already succeeded and must not be discarded because the additional
        # validation-threshold scoring step failed; record the failure explicitly instead.
        validation_metrics = {"FAILED": f"{type(exc).__name__}: {exc}"}

    manifest.update({
        "status": "completed",
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "minutes": result["minutes"],
        "peak_gpu_gib": result["peak_gpu_gib"],
        "epochs_trained": result["epochs_trained"],
        "selected_checkpoint": {
            "path": os.path.relpath(checkpoint_path, ROOT),
            "sha256": R.sha256_file(checkpoint_path),
            "epoch": best_epoch,
            "criterion": "minimum validation loss",
        },
        "test_roc_metrics": saved_test_roc_metrics(results_path),
        "validation_threshold_metrics": validation_metrics,
        "output_sha256": hash_run_outputs(out_dir),
        "missing_required_metrics": "training-seed uncertainty requires seeds 1 and 2",
    })
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1, default=str)

    return result


def main():
    p = argparse.ArgumentParser("P1 2x2 attacker-pairing audit runner")
    p.add_argument("--config", required=True,
                   help="comma-separated config names from pairing_resolver.PRIORITY_CONFIGS, "
                        "or 'all' for the full planned design (locked before running; not "
                        "pre-registered in any timestamped external registry)")
    p.add_argument("--seeds", default="0", help="comma-separated seeds, e.g. 0,1,2")
    p.add_argument("--design", choices=("pilot", "factorial"), default="pilot",
                   help="'factorial' runs the locked DEC-2026-11-02-001 design (verifier-init; "
                        "--config/--seeds are ignored)")
    p.add_argument("--only", default="", help="factorial only: substring filter on run_id")
    p.add_argument("--shard", default="0/1", help="factorial only: i/n, run every n-th cell")
    p.add_argument("--no-resume", action="store_true",
                   help="re-run a cell even if its results file already exists")
    args = p.parse_args()

    config_names = None if args.config == "all" else args.config.split(",")
    seeds = tuple(int(s) for s in args.seeds.split(","))

    if not torch.cuda.is_available():
        raise SystemExit("REFUSING TO RUN: no GPU available. This script trains real networks; "
                         "see pairing_resolver.py for the CPU-only config-resolution surface, "
                         "which is fully testable without one.")

    if args.design == "factorial":
        i, n = (int(x) for x in args.shard.split("/"))
        cells = [c for c in R.factorial_cells() if args.only in R.cell_id(c[0], c[1], c[2], c[3], c[4], c[5])]
        cells = cells[i::n]
    else:
        cells = [(*c, None, "orig") for c in R.all_cells(config_names=config_names, seeds=seeds)]
    print(f"resolved {len(cells)} cells: {cells}", flush=True)

    # Merge into any existing run_summary.json rather than overwriting it: a prior invocation's
    # entries for cells not included in THIS invocation (e.g. a previous --config subset) must
    # not disappear just because this process started with an empty in-memory list.
    summary_path = os.path.join(OUT_ROOT, "run_summary.json")
    summary_by_id = {}
    if os.path.isfile(summary_path):
        try:
            with open(summary_path) as f:
                for row in json.load(f):
                    if "run_id" in row:
                        summary_by_id[row["run_id"]] = row
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: existing {summary_path} unreadable ({exc}); starting a fresh summary "
                  "without discarding the file itself", flush=True)

    for name, init, pairing, seed, val_pairing, val_list in cells:
        try:
            spec = R.resolve_run(name, init, pairing, seed, val_pairing, val_list)
        except R.IncompatibleEnvOverrideError as exc:
            print(f"===== {name}/{init}/{pairing}/seed{seed} REFUSED: {exc} =====", flush=True)
            raise
        print(f"\n===== {spec['run_id']} starting at {time.strftime('%F %T')} =====", flush=True)
        try:
            result = run_one_cell(spec, resume=not args.no_resume)
        except Exception as exc:  # noqa: BLE001
            print(f"===== {spec['run_id']} FAILED: {type(exc).__name__}: {exc} =====", flush=True)
            result = {"run_id": spec["run_id"], "status": "FAILED", "error": str(exc)}
            summary_by_id[result["run_id"]] = result
            with open(summary_path, "w") as f:
                json.dump(list(summary_by_id.values()), f, indent=1, default=str)
            continue
        print(f"===== {spec['run_id']}: {result.get('status')} "
              f"({result.get('minutes', '?')} min) =====", flush=True)
        summary_by_id[result["run_id"]] = result
        with open(summary_path, "w") as f:
            json.dump(list(summary_by_id.values()), f, indent=1, default=str)

    print("\nDone. Summary:")
    for r in summary_by_id.values():
        print(" ", r.get("run_id"), r.get("status"), r.get("minutes"))


if __name__ == "__main__":
    main()
