"""Config resolution and provenance for the P1 2x2 attacker-pairing audit.

The 2x2 design crosses two independently-implemented axes:

  snn_init:      'imagenet' | 'pretrained_ver'          (which checkpoint the attacker starts from)
  train_pairing: 'sanitized_sanitized' | 'sanitized_raw' (whether the training/validation
                                                           reference endpoint is sanitized)

Both are explicit string enums, never a bare boolean: 'sanitized=true' does not say relative to
what, and a boolean can't be extended if a third pairing regime is ever added. The test endpoint
is NOT a variable of this study -- it is always query-sanitized / reference-raw by construction,
for every cell, matching the released threat model. This module documents that as a constant
rather than a config field, so it cannot be silently varied by mistake.

Every function here is pure (no GPU, no filesystem writes, no training). It only reads existing
config/checkpoint/pair-list files to compute hashes and resolve paths. `run_pairing_audit.py`
is the only file in this directory that touches CUDA or trains anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIG_DIR = os.path.join(ROOT, "config_files")
ARCHIVE_DIR = os.path.join(ROOT, "archive")

SNN_INIT_CHOICES = ("imagenet", "pretrained_ver", "vstar")
TRAIN_PAIRING_CHOICES = ("sanitized_sanitized", "sanitized_raw")

# Fixed by the threat model this audit evaluates, not a config axis. Recorded in every
# manifest as a constant so it is never confused with something this study varies.
TEST_PAIRING = "sanitized_raw"  # query sanitized, reference raw -- identical across all cells

PRETRAINED_VER_MODEL = os.path.join(ROOT, "networks", "pretrained_verification_model.pth")
# DEC-2026-11-09-001 E1: independent verifier trained here on patients disjoint from val/test
VSTAR_MODEL = os.path.join(ROOT, "networks", "vstar_verifier.pth")

# The four priority configurations locked before any run, and the base config file that
# defines each one's sanitizer. Reusing the EXACT files that produced Table 1's ImageNet-init
# arm (config_ver_*.json), rather than writing new ones, so a mismatch in preprocessing or
# hyperparameters between the P1 cells and the existing baselines cannot occur by construction.
PRIORITY_CONFIGS = {
    "none": "config_ver_none.json",
    "ff_mu0.01": "config_ver_ff_mu0.01.json",
    "blur_s2": "config_ver_blur_s2.json",
    "elastic_m004": "config_ver_elastic_m004.json",
}

PAIR_LIST_FILES = {
    "training": os.path.join(ROOT, "image_pairs", "image_pairs_training_10000.txt"),
    "validation": os.path.join(ROOT, "image_pairs", "image_pairs_validation_2000.txt"),
    "testing": os.path.join(ROOT, "image_pairs", "image_pairs_testing_5000.txt"),
}


class UnknownConfigError(KeyError):
    pass


def sha256_file(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"cannot hash a file that does not exist: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitizer_checkpoint_for(base_config: dict) -> str | None:
    """The sanitizer checkpoint a config depends on, if it has a network-based one.

    'none', 'blur', 'elastic', 'dp_pix' apply no network (blur/elastic/dp_pix are closed-form,
    applied inside the dataset); only 'flow_field' and 'privacy_net' load a checkpoint file.
    """
    if base_config.get("perturbation_type") in ("flow_field", "privacy_net"):
        return base_config["perturbation_model_file"]
    return None


VAL_LIST_CHOICES = ("orig", "disjoint")
VAL_LIST_FILES = {
    "orig": os.path.join(ROOT, "image_pairs", "image_pairs_validation_2000.txt"),
    # 700 pairs whose patients are absent from the verifier pretraining list (see
    # tools/build_disjoint_validation.py)
    "disjoint": os.path.join(ROOT, "image_pairs", "image_pairs_validation_disjoint_700.txt"),
}
_PAIRING_TAG = {"sanitized_sanitized": "SS", "sanitized_raw": "SR"}


def cell_id(config_name: str, snn_init: str, train_pairing: str, seed: int,
            val_pairing: str | None = None, val_list: str = "orig") -> str:
    base = f"p1_{config_name}_{train_pairing}_{snn_init}_seed{seed}"
    if (val_pairing in (None, train_pairing)) and val_list == "orig":
        return base  # identical to the pre-factorial id, so old manifests/outputs stay addressable
    vp = val_pairing or train_pairing
    return f"p1f_{config_name}_{_PAIRING_TAG[train_pairing]}train_{_PAIRING_TAG[vp]}val_{val_list}_{snn_init}_seed{seed}"


class IncompatibleEnvOverrideError(RuntimeError):
    pass


def check_env_overrides() -> dict:
    """Resolve or reject the three environment variables the training/eval code reads directly
    (utils.SiameseDataset's TRAIN_PAIRS_FILE, utils.train_snn's GRADIENT_ACCUMULATION_STEPS,
    utils.elastic_warp's ELASTIC_SALT), none of which pairing hashes below would otherwise
    notice, since those hash fixed default paths regardless of what is actually loaded at
    runtime.

    A value equal to the locked protocol's default is recorded (so the manifest is explicit,
    not silent) but not rejected. Any other value raises: this study does not permit a cell to
    silently swap its training pair list, gradient-accumulation schedule, or elastic-field
    realization mid-design. A deliberate override belongs in an updated locked protocol
    document, not an ambient shell variable.
    """
    overrides = {}

    train_pairs_file = os.environ.get("TRAIN_PAIRS_FILE")
    expected_pairs = os.path.normpath(PAIR_LIST_FILES["training"])
    if train_pairs_file is not None:
        actual = train_pairs_file if os.path.isabs(train_pairs_file) else os.path.join(ROOT, train_pairs_file)
        if os.path.normpath(actual) != expected_pairs:
            raise IncompatibleEnvOverrideError(
                f"TRAIN_PAIRS_FILE={train_pairs_file!r} does not resolve to the locked training "
                f"pair list ({PAIR_LIST_FILES['training']}); this audit's protocol does not "
                "permit swapping the pair list mid-design. Unset it, or update the locked "
                "protocol explicitly before resolving new cells."
            )
    overrides["TRAIN_PAIRS_FILE"] = train_pairs_file if train_pairs_file is not None else "(unset; default)"

    accum = os.environ.get("GRADIENT_ACCUMULATION_STEPS")
    if accum is not None and accum != "1":
        raise IncompatibleEnvOverrideError(
            f"GRADIENT_ACCUMULATION_STEPS={accum!r} changes the effective optimization schedule "
            "relative to the locked protocol (implicit default '1'). Unset it, or update the "
            "locked protocol explicitly before resolving new cells."
        )
    overrides["GRADIENT_ACCUMULATION_STEPS"] = accum if accum is not None else "(unset; default '1')"

    salt = os.environ.get("ELASTIC_SALT")
    if salt is not None and salt != "0":
        raise IncompatibleEnvOverrideError(
            f"ELASTIC_SALT={salt!r} draws a different elastic-field realization than the locked "
            "protocol (default '0'). That is legitimate for an explicit realization-sensitivity "
            "check (see artifacts/paper_upgrade/phase3_random_fields/), but not for a cell "
            "resolved as part of this design. Unset it, or use the dedicated sensitivity tooling."
        )
    overrides["ELASTIC_SALT"] = salt if salt is not None else "(unset; default '0')"

    return overrides


def resolve_run(config_name: str, snn_init: str, train_pairing: str, seed: int,
                val_pairing: str | None = None, val_list: str = "orig") -> dict:
    """Resolve one 2x2-design cell into a fully concrete, provenance-complete run spec.

    Raises rather than guessing on any ambiguous or unknown input: an unknown config name,
    an invalid enum value, a missing file, or an incompatible environment override
    (`check_env_overrides`) all raise immediately. Nothing here launches a process or touches
    the GPU.
    """
    if config_name not in PRIORITY_CONFIGS:
        raise UnknownConfigError(
            f"{config_name!r} is not one of the locked priority configs: "
            f"{sorted(PRIORITY_CONFIGS)}"
        )
    if snn_init not in SNN_INIT_CHOICES:
        raise ValueError(f"snn_init must be one of {SNN_INIT_CHOICES}, got {snn_init!r}")
    if train_pairing not in TRAIN_PAIRING_CHOICES:
        raise ValueError(
            f"train_pairing must be one of {TRAIN_PAIRING_CHOICES}, got {train_pairing!r}"
        )
    val_pairing = val_pairing or train_pairing
    if val_pairing not in TRAIN_PAIRING_CHOICES:
        raise ValueError(f"val_pairing must be one of {TRAIN_PAIRING_CHOICES}, got {val_pairing!r}")
    if val_list not in VAL_LIST_CHOICES:
        raise ValueError(f"val_list must be one of {VAL_LIST_CHOICES}, got {val_list!r}")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")

    base_config_path = os.path.join(CONFIG_DIR, PRIORITY_CONFIGS[config_name])
    with open(base_config_path) as f:
        base_config = json.load(f)

    run_id = cell_id(config_name, snn_init, train_pairing, seed, val_pairing, val_list)
    resolved_config = dict(base_config)
    resolved_config["val_pairing"] = val_pairing
    resolved_config["experiment_description"] = run_id
    resolved_config["snn_init"] = snn_init
    resolved_config["train_pairing"] = train_pairing

    sanitizer_ckpt = sanitizer_checkpoint_for(base_config)
    sanitizer_ckpt_abs = (
        os.path.join(ROOT, sanitizer_ckpt.lstrip("./")) if sanitizer_ckpt else None
    )

    checkpoint_hashes = {}
    if sanitizer_ckpt_abs:
        checkpoint_hashes["sanitizer_checkpoint"] = sha256_file(sanitizer_ckpt_abs)
    if snn_init == "pretrained_ver":
        checkpoint_hashes["init_checkpoint"] = sha256_file(PRETRAINED_VER_MODEL)
    elif snn_init == "vstar":
        checkpoint_hashes["init_checkpoint"] = sha256_file(VSTAR_MODEL)

    pair_list_files = dict(PAIR_LIST_FILES)
    pair_list_files["validation"] = VAL_LIST_FILES[val_list]
    pair_list_hashes = {phase: sha256_file(path) for phase, path in pair_list_files.items()}

    env_overrides = check_env_overrides()

    # 'none' has no sanitizer, so train_pairing has no code path that can affect it: SiameseDataset
    # only reads train_pairing inside blur/elastic/DP-Pix gates that all require a configured
    # sanitizer (config_ver_none.json sets blur_sigma/elastic_mu/b/m/eps all to null), and
    # utils.train_snn/validate_snn only consult it inside `if perturbation_type == 'flow_field'`
    # / `'privacy_net'` blocks that never execute for 'none'. Proven empirically (not just argued)
    # in artifacts/paper_upgrade/phase5_pairing_audit/tests/test_dataset_train_pairing.py
    # (dataset output bit-identical) and test_train_snn_flow_field_pairing.py (validate_snn's
    # net() inputs bit-identical), for both train/validation phases. A 'sanitized_raw' cell of
    # 'none' is therefore the SAME run as its 'sanitized_sanitized' twin at the SAME seed, not a
    # merely-similar one -- run_pairing_audit.py must alias it rather than train it independently,
    # and nothing may count it as an independent training run (results.tsv, run_summary.json).
    train_pairing_equivalence = None
    if config_name == "none" and train_pairing == "sanitized_raw":
        train_pairing_equivalence = {
            "equivalent_to_train_pairing": "sanitized_sanitized",
            "equivalent_to_run_id": cell_id(config_name, snn_init, "sanitized_sanitized", seed),
            "proof": (
                "no sanitizer is configured (config_ver_none.json: blur_sigma/elastic_mu/b/m/eps "
                "all null), and every train_pairing-reading code path in SiameseDataset and "
                "utils.train_snn/validate_snn is itself gated on a sanitizer being configured; "
                "empirically verified bit-identical in "
                "artifacts/paper_upgrade/phase5_pairing_audit/tests/"
                "test_dataset_train_pairing.py and test_train_snn_flow_field_pairing.py"
            ),
        }

    return {
        "run_id": run_id,
        "config_name": config_name,
        "base_config_file": os.path.relpath(base_config_path, ROOT),
        "base_config_sha256": sha256_file(base_config_path),
        "resolved_config": resolved_config,
        "snn_init": snn_init,
        "train_pairing": train_pairing,
        "val_pairing": val_pairing,
        "val_list": val_list,
        "val_list_file": os.path.relpath(VAL_LIST_FILES[val_list], ROOT),
        "train_pairing_equivalence": train_pairing_equivalence,
        "test_pairing": TEST_PAIRING,
        "seed": seed,
        "sanitizer_checkpoint_file": (
            os.path.relpath(sanitizer_ckpt_abs, ROOT) if sanitizer_ckpt_abs else None
        ),
        "checkpoint_sha256": checkpoint_hashes,
        "pair_list_sha256": pair_list_hashes,
        "env_overrides": env_overrides,
        "output_dir": os.path.relpath(os.path.join(ARCHIVE_DIR, run_id), ROOT),
        "results_file": os.path.relpath(
            os.path.join(ARCHIVE_DIR, run_id, f"{run_id}_results.txt"), ROOT
        ),
    }


def all_cells(config_names=None, seeds=(0, 1, 2)):
    """Every (config, snn_init, train_pairing, seed) cell of the 2x2 design, deterministically
    ordered. Grouped by (config_name, train_pairing) first, since those two together determine
    the data loaders an actual run would need to build -- adjacent cells in this order can share
    a single AgentSiameseNetwork construction, varying only snn_init and seed."""
    names = list(PRIORITY_CONFIGS) if config_names is None else list(config_names)
    for name in names:
        for pairing in TRAIN_PAIRING_CHOICES:
            for init in SNN_INIT_CHOICES:
                for seed in seeds:
                    yield name, init, pairing, seed


def factorial_cells(seeds_ff=(0, 1, 2, 3, 4), seeds_ctrl=(0, 1, 2)):
    """Locked factorial design (DEC-2026-11-02-001), verifier-init only.

    Tier A: ff_mu0.01, train x val pairing, original val list.
    Tier B: ff_mu0.01, train=val pairing, disjoint val list.
    Tier C: blur_s2, elastic_m004: tiers A+B cells, fewer seeds.
    Yields (config, init, train_pairing, seed, val_pairing, val_list)."""
    SS, SR = TRAIN_PAIRING_CHOICES
    def cells(cfg, seeds):
        for seed in seeds:
            for tp in (SS, SR):
                for vp in (SS, SR):
                    yield cfg, "pretrained_ver", tp, seed, vp, "orig"
            for tp in (SS, SR):
                yield cfg, "pretrained_ver", tp, seed, tp, "disjoint"
    yield from cells("ff_mu0.01", seeds_ff)
    for cfg in ("blur_s2", "elastic_m004"):
        yield from cells(cfg, seeds_ctrl)


def git_provenance() -> dict:
    def _run(args):
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True).strip()
        except Exception as exc:  # noqa: BLE001
            return f"unavailable: {exc}"

    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_worktree_dirty": bool(_run(["git", "status", "--porcelain"])),
    }
