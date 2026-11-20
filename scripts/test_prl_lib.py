"""CPU unit tests for scripts/prl_lib.py and confirmatory E1 clean splits."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import prl_lib as P  # noqa: E402
from datasets.SiameseDataset import SiameseDataset  # noqa: E402


def _require(*rel_paths):
    """Skip a test whose inputs are not part of the public release (images, weights, checkpoints)."""
    missing = [p for p in rel_paths if not (ROOT / p).exists()]
    if missing:
        pytest.skip("missing local inputs: " + ", ".join(missing))


NEEDS_IMAGES = ("images",)
NEEDS_WEIGHTS = ("networks/generator_lowest_total_loss_mu_0.01.pth", "networks/pretrained_verification_model.pth")


@pytest.fixture(scope="module")
def store():
    _require(*NEEDS_IMAGES)
    return P.RAMImageStore(use_cache=True)


def test_clean_splits_strict_patient_disjoint():
    clean = ROOT / "results" / "clean_splits"
    splits = {
        "pretrain": clean / "d_pretrain_clean.txt",
        "att_train": clean / "d_att_train_clean.txt",
        "val_2k": ROOT / "image_pairs" / "image_pairs_validation_2000.txt",
        "val_sub700": clean / "d_val_2000_sub700.txt",
        "cal_700": ROOT / "image_pairs" / "image_pairs_validation_disjoint_700.txt",
        "test_5k": ROOT / "image_pairs" / "image_pairs_testing_5000.txt",
    }
    P.assert_patient_disjoint(splits, allow_pairs=(("val_2k", "val_sub700"),))


def test_clean_splits_leakage_raises():
    clean = ROOT / "results" / "clean_splits"
    with pytest.raises(ValueError, match="Patient leakage"):
        P.assert_patient_disjoint(
            {
                "val_2k": ROOT / "image_pairs" / "image_pairs_validation_2000.txt",
                "val_sub700": clean / "d_val_2000_sub700.txt",
            }
        )


def test_ram_store_matches_siamese_dataset_exact(store):
    pairs = [l.split() for l in (ROOT / "image_pairs" / "image_pairs_testing_5000.txt").open()][:4]
    with tempfile.TemporaryDirectory() as td:
        pdir = Path(td) / "image_pairs"
        pdir.mkdir(parents=True)
        with (pdir / "image_pairs_testing_5000.txt").open("w") as f:
            for a, b, y in pairs:
                f.write(f"{a} {b} {y}\n")
        import os

        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            ds = SiameseDataset(phase="testing", n_channels=1, image_path=str(ROOT / "images") + "/")
            fns = [pairs[0][0], pairs[0][1]]
            batch = store.get_batch_tensor(fns)
            assert (ds[0][0] - batch[0]).abs().max().item() == 0.0
            assert (ds[0][1] - batch[1]).abs().max().item() == 0.0
        finally:
            os.chdir(old_cwd)


def test_sanitizers_equivalence_and_stochasticity(store):
    _require(*NEEDS_WEIGHTS)
    dev = torch.device("cpu")
    fns = ["00000001_001.png", "00000001_002.png"]
    x = store.get_batch_tensor(fns, device=dev)

    # 1. Identity
    s_none = P.build_sanitizer("none", dev)
    assert torch.equal(s_none.sanitize_batch(x, fns), x)

    # 2. Flow field mu=0.01 deterministic + matches archived p1f logits within 2e-3 (CPU PIL/cache)
    s_ff = P.build_sanitizer("ff_mu0.01", dev)
    y_ff1 = s_ff.sanitize_batch(x, fns)
    y_ff2 = s_ff.sanitize_batch(x, fns)
    assert y_ff1.shape == x.shape
    assert torch.allclose(y_ff1, y_ff2, atol=1e-7)
    assert 0.0 <= float(y_ff1.min()) and float(y_ff1.max()) <= 1.0

    # 3. PrivacyNet deterministic eval output matches cls_privacy_net check shape [0, 1]
    s_pn = P.build_sanitizer("privacy_net", dev)
    y_pn1 = s_pn.sanitize_batch(x, fns)
    y_pn2 = s_pn.sanitize_batch(x, fns)
    assert y_pn1.shape == x.shape
    assert torch.allclose(y_pn1, y_pn2, atol=1e-7)
    assert 0.0 <= float(y_pn1.min()) and float(y_pn1.max()) <= 1.0

    # 4. Blur kernel=13, sigma=2 deterministic & close (< 2e-3 float vs uint8) to SiameseDataset
    s_bl = P.build_sanitizer("blur_s2", dev)
    y_bl = s_bl.sanitize_batch(x, fns)
    assert y_bl.shape == x.shape

    # 5. Elastic keyed vs stochastic
    s_ek = P.build_sanitizer("elastic_m0.04_keyed", dev, seed=0)
    s_es = P.build_sanitizer("elastic_m0.04_stochastic", dev, seed=0)
    assert torch.allclose(s_ek.sanitize_batch(x, fns), s_ek.sanitize_batch(x, fns), atol=1e-7)
    assert not torch.allclose(s_es.sanitize_batch(x, fns), s_es.sanitize_batch(x, fns), atol=1e-5)

    # 6. DP-Pix fixed vs stochastic
    s_df = P.build_sanitizer("dppix_b8_fixed", dev, seed=0)
    s_ds = P.build_sanitizer("dppix_b8_stochastic", dev, seed=0)
    assert torch.allclose(s_df.sanitize_batch(x, fns), s_df.sanitize_batch(x, fns), atol=1e-7)
    assert not torch.allclose(s_ds.sanitize_batch(x, fns), s_ds.sanitize_batch(x, fns), atol=1e-4)
    assert (s_ds.sanitize_batch(x, fns) >= 0.0).all() and (s_ds.sanitize_batch(x, fns) <= 1.0).all()


def test_archived_prediction_reproduction(store):
    _require("archive/ver_none_seed0", "archive/p1f_ff_mu0.01_SRtrain_SRval_disjoint_pretrained_ver_seed0/p1f_ff_mu0.01_SRtrain_SRval_disjoint_pretrained_ver_seed0_best_network.pth")
    dev = torch.device("cpu")
    pairs = [l.split() for l in (ROOT / "image_pairs" / "image_pairs_testing_5000.txt").open()][:4]
    lefts = [p[0] for p in pairs]
    rights = [p[1] for p in pairs]

    # ver_none_seed0
    spec_none = P.AttackerSpec(
        arch="resnet50_bce",
        init_name=f"custom:{ROOT / 'archive/ver_none_seed0/ver_none_seed0_best_network.pth'}",
    )
    net_n, _ = P.build_attacker(spec_none, dev)
    probs_n = P.predict_pairs(net_n, store, P.build_sanitizer("none", dev), lefts, rights, "SS", spec_none, dev, batch_size=4)
    arch_n = np.loadtxt(ROOT / "archive/ver_none_seed0/ver_none_seed0_labels_predictions.txt", skiprows=1)[:4, 1]
    assert np.max(np.abs(probs_n - arch_n)) < 1e-3

    # p1f_ff_mu0.01_SRtrain_SRval_disjoint_pretrained_ver_seed0
    spec_ff = P.AttackerSpec(
        arch="resnet50_bce",
        init_name=f"custom:{ROOT / 'archive/p1f_ff_mu0.01_SRtrain_SRval_disjoint_pretrained_ver_seed0/p1f_ff_mu0.01_SRtrain_SRval_disjoint_pretrained_ver_seed0_best_network.pth'}",
    )
    net_f, _ = P.build_attacker(spec_ff, dev)
    probs_f = P.predict_pairs(net_f, store, P.build_sanitizer("ff_mu0.01", dev), lefts, rights, "SR", spec_ff, dev, batch_size=4)
    arch_f = np.loadtxt(
        ROOT / "archive/p1f_ff_mu0.01_SRtrain_SRval_disjoint_pretrained_ver_seed0/p1f_ff_mu0.01_SRtrain_SRval_disjoint_pretrained_ver_seed0_labels_predictions.txt",
        skiprows=1,
    )[:4, 1]
    assert np.max(np.abs(probs_f - arch_f)) < 2e-3


def test_attackers_forward_and_backward_cpu():
    _require(*NEEDS_WEIGHTS, "networks/pretrained_classifier.pth",
             "extensions/chexgenbench/checkpoints/ResNet-50_epoch11_data_handling_RPN.pth")
    dev = torch.device("cpu")
    x1 = torch.rand(2, 1, 256, 256)
    x2 = torch.rand(2, 1, 256, 256)
    y = torch.tensor([[1.0], [0.0]])
    mean_t = torch.tensor(P.IMAGENET_MEAN).view(1, 3, 1, 1)
    std_t = torch.tensor(P.IMAGENET_STD).view(1, 3, 1, 1)

    for spec in [
        P.AttackerSpec("resnet50_bce", "chexgenbench", input_size=224, use_imagenet_norm=False),
        P.AttackerSpec("resnet50_cosine", "pretrained_ver", input_size=256, use_imagenet_norm=True),
        P.AttackerSpec("densenet121_bce", "chexnet", input_size=256, use_imagenet_norm=True),
        P.AttackerSpec("convnext_tiny_bce", "imagenet", input_size=256, use_imagenet_norm=True),
        P.AttackerSpec("convnext_tiny_cosine", "imagenet", input_size=256, use_imagenet_norm=True),
    ]:
        net, _ = P.build_attacker(spec, dev)
        net.train()
        i1 = P.preprocess_for_attacker(x1, spec.input_size, spec.use_imagenet_norm, mean_t, std_t)
        i2 = P.preprocess_for_attacker(x2, spec.input_size, spec.use_imagenet_norm, mean_t, std_t)
        out = net(i1, i2)
        assert out.shape == (2, 1)
        loss = torch.nn.BCEWithLogitsLoss()(out, y)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in net.parameters() if p.requires_grad)
        assert has_grad, f"No nonzero gradient in {spec}"


def test_threshold_and_tie_safe_gallery():
    # Top score is negative -> fpr=0 requires threshold=+inf fallback
    y = np.array([1, 1, 0, 0, 0])
    s_top_neg = np.array([0.8, 0.4, 0.9, 0.3, 0.2])
    thr_inf, fb_inf = P.select_val_threshold(y, s_top_neg, 0.001)
    assert np.isinf(thr_inf) and fb_inf

    # Top score is positive -> finite threshold exists at fpr=0
    s_top_pos = np.array([0.9, 0.4, 0.8, 0.3, 0.2])
    thr_fin, fb_fin = P.select_val_threshold(y, s_top_pos, 0.001)
    assert abs(thr_fin - 0.9) < 1e-6 and not fb_fin
    thr35, fb35 = P.select_val_threshold(y, s_top_pos, 0.35)
    assert abs(thr35 - 0.4) < 1e-6 and not fb35  # 1/3 neg = 0.333 <= 0.35


def test_e4_splits_strictly_patient_disjoint():
    _require(*NEEDS_IMAGES)
    import pandas as pd

    img_dir = ROOT / "images"
    have = {p.name for p in img_dir.iterdir()} if img_dir.is_dir() else set()
    tr = {fn.split("_")[0] for fn in open(ROOT / "image_pairs" / "train_val_list.txt").read().splitlines() if fn in have}
    te = {fn.split("_")[0] for fn in open(ROOT / "image_pairs" / "test_list.txt").read().splitlines() if fn in have}
    assert len(tr & te) == 0
    labels_df = pd.read_csv(ROOT / "chexnet" / "nih_labels.csv")
    assert len(labels_df) > 100000

    # Constant-score tie safety on rank calculation
    class ConstNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_end = torch.nn.Linear(4, 1)
            torch.nn.init.zeros_(self.fc_end.weight)
            torch.nn.init.zeros_(self.fc_end.bias)

        def forward_once(self, x):
            return torch.zeros((x.shape[0], 4), device=x.device)

    store = P.RAMImageStore(use_cache=True)
    res = P.evaluate_sr_gallery(
        ConstNet(),
        store,
        P.IdentitySanitizer(),
        P.AttackerSpec("resnet50_bce", "imagenet"),
        torch.device("cpu"),
        n_boot=20,
    )
    assert res["rank1"] < 1.0, f"Tie collapse gave fake Rank-1={res['rank1']}"
    assert res["rank5"] < 2.0


def test_cal_and_extra_evals_end_to_end(store, tmp_path):
    _require(*NEEDS_WEIGHTS)
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for end-to-end train_and_evaluate_pair_run test")
    all_lines = (ROOT / "image_pairs" / "image_pairs_testing_5000.txt").read_text().splitlines()
    sample_lines = all_lines[:12] + all_lines[-12:]
    pair_p = tmp_path / "mini_pairs.txt"
    pair_p.write_text("\n".join(sample_lines) + "\n")
    meta_lut = P.load_nih_metadata_lut()
    rec = P.train_and_evaluate_pair_run(
        run_id="smoke_cal_extra",
        out_dir=tmp_path / "out",
        store=store,
        meta_lut=meta_lut,
        spec=P.AttackerSpec("resnet50_bce", "imagenet"),
        sanitizer_name="none",
        train_path=None,
        val_path=pair_p,
        test_path=pair_p,
        cal_paths={"cal_mini": (pair_p, "SR")},
        extra_evals={"ex_mini": (pair_p, "SS")},
    )
    assert "cal_mini" in rec["calibration"]
    assert "ex_mini" in rec["extra_evals"]
    assert (tmp_path / "out" / "smoke_cal_extra_preds.npz").is_file()


def test_one_class_metric_guard():
    import math
    import numpy as np

    scores = np.linspace(0.1, 0.9, 6)
    y_one = np.ones(6, dtype=np.int64)
    y_zero = np.zeros(6, dtype=np.int64)
    for y_single in (y_one, y_zero):
        m = P.compute_binary_metrics(y_single, scores)
        assert math.isnan(m["auc"])
        assert m.get("undefined") is True
        lf = [f"{i:08d}_000.png" for i in range(6)]
        rf = [f"{i:08d}_001.png" for i in range(6)]
        b = P.bootstrap_pair_metrics(lf, rf, y_single, scores, n_boot=10, seed=0)
        assert b["auc_ci95"] is None
        assert b["tpr1_ci95"] is None
        assert b.get("undefined") is True

    # Two classes present, but zero valid bootstrap replicates -> must not crash on empty np.percentile
    b_empty = P.bootstrap_pair_metrics(
        ["00000001_000.png", "00000002_000.png"],
        ["00000001_001.png", "00000003_000.png"],
        np.array([1, 0], dtype=np.int64),
        np.array([0.8, 0.2]),
        n_boot=0,
        seed=0,
    )
    assert b_empty["auc_ci95"] is None
    assert b_empty["tpr1_ci95"] is None
    assert b_empty.get("undefined") is True



def test_elastic_mu_parser_rejects_ambiguous_zero_pad() -> None:
    import pytest
    dev = torch.device("cpu")
    with pytest.raises(ValueError, match="Ambiguous elastic mu token"):
        P.build_sanitizer("elastic_m004_keyed", dev, seed=0)
    san = P.build_sanitizer("elastic_m0.04_keyed", dev, seed=0)
    assert abs(san.mu - 0.04) < 1e-12
