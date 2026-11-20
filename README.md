# cxr-verifier-audit

Code, split and pair lists, and pair-level scores for the manuscript

> **How verifier choice and training pairs affect estimates of identity leakage in sanitized chest radiographs**
> Tuan Minh Tran, Quan Duc Tran, Quan Dinh Nguyen, Hoai Nam Vu (2026)

The study audits the PriCheXy-Net sanitizer on NIH ChestX-ray14. It varies the pairing used to train a patient-verification network, compares ten verifier configurations on common test pairs, and compares learned deformation with blur and random elastic deformation at similar diagnostic utility.

Every number in the manuscript and its supplement can be regenerated from the files in this repository **without a GPU and without the images**. See [Reproduce the tables and figures](#reproduce-the-tables-and-figures). Re-running the trainings requires the images and model weights described in [Re-run the experiments](#re-run-the-experiments).

## Repository layout

| Path | Contents |
|------|----------|
| `scripts/prl_lib.py` | Sanitizers (learned field, Privacy-Net, blur, keyed elastic, DP-Pix), verifier architectures and recipes, training loop, metrics, calibration and identification. |
| `scripts/e1_build_clean_splits.py` | Builds the de-duplicated, patient-disjoint verifier splits in `results/clean_splits/`. |
| `scripts/run_e1.py`, `scripts/run_prl_final.py` | Verifier trainings of the main comparison (37 conditions × 3 seeds = 111 trainings). |
| `scripts/run_e4.py` | Diagnostic utility: frozen classifier and classifiers retrained on sanitized images. |
| `artifacts/paper_upgrade/phase5_pairing_audit/` | Training runner and cell definitions of the 2×2 pairing factorial. |
| `tools/` | Analyses: factorial bootstrap, paired patient-clustered intervals, tables, supplement tables and the frontier figure. |
| `image_pairs/` | Released PriCheXy-Net pair lists and official NIH split, plus the calibration list `image_pairs_validation_disjoint_700.txt`. |
| `results/clean_splits/` | Verifier training, local-verifier pretraining and auxiliary lists, with a manifest of sizes and hashes. |
| `results/e1/runs/`, `results/prl_final/runs/` | Per-run summaries (`*_summary.json`) and pair-level scores (`*_preds.npz`) of the main comparison. |
| `archive/p1_*`, `archive/p1f_*` | Pair-level test scores of the pairing factorial (`*_labels_predictions.txt`). |
| `results/e4/` | Utility summaries and per-image predictions of the frozen classifier. |
| `results/prl_final/*.json`, `artifacts/factorial_results.json` | Bootstrap results used in the manuscript. |
| `agents/`, `datasets/`, `networks/`, `utils/`, `chexnet/`, `config_files/` | PriCheXy-Net code at commit `29245d1`, with the patches listed in [UPSTREAM.md](UPSTREAM.md). |
| `patches/prichexy-net_29245d1.diff` | Every change to the upstream files. |

## Installation

Tested with Python 3.13.5 on Linux with one NVIDIA RTX 4090.

```bash
pip install -r requirements.txt
```

## Reproduce the tables and figures

These steps read only the released scores. They take about five minutes on a CPU.

```bash
python3 tools/analyze_factorial.py      # pairing factorial -> artifacts/factorial_results.json
python3 tools/prl_paired_ci.py          # learned field vs controls -> results/prl_final/paired_ci.json
python3 tools/prl_rank_reversal_ci.py   # exploratory verifier contrast -> results/prl_final/rank_reversal_ci.json
python3 tools/prl_tables.py             # main tables and every number -> outputs/generated/
python3 tools/prl_supplement.py         # supplementary tables -> outputs/generated/
python3 tools/prl_figures.py            # privacy-utility figure -> outputs/figures/
```

The bootstrap scripts use fixed random seeds, so the first three commands rewrite the released JSON files with identical content.

| Manuscript item | Output |
|-----------------|--------|
| Table 1 (pairing factorial) | `outputs/generated/tab_factorial.tex` |
| Table 2 (ten verifier configurations) | `outputs/generated/tab_construction.tex` (the manuscript omits its last column) |
| Table 3 (diagnostic utility) | `outputs/generated/tab_utility.tex` |
| Fig. 2 (leakage vs utility) | `outputs/figures/fig_frontier.pdf` |
| Every number in the text | `outputs/generated/numbers.tex` |
| Tables S1–S6 | `supp_exposure.tex`, `supp_cells.tex`, `supp_calibrated.tex`, `supp_factorial.tex`, `supp_paired.tex`, `supp_rho.tex` in `outputs/generated/` |
| Table S7 | `results/prl_final/rank_reversal_ci.json` |

The typeset tables in the manuscript differ from these files only in row labels and minus signs.

## Pair-level scores

Each pair list has one pair per line: `image_A image_B label`. Image A is the query, which is sanitized at test time; image B is the raw reference. File names start with the NIH patient ID.

- `results/{e1,prl_final}/runs/<run_id>_preds.npz` holds verifier scores in the order of the corresponding list:
  `test_y`, `test_p` for `image_pairs_testing_5000.txt`; `val_y`, `val_p` for `image_pairs_validation_2000.txt`;
  `cal_disjoint700_SR_y`, `cal_disjoint700_SR_p` for `image_pairs_validation_disjoint_700.txt`.
- `<run_id>_summary.json` records the initialization, recipe, sanitizer, seed, learning rate, selected epoch, SHA-256 of the initialization and sanitizer weights, test metrics, calibrated operating points and closed-set identification.
- `archive/<cell>/<cell>_labels_predictions.txt` holds `Label`, `Prediction` and `PredictionThresholded` for the 5,000 test pairs of one factorial cell. Cell names follow `artifacts/paper_upgrade/phase5_pairing_audit/pairing_resolver.py:cell_id`.
- `results/e4/frozen/u_frozen_<sanitizer>_test_preds.npz` holds `te_probs`, the frozen classifier's probabilities for the 14 findings on the 25,596 official test images, in the row order of `chexnet/nih_labels.csv` restricted to `image_pairs/test_list.txt`.

Run IDs encode the protocol, for example
`frontier_ver__resnet50_bce__pretrained_ver__ff_mu0.01__att_train_clean__pat5__lr0.0001__s0`:
group, architecture and recipe, initialization, sanitizer, training list, patience, learning rate and seed.
Initializations are `imagenet`, `vstar_clean` (local verifier), `chexgenbench` (MIMIC-CXR verifier) and `pretrained_ver` (reference verifier released with PriCheXy-Net).

## Re-run the experiments

### Data

1. Download NIH ChestX-ray14 from <https://nihcc.app.box.com/v/ChestXray-NIHCC> and put the PNG files in `images/`.
2. Build the image cache used by `scripts/prl_lib.py`:
   ```bash
   python3 build_image_cache.py
   ```

### Model weights

Weights are not redistributed here. Put them at the paths below and check their SHA-256.

| File | Source | SHA-256 |
|------|--------|---------|
| `networks/pretrained_verification_model.pth` | [PriCheXy-Net](https://github.com/kaipackhaeuser/PriCheXy-Net) (Git LFS) | `331efaed0c0433c69941ddc003a14a936c688d94fd4ecfbefd34e53bfa7c051a` |
| `networks/pretrained_classifier.pth` | PriCheXy-Net | `8ad15b38286f734ea135394ac5e7c79f4a6c1d2db4d563fbe1f81cf3dbe5e663` |
| `networks/generator_lowest_total_loss_mu_0.001.pth` | PriCheXy-Net | `2452f102d6ec75853c9f71a552175ab1672c6c3e72c6a6e8bb6b07b73ca63ead` |
| `networks/generator_lowest_total_loss_mu_0.005.pth` | PriCheXy-Net | `4d368637426e0c33606090d37d3338e472ff0f6a64a5791b9f22e25c3aceae3b` |
| `networks/generator_lowest_total_loss_mu_0.01.pth` | PriCheXy-Net | `4d82dcdd1c1b5856d6361fd08b7a6838b044ffc7db89e8bf953a6279cb3bf153` |
| `networks/generator_lowest_total_loss_privacy_net.pth` | PriCheXy-Net | `50766e0607711f101fa25f0d1fe1858933683727c84a49919614ff1f4ca3ee4d` |
| `extensions/chexgenbench/checkpoints/ResNet-50_epoch11_data_handling_RPN.pth` | [raman07/Patient_ReIdentification_MIMIC](https://huggingface.co/raman07/Patient_ReIdentification_MIMIC) (CheXGenBench) | `6f12e8ff138611e4d70b4df509c7a4d7a634361cd8491304bff452bab66defa6` |

```bash
git clone https://github.com/kaipackhaeuser/PriCheXy-Net /tmp/prichexy && (cd /tmp/prichexy && git lfs pull)
cp /tmp/prichexy/networks/*.pth networks/
huggingface-cli download raman07/Patient_ReIdentification_MIMIC ResNet-50_epoch11_data_handling_RPN.pth \
    --local-dir extensions/chexgenbench/checkpoints
```

The local verifier (`results/e1/vstar_clean_best.pth`) is trained by the first step of `scripts/run_e1.py`.

### Trainings

Run every command from the repository root. Finished runs are skipped on restart, so each command can be interrupted and resumed.

```bash
python3 tools/build_disjoint_validation.py           # calibration list (already included)
python3 scripts/e1_build_clean_splits.py              # clean splits (already included)
python3 -u scripts/run_e1.py --pruned --seeds 0,1,2   # local verifier, initialization comparison
python3 -u scripts/run_prl_final.py --seeds 0,1,2     # recipe ablations, frontier, controls, ConvNeXt-T, learning-rate check
python3 -u scripts/run_e4.py --seeds 0,1,2 --sanitizers none,ff_mu0.01,blur_s2,elastic_m0.04_keyed   # utility

# Pairing factorial (reference-verifier initialization, released recipe and training list):
# 66 cells = learned field x 5 seeds + blur and elastic x 3 seeds, each with 4 pairing cells and 2 disjoint-validation cells
python3 artifacts/paper_upgrade/phase5_pairing_audit/run_pairing_audit.py --config all --design factorial
```

One verifier training takes 15–40 minutes on an RTX 4090. Results can differ slightly from the released scores across GPUs and library versions.

## Tests

```bash
python3 -m pytest -q scripts/test_prl_lib.py
```

Tests that need the images or the weights are skipped when those files are absent.

## Data terms and ethics

ChestX-ray14 is provided by the NIH Clinical Center; please follow its terms of use and cite Wang et al. (2017). This repository contains no images and no sanitized images. The pair lists contain only file names. Re-identification experiments measure statistical linkage between images; they do not identify individuals.

## License and acknowledgements

This repository builds on [PriCheXy-Net](https://github.com/kaipackhaeuser/PriCheXy-Net) (Packhäuser et al., MICCAI 2023), released under the GNU General Public License v3.0, and is distributed under the same license (see [LICENSE](LICENSE)). The classifier code in `chexnet/` keeps its original license (`chexnet/LICENSE`). The MIMIC-CXR verifier is from CheXGenBench (Dutt et al., 2026).
