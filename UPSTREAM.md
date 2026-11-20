# Relation to PriCheXy-Net

The directories `agents/`, `chexnet/`, `config_files/`, `datasets/`, `networks/` and `utils/`, the scripts `eval_classifier.py`, `retrain_SNN.py`, `pretrain_generator.py` and `train_architecture.py`, the pair lists `image_pairs/image_pairs_{training_10000,validation_2000,testing_5000}.txt`, the NIH lists `image_pairs/{train_val_list,test_list}.txt`, `chexnet/nih_labels.csv` and `Data_Entry_2017_v2020.csv` come from [kaipackhaeuser/PriCheXy-Net](https://github.com/kaipackhaeuser/PriCheXy-Net) at commit `29245d1` (GPL-3.0). The data files and lists are byte-identical to that commit.

`patches/prichexy-net_29245d1.diff` contains every change to the upstream source files. In summary:

| File | Change |
|------|--------|
| `utils/utils.py` | Adds Gaussian blur and keyed random elastic deformation as sanitizers; `dp_pix` follows the input tensor's device (the released code crashed when DP-Pix was applied on the CPU while building verification datasets); training and validation pairs can be sanitized–sanitized (SS, released behaviour) or sanitized–raw (SR). |
| `datasets/SiameseDataset.py`, `datasets/Dataset.py` | Blur and elastic sanitizers inside the dataset; SS or SR pairing for training and validation; the training pair list can be set with `TRAIN_PAIRS_FILE`. |
| `agents/AgentSiameseNetwork.py` | Verifier initialization from ImageNet (released behaviour), the released reference verifier or a local verifier; separate SS/SR pairing for training and for checkpoint selection; blur and elastic controls. The constructor still seeds with 42, so the runners set each run's seed after constructing the agent. |
| `networks/SiameseNetwork.py` | Optional construction without downloading ImageNet weights, for loading a full checkpoint. |
| `chexnet/eval_model.py`, `chexnet/cxr_dataset.py`, `eval_classifier.py` | Frozen-classifier utility for the identity, blur and elastic controls; evaluation under `torch.no_grad()` (predictions unchanged, lower GPU memory); `torch.load(..., weights_only=False)` for current PyTorch. |
| `config_files/config_eval_classifier.json`, `config_files/config_retrainSNN.json` | Local image path and the evaluated sanitizer. |

The empty `__init__.py` files in `agents/`, `datasets/`, `networks/` and `utils/` were added so that these directories import as packages and do not clash with installed packages of the same name.

All other code (`scripts/`, `tools/`, `artifacts/`, `build_image_cache.py`) and the files in `results/`, `archive/` and `image_pairs/image_pairs_validation_disjoint_700.txt` were written for this study and are released under the same GPL-3.0 license.
