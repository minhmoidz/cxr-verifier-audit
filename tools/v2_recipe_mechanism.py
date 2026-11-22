"""T4: evidence on why the released verification head fails from ImageNet weights (supplement S10).

Two kinds of evidence, no new training:
  1. Training dynamics of the ImageNet-initialized recipes (published runs, clean protocol, raw and
     mu=0.01 images, seeds 0-2): selected epoch, epochs run, BCE loss at epoch 1 and at the
     selected epoch (training), and validation loss at epoch 1 and its minimum; test AUC.
  2. Signs of the trained head weights w (score = w . |e1 - e2| + b): fraction of negative weights,
     summed positive and negative weight, bias. A non-positive w makes the score non-increasing in
     every embedding distance. Checkpoints: the published ImageNet released-recipe runs (seed 0), the
     ImageNet-initialized released-protocol runs of E3 (results/E3/cells), the reference verifier, and
     the untrained head at the same seeds (PyTorch default initialization), for comparison.

Output: results/T4/recipe_mechanism.json
Usage:  python3 tools/v2_recipe_mechanism.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from networks.SiameseNetwork import SiameseNetwork  # noqa: E402

OUT = os.path.join(ROOT, "results", "T4", "recipe_mechanism.json")
RECIPES = {"resnet50_bce": "released", "resnet50_monohead": "head change", "resnet50_adamw_headlr": "optimizer change",
           "resnet50_metric_l1": "full recipe"}


def head_stats(w, b):
    w = np.asarray(w, float).ravel()
    return {"frac_negative": float(np.mean(w < 0)), "sum_positive": float(w[w > 0].sum()),
            "sum_negative": float(w[w < 0].sum()), "bias": float(b)}


def load_head(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        sd = sd.state_dict()
    return sd["fc_end.weight"].numpy(), float(sd["fc_end.bias"].numpy().ravel()[0])


def main():
    dyn = defaultdict(list)
    for pat in ("results/e1/runs/*_summary.json", "results/prl_final/runs/*_summary.json"):
        for f in glob.glob(os.path.join(ROOT, pat)):
            r = json.load(open(f))
            if (r.get("train_path") is None or "att_train_clean" not in r["train_path"] or r["init_name"] != "imagenet"
                    or r["arch"] not in RECIPES or float(r["lr"]) != 1e-4 or r["sanitizer"] not in ("none", "ff_mu0.01")):
                continue
            se = int(r["selected_epoch"])
            dyn[f"{RECIPES[r['arch']]}/{r['sanitizer']}"].append({
                "seed": int(r["seed"]), "selected_epoch": se, "epochs_ran": int(r["epochs_ran"]),
                "train_loss_ep1": r["train_losses"][0], "train_loss_selected": r["train_losses"][se - 1],
                "val_loss_ep1": r["val_losses"][0], "val_loss_min": float(min(r["val_losses"])),
                "test_auc": r["test_metrics"]["auc"], "run_id": r["run_id"]})
    for v in dyn.values():
        v.sort(key=lambda d: d["seed"])
    summary = {}
    for k, v in sorted(dyn.items()):
        assert [d["seed"] for d in v] == [0, 1, 2], k
        arr = lambda key: [d[key] for d in v]
        summary[k] = {"selected_epoch": arr("selected_epoch"), "epochs_ran": arr("epochs_ran"),
                      "val_loss_ep1_mean": float(np.mean(arr("val_loss_ep1"))),
                      "val_loss_min_mean": float(np.mean(arr("val_loss_min"))),
                      "train_loss_ep1_mean": float(np.mean(arr("train_loss_ep1"))),
                      "test_auc_mean": float(np.mean(arr("test_auc")))}

    heads = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "e1", "runs", "init_clean10k__imagenet__*_best.pth"))):
        heads["imagenet_released/" + os.path.basename(p).split("__")[2]] = head_stats(*load_head(p))
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "E3", "cells", "*", "*_best_network.pth"))):
        heads["e3_imagenet_released/" + os.path.basename(os.path.dirname(p))] = head_stats(*load_head(p))
    heads["reference_verifier"] = head_stats(*load_head(os.path.join(ROOT, "networks", "pretrained_verification_model.pth")))
    for s in (0, 1, 2):
        torch.manual_seed(s)
        net = SiameseNetwork(pretrained=False)
        heads[f"untrained_default_init/seed{s}"] = head_stats(net.fc_end.weight.detach().numpy(),
                                                             float(net.fc_end.bias.detach().numpy()[0]))

    def agg(prefix):
        rows = [v for k, v in heads.items() if k.startswith(prefix)]
        return {"n": len(rows), **{f"{m}_range": [min(r[m] for r in rows), max(r[m] for r in rows)]
                                   for m in ("frac_negative", "sum_positive", "sum_negative", "bias")}}

    out = {"dynamics": dyn, "dynamics_summary": summary, "heads": heads,
           "head_groups": {g: agg(g) for g in ("imagenet_released/", "e3_imagenet_released/", "reference_verifier",
                                               "untrained_default_init/")}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out["dynamics_summary"], indent=1))
    print(json.dumps(out["head_groups"], indent=1))


if __name__ == "__main__":
    main()
