"""E1/E2: embed once per image and score the extended negative set and SS test pairs.

A ResNet-50 Siamese verifier scores a pair as  logit = w . |e(x1) - e(x2)| + b, with e() the
sigmoid embedding of one branch (networks/SiameseNetwork.py). Each image is therefore embedded
once and every pair is scored from embeddings, in float64, on the logit scale (no sigmoid, so
strong verifiers produce no saturation ties at very low FPR).

Pair sets, all built from the released test list (testing_5000) and its 1,307 patients:
  test5k SR / SS  the 5,000 released test pairs; query sanitized; reference raw (SR) or also
                  sanitized (SS, E2). SR reproduces the published scores (checked in the output).
  gallery         one sanitized query and one raw reference for each of the 1,073 test patients
                  with two images (tools/gallery_identification.py). Negatives = all
                  cross-patient query-reference combinations, 1,073 x 1,072 = 1,150,256 (E1).
                  Positives for the extended set are the 2,500 released positive pairs.
                  The same with the reference also sanitized gives the SS extended set (E2).

Per model, results/E1E2/scores/<model_id>.npz stores: the 5,000 SR and SS logits, the float32
gallery embeddings (query sanitized, reference raw, reference sanitized) and the head (w, b).
The extended negatives are recomputed from these by tools/v2_extended_ci.py, exactly.

Model list: results/E1E2/models.json (written by --build-list from checkpoints on disk).
Usage: python3 tools/v2_embed_score.py --build-list
       python3 tools/v2_embed_score.py [--only SUBSTRING]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prl_lib as P  # noqa: E402
from tools.gallery_identification import gallery_identities  # noqa: E402

OUTD = ROOT / "results" / "E1E2"
TEST_5K = ROOT / "image_pairs" / "image_pairs_testing_5000.txt"
FACT_SAN = {"ff_mu0.01": "ff_mu0.01", "blur_s2": "blur_s2", "elastic_m004": "elastic_m0.04_keyed"}


def build_list():
    models = []
    # Pairing factorial, reference initialization, original validation list (four main cells).
    for cfg, san in FACT_SAN.items():
        for d in sorted(glob.glob(str(ROOT / "archive" / f"p1*_{cfg}_*_pretrained_ver_seed*"))):
            cid = os.path.basename(d)
            m = re.match(rf"p1_{re.escape(cfg)}_sanitized_(sanitized|raw)_pretrained_ver_seed(\d+)$", cid)
            m2 = re.match(rf"p1f_{re.escape(cfg)}_(SS|SR)train_(SS|SR)val_orig_pretrained_ver_seed(\d+)$", cid)
            if m:
                tp = vp = {"sanitized": "SS", "raw": "SR"}[m.group(1)]; seed = int(m.group(2))
            elif m2:
                tp, vp, seed = m2.group(1), m2.group(2), int(m2.group(3))
            else:
                continue                                    # disjoint-validation cells are not used here
            ck = Path(d) / f"{cid}_best_network.pth"
            pred = Path(d) / f"{cid}_labels_predictions.txt"
            if ck.exists():
                models.append({"model_id": f"factorial__{cfg}__{tp}{vp}__s{seed}", "family": "factorial_ref",
                               "sanitizer": san, "train_pairing": tp, "val_pairing": vp, "seed": seed,
                               "ckpt": str(ck.relative_to(ROOT)), "published_pred": str(pred.relative_to(ROOT)),
                               "arch": "resnet50_bce", "input_size": 256, "use_norm": True})
    # E3: ImageNet initialization, released recipe (learned field).
    for d in sorted(glob.glob(str(ROOT / "results" / "E3" / "cells" / "p1*_ff_mu0.01_*_imagenet_seed*"))):
        cid = os.path.basename(d)
        m = re.match(r"p1_ff_mu0.01_sanitized_(sanitized|raw)_imagenet_seed(\d+)$", cid)
        m2 = re.match(r"p1f_ff_mu0.01_(SS|SR)train_(SS|SR)val_orig_imagenet_seed(\d+)$", cid)
        if m:
            tp = vp = {"sanitized": "SS", "raw": "SR"}[m.group(1)]; seed = int(m.group(2))
        elif m2:
            tp, vp, seed = m2.group(1), m2.group(2), int(m2.group(3))
        else:
            continue
        ck = Path(d) / f"{cid}_best_network.pth"
        if ck.exists():
            models.append({"model_id": f"e3__ff_mu0.01__{tp}{vp}__s{seed}", "family": "factorial_imagenet",
                           "sanitizer": "ff_mu0.01", "train_pairing": tp, "val_pairing": vp, "seed": seed,
                           "ckpt": str(ck.relative_to(ROOT)),
                           "published_pred": str((Path(d) / f"{cid}_labels_predictions.txt").relative_to(ROOT)),
                           "arch": "resnet50_bce", "input_size": 256, "use_norm": True})
    # Reference verifier and ImageNet full recipe at the Fig. 2 settings: published checkpoints
    # where they exist, otherwise the retrained replicates (results/R_retrain).
    from scripts.run_v2_retrain import published_runs
    for (ver, san, seed), (path, r) in sorted(published_runs().items()):
        ck = path.with_name(path.name.replace("_summary.json", "_best.pth"))
        src = "published"
        pred = path.with_name(path.name.replace("_summary.json", "_preds.npz"))
        if not ck.exists():
            rid = "retrain__" + r["run_id"]
            ck = ROOT / "results" / "R_retrain" / "runs" / f"{rid}_best.pth"
            pred = ROOT / "results" / "R_retrain" / "runs" / f"{rid}_preds.npz"
            src = "retrained"
        if ck.exists():
            models.append({"model_id": f"{ver}__{san}__s{seed}", "family": ver, "sanitizer": san, "seed": seed,
                           "source": src, "ckpt": str(ck.relative_to(ROOT)), "published_pred": str(pred.relative_to(ROOT)),
                           "arch": r["arch"], "input_size": int(r["input_size"]), "use_norm": bool(r["use_imagenet_norm"])})
    # Zero-shot released reference verifier on raw pairs (T3 anchor; no stored predictions to compare).
    models.append({"model_id": "zeroshot__reference__none__s0", "family": "zeroshot_reference", "sanitizer": "none",
                   "seed": 0, "ckpt": "networks/pretrained_verification_model.pth", "published_pred": None,
                   "arch": "resnet50_bce", "input_size": 256, "use_norm": True})
    # E5: independent strong verifier, sanitized runs (results/E5/runs).
    for f in sorted(glob.glob(str(ROOT / "results" / "E5" / "runs" / "*_summary.json"))):
        r = json.load(open(f))
        ck = Path(f.replace("_summary.json", "_best.pth"))
        if ck.exists():
            models.append({"model_id": f"e5__{r['sanitizer']}__s{r['seed']}", "family": "e5", "sanitizer": r["sanitizer"],
                           "seed": int(r["seed"]), "source": "new", "ckpt": str(ck.relative_to(ROOT)),
                           "published_pred": str(Path(f.replace("_summary.json", "_preds.npz")).relative_to(ROOT)),
                           "arch": "resnet50_bce", "input_size": 256, "use_norm": True})
    # V3-B: seeds 3 and 4 of the main arms (results/V3B/runs); same families as seeds 0-2.
    for f in sorted(glob.glob(str(ROOT / "results" / "V3B" / "runs" / "*_summary.json"))):
        r = json.load(open(f))
        ck = Path(f.replace("_summary.json", "_best.pth"))
        ver = os.path.basename(f).split("__")[1]
        if ck.exists():
            models.append({"model_id": f"{ver}__{r['sanitizer']}__s{r['seed']}", "family": ver, "sanitizer": r["sanitizer"],
                           "seed": int(r["seed"]), "source": "v3b", "ckpt": str(ck.relative_to(ROOT)),
                           "published_pred": str(Path(f.replace("_summary.json", "_preds.npz")).relative_to(ROOT)),
                           "arch": r["arch"], "input_size": int(r["input_size"]), "use_norm": bool(r["use_imagenet_norm"])})
    # V3-C: verifiers trained on the retrained generators (results/V3C/runs); family v3c_<verifier>.
    for f in sorted(glob.glob(str(ROOT / "results" / "V3C" / "runs" / "*_summary.json"))):
        r = json.load(open(f))
        ck = Path(f.replace("_summary.json", "_best.pth"))
        ver = os.path.basename(f).split("__")[1]
        if ck.exists():
            models.append({"model_id": f"v3c_{ver}__{r['sanitizer']}__s{r['seed']}", "family": f"v3c_{ver}",
                           "sanitizer": r["sanitizer"], "seed": int(r["seed"]), "source": "v3c", "ckpt": str(ck.relative_to(ROOT)),
                           "published_pred": str(Path(f.replace("_summary.json", "_preds.npz")).relative_to(ROOT)),
                           "arch": r["arch"], "input_size": int(r["input_size"]), "use_norm": bool(r["use_imagenet_norm"])})
    # V3-D: verifiers on the second learned sanitizer family (results/V3D/runs); family v3d_<verifier>, where
    # referenceSS is the reference initialization trained and selected on SS pairs.
    for f in sorted(glob.glob(str(ROOT / "results" / "V3D" / "runs" / "*_summary.json"))):
        r = json.load(open(f))
        ck = Path(f.replace("_summary.json", "_best.pth"))
        ver = os.path.basename(f).split("__")[1]
        if ck.exists():
            models.append({"model_id": f"v3d_{ver}__{r['sanitizer']}__s{r['seed']}", "family": f"v3d_{ver}",
                           "sanitizer": r["sanitizer"], "seed": int(r["seed"]), "source": "v3d", "ckpt": str(ck.relative_to(ROOT)),
                           "published_pred": str(Path(f.replace("_summary.json", "_preds.npz")).relative_to(ROOT)),
                           "arch": r["arch"], "input_size": int(r["input_size"]), "use_norm": bool(r["use_imagenet_norm"])})
    OUTD.mkdir(parents=True, exist_ok=True)
    json.dump(models, open(OUTD / "models.json", "w"), indent=1)
    from collections import Counter
    print(len(models), "models:", dict(Counter(m["family"] for m in models)))


class Blur8(P.BaseSanitizer):
    """Gaussian blur as the released dataset applies it (datasets/SiameseDataset.py): torchvision
    GaussianBlur on the 8-bit image, so the output is rounded to 8 bits. Used for the factorial cells,
    which were trained through the released code; prl_lib's GaussianBlurSanitizer blurs unrounded floats."""

    def __init__(self, sigma: float):
        import math
        from torchvision import transforms
        self.blur = transforms.GaussianBlur(kernel_size=int(2 * math.ceil(3.0 * sigma) + 1), sigma=float(sigma))
        self.name = f"blur8_s{sigma:g}"

    @torch.no_grad()
    def sanitize_batch(self, x, filenames):
        return self.blur((x * 255).round().to(torch.uint8)).to(torch.float32) / 255.0


class Elastic8(P.BaseSanitizer):
    """Random elastic deformation as the released dataset applies it: keyed warp, clamp to [0, 1], then
    ToPILImage, which truncates to 8 bits. Used for the factorial cells (see Blur8)."""

    def __init__(self, name: str, device, seed: int):
        self.inner = P.build_sanitizer(name, device, seed=seed)
        self.name = name.replace("elastic_", "elastic8_")

    @torch.no_grad()
    def sanitize_batch(self, x, filenames):
        y = self.inner.sanitize_batch(x, filenames).clamp(0, 1)
        return (y * 255).to(torch.uint8).to(torch.float32) / 255.0


def scoring_sanitizer(m, device):
    """The sanitizer each model was trained with: released-code 8-bit variants for factorial blur/elastic."""
    if m["family"].startswith("factorial") and m["sanitizer"].startswith("blur_s"):
        return Blur8(float(m["sanitizer"][len("blur_s"):]))
    if m["family"].startswith("factorial") and m["sanitizer"].startswith("elastic_m"):
        return Elastic8(m["sanitizer"], device, int(m["seed"]))
    return P.build_sanitizer(m["sanitizer"], device, seed=int(m["seed"]))


def load_published(path: Path):
    if path.suffix == ".npz":
        d = np.load(path)
        return d["test_y"].astype(float), d["test_p"].astype(np.float64)
    a = np.loadtxt(path, skiprows=1)
    return a[:, 0], a[:, 1]


@torch.no_grad()
def embed(net, store, sanitizer, fns, do_san, spec, device, bs=32):
    mean_t = torch.tensor(P.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(P.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(fns), bs):
        sub = fns[i:i + bs]
        t = store.get_batch_tensor(sub, device=device)
        if do_san:
            t = sanitizer.sanitize_batch(t, sub).to(device)
        out.append(net.forward_once(P.preprocess_for_attacker(t, spec.input_size, spec.use_imagenet_norm, mean_t, std_t)).float().cpu())
    return torch.cat(out).numpy()


def logits(w, b, A, B):
    return np.abs(A.astype(np.float64) - B.astype(np.float64)) @ w.astype(np.float64) + float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-list", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    if args.build_list:
        build_list()
        return
    models = json.load(open(OUTD / "models.json"))
    (OUTD / "scores").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    store = P.RAMImageStore(use_cache=True)
    tl, tr, ty = P.load_pair_file(TEST_5K)
    gid = gallery_identities()
    g_ref = [v[0] for v in gid.values()]
    g_qry = [v[1] for v in gid.values()]
    g_pat = list(gid.keys())
    left_u = sorted(set(tl) | set(g_qry))
    right_u = sorted(set(tr) | set(g_ref))
    store.preload(left_u + right_u)
    li = {f: i for i, f in enumerate(left_u)}
    ri = {f: i for i, f in enumerate(right_u)}
    for m in models:
        if args.only not in m["model_id"]:
            continue
        dst = OUTD / "scores" / f"{m['model_id']}.npz"
        if dst.exists():
            continue
        t0 = time.time()
        spec = P.AttackerSpec(m["arch"], f"custom:{ROOT / m['ckpt']}", m["input_size"], m["use_norm"])
        net, _ = P.build_attacker(spec, device)
        net.eval()
        san = scoring_sanitizer(m, device)
        EL = embed(net, store, san, left_u, True, spec, device)           # queries, sanitized
        ER = embed(net, store, san, right_u, False, spec, device)         # references, raw
        ERS = embed(net, store, san, right_u, True, spec, device)         # references, sanitized
        w = net.fc_end.weight.detach().cpu().numpy().reshape(-1)
        b = float(net.fc_end.bias.detach().cpu().numpy().reshape(-1)[0])
        Li = [li[f] for f in tl]; Ri = [ri[f] for f in tr]
        sr = logits(w, b, EL[Li], ER[Ri])
        ss = logits(w, b, EL[Li], ERS[Ri])
        from sklearn.metrics import roc_auc_score
        auc_new = 100 * roc_auc_score(ty, sr)
        if m["published_pred"]:
            py, pp = load_published(ROOT / m["published_pred"])
            assert np.array_equal(py, ty.astype(float))
            auc_pub = 100 * roc_auc_score(ty, pp)
        else:
            auc_pub = float("nan")
        gq = [li[f] for f in g_qry]; gr = [ri[f] for f in g_ref]
        np.savez_compressed(dst, test_y=ty.astype(np.int8), test_SR_logit=sr, test_SS_logit=ss,
                            gal_q_san=EL[gq], gal_r_raw=ER[gr], gal_r_san=ERS[gr], head_w=w, head_b=np.array([b]),
                            gal_patients=np.array(g_pat))
        meta = {**m, "scoring_sanitizer": san.name if hasattr(san, "name") else m["sanitizer"], "auc_SR_from_embeddings": auc_new, "auc_SR_published_pred": auc_pub,
                "auc_abs_diff": abs(auc_new - auc_pub) if m["published_pred"] else None, "seconds": round(time.time() - t0, 1)}
        (OUTD / "scores" / f"{m['model_id']}.json").write_text(json.dumps(meta, indent=1))
        print(f"{m['model_id']}: SR AUC {auc_new:.3f} (stored predictions {auc_pub:.3f}), "
              f"SS AUC {100 * roc_auc_score(ty, ss):.3f}  [{meta['seconds']}s]", flush=True)
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
