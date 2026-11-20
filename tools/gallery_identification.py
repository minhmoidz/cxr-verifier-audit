"""Closed-set gallery identification, following PROTOCOL_ADDENDUM.md section A.

The paper's existing task is pairwise verification: given two records, same patient or not.
A re-identification risk claim is usually read as something stronger -- given a released record,
which patient is it? This measures that directly, on a 1,073-identity closed-set gallery built
from the shipped test split, using the attacker checkpoints that already exist. No training.

Scoring uses the attacker's own head, fc_end(|e_q - e_r|), exactly as utils.test_snn scores a
pair. Using cosine similarity instead would measure a different attacker than the one audited.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.SiameseNetwork import SiameseNetwork
from networks.UNet_PriCheXyNet import UNet
from utils.GaussianSmoothing import GaussianSmoothing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images")
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
RESIZE = transforms.Resize((256, 256))
TT = transforms.ToTensor()


def gallery_identities():
    """Test patients with >= 2 images on disk: one reference, one query."""
    pairs = [l.split() for l in open(os.path.join(ROOT, "image_pairs/image_pairs_testing_5000.txt"))]
    test_pat = {f.split("_")[0] for p in pairs for f in (p[0], p[1])}
    by = {}
    for f in sorted(os.listdir(IMG)):
        m = re.match(r"(\d{8})_\d+\.png$", f)
        if m and m.group(1) in test_pat:
            by.setdefault(m.group(1), []).append(f)
    return {p: v[:2] for p, v in sorted(by.items()) if len(v) >= 2}


def load(fname):
    with open(os.path.join(IMG, fname), "rb") as fh:
        return TT(RESIZE(Image.open(fh).convert("L")))


class Deform:
    """The mu-scaled, Gaussian-smoothed flow field, as utils.test_snn applies it."""

    def __init__(self, model_file, mu):
        self.mu = mu
        self.net = UNet(1, 2, 32).cuda()
        self.net.load_state_dict(torch.load(model_file))
        self.net.eval()
        d = torch.linspace(-1, 1, 256)
        mx, my = torch.meshgrid((d, d), indexing="ij")
        self.grid_id = torch.stack((my, mx), 2).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        self.gauss = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).cuda()

    @torch.no_grad()
    def __call__(self, x):
        g = self.grid_id - self.mu * self.net(x)
        g = self.gauss(g).permute(0, 2, 3, 1)
        return torch.nn.functional.grid_sample(x, g, padding_mode="border", align_corners=True)


@torch.no_grad()
def embed(net, files, sanitize=None, batch=16):
    out = []
    for i in range(0, len(files), batch):
        x = torch.stack([load(f) for f in files[i:i + batch]]).cuda()
        if sanitize is not None:
            x = sanitize(x)
        out.append(net.forward_once(NORM(x.expand(-1, 3, -1, -1))).cpu())
    return torch.cat(out)


@torch.no_grad()
def ranks(net, q, r, chunk=64):
    """Rank of each query's true reference, scored with the attacker's own head."""
    out = []
    for i in range(0, len(q), chunk):
        qq = q[i:i + chunk].cuda()
        s = torch.stack([net.fc_end(torch.abs(v.unsqueeze(0) - r.cuda())).squeeze(1)
                         for v in qq])                      # (chunk, n_gallery)
        for j in range(len(qq)):
            order = torch.argsort(s[j], descending=True)
            out.append(int((order == (i + j)).nonzero()[0, 0]) + 1)
    return np.array(out)


def boot(rk, k, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = [(rk[rng.integers(0, len(rk), len(rk))] <= k).mean() * 100 for _ in range(n)]
    return [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--conditions", default="",
                    help="comma-separated subset; default is every deformation level")
    ap.add_argument("--out", default=os.path.join(ROOT, "sanitizer_audit", "gallery.json"))
    a = ap.parse_args()

    ids = gallery_identities()
    refs = [v[0] for v in ids.values()]
    qrys = [v[1] for v in ids.values()]
    print(f"gallery: {len(ids)} identities, one real reference and one query each")

    G = lambda name: os.path.join(ROOT, f"networks/generator_lowest_total_loss_{name}.pth")
    ALL = {
        "unprotected": ("none", "e11_none", None),
        "mu=0.001": ("ff_mu0.001", "e11_ff_mu0.001", (G("mu_0.001"), 0.001)),
        "mu=0.005": ("ff_mu0.005", "e11_ff_mu0.005", (G("mu_0.005"), 0.005)),
        "mu=0.01": ("ff_mu0.01", "e11_ff_mu0.01", (G("mu_0.01"), 0.01)),
        # Utility-matched comparison. Blur is applied directly, not through a flow field.
        "blur_s2": ("blur_s2", "e11_blur_s2", "blur2"),
    }
    want = a.conditions.split(",") if a.conditions else list(ALL)
    CONDS = [(c, *ALL[c]) for c in want if c in ALL]

    res = json.load(open(a.out)) if os.path.isfile(a.out) else {}
    for cond, naive_tag, adapt_tag, san in CONDS:
        if san == "blur2":
            import torchvision.transforms as _T
            sanitize = _T.GaussianBlur(kernel_size=13, sigma=2.0)   # 2*ceil(3*2)+1 = 13
        elif san:
            sanitize = Deform(*san)
        else:
            sanitize = None
        for arm, tag in (("naive", naive_tag), ("adapted", adapt_tag)):
            r1, r5, ci1, ci5, used = [], [], [], [], []
            for sd in a.seeds.split(","):
                ck = glob.glob(os.path.join(ROOT, f"archive/ver_{tag}_seed{sd}/*_best_network.pth"))
                if not ck:
                    print(f"  MISSING checkpoint: {tag} seed {sd}")
                    continue
                net = SiameseNetwork().cuda()
                net.load_state_dict(torch.load(ck[0]))
                net.eval()
                R = embed(net, refs)                       # references are never sanitized
                Q = embed(net, qrys, sanitize)
                rk = ranks(net, Q, R)
                r1.append((rk <= 1).mean() * 100); r5.append((rk <= 5).mean() * 100)
                ci1.append(boot(rk, 1)); ci5.append(boot(rk, 5))
                used.append(sd)
                del net; torch.cuda.empty_cache()
            if not r1:
                continue
            m = lambda v: round(float(np.mean(v)), 2)
            sd_ = lambda v: round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 2)
            res[f"{cond} / {arm}"] = dict(
                rank1=m(r1), rank1_sd=sd_(r1), rank5=m(r5), rank5_sd=sd_(r5),
                rank1_ci=[m([c[0] for c in ci1]), m([c[1] for c in ci1])],
                rank5_ci=[m([c[0] for c in ci5]), m([c[1] for c in ci5])],
                seeds=used, n_identities=len(ids))
            r = res[f"{cond} / {arm}"]
            print(f"  {cond:12s} {arm:8s} rank-1 {r['rank1']:6.2f} +-{r['rank1_sd']:4.2f} "
                  f"CI {r['rank1_ci']}   rank-5 {r['rank5']:6.2f} +-{r['rank5_sd']:4.2f}")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
