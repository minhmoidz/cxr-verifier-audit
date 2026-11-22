"""Shared library for confirmatory Pattern Recognition Letters (PRL) experiments E1-E5.

Provides:
- Fast RAM uint8 (256, 256) ImageStore backed by cache/images_256.npy + lazy PIL fallback
- Bit/numerically verified sanitizers (none, flow_field, privacy_net, gaussian_blur,
  elastic_keyed, elastic_stochastic, dppix_fixed, dppix_stochastic)
- Unified attacker backbones & objectives (ResNet-50 BCE/Cosine, DenseNet-121 BCE,
  ConvNeXt-Tiny BCE/Cosine, native CheXGenBench RPN ResNet-50)
- Confirmatory pair trainer with deterministic seeding, validation checkpoint selection,
  independent D_cal threshold calibration, D_test ROC/TPR/PR-AUC + subgroup metrics,
  MC multi-draw stochastic inference, and closed-set SR gallery Rank-1/Rank-5 evaluation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torchvision import models, transforms

# Prevent circular import between utils.utils and datasets.SiameseDataset.
from utils import utils  # noqa: E402
from networks.SiameseNetwork import SiameseNetwork  # noqa: E402
from networks.UNet_PriCheXyNet import UNet  # noqa: E402
from networks.UNet_PrivacyNet import Unet2D_encoder  # noqa: E402
from utils.GaussianSmoothing import GaussianSmoothing  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "images"
# 256 x 256 copies of NIH train/val images not in images/ (tools/fetch_nih_trainval_256.py, E5).
# Loading one gives the same pixels as loading the original: a 256 x 256 image is not resized again.
EXTRA_IMAGE_DIR = ROOT / "images_256_extra"
CACHE_NPY = ROOT / "cache" / "images_256.npy"
CACHE_IDX = ROOT / "cache" / "index.json"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def patient_id_from_name(filename: str) -> str:
    return filename.split("_")[0]


def study_id_from_name(filename: str) -> str:
    # For NIH CXR filenames like 00000001_001.png, study proxy is the filename stem;
    # metadata Follow-up # is in Data_Entry_2017_v2020.csv when needed.
    return filename


def load_pair_file(path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    lefts, rights, labels = [], [], []
    with Path(path).open("r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            lefts.append(parts[0])
            rights.append(parts[1])
            labels.append(float(parts[2]))
    return lefts, rights, np.asarray(labels, dtype=np.float32)


def split_patients(path: Path) -> set[str]:
    lefts, rights, _ = load_pair_file(path)
    return {patient_id_from_name(x) for x in lefts} | {patient_id_from_name(x) for x in rights}


def assert_patient_disjoint(named_splits: Dict[str, Path], allow_pairs: Tuple[Tuple[str, str], ...] = ()) -> None:
    allowed = {tuple(sorted(pair)) for pair in allow_pairs}
    pts = {k: split_patients(v) for k, v in named_splits.items()}
    names = list(named_splits.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = tuple(sorted((names[i], names[j])))
            if pair in allowed:
                continue
            overlap = pts[names[i]] & pts[names[j]]
            if overlap:
                raise ValueError(
                    f"Patient leakage detected between {names[i]} and {names[j]}: "
                    f"{len(overlap)} shared patients (e.g. {sorted(overlap)[:5]})"
                )


class RAMImageStore:
    """In-memory uint8 (256, 256) grayscale store backed by cache/images_256.npy + PIL fallback."""

    def __init__(self, image_dir: Path = IMAGE_DIR, use_cache: bool = True):
        self.image_dir = Path(image_dir)
        self.cache_idx: Dict[str, int] = {}
        self.cache_arr: Optional[np.ndarray] = None
        self.extra: Dict[str, np.ndarray] = {}
        self.san_caches: Dict[str, Dict[str, torch.Tensor]] = {}
        if use_cache and CACHE_NPY.is_file() and CACHE_IDX.is_file():
            with CACHE_IDX.open("r") as f:
                self.cache_idx = json.load(f)
            # Load full 1.94 GB array into RAM for fast zero-I/O slicing.
            self.cache_arr = np.load(CACHE_NPY)

    def _load_pil_uint8(self, filename: str) -> np.ndarray:
        path = self.image_dir / filename
        if not path.is_file() and (EXTRA_IMAGE_DIR / filename).is_file():
            path = EXTRA_IMAGE_DIR / filename
        with Image.open(path) as im:
            g = im.convert("L")
            if g.size != (256, 256):
                g = transforms.Resize((256, 256))(g)
            return np.asarray(g, dtype=np.uint8)

    def preload(self, filenames: Iterable[str], max_workers: int = 8) -> None:
        missing = sorted({fn for fn in filenames if fn not in self.cache_idx and fn not in self.extra})
        if not missing:
            return
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            arrays = list(ex.map(self._load_pil_uint8, missing))
        for fn, arr in zip(missing, arrays):
            self.extra[fn] = arr

    def get_uint8(self, filename: str) -> np.ndarray:
        if filename in self.cache_idx and self.cache_arr is not None:
            return self.cache_arr[self.cache_idx[filename]]
        if filename not in self.extra:
            self.extra[filename] = self._load_pil_uint8(filename)
        return self.extra[filename]

    def get_batch_tensor(self, filenames: Sequence[str], device: Optional[torch.device] = None) -> torch.Tensor:
        stack = np.stack([self.get_uint8(fn) for fn in filenames], axis=0)
        t = torch.from_numpy(stack).unsqueeze(1).to(dtype=torch.float32) / 255.0
        if device is not None:
            t = t.to(device, non_blocking=True)
        return t


class BaseSanitizer:
    name: str
    is_deterministic: bool = True
    checkpoint_path: Optional[Path] = None

    def sanitize_batch(self, x: torch.Tensor, filenames: Sequence[str]) -> torch.Tensor:
        raise NotImplementedError

    def to(self, device: torch.device) -> "BaseSanitizer":
        return self


class IdentitySanitizer(BaseSanitizer):
    def __init__(self):
        self.name = "none"
        self.is_deterministic = True

    def sanitize_batch(self, x: torch.Tensor, filenames: Sequence[str]) -> torch.Tensor:
        return x


class FlowFieldSanitizer(BaseSanitizer):
    def __init__(self, mu: float, ckpt_path: Path, device: torch.device):
        self.name = f"ff_mu{mu:g}"
        self.mu = float(mu)
        self.checkpoint_path = Path(ckpt_path)
        self.is_deterministic = True
        self.device = device
        self.gen = UNet(1, 2, 32).to(device).eval()
        self.gen.load_state_dict(torch.load(self.checkpoint_path, map_location=device, weights_only=True))
        for p in self.gen.parameters():
            p.requires_grad_(False)
        self.gauss = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).to(device)
        axis = torch.linspace(-1, 1, 256, device=device)
        mx, my = torch.meshgrid(axis, axis, indexing="ij")
        self.gid = torch.stack((my, mx), 2).unsqueeze(0).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def sanitize_batch(self, x: torch.Tensor, filenames: Sequence[str]) -> torch.Tensor:
        x_dev = x.to(self.device)
        grid = self.gauss(self.gid - self.mu * self.gen(x_dev)).permute(0, 2, 3, 1)
        return F.grid_sample(x_dev, grid, padding_mode="border", align_corners=True)


class PrivacyNetSanitizer(BaseSanitizer):
    def __init__(self, ckpt_path: Path, device: torch.device):
        self.name = "privacy_net"
        self.checkpoint_path = Path(ckpt_path)
        self.is_deterministic = True
        self.device = device
        self.gen = Unet2D_encoder(1, 1, 16).to(device).eval()
        self.gen.load_state_dict(torch.load(self.checkpoint_path, map_location=device, weights_only=True))
        for p in self.gen.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def sanitize_batch(self, x: torch.Tensor, filenames: Sequence[str]) -> torch.Tensor:
        return self.gen(x.to(self.device))


class GaussianBlurSanitizer(BaseSanitizer):
    def __init__(self, sigma: float):
        self.sigma = float(sigma)
        self.name = f"blur_s{self.sigma:g}"
        self.is_deterministic = True
        k = int(2 * math.ceil(3.0 * self.sigma) + 1)
        self.blur = transforms.GaussianBlur(kernel_size=k, sigma=self.sigma)

    @torch.no_grad()
    def sanitize_batch(self, x: torch.Tensor, filenames: Sequence[str]) -> torch.Tensor:
        return self.blur(x)


class ElasticSanitizer(BaseSanitizer):
    def __init__(self, mu: float, stochastic: bool = False, seed: int = 0):
        self.mu = float(mu)
        self.stochastic = stochastic
        self.is_deterministic = not stochastic
        self.name = f"elastic_m{self.mu:g}_{'stochastic' if stochastic else 'keyed'}"
        self.rng = np.random.default_rng(seed)
        self.counter = 0

    @torch.no_grad()
    def sanitize_batch(self, x: torch.Tensor, filenames: Sequence[str]) -> torch.Tensor:
        if self.stochastic:
            # Generate unique fresh keys per image draw so utils.elastic_warp draws fresh fields.
            keys = []
            for _ in filenames:
                self.counter += 1
                keys.append(f"stoch_{self.counter}_{int(self.rng.integers(0, 1 << 31))}")
        else:
            keys = list(filenames)
        return utils.elastic_warp(x, self.mu, keys)


class DPPixSanitizer(BaseSanitizer):
    def __init__(self, b: int = 8, m: int = 1, eps: float = 0.1, stochastic: bool = False, seed: int = 0):
        self.b = int(b)
        self.m = int(m)
        self.eps = float(eps)
        self.stochastic = stochastic
        self.is_deterministic = not stochastic
        # Even static per-dataset dp_pix draws once per instance, so we mark fixed vs stochastic explicitly.
        self.name = f"dppix_b{self.b}_{'stochastic' if stochastic else 'fixed'}"
        self.seed = int(seed)
        self.static_cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def sanitize_batch(self, x: torch.Tensor, filenames: Sequence[str]) -> torch.Tensor:
        out = []
        dev = x.device
        for i, fn in enumerate(filenames):
            xi = x[i : i + 1].detach().cpu()
            if not self.stochastic:
                if fn not in self.static_cache:
                    # Seed torch deterministically from seed + filename for reproducible static draw.
                    h = int.from_bytes(hashlib.sha256(f"{self.seed}:{fn}".encode()).digest()[:4], "little")
                    state = torch.random.get_rng_state()
                    torch.manual_seed(h)
                    try:
                        self.static_cache[fn] = utils.dp_pix(xi, b=self.b, m=self.m, eps=self.eps, plot=False)
                    finally:
                        torch.random.set_rng_state(state)
                out.append(self.static_cache[fn])
            else:
                out.append(utils.dp_pix(xi, b=self.b, m=self.m, eps=self.eps, plot=False))
        return torch.cat(out, dim=0).to(dev)


def build_sanitizer(name: str, device: torch.device, seed: int = 0) -> BaseSanitizer:
    if name == "none":
        return IdentitySanitizer()
    if name.startswith("ff_mu"):
        mu_str = name[len("ff_mu"):]
        ckpt = ROOT / "networks" / f"generator_lowest_total_loss_mu_{mu_str}.pth"
        return FlowFieldSanitizer(float(mu_str), ckpt, device)
    if name == "privacy_net":
        return PrivacyNetSanitizer(ROOT / "networks" / "generator_lowest_total_loss_privacy_net.pth", device)
    if name.startswith("blur_s"):
        return GaussianBlurSanitizer(float(name[len("blur_s"):]))
    if name.startswith("elastic_m"):
        m = re.match(r"^elastic_m([0-9.]+?)(?:_(keyed|stochastic))?$", name)
        if not m:
            raise ValueError(f"Unrecognized elastic sanitizer spec: {name}")
        mu_tok = m.group(1)
        if "." not in mu_tok and len(mu_tok) > 1 and mu_tok.startswith("0"):
            raise ValueError(
                f"Ambiguous elastic mu token '{mu_tok}' in '{name}'; use explicit decimal notation like 'elastic_m0.04_keyed'."
            )
        mu = float(mu_tok)
        stochastic = m.group(2) == "stochastic"
        return ElasticSanitizer(mu=mu, stochastic=stochastic, seed=seed)
    if name.startswith("dppix_b"):
        m = re.match(r"^dppix_b(\d+)(?:_(fixed|stochastic))?$", name)
        if not m:
            raise ValueError(f"Unrecognized dp_pix sanitizer spec: {name}")
        b = int(m.group(1))
        stochastic = m.group(2) == "stochastic"
        return DPPixSanitizer(b=b, m=1, eps=0.1, stochastic=stochastic, seed=seed)
    raise ValueError(f"Unknown sanitizer name: {name}")


def precompute_sanitized_map(
    store: RAMImageStore,
    sanitizer: BaseSanitizer,
    filenames: Iterable[str],
    device: torch.device,
    batch_size: int = 128,
) -> Optional[Dict[str, torch.Tensor]]:
    """Pre-sanitize deterministic filenames into CPU uint8/float32 RAM dict for ~10x faster epochs."""
    if not sanitizer.is_deterministic or sanitizer.name == "none":
        return None
    cache_key = f"{sanitizer.name}__s{getattr(sanitizer, 'seed', 0) if sanitizer.name.startswith('dppix') else 0}"
    cached = store.san_caches.get(cache_key)
    if cached is None:
        if len(store.san_caches) >= 2:
            oldest_key = next(iter(store.san_caches))
            del store.san_caches[oldest_key]
        cached = {}
        store.san_caches[cache_key] = cached
    uniq = sorted(fn for fn in set(filenames) if fn not in cached)
    if not uniq:
        return cached
    store.preload(uniq)
    with torch.no_grad():
        for i in range(0, len(uniq), batch_size):
            batch_fns = uniq[i : i + batch_size]
            xb = store.get_batch_tensor(batch_fns, device=device)
            sb = sanitizer.sanitize_batch(xb, batch_fns).detach().cpu()
            for fn, tensor in zip(batch_fns, sb):
                cached[fn] = tensor  # (1, 256, 256) float32 in [0, 1]
    return cached


class CosineSiameseNet(nn.Module):
    """Metric-learning cosine verifier supporting ResNet-50 or ConvNeXt-Tiny backbones."""

    def __init__(self, backbone_name: str, emb_dim: int = 128, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            m = models.resnet50(weights=weights)
            m.fc = nn.Linear(m.fc.in_features, emb_dim)
            self.model = m
        elif backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            m = models.convnext_tiny(weights=weights)
            in_f = m.classifier[2].in_features
            m.classifier[2] = nn.Linear(in_f, emb_dim)
            self.model = m
        else:
            raise ValueError(f"Unsupported cosine backbone: {backbone_name}")
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model(x), p=2, dim=-1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        e1 = self.forward_once(x1)
        e2 = self.forward_once(x2)
        cos = (e1 * e2).sum(dim=-1, keepdim=True)
        return self.logit_scale.exp() * cos + self.logit_bias

    def matrix_logits(self, q_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
        return self.logit_scale.exp() * (q_emb @ r_emb.T) + self.logit_bias


class BCESiameseConvNeXt(nn.Module):
    """ConvNeXt-Tiny matching PriCheXy Siamese head: sigmoid(Linear(768->128)) -> |e1-e2| -> fc_end."""

    def __init__(self, emb_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.model = models.convnext_tiny(weights=weights)
        in_f = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_f, emb_dim)
        self.fc_end = nn.Linear(emb_dim, 1)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        e1 = self.forward_once(x1)
        e2 = self.forward_once(x2)
        return self.fc_end(torch.abs(e1 - e2))


class BCESiameseDenseNet(nn.Module):
    """Clean DenseNet-121 Siamese verifier supporting CheXNet or ImageNet init."""

    def __init__(self, chexnet_ckpt: Optional[Path] = None, emb_dim: int = 128):
        super().__init__()
        if chexnet_ckpt is not None:
            raw_obj = torch.load(chexnet_ckpt, map_location="cpu", weights_only=False)
            if isinstance(raw_obj, dict) and "model" in raw_obj:
                base = raw_obj["model"]
            elif isinstance(raw_obj, dict):
                base = models.densenet121(weights=None)
                base.load_state_dict(raw_obj, strict=False)
            else:
                base = raw_obj
            self.features = base.features
        else:
            self.features = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).features
        self.proj = nn.Linear(1024, emb_dim)
        self.fc_end = nn.Linear(emb_dim, 1)
        for p in self.parameters():
            p.requires_grad_(True)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        out = F.relu(feat, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        return torch.sigmoid(self.proj(out))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.fc_end(torch.abs(self.forward_once(x1) - self.forward_once(x2)))


def _load_siamese_resnet_state(net: nn.Module, ckpt_path: Path, strict: bool = True) -> None:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net.load_state_dict(sd, strict=strict)


def resolve_init_path(init_name: str) -> Optional[Path]:
    mapping = {
        "imagenet": None,
        "pretrained_ver": ROOT / "networks" / "pretrained_verification_model.pth",
        "vstar": ROOT / "networks" / "vstar_verifier.pth",
        "vstar_clean": ROOT / "results" / "e1" / "vstar_clean_best.pth",
        "chexgenbench": ROOT / "extensions" / "chexgenbench" / "checkpoints" / "ResNet-50_epoch11_data_handling_RPN.pth",
        "chexnet": ROOT / "networks" / "pretrained_classifier.pth",
    }
    if init_name in mapping:
        p = mapping[init_name]
        if p is not None and not p.is_file():
            raise FileNotFoundError(f"Init checkpoint missing for '{init_name}': {p}")
        return p
    if init_name.startswith("custom:"):
        p = Path(init_name[len("custom:"):])
        if not p.is_file():
            raise FileNotFoundError(f"Custom init checkpoint missing: {p}")
        return p
    raise ValueError(f"Unknown init_name: {init_name}")


@dataclass
class AttackerSpec:
    arch: str          # resnet50_bce, resnet50_cosine, densenet121_bce, convnext_tiny_bce, convnext_tiny_cosine
    init_name: str     # imagenet, pretrained_ver, vstar, vstar_clean, chexgenbench, chexnet, custom:<path>
    input_size: int = 256
    use_imagenet_norm: bool = True


# Same Siamese ResNet-50 network (|e1 - e2| -> fc_end); the variants differ only in training recipe.
#   resnet50_bce           released recipe: default head init, Adam(lr) on all parameters
#   resnet50_metric_l1     monotone head (init w=-0.15, b=3 for ImageNet; w <= 0 enforced) +
#                          AdamW(wd 1e-4) with head lr = max(25*lr, 2e-3) (+ stage-0 head fit for
#                          non-ImageNet inits)
#   resnet50_monohead      ablation: monotone head only, released optimizer
#   resnet50_adamw_headlr  ablation: AdamW + boosted head lr only, default (unconstrained) head
RESNET50_SIAMESE_ARCHS = ("resnet50_bce", "resnet50_metric_l1", "resnet50_monohead",
                          "resnet50_adamw_headlr")
MONOTONE_HEAD_ARCHS = ("resnet50_metric_l1", "resnet50_monohead")
BOOSTED_OPT_ARCHS = ("resnet50_metric_l1", "resnet50_adamw_headlr")


def build_attacker(spec: AttackerSpec, device: torch.device) -> Tuple[nn.Module, Optional[Path]]:
    init_path = resolve_init_path(spec.init_name)
    if spec.arch in RESNET50_SIAMESE_ARCHS:
        if spec.init_name == "imagenet":
            net = SiameseNetwork(pretrained=True)
        else:
            net = SiameseNetwork(pretrained=False)
            assert init_path is not None
            _load_siamese_resnet_state(net, init_path, strict=True)
        if spec.arch in MONOTONE_HEAD_ARCHS:
            with torch.no_grad():
                if spec.init_name == "imagenet":
                    net.fc_end.bias.fill_(3.0)
                    net.fc_end.weight.fill_(-0.15)
                else:
                    net.fc_end.weight.clamp_(max=0.0)
    elif spec.arch == "resnet50_cosine":
        net = CosineSiameseNet("resnet50", emb_dim=128, pretrained=(spec.init_name == "imagenet"))
        if init_path is not None:
            sd = torch.load(init_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # Copy matching model.* weights (excluding fc_end) into CosineSiameseNet.
            filtered = {k: v for k, v in sd.items() if k.startswith("model.")}
            net.load_state_dict(filtered, strict=False)
    elif spec.arch == "densenet121_bce":
        if spec.init_name not in ("chexnet", "imagenet"):
            raise ValueError("densenet121_bce expects init_name 'chexnet' or 'imagenet'")
        net = BCESiameseDenseNet(chexnet_ckpt=init_path)
    elif spec.arch == "convnext_tiny_bce":
        if spec.init_name != "imagenet":
            raise ValueError("convnext_tiny_bce expects init_name 'imagenet'")
        net = BCESiameseConvNeXt(pretrained=True)
    elif spec.arch == "convnext_tiny_cosine":
        if spec.init_name != "imagenet":
            raise ValueError("convnext_tiny_cosine expects init_name 'imagenet'")
        net = CosineSiameseNet("convnext_tiny", emb_dim=128, pretrained=True)
    else:
        raise ValueError(f"Unknown arch: {spec.arch}")
    return net.to(device), init_path


def preprocess_for_attacker(
    x: torch.Tensor,
    input_size: int,
    use_imagenet_norm: bool,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
) -> torch.Tensor:
    # x: (B, 1, 256, 256) float32 on device in [0, 1]
    if x.shape[-2:] != (input_size, input_size):
        x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
    rgb = x.expand(-1, 3, -1, -1)
    if use_imagenet_norm:
        rgb = (rgb - mean_t) / std_t
    return rgb


def select_val_threshold(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> Tuple[float, bool]:
    fpr, _, thresholds = roc_curve(y_true, scores)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return float("inf"), True
    thr = float(thresholds[valid[-1]])
    fallback = bool(not np.isfinite(thr))
    return thr, fallback


def eval_threshold_op_point(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    pred = scores >= threshold
    neg = y_true == 0
    pos = y_true == 1
    fpr = float((pred & neg).sum() / max(int(neg.sum()), 1)) * 100.0
    tpr = float((pred & pos).sum() / max(int(pos.sum()), 1)) * 100.0
    return {"fpr_pct": fpr, "tpr_pct": tpr, "threshold": float(threshold)}


def compute_binary_metrics(y: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y)) < 2:
        nan = float("nan")
        return {
            "auc": nan,
            "pr_auc": nan,
            "tpr_at_1pct": nan,
            "tpr_at_0_5pct": nan,
            "tpr_at_0_1pct": nan,
            "undefined": True,
        }
    auc_val = float(roc_auc_score(y, scores)) * 100.0
    pr_auc = float(average_precision_score(y, scores)) * 100.0
    fpr, tpr, _ = roc_curve(y, scores)
    return {
        "auc": auc_val,
        "pr_auc": pr_auc,
        "tpr_at_1pct": float(np.interp(0.01, fpr, tpr)) * 100.0,
        "tpr_at_0_5pct": float(np.interp(0.005, fpr, tpr)) * 100.0,
        "tpr_at_0_1pct": float(np.interp(0.001, fpr, tpr)) * 100.0,
    }


def _bootstrap_patient_weights(rng: np.random.Generator, left_code: np.ndarray, right_code: np.ndarray, n_patients: int) -> np.ndarray:
    counts = np.bincount(rng.integers(0, n_patients, n_patients), minlength=n_patients)
    w = counts[left_code] * counts[right_code]
    same = left_code == right_code
    w[same] = counts[left_code[same]]
    return w


def bootstrap_pair_metrics(
    left_files: Sequence[str],
    right_files: Sequence[str],
    y: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, List[float]]:
    l_pts = [patient_id_from_name(x) for x in left_files]
    r_pts = [patient_id_from_name(x) for x in right_files]
    uniq = sorted(set(l_pts) | set(r_pts))
    lut = {p: i for i, p in enumerate(uniq)}
    l_code = np.asarray([lut[p] for p in l_pts], dtype=np.int32)
    r_code = np.asarray([lut[p] for p in r_pts], dtype=np.int32)
    rng = np.random.default_rng(seed)
    aucs, tprs = [], []
    if len(np.unique(y)) < 2:
        return {"auc_ci95": None, "tpr1_ci95": None, "undefined": True}
    for _ in range(n_boot):
        w = _bootstrap_patient_weights(rng, l_code, r_code, len(uniq))
        if not np.any(w[y == 0]) or not np.any(w[y == 1]):
            continue
        aucs.append(float(roc_auc_score(y, scores, sample_weight=w)) * 100.0)
        fpr, tpr, _ = roc_curve(y, scores, sample_weight=w)
        tprs.append(float(np.interp(0.01, fpr, tpr)) * 100.0)
    if not aucs:
        return {"auc_ci95": None, "tpr1_ci95": None, "undefined": True}
    return {
        "auc_ci95": [round(float(np.percentile(aucs, 2.5)), 2), round(float(np.percentile(aucs, 97.5)), 2)],
        "tpr1_ci95": [round(float(np.percentile(tprs, 2.5)), 2), round(float(np.percentile(tprs, 97.5)), 2)],
    }


def load_nih_metadata_lut() -> Dict[str, Tuple[int, str]]:
    df = pd.read_csv(ROOT / "Data_Entry_2017_v2020.csv", usecols=["Image Index", "Follow-up #", "Patient Gender"])
    return {
        row["Image Index"]: (int(row["Follow-up #"]), str(row["Patient Gender"]))
        for _, row in df.iterrows()
    }


def compute_subgroup_metrics(
    left_files: Sequence[str],
    right_files: Sequence[str],
    y: np.ndarray,
    scores: np.ndarray,
    meta_lut: Dict[str, Tuple[int, str]],
) -> Dict[str, Dict[str, float]]:
    dfu = np.array([abs(meta_lut[a][0] - meta_lut[b][0]) for a, b in zip(left_files, right_files)])
    same_sex = np.array([meta_lut[a][1] == meta_lut[b][1] for a, b in zip(left_files, right_files)])
    neg_mask = y == 0
    pos_mask = y == 1
    out: Dict[str, Dict[str, float]] = {}
    for label, pmask in [("fu1_4", pos_mask & (dfu >= 1) & (dfu <= 4)), ("fu_ge5", pos_mask & (dfu >= 5))]:
        sub = pmask | neg_mask
        if pmask.sum() > 0:
            m = compute_binary_metrics(y[sub], scores[sub])
            m["n_pos"] = int(pmask.sum())
            m["n_neg"] = int(neg_mask.sum())
            out[label] = m
    sex_mask = same_sex & (pos_mask | neg_mask)
    if (same_sex & neg_mask).sum() > 0 and (same_sex & pos_mask).sum() > 0:
        m = compute_binary_metrics(y[sex_mask], scores[sex_mask])
        m["n_pos"] = int((same_sex & pos_mask).sum())
        m["n_neg"] = int((same_sex & neg_mask).sum())
        out["same_sex"] = m
    return out


def pairing_flags(pairing: str) -> Tuple[bool, bool]:
    mapping = {
        "SS": (True, True),
        "sanitized_sanitized": (True, True),
        "SR": (True, False),
        "sanitized_raw": (True, False),
        "RS": (False, True),
        "RR": (False, False),
        "raw_raw": (False, False),
    }
    if pairing not in mapping:
        raise ValueError(f"Unknown pairing '{pairing}'")
    return mapping[pairing]


@torch.no_grad()
def predict_pairs(
    net: nn.Module,
    store: RAMImageStore,
    sanitizer: BaseSanitizer,
    left_files: Sequence[str],
    right_files: Sequence[str],
    pairing: str,
    spec: AttackerSpec,
    device: torch.device,
    batch_size: int = 128,
    san_cache: Optional[Dict[str, torch.Tensor]] = None,
    mc_k: int = 1,
) -> np.ndarray:
    net.eval()
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    san_l, san_r = pairing_flags(pairing)
    if not sanitizer.is_deterministic and mc_k > 1:
        k_draws = mc_k
    else:
        k_draws = 1

    store.preload(left_files)
    store.preload(right_files)

    def _side_tensor(fns: Sequence[str], do_san: bool) -> torch.Tensor:
        if do_san and san_cache is not None:
            t = torch.stack([san_cache[fn] for fn in fns], dim=0).to(device, non_blocking=True)
        elif do_san:
            raw = store.get_batch_tensor(fns, device=device)
            t = sanitizer.sanitize_batch(raw, fns).to(device, non_blocking=True)
        else:
            t = store.get_batch_tensor(fns, device=device)
        return preprocess_for_attacker(t, spec.input_size, spec.use_imagenet_norm, mean_t, std_t)

    probs_all = np.zeros(len(left_files), dtype=np.float64)
    for _draw in range(k_draws):
        preds = []
        for i in range(0, len(left_files), batch_size):
            bf_l = left_files[i : i + batch_size]
            bf_r = right_files[i : i + batch_size]
            x1 = _side_tensor(bf_l, san_l)
            x2 = _side_tensor(bf_r, san_r)
            logits = net(x1, x2).flatten()
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        probs_all += np.concatenate(preds) / k_draws
    return probs_all.astype(np.float32)


@torch.no_grad()
def evaluate_sr_gallery(
    net: nn.Module,
    store: RAMImageStore,
    sanitizer: BaseSanitizer,
    spec: AttackerSpec,
    device: torch.device,
    san_cache: Optional[Dict[str, torch.Tensor]] = None,
    batch_size: int = 128,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, object]:
    """Closed-set sanitized-to-raw 1,073-patient identification on testing_5000 positive anchors."""
    from tools.gallery_identification import gallery_identities

    ids = gallery_identities()
    refs = [v[0] for v in ids.values()]
    qrys = [v[1] for v in ids.values()]
    store.preload(refs + qrys)
    net.eval()
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def _embed(fns: Sequence[str], do_san: bool) -> torch.Tensor:
        embs = []
        for i in range(0, len(fns), batch_size):
            sub = fns[i : i + batch_size]
            if do_san and san_cache is not None and all(fn in san_cache for fn in sub):
                t = torch.stack([san_cache[fn] for fn in sub], dim=0).to(device)
            elif do_san:
                t = sanitizer.sanitize_batch(store.get_batch_tensor(sub, device=device), sub).to(device)
            else:
                t = store.get_batch_tensor(sub, device=device)
            inp = preprocess_for_attacker(t, spec.input_size, spec.use_imagenet_norm, mean_t, std_t)
            embs.append(net.forward_once(inp))
        return torch.cat(embs, dim=0)

    R = _embed(refs, do_san=False)
    Q = _embed(qrys, do_san=True)
    # Use deterministic tiny noise-free tie-safe ranking: strict count above + tie count.
    ranks_low = np.empty(len(Q), dtype=np.float64)
    ranks_high = np.empty(len(Q), dtype=np.float64)
    ranks_mid = np.empty(len(Q), dtype=np.float64)
    if hasattr(net, "matrix_logits"):
        scores_mat = net.matrix_logits(Q, R)
    else:
        scores_mat = None

    def _fill_rank(i: int, s: torch.Tensor) -> None:
        above = float((s > s[i]).sum().item())
        equal = float((s == s[i]).sum().item())  # >= 1 (includes self)
        ranks_low[i] = above + 1.0
        ranks_high[i] = above + equal
        ranks_mid[i] = above + 0.5 * (equal + 1.0)

    for i in range(len(Q)):
        if scores_mat is not None:
            _fill_rank(i, scores_mat[i])
        else:
            s = torch.sigmoid(net.fc_end(torch.abs(Q[i : i + 1] - R))).squeeze(1)
            _fill_rank(i, s)

    # Expected hit probability at Rank<=K under random tie-breaking (equals 1/0 when no ties).
    def _hit_at(k: int) -> np.ndarray:
        n_ties = np.maximum(ranks_high - ranks_low + 1.0, 1.0)
        slots_in_topk = np.clip(k - ranks_low + 1.0, 0.0, n_ties)
        return np.where(ranks_high <= k, 1.0, np.where(ranks_low > k, 0.0, slots_in_topk / n_ties))

    h1 = _hit_at(1)
    h5 = _hit_at(5)
    aps = 1.0 / ranks_mid

    rng = np.random.default_rng(seed)
    n = len(ranks_mid)
    r1_b, r5_b = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r1_b.append(float(h1[idx].mean() * 100.0))
        r5_b.append(float(h5[idx].mean() * 100.0))

    return {
        "n_identities": len(ids),
        "rank1": round(float(h1.mean() * 100.0), 2),
        "rank1_ci95": [round(float(np.percentile(r1_b, 2.5)), 2), round(float(np.percentile(r1_b, 97.5)), 2)],
        "rank5": round(float(h5.mean() * 100.0), 2),
        "rank5_ci95": [round(float(np.percentile(r5_b, 2.5)), 2), round(float(np.percentile(r5_b, 97.5)), 2)],
        "mAP": round(float(aps.mean() * 100.0), 2),
        "mean_ties_per_query": round(float((ranks_high - ranks_low).mean()), 3),
    }


def train_and_evaluate_pair_run(
    run_id: str,
    out_dir: Path,
    store: RAMImageStore,
    meta_lut: Dict[str, Tuple[int, str]],
    spec: AttackerSpec,
    sanitizer_name: str,
    train_path: Optional[Path],
    val_path: Path,
    test_path: Path,
    cal_paths: Dict[str, Tuple[Path, str]],  # name -> (path, pairing)
    train_pairing: str = "SR",
    val_pairing: str = "SR",
    test_pairing: str = "SR",
    seed: int = 0,
    lr: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 100,
    patience: Optional[int] = 5,
    fixed_epochs: Optional[int] = None,
    save_ckpt: bool = False,
    mc_k: int = 1,
    mc_eval_ks: Sequence[int] = (),
    grad_clip: Optional[float] = None,
    disjoint_check_splits: Optional[Dict[str, Path]] = None,
    extra_evals: Optional[Dict[str, Tuple[Path, str]]] = None,
) -> Dict[str, object]:
    """Train (unless train_path is None for zero-shot) and evaluate one attacker run."""
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if disjoint_check_splits:
        assert_patient_disjoint(disjoint_check_splits, allow_pairs=(("val_2k", "val_sub700"),))

    utils.seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for attacker runs")

    sanitizer = build_sanitizer(sanitizer_name, device=device, seed=seed)
    net, init_path = build_attacker(spec, device=device)

    val_l, val_r, val_y = load_pair_file(val_path)
    test_l, test_r, test_y = load_pair_file(test_path)
    cal_data = {k: (v[1], *load_pair_file(v[0])) for k, v in cal_paths.items()}
    extra_data = {k: (v[1], *load_pair_file(v[0])) for k, v in (extra_evals or {}).items()}

    all_needed = list(val_l) + list(val_r) + list(test_l) + list(test_r)
    if train_path is not None:
        tr_l, tr_r, tr_y = load_pair_file(train_path)
        all_needed.extend(tr_l)
        all_needed.extend(tr_r)
    for _, cl, cr, _ in cal_data.values():
        all_needed.extend(cl)
        all_needed.extend(cr)
    for _, el, er, _ in extra_data.values():
        all_needed.extend(el)
        all_needed.extend(er)

    store.preload(all_needed)
    san_cache = precompute_sanitized_map(store, sanitizer, all_needed, device=device)

    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    train_losses, val_losses = [], []
    selected_epoch = 0

    if train_path is not None:
        is_metric_l1 = spec.arch == "resnet50_metric_l1"
        mono_head = spec.arch in MONOTONE_HEAD_ARCHS
        if spec.arch in BOOSTED_OPT_ARCHS:
            lr_head = max(lr * 25.0, 2e-3)
            optimizer = torch.optim.AdamW([
                {"params": net.model.parameters(), "lr": lr, "weight_decay": 1e-4},
                {"params": net.fc_end.parameters(), "lr": lr_head, "weight_decay": 0.0},
            ])
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        san_tr_l, san_tr_r = pairing_flags(train_pairing)
        san_va_l, san_va_r = pairing_flags(val_pairing)

        def _prep_batch(fns: Sequence[str], do_san: bool) -> torch.Tensor:
            if do_san and san_cache is not None:
                t = torch.stack([san_cache[fn] for fn in fns], dim=0).to(device, non_blocking=True)
            elif do_san:
                raw = store.get_batch_tensor(fns, device=device)
                t = sanitizer.sanitize_batch(raw, fns).to(device, non_blocking=True)
            else:
                t = store.get_batch_tensor(fns, device=device)
            return preprocess_for_attacker(t, spec.input_size, spec.use_imagenet_norm, mean_t, std_t)

        best_val = float("inf")
        best_state = copy.deepcopy(net.state_dict())
        no_imp = 0
        rng = np.random.default_rng(seed)
        n_tr = len(tr_y)
        total_epochs = fixed_epochs if fixed_epochs is not None else max_epochs

        if is_metric_l1 and spec.init_name != "imagenet":
            # Stage 0: fast linear-probe head calibration (w <= 0, b) so initial SR logits are calibrated
            # before backpropagating into the ResNet-50 backbone.
            net.eval()
            for p in net.model.parameters():
                p.requires_grad_(False)
            head_opt = torch.optim.Adam(net.fc_end.parameters(), lr=0.02)
            perm0 = rng.permutation(n_tr)
            for start in range(0, n_tr, batch_size):
                idx = perm0[start : start + batch_size]
                bf_l = [tr_l[k] for k in idx]
                bf_r = [tr_r[k] for k in idx]
                yb = torch.from_numpy(tr_y[idx]).to(device=device, dtype=torch.float32).unsqueeze(1)
                with torch.no_grad():
                    e1 = net.forward_once(_prep_batch(bf_l, san_tr_l))
                    e2 = net.forward_once(_prep_batch(bf_r, san_tr_r))
                    diff = torch.abs(e1 - e2)
                head_opt.zero_grad()
                loss0 = criterion(net.fc_end(diff), yb)
                loss0.backward()
                head_opt.step()
                with torch.no_grad():
                    net.fc_end.weight.clamp_(max=0.0)
            for p in net.model.parameters():
                p.requires_grad_(True)

        for ep in range(1, total_epochs + 1):
            net.train()
            perm = rng.permutation(n_tr)
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, n_tr, batch_size):
                idx = perm[start : start + batch_size]
                bf_l = [tr_l[k] for k in idx]
                bf_r = [tr_r[k] for k in idx]
                yb = torch.from_numpy(tr_y[idx]).to(device=device, dtype=torch.float32).unsqueeze(1)
                x1 = _prep_batch(bf_l, san_tr_l)
                x2 = _prep_batch(bf_r, san_tr_r)
                optimizer.zero_grad()
                logits = net(x1, x2)
                loss = criterion(logits, yb)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                optimizer.step()
                if mono_head:
                    with torch.no_grad():
                        net.fc_end.weight.clamp_(max=0.0)
                ep_loss += float(loss.item())
                n_batches += 1
            train_losses.append(ep_loss / max(n_batches, 1))

            net.eval()
            v_loss = 0.0
            v_batches = 0
            with torch.no_grad():
                for start in range(0, len(val_y), batch_size):
                    bf_l = val_l[start : start + batch_size]
                    bf_r = val_r[start : start + batch_size]
                    yb = torch.from_numpy(val_y[start : start + batch_size]).to(device=device, dtype=torch.float32).unsqueeze(1)
                    x1 = _prep_batch(bf_l, san_va_l)
                    x2 = _prep_batch(bf_r, san_va_r)
                    logits = net(x1, x2)
                    v_loss += float(criterion(logits, yb).item())
                    v_batches += 1
            mean_v = v_loss / max(v_batches, 1)
            val_losses.append(mean_v)

            if mean_v < best_val:
                best_val = mean_v
                selected_epoch = ep
                best_state = copy.deepcopy(net.state_dict())
                no_imp = 0
            else:
                no_imp += 1
                if fixed_epochs is None and patience is not None and no_imp >= patience:
                    break

        if fixed_epochs is None:
            net.load_state_dict(best_state)
        else:
            # In fixed-epoch regime, evaluate final epoch weights directly AND log best_val_epoch.
            selected_epoch = total_epochs

    ckpt_saved_path = None
    if save_ckpt:
        ckpt_saved_path = out_dir / f"{run_id}_best.pth"
        torch.save(net.state_dict(), ckpt_saved_path)

    val_p = predict_pairs(net, store, sanitizer, val_l, val_r, val_pairing, spec, device, batch_size=128, san_cache=san_cache, mc_k=mc_k)
    test_p = predict_pairs(net, store, sanitizer, test_l, test_r, test_pairing, spec, device, batch_size=128, san_cache=san_cache, mc_k=mc_k)

    cal_results = {}
    npz_payload = {"val_y": val_y, "val_p": val_p, "test_y": test_y, "test_p": test_p}
    for cal_name, (cal_pair_mode, cl, cr, cy) in cal_data.items():
        cp = predict_pairs(net, store, sanitizer, cl, cr, cal_pair_mode, spec, device, batch_size=128, san_cache=san_cache, mc_k=mc_k)
        npz_payload[f"cal_{cal_name}_y"] = cy
        npz_payload[f"cal_{cal_name}_p"] = cp
        op_points = {}
        for tgt in (0.01, 0.005, 0.001):
            thr, fb = select_val_threshold(cy, cp, tgt)
            op = eval_threshold_op_point(test_y, test_p, thr)
            op["fallback_inf"] = fb
            op["n_cal_neg"] = int((cy == 0).sum())
            op_points[f"fpr_{tgt*100:g}pct"] = op
        cal_results[cal_name] = {
            "pairing": cal_pair_mode,
            "n_pairs": len(cy),
            "cal_auc": round(compute_binary_metrics(cy, cp)["auc"], 4),
            "operating_points": op_points,
        }

    extra_results = {}
    for ex_name, (ex_pair_mode, el, er, ey) in extra_data.items():
        ep = predict_pairs(net, store, sanitizer, el, er, ex_pair_mode, spec, device, batch_size=128, san_cache=san_cache, mc_k=mc_k)
        npz_payload[f"extra_{ex_name}_y"] = ey
        npz_payload[f"extra_{ex_name}_p"] = ep
        em = compute_binary_metrics(ey, ep)
        extra_results[ex_name] = {
            "pairing": ex_pair_mode,
            "n_pairs": len(ey),
            "metrics": {k: round(v, 4) for k, v in em.items()},
        }

    mc_evals = {}
    for k_val in mc_eval_ks:
        if k_val == mc_k or sanitizer.is_deterministic:
            continue
        mp = predict_pairs(net, store, sanitizer, test_l, test_r, test_pairing, spec, device, batch_size=128, san_cache=san_cache, mc_k=int(k_val))
        npz_payload[f"test_p_mc{k_val}"] = mp
        mm = compute_binary_metrics(test_y, mp)
        mb = bootstrap_pair_metrics(test_l, test_r, test_y, mp, n_boot=1000, seed=seed)
        mc_evals[f"k{k_val}"] = {"metrics": {k: round(v, 4) for k, v in mm.items()}, "bootstrap": mb}

    np.savez_compressed(out_dir / f"{run_id}_preds.npz", **npz_payload)

    test_metrics = compute_binary_metrics(test_y, test_p)
    test_boot = bootstrap_pair_metrics(test_l, test_r, test_y, test_p, n_boot=1000, seed=seed)
    subgroups = compute_subgroup_metrics(test_l, test_r, test_y, test_p, meta_lut)
    gallery = evaluate_sr_gallery(net, store, sanitizer, spec, device, san_cache=san_cache, n_boot=1000, seed=seed)

    runtime_sec = round(time.time() - t0, 2)
    record = {
        "run_id": run_id,
        "arch": spec.arch,
        "init_name": spec.init_name,
        "init_ckpt_sha256": sha256_file(init_path),
        "input_size": spec.input_size,
        "use_imagenet_norm": spec.use_imagenet_norm,
        "sanitizer": sanitizer_name,
        "sanitizer_ckpt_sha256": sha256_file(sanitizer.checkpoint_path),
        "train_path": str(train_path) if train_path else None,
        "train_path_sha256": sha256_file(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
        "train_pairing": train_pairing,
        "val_pairing": val_pairing,
        "test_pairing": test_pairing,
        "seed": seed,
        "lr": lr,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "fixed_epochs": fixed_epochs,
        "selected_epoch": selected_epoch,
        "epochs_ran": len(train_losses),
        "mc_k": mc_k,
        "train_losses": [round(x, 6) for x in train_losses],
        "val_losses": [round(x, 6) for x in val_losses],
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        "test_bootstrap_single_seed": test_boot,
        "subgroups": subgroups,
        "calibration": cal_results,
        "extra_evals": extra_results,
        "mc_evals": mc_evals,
        "sr_gallery": gallery,
        "ckpt_saved_path": str(ckpt_saved_path) if ckpt_saved_path else None,
        "runtime_sec": runtime_sec,
    }
    with (out_dir / f"{run_id}_summary.json").open("w") as f:
        json.dump(record, f, indent=2)

    del net, sanitizer, san_cache
    torch.cuda.empty_cache()
    return record
