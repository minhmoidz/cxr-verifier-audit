"""V3-A (docs/PREREGISTRATION_v3.md): embed the CheXpert pair set with every NIH-trained verifier.

For each model of results/E1E2/models.json (which also lists the V3-B and V3-C models), the 2,000 queries are sanitized with the model's own sanitizer, and the 2,000 references are embedded
raw and sanitized. results/V3A/scores/<model_id>.npz stores the float32 embeddings and the head (w, b); the
4,000,000 pair logits w . |e_q - e_r| + b are recomputed from them by tools/v3_chexpert_ci.py.

Preprocessing (pre-specified): grayscale, center crop to a square of side min(width, height), bilinear
resize to 256 x 256; then the NIH pipeline unchanged. Elastic fields are keyed on the CheXpert path.

Usage: python3 tools/v3_chexpert_score.py --chexpert-dir /path/to/data_chexpert [--only S]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prl_lib as P  # noqa: E402
from tools.v2_embed_score import embed, scoring_sanitizer  # noqa: E402

OUTD = ROOT / "results" / "V3A"
FAMILIES = ("reference", "e5", "imagenet_full", "factorial_ref")


class CheXpertStore(P.RAMImageStore):
    def __init__(self, base: Path):
        super().__init__(image_dir=base, use_cache=False)

    def _load_pil_uint8(self, filename: str) -> np.ndarray:
        # tools/v3_remote_zip.py stores each selected image already preprocessed (256 x 256 PNG) at
        # <dir>/patientN/studyN/viewN_frontal.png; the crop and resize below are then the identity.
        rel = re.search(r"(patient\d+/study\d+/view\d+_\w+)\.jpg$", filename).group(1) + ".png"
        with Image.open(self.image_dir / rel) as im:
            g = im.convert("L")
            w, h = g.size
            s = min(w, h)
            g = g.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
            g = transforms.Resize((256, 256))(g)
            return np.asarray(g, dtype=np.uint8)


def model_list():
    ms = []
    ms = json.load(open(ROOT / "results" / "E1E2" / "models.json"))   # includes V3-B and V3-C models
    keep = [m for m in ms if (m["family"] in FAMILIES or m["family"].startswith("v3c_")) and not (m["family"] == "factorial_ref" and m["sanitizer"] != "ff_mu0.01")]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chexpert-dir", required=True, help="output directory of tools/v3_remote_zip.py pairs")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    pairs = json.load(open(OUTD / "chexpert_pairs.json"))["pairs"]
    refs = [p["reference"] for p in pairs]
    qrys = [p["query"] for p in pairs]
    (OUTD / "scores").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    store = CheXpertStore(Path(a.chexpert_dir))
    store.preload(refs + qrys)
    for m in model_list():
        if a.only not in m["model_id"]:
            continue
        dst = OUTD / "scores" / f"{m['model_id']}.npz"
        if dst.exists():
            continue
        t0 = time.time()
        spec = P.AttackerSpec(m["arch"], f"custom:{ROOT / m['ckpt']}", m["input_size"], m["use_norm"])
        net, _ = P.build_attacker(spec, device)
        net.eval()
        san = scoring_sanitizer(m, device)
        q_san = embed(net, store, san, qrys, True, spec, device)
        r_raw = embed(net, store, san, refs, False, spec, device)
        r_san = embed(net, store, san, refs, True, spec, device)
        w = net.fc_end.weight.detach().cpu().numpy().reshape(-1)
        b = float(net.fc_end.bias.detach().cpu().numpy().reshape(-1)[0])
        np.savez_compressed(dst, q_san=q_san, r_raw=r_raw, r_san=r_san, head_w=w, head_b=np.array([b]),
                            patients=np.array([p["patient"] for p in pairs]))
        pos = np.abs(q_san.astype(np.float64) - r_raw) @ w + b
        meta = {**m, "seconds": round(time.time() - t0, 1), "mean_pos_logit_SR": float(pos.mean())}
        (OUTD / "scores" / f"{m['model_id']}.json").write_text(json.dumps(meta, indent=1))
        print(f"{m['model_id']}: done [{meta['seconds']}s]", flush=True)
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
