"""Build a checkpoint-selection pair list whose patients the reference verifier never saw.

The reference verifier's supplied pretraining list (the official NIH train_val_list.txt, 28,008
patients) contains every patient of the released 2,000-pair validation list. Validation loss,
early stopping and validation-selected thresholds for verifier-init attackers are therefore
computed on patients the initialization has already seen. This script draws a replacement
validation list from patients that are in the official NIH test list, absent from the linkage
test pairs, and absent from the pretraining list. Deterministic (fixed RNG seed).

Usage: python3 tools/build_disjoint_validation.py
"""
import collections
import hashlib
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20261102
N_POS = N_NEG = 350
OUT = os.path.join(ROOT, "image_pairs", "image_pairs_validation_disjoint_700.txt")

pid = lambda f: f.split("_")[0]
on_disk = set(os.listdir(os.path.join(ROOT, "images")))

ver = {pid(l.split()[0]) for l in open(os.path.join(ROOT, "image_pairs", "train_val_list.txt"))}
linkage = {}
for phase, fn in (("train", "image_pairs_training_10000.txt"), ("val", "image_pairs_validation_2000.txt"),
                  ("test", "image_pairs_testing_5000.txt")):
    linkage[phase] = {pid(x) for l in open(os.path.join(ROOT, "image_pairs", fn)) for x in l.split()[:2]}

by_patient = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "image_pairs", "test_list.txt")):
    f = l.strip()
    p = pid(f)
    if f in on_disk and p not in ver and not any(p in s for s in linkage.values()):
        by_patient[p].append(f)

multi = sorted(p for p, v in by_patient.items() if len(v) > 1)
patients = sorted(by_patient)
rng = random.Random(SEED)
assert len(multi) >= N_POS, (len(multi), N_POS)

pairs = []
for p in rng.sample(multi, N_POS):
    a, b = rng.sample(by_patient[p], 2)
    pairs.append((a, b, "1.0"))
seen = set()
while len(seen) < N_NEG:
    p, q = rng.sample(patients, 2)
    a, b = rng.choice(by_patient[p]), rng.choice(by_patient[q])
    if (a, b) not in seen:
        seen.add((a, b))
        pairs.append((a, b, "0.0"))
rng.shuffle(pairs)

with open(OUT, "w") as f:
    for a, b, y in pairs:
        f.write(f"{a}\t{b}\t{y}\n")

used = {pid(x) for a, b, _ in pairs for x in (a, b)}
assert not used & ver and not used & linkage["test"] and not used & linkage["train"] and not used & linkage["val"]
print(f"wrote {OUT}: {len(pairs)} pairs, {len(used)} patients, candidates {len(patients)} ({len(multi)} multi-image)")
print("sha256", hashlib.sha256(open(OUT, "rb").read()).hexdigest())
