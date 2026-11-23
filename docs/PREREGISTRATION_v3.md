# Pre-specified analysis plan, revision v3

Written and committed on 2026-11-23, after the revision-v2 results (commit e1c1633) and **before** any
CheXpert image was downloaded, any additional seed was trained, or any sanitizer was retrained. The commit
that adds this file is pushed to the public repository so that its timestamp precedes the new data. Any
later change to this plan is recorded at the end of this file with its date and reason; analyses that
deviate from it are reported as exploratory.

Three studies address the limitations of revision v2: one cohort, three seeds per arm, and a test list used
during exploration (V3-A, V3-B), and an untested explanation for the attack-dependent advantage of the
learned field (V3-C).

## Common definitions

- Learned field: the released PriCheXy-Net generator at mu = 0.01 unless stated. Controls: Gaussian blur
  sigma = 2; random elastic deformation at strength 0.045 (reference and independent verifiers) and 0.04
  (all verifiers), with the keyed fields used in revision v2.
- Verifiers: the networks of revision v2, trained on NIH with SR pairs for each sanitizer (released recipe):
  reference initialization ("reference"), the independent verifier E5 ("independent"), and ImageNet with
  the full recipe ("ImageNet"). Scoring uses logits = w . |e1 - e2| + b from embeddings, as in
  `tools/v2_embed_score.py`.
- SR attack: sanitized query, raw reference. SS attack: query and reference both sanitized with the same
  sanitizer.
- Metrics: primary TPR at 1% FPR; secondary AUC and TPR at 0.1% FPR. Each metric is computed per trained
  network on all pairs of the set and averaged over seeds.
- Intervals: 2,000 bootstrap replicates, RNG seed 20261124. Patients are resampled with replacement; a
  positive pair has weight m_p and a negative pair m_a m_b (dyadic weights, as in revision v2). The
  patient-and-seed interval also resamples seeds with replacement within each arm (independently across
  arms, except arms that share a seed index by design). Contrasts use the same patient draw for every arm.
  Decisions use the 95% patient-and-seed percentile interval.

## V3-A: external confirmation on CheXpert

Data: CheXpert-v1.0-small, training split only (the validation split is not used). The download is
recorded with its SHA-256.

Pair set (fixed before images are opened; built from `train.csv` only):
1. Keep frontal images (Frontal/Lateral = Frontal). Order a patient's studies by study number.
2. Eligible patients have frontal images in at least two studies. The reference image is the first frontal
   image (by file name) of the first such study; the query is the first frontal image of the second.
3. Draw 2,000 eligible patients without replacement with `numpy.random.default_rng(20261124)` from the
   sorted list of eligible patient IDs.
4. Positives: the 2,000 query-reference pairs of the same patient. Negatives: all 2,000 x 1,999 =
   3,998,000 cross-patient query-reference combinations.
5. Preprocessing: grayscale, center crop to a square of side min(width, height), bilinear resize to
   256 x 256, then the NIH pipeline unchanged.

No CheXpert image is used for training or checkpoint selection. All verifiers are the NIH-trained networks
(a transfer attack: the attacker trains on public data and attacks a new release).

Hypotheses (each tested under the reference and the independent verifier; a hypothesis is confirmed only if
it holds under both; ImageNet results are reported as secondary):
- **H1 (SR attack).** TPR(field) - TPR(blur) < 0 under the SR attack: the upper limit of the 95% interval
  is below zero. Secondary: the same for elastic 0.045 and 0.04.
- **H2 (attack dependence).** [TPR_SS(field) - TPR_SR(field)] - [TPR_SS(blur) - TPR_SR(blur)] > 0: the lower
  limit of the 95% interval is above zero. Secondary: the same with elastic 0.045. Under the SS attack,
  TPR(field) - TPR(blur) is reported with its interval; no equivalence claim is made.
- **H3 (training pairs).** For the reference-initialized learned-field factorial networks (NIH, five seeds
  per cell), the training effect (SR minus SS training, averaged over validation pairings) on AUC of
  CheXpert SR pairs is above zero (lower limit above zero). Seeds are resampled jointly across the four
  cells, as in Table S4.

Reported regardless of outcome: all hypotheses, both verifiers, all three metrics, and raw-image linkage on
the same pairs.

## V3-B: five seeds for the main NIH contrasts

Seeds 3 and 4 are added, with the arguments of seeds 0-2, for: reference {raw, field, blur 2, elastic 0.045,
elastic 0.04}; independent {raw, field, blur 2, elastic 0.045, elastic 0.04}; ImageNet {raw, field, blur 2,
elastic 0.04}. All are scored on the NIH extended set (SR and SS). The five-seed contrasts of Table 4
(field minus blur and minus elastic, SR and SS) replace the three-seed contrasts in the manuscript, whatever
their direction; the three-seed values remain in the supplement. H1 and H2 above are also reported on NIH
with five seeds. Seeds 3 and 4 are also scored on CheXpert and included in V3-A if they finish before the
V3-A analysis is run; otherwise V3-A uses seeds 0-2 and this is stated.

## V3-C: does the adversary's pairing decide which attack the sanitizer defeats?

PriCheXy-Net updates its adversary on SR pairs (sanitized image, raw partner) and trains the generator
against it. Hypothesis: a generator trained against an SS adversary (both images sanitized) leaks less under
the SS attack.

Sanitizers trained (same code, `adversary_pairing` option; everything else as released): mu = 0.01; start
from the released pretrained generator; adversary initialized from the released verification weights;
auxiliary classifier from the released classifier; released 10,000 training and 2,000 validation pairs;
the released number of epochs, batch size and learning rate; checkpoint with the lowest validation total
loss, as released.
- **SR-adv**: the released objective (replication).
- **SS-adv**: the generator loss and the adversary update use (G(x1), G(x2)) instead of (G(x1), x2); the
  validation loss uses the same pairing.
Seeds: the released code seeds every run with 42; both variants are trained first with seed 42. A second
seed (43) per variant is trained if the first pair finishes within 48 hours of starting; otherwise one seed per variant is reported and the limitation stated. If GPU
memory forces a smaller batch, both variants use the same batch and this is stated.

Evaluation per trained sanitizer (identical to revision v2): frozen-classifier macro AUC on the official
test images; verifiers with reference and with independent initialization, released recipe, SR pairs, three
seeds each; SR and SS attacks on the NIH extended set and on the V3-A CheXpert set.

Hypotheses (reference and independent verifiers; confirmed only if both hold):
- **H4.** Under the SS attack, TPR(SS-adv) - TPR(SR-adv) < 0 (upper limit below zero), on NIH and on CheXpert.
- **H5 (secondary, two-sided).** Under the SR attack, TPR(SS-adv) - TPR(SR-adv) is reported with its
  interval; no direction is predicted.
- Replication check (descriptive): SR-adv versus the released mu = 0.01 generator on frozen utility and on
  SR- and SS-attack TPR.
The bootstrap treats the sanitizer seed as fixed when there is one seed per variant; with two seeds, sanitizer
seeds are resampled with the verifier seeds nested within them.

## Amendments

**A1 (2026-11-23, before any CheXpert file was read).** Stanford AIMI now distributes only the
full-resolution `CheXpert-v1.0.zip` (471 GB), not CheXpert-v1.0-small, and the analysis machine has about
22 GB free. `train.csv` and the selected JPEG files are therefore read from the remote zip with HTTP range
requests (no other member is read). The pair-selection rule, the seed and the preprocessing are unchanged;
the center crop and the bilinear resize to 256 x 256 are applied to the full-resolution images instead of
the downsampled ones. The SHA-256 of `train.csv` and of every image read is recorded.

**A2 (2026-11-23, before any sanitizer training finished).** The first timing epoch of V3-C ran out of GPU
memory because other users' jobs occupied about 18.9 of the 24 GB GPU. If the released batch size of 64
cannot run, both variants are trained with the same smaller batch (32), as the plan allows, and this is
stated.

**A3 (2026-11-23, before any CheXpert file was read).** The `CheXpert-v1.0.zip` link returned
BlobNotFound. The AIMI container instead holds `CheXpert-v1.0 batch 1 (validate & csv).zip` and three
training batches (`batch 2 (train 1)`, `batch 3 (train 2)`, `batch 4 (train 3)`). `train.csv` is read from
batch 1 and each selected image from the batch that contains it, with range requests as in A1. Nothing
else changes.
