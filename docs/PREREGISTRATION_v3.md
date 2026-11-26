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

**A4 (2026-11-23, before any sanitizer training finished).** Peak GPU memory measured on this card
(two training steps, batch 2 and 4, extrapolated linearly): SR adversary about 0.53 GiB per image, SS adversary
about 0.69 GiB per image, so batch 64 needs about 34 and 45 GiB, and batch 32 with the SS adversary about
22.7 GiB, more than the 23 GiB a free 24 GB card offers. Both variants are therefore trained with batch 16
(learning rate, epochs and every other setting as released). This replaces the batch 32 of A2. The comparison
of SR-adv and SS-adv is unaffected because both use the same settings; the replication check against the
released generator (batch 64) is descriptive only.

**A5 (2026-11-23, before any sanitizer training started).** Execution only; no analysis changes. On the
shared GPU a long run can be killed by another user's job growing (out of memory), and the GPU is ours only in
part. (i) The sanitizer trainer saves its full state after every epoch (networks, the three optimizers, the
best-loss values, the loss curves and the random-number states) and, after an interruption, resumes at the next
epoch (`train_architecture.py --resume`) instead of restarting; the selected generator is still the one with
the lowest validation total loss over all 250 epochs. (ii) Until 06:00 on 2026-11-24 two of our jobs may share the
GPU (the V3-B queue and the SR-adversary generator); afterwards one at a time. If V3-B has not finished by
then, the generator is stopped and resumed after V3-B (`tools/v3_overnight.py`). Settings are unchanged.

## V3-D (added 2026-11-26, after V3-C and before any V3-D training): a second learned sanitizer family

Written after the V3-A to V3-C results were known (commit a8be662) and before any V3-D sanitizer was trained or
evaluated; only a four-minute timing run of the trainer has been executed (no checkpoint kept). Every result of
revision v2 and V3-A to V3-C concerns one learned sanitizer family, the learned deformation field. V3-D asks
whether the audit's three central findings hold for a learned sanitizer that rewrites intensities instead of
deforming the image.

Sanitizer: the Privacy-Net generator (Kim et al. 2021) as implemented in the released PriCheXy-Net code
(`generator_type: privacy_net`, `Unet2D_encoder(1, 1, 16)`, started from the released
`pretrained_generator_privacy_net.pth`), trained with the released objective (auxiliary classifier plus an
adversary updated on SR pairs) and the released starting networks, pairs and learning rate, batch 16 (as in
V3-C), seed 42, 100 epochs (instead of 250, for computing time), saving the generator after every epoch. The
released Privacy-Net weights (verification-loss weight 1) give a frozen test macro AUC of 57.55, far below the
learned field (76.19), so a utility-matched operating point is defined as follows, using validation data only.

- Target T: frozen-classifier macro AUC of the learned field (mu = 0.01) on the validation images of the
  retrained-utility split (3,484 images, `load_e4_splits()["val"]`).
- Verification-loss weight w: start with w = 0.1. If the validation utility of the generator after epoch 1 is
  below T, the run is discarded and restarted with w divided by 10 (at most three attempts: 0.1, 0.01, 0.001).
- Checkpoint: among the 100 epoch generators with validation utility >= T, the one with the lowest validation
  verification term -log(1 - p) (the released loss term; p = the adversary's same-patient probability). If no
  epoch reaches T, the epoch with the highest validation utility is used and this is reported.
- Matched blur: the blur strength (sigma 1.5, 2 or 2.5) whose frozen validation utility is closest to that of
  the selected checkpoint. Matched elastic deformation (secondary): strength 0.035, 0.04 or 0.045, chosen the
  same way.

Evaluation, as in V3-C: frozen test utility; verifiers with reference and with independent initialization,
released recipe, trained and selected on SR pairs, seeds 0-2; in addition, reference-initialization verifiers
trained and selected on SS pairs (the released protocol), seeds 0-2. SR and SS attacks on the NIH extended set
and on the V3-A CheXpert set; bootstrap as above, with the sanitizer treated as fixed (one seed). Control arms
are those already scored (all available verifier seeds).

Hypotheses (PN = selected Privacy-Net checkpoint; TPR at 1% FPR unless stated):
- **H6** (advantage against the SR attack). TPR(PN) - TPR(matched blur) < 0 under the SR attack (upper limit
  below zero), on NIH and on CheXpert, under the reference and the independent verifier.
- **H7** (attack dependence). [TPR_SS - TPR_SR](PN) - [TPR_SS - TPR_SR](matched blur) > 0 (lower limit above
  zero), same data sets and verifiers.
- **H8** (training pairs). Reference initialization, SR attack: AUC(SR-trained) - AUC(SS-trained) > 0 (lower
  limit above zero), on NIH and on CheXpert.
A hypothesis is confirmed only if it holds in every listed data set and verifier. Secondary, descriptive:
the same contrasts against matched elastic deformation; TPR under the SR and SS attacks for the released
Privacy-Net weights (reference and independent verifiers, SR pairs, seeds 0-2).
