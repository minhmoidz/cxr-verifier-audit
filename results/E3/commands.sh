#!/bin/bash
# E3: released PriCheXy-Net evaluation protocol inside this study (ImageNet init, released recipe,
# released 10,000/2,000 pair lists, SR test), 2x2 train x validation pairing, learned field mu=0.01, seeds 0-4.
#
# SS/SS and SR/SR at seeds 0-2 were trained on 2026-09-05 by the pilot design
#   python3 artifacts/paper_upgrade/phase5_pairing_audit/run_pairing_audit.py --config ff_mu0.01 --seeds 0,1,2
# and are copied into results/E3/cells/ (their manifests are in results/E3/manifests/).
#
# The 14 remaining cells. The released agent writes cell outputs to archive/<run_id>/; manifests go to
# results/E3/manifests/. A first attempt (2026-11-22) wrote outputs to archive/ while the runner looked in
# results/E3/cells/, so one finished cell had no manifest; it was moved to results/E3/quarantine/ and retrained.
cd "$(dirname "$0")/../.."
PAIRING_OUT_ROOT=results/E3/manifests \
  python3 -u artifacts/paper_upgrade/phase5_pairing_audit/run_pairing_audit.py --config ff_mu0.01 --design imagenet_factorial

# Copy predictions of the new cells into results/E3/cells/ (checkpoints are linked, not copied).
for d in archive/p1f_ff_mu0.01_*_imagenet_seed* archive/p1_ff_mu0.01_*_imagenet_seed[34]; do
  [ -d "$d" ] || continue
  n=$(basename "$d"); mkdir -p results/E3/cells/$n
  cp -p "$d/${n}_labels_predictions.txt" "$d/${n}_results.txt" "$d/${n}_loss_values.txt" results/E3/cells/$n/
  ln -sfn "$(pwd)/$d/${n}_best_network.pth" results/E3/cells/$n/${n}_best_network.pth
done
