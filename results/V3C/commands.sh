#!/bin/bash
# V3 GPU queue (docs/PREREGISTRATION_v3.md), run one job at a time on the shared GPU. Thresholds are the measured
# peak memory plus a margin: PriCheXy-Net batch 16 about 9 GiB (SR) and 11.6 GiB (SS); verifier batch 32 about 8 GiB.
cd "$(dirname "$0")/../.."
W="python3 -u tools/gpu_wait_run.py"
$W --min-free-gb 10.5 --log results/V3C/run.log -- python3 -u scripts/run_v3c.py gen SR
$W --min-free-gb 13   --log results/V3C/run.log -- python3 -u scripts/run_v3c.py gen SS
$W --min-free-gb 9    --log results/V3C/run.log -- python3 -u scripts/run_v3c.py verifiers
$W --min-free-gb 9    --log results/V3B/run.log -- python3 -u scripts/run_v3_seeds.py
$W --min-free-gb 3    --log results/V3C/run.log -- python3 -u scripts/run_v3c.py utility
