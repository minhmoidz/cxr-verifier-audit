#!/bin/bash
# V3-C (docs/PREREGISTRATION_v3.md): retrained generators (SR / SS adversary), verifiers on them, frozen utility.
# Each step waits for free memory on the shared GPU and retries after out-of-memory failures.
cd "$(dirname "$0")/../.."
W="python3 -u tools/gpu_wait_run.py --log results/V3C/run.log"
$W --min-free-gb 4 -- python3 -u scripts/run_v3c.py gen SR
$W --min-free-gb 4 -- python3 -u scripts/run_v3c.py gen SS
$W --min-free-gb 6 -- python3 -u scripts/run_v3c.py verifiers
$W --min-free-gb 3 -- python3 -u scripts/run_v3c.py utility
