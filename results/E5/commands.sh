#!/bin/bash
# E5: independent strong verifier. Started after E3 finishes (one GPU, shared with the retraining queue).
cd "$(dirname "$0")/../.."
python3 tools/fetch_nih_trainval_256.py          # no-op once all parts are fetched (see results/E5/fetch/)
python3 scripts/e5_build_pairs.py                 # results/E5/splits/ (deterministic, seed 20261122)
python3 -u scripts/run_e5.py --step all           # pretraining, then 30 sanitized runs
