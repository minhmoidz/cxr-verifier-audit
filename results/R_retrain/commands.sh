#!/bin/bash
# Retrain the 43 Fig. 2 verifier runs whose checkpoints were not saved (replicates of published runs).
cd "$(dirname "$0")/../.."
python3 -u scripts/run_v2_retrain.py
