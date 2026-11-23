#!/bin/bash
cd "$(dirname "$0")/../.."
python3 -u tools/v2_secondary_ci.py --n_boot 2000 --seed 20261122
