#!/bin/bash
# V3-B (docs/PREREGISTRATION_v3.md): seeds 3 and 4 for the main NIH arms, then scoring on the NIH extended set.
cd "$(dirname "$0")/../.."
python3 -u scripts/run_v3_seeds.py
python3 tools/v2_embed_score.py --build-list
python3 -u tools/v2_embed_score.py --only __s3
python3 -u tools/v2_embed_score.py --only __s4
