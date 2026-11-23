#!/bin/bash
# E1/E2 scoring: embed once per image; extended negatives and SS test pairs (rerun after new checkpoints appear).
cd "$(dirname "$0")/../.."
python3 tools/v2_embed_score.py --build-list
python3 -u tools/v2_embed_score.py
