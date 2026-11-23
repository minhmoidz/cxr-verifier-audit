#!/bin/bash
# Fetch the NIH train/val images not on disk, at 256 px (three parallel lanes over the 12 tarballs).
cd "$(dirname "$0")/../../.."
python3 -u tools/fetch_nih_trainval_256.py --parts 1,2,3,4 > results/E5/fetch/lane1.log 2>&1 &
python3 -u tools/fetch_nih_trainval_256.py --parts 5,6,7,8 > results/E5/fetch/lane2.log 2>&1 &
python3 -u tools/fetch_nih_trainval_256.py --parts 9,10,11,12 > results/E5/fetch/lane3.log 2>&1 &
wait
python3 -u tools/fetch_nih_trainval_256.py   # final pass: confirms all parts done and writes still_missing.txt
