"""E6 feasibility: official-test patients outside the released test list and the calibration list.

A held-out confirmatory set needs patients with at least two radiographs (to form positive pairs).
Counts the official-test patients absent from both lists and how many images each has.
Output: results/E6/e6_count.json.  Usage: python3 tools/v2_e6_count.py
"""
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pid = lambda f: f.split("_")[0]


def pts(path):
    return {pid(x) for l in open(path) for x in l.split()[:2]}


def main():
    test = [l.strip() for l in open(os.path.join(ROOT, "image_pairs", "test_list.txt")) if l.strip()]
    n_img = collections.Counter(pid(f) for f in test)
    t5k = pts(os.path.join(ROOT, "image_pairs", "image_pairs_testing_5000.txt"))
    cal = pts(os.path.join(ROOT, "image_pairs", "image_pairs_validation_disjoint_700.txt"))
    held = set(n_img) - t5k - cal
    out = {"official_test_patients": len(n_img), "in_test_list": len(set(n_img) & t5k),
           "in_calibration_list": len(set(n_img) & cal), "held_out": len(held),
           "held_out_single_image": sum(n_img[p] == 1 for p in held),
           "held_out_two_or_more_images": sum(n_img[p] >= 2 for p in held), "threshold_in_brief": 300,
           "decision": "E6 not run: too few held-out patients can form positive pairs"}
    os.makedirs(os.path.join(ROOT, "results", "E6"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "E6", "e6_count.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
