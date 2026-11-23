"""Run a command when the shared GPU has enough free memory; retry after CUDA out-of-memory failures.

The GPU is shared with other users' jobs, so a run can start only when enough memory is free, and it can
still fail when another job grows. Runners in this repository skip finished runs, so a retry resumes the
queue. Training settings are never changed to fit memory.

Usage: python3 tools/gpu_wait_run.py --min-free-gb 7 --log results/V3B/run.log -- python3 -u scripts/run_v3_seeds.py
"""
import argparse
import subprocess
import sys
import time


def free_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0]) / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-gb", type=float, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--max-attempts", type=int, default=200)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    for attempt in range(1, a.max_attempts + 1):
        ok = 0
        while ok < 3:                                        # free memory on three checks one minute apart
            ok = ok + 1 if free_gb() >= a.min_free_gb else 0
            time.sleep(60 if ok < 3 else 0)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(a.log, "a") as lf:
            lf.write(f"\n=== attempt {attempt} at {stamp}, {free_gb():.1f} GB free ===\n")
            lf.flush()
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            print(f"done after {attempt} attempt(s)")
            return
        tail = open(a.log, errors="ignore").read()[-4000:]
        if "OutOfMemoryError" not in tail and "out of memory" not in tail:
            sys.exit(f"failed with exit code {rc} (not out of memory); see {a.log}")
        print(f"attempt {attempt}: out of memory, waiting", flush=True)
        time.sleep(300)
    sys.exit("too many attempts")


if __name__ == "__main__":
    main()
