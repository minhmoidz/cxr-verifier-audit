"""GPU schedule for the night of 2026-11-23/24 (docs/PREREGISTRATION_v3.md, amendment A5).

Until --deadline two of our jobs may share the GPU; afterwards only one may run.
  lane 1  the V3-B queue that is already running (--v3b-pid: its tools/gpu_wait_run.py process), then the
          scoring of its runs (tools/v2_embed_score.py on NIH, tools/v3_chexpert_score.py on CheXpert)
  lane 2  the first V3-C generator (scripts/run_v3c.py gen SR), started now
At the deadline, if lane 1 has not finished (V3-B and its scoring), the generator is stopped; it resumes from its
last finished epoch (agents/Agent.py resume state) after lane 1. From then on the queue runs one job at a time: scoring (if not done), generator
SR (waits for it if it is still running), generator SS, V3-C verifiers, V3-C utility. Training settings are
never changed; every job goes through tools/gpu_wait_run.py (waits for free memory, retries out-of-memory).

Usage: python3 -u tools/v3_overnight.py --v3b-pid PID [--deadline "2026-11-24 06:00"]
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = [sys.executable, "-u", "tools/gpu_wait_run.py"]
GEN_SR = W + ["--min-free-gb", "10.5", "--log", "results/V3C/run.log", "--", sys.executable, "-u", "scripts/run_v3c.py", "gen", "SR"]
SCORING = [
    [sys.executable, "-u", "tools/v2_embed_score.py", "--build-list"],
    W + ["--min-free-gb", "4", "--log", "results/V3B/score.log", "--", sys.executable, "-u", "tools/v2_embed_score.py", "--only", "__s3"],
    W + ["--min-free-gb", "4", "--log", "results/V3B/score.log", "--", sys.executable, "-u", "tools/v2_embed_score.py", "--only", "__s4"],
    W + ["--min-free-gb", "4", "--log", "results/V3B/score.log", "--", sys.executable, "-u", "tools/v3_chexpert_score.py",
         "--chexpert-dir", "../data_chexpert", "--only", "__s3"],
    W + ["--min-free-gb", "4", "--log", "results/V3B/score.log", "--", sys.executable, "-u", "tools/v3_chexpert_score.py",
         "--chexpert-dir", "../data_chexpert", "--only", "__s4"],
]
REST = [
    W + ["--min-free-gb", "13", "--log", "results/V3C/run.log", "--", sys.executable, "-u", "scripts/run_v3c.py", "gen", "SS"],
    W + ["--min-free-gb", "9", "--log", "results/V3C/run.log", "--", sys.executable, "-u", "scripts/run_v3c.py", "verifiers"],
    W + ["--min-free-gb", "3", "--log", "results/V3C/run.log", "--", sys.executable, "-u", "scripts/run_v3c.py", "utility"],
]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    st = Path(f"/proc/{pid}/stat")
    return st.is_file() and st.read_text().split()[2] != "Z"


def start(cmd):
    log("start: " + " ".join(cmd[cmd.index("--") + 1:] if "--" in cmd else cmd))
    return subprocess.Popen(cmd, cwd=ROOT, start_new_session=True)


def run(cmd):
    rc = start(cmd).wait()
    log(f"exit {rc}")
    if rc != 0:
        sys.exit(f"stopped: exit {rc}")


def stop(p):
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(120)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    time.sleep(30)                                                    # let the GPU memory be released


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3b-pid", type=int, required=True)
    ap.add_argument("--deadline", default="2026-11-24 06:00")
    a = ap.parse_args()
    deadline = datetime.strptime(a.deadline, "%Y-%m-%d %H:%M")
    lane2 = start(GEN_SR)
    scoring, step, v3b_seen_done = None, 0, False
    while datetime.now() < deadline and step <= len(SCORING):
        if not alive(a.v3b_pid):
            if not v3b_seen_done:
                v3b_seen_done = True
                n = len(list((ROOT / "results" / "V3B" / "runs").glob("*_summary.json")))
                log(f"V3-B queue ended: {n}/28 runs" + ("" if n == 28 else "  WARNING: incomplete"))
            if scoring is None or scoring.poll() is not None:
                if scoring is not None:
                    log(f"exit {scoring.returncode}")
                    if scoring.returncode != 0:
                        sys.exit(f"scoring failed: exit {scoring.returncode}")
                if step == len(SCORING):
                    step += 1
                    log("lane 1 finished (V3-B and its scoring)")
                    break
                scoring = start(SCORING[step])
                step += 1
        if lane2.poll() is not None:
            break
        time.sleep(30)
    lane1_done = step > len(SCORING)
    if lane2.poll() is not None:
        log(f"generator SR process ended (exit {lane2.returncode})")
    if not lane1_done:
        while datetime.now() < deadline and lane2.poll() is None:
            time.sleep(30)
        if lane2.poll() is None:
            log("deadline reached and lane 1 still busy: stopping generator SR (it resumes from its last epoch)")
            stop(lane2)
            lane2 = None
        while alive(a.v3b_pid):
            time.sleep(60)
        if scoring is not None and scoring.poll() is None:
            scoring.wait()
            log(f"exit {scoring.returncode}")
        log("lane 1: finishing alone")
        for cmd in SCORING[step:]:
            run(cmd)
    if lane2 is not None:
        rc = lane2.wait()
        log(f"generator SR exit {rc}")
        if rc != 0:
            sys.exit("generator SR failed")
    else:
        run(GEN_SR)
    for cmd in REST:
        run(cmd)
    log("all V3 GPU jobs done")


if __name__ == "__main__":
    main()
