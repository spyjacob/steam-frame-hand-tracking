#!/usr/bin/env python3
"""Driver ablation runner (PC): for each driver configuration, rewrite the live ini, then run
replay_sender.py (known synthetic motion) and pose_recorder.exe (what SteamVR presents)
simultaneously.  Restores the original ini at the end.

Synthetic data only: this measures the PC driver's presentation, not hand tracking.
usage: driver_ablation.py OUTDIR
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI = os.path.join(ROOT, "pcdriver", "framehands", "framehands_live.ini")
REC = os.path.join(ROOT, "pcdriver", "build", "pose_recorder.exe")
SENDER = os.path.join(ROOT, "tools", "replay_sender.py")

CONFIGS = {
    "C0_default": dict(predictLinear=1, predictAngular=0, maxPredictMs=50, interpDelayMs=0),
    "C1_nopredict": dict(predictLinear=0, predictAngular=0, maxPredictMs=50, interpDelayMs=0),
    "C2_interp17": dict(predictLinear=0, predictAngular=0, maxPredictMs=50, interpDelayMs=17),
    "C3_old": dict(predictLinear=1, predictAngular=1, maxPredictMs=100, interpDelayMs=0),
}
PATTERNS = [("sine", 12), ("reversal", 12), ("rotate", 12), ("still", 8)]
NETWORK = [("C0_default", "jitter"), ("C2_interp17", "jitter")]


def write_ini(base, over):
    lines = []
    for line in base.splitlines():
        k = line.split("=")[0].strip()
        if k in over:
            line = "%s = %s" % (k, over[k])
        lines.append(line)
    open(INI, "w").write("\n".join(lines) + "\n")


def run(out, name, pattern, secs, extra=()):
    csv = os.path.join(out, "%s__%s.csv" % (name, pattern))
    rec = subprocess.Popen([REC, csv, str(secs - 1)], stdout=subprocess.PIPE, text=True)
    snd = subprocess.run([sys.executable, SENDER, "--pattern", pattern, "--seconds", str(secs), *extra],
                         capture_output=True, text=True)
    r = rec.communicate()[0]
    print("  %-14s %-9s %s | %s" % (name, pattern, snd.stdout.strip().splitlines()[-1] if snd.stdout else snd.stderr[-200:],
                                    r.strip().splitlines()[-1] if r else "-"), flush=True)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    base = open(INI).read()
    shutil.copy(INI, os.path.join(out, "framehands_live.ini.orig"))
    try:
        for name, cfg in CONFIGS.items():
            write_ini(base, dict(cfg, headsetHost="127.0.0.1", forceHandMode=1))
            time.sleep(2.5)                       # driver re-reads once a second; let clock sync settle
            for pattern, secs in PATTERNS:
                run(out, name, pattern, secs)
        for name, tag in NETWORK:
            write_ini(base, dict(CONFIGS[name], headsetHost="127.0.0.1", forceHandMode=1))
            time.sleep(2.5)
            run(out, name + "_" + tag, "sine", 12, ("--jitter-ms", "15", "--loss", "0.02", "--reorder", "0.02"))
    finally:
        open(INI, "w").write(base)              # back to the real headset
        print("restored live ini (real headset, forceHandMode as before)")


if __name__ == "__main__":
    main()
