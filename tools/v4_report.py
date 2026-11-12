#!/usr/bin/env python3
"""Summarise the v4 tracker's telemetry (~/handtrack/data/telemetry_v4.csv, written by handtrack.main).

  python tools/v4_report.py [CSV] [--run N] [--last SECONDS] [--from T --to T]   (T = seconds from the run's start)

By default only the newest run (since the tracker last started) is summarised; --run -2 is the one
before, --run 1 the first in the file.

Per hand: time sent / missing while enabled, gaps, frame-to-frame position and rotation step (with a
still hand that is the jitter), capture-to-send time, observation rate and age, why crops were rejected, acquisitions / losses, fit residuals and
heatmap confidence.  Poses and events only: the file never contains images.
"""
import argparse
import collections
import os

import numpy as np


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=os.path.expanduser("~/handtrack/data/telemetry_v4.csv"))
    ap.add_argument("--run", type=int, default=-1, help="which run (tracker start): -1 newest (default), 1 first")
    ap.add_argument("--last", type=float, default=0, help="only the last N seconds")
    ap.add_argument("--from", dest="t_from", type=float, default=None)
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    a = ap.parse_args()
    runs, prev = [[]], None
    for ln in open(a.csv):
        if ln.startswith("R,"):                    # tracker (re)started
            if runs[-1]:
                runs.append([])
            prev = None
            continue
        if ln[:2] not in ("F,", "O,", "N,", "A,", "L,", "C,", "X,", "V,"):
            continue
        r = ln.rstrip("\n").split(",")
        t = int(r[1])
        if prev is not None and t < prev - 2_000_000_000 and runs[-1]:
            runs.append([])                        # the clock went back: a reboot (or a file without R rows)
            prev = None
        prev = t if prev is None else max(prev, t)
        runs[-1].append(r)
    runs = [r for r in runs if r]
    if not runs:
        print("no rows")
        return
    try:
        rows = runs[a.run if a.run < 0 else a.run - 1]
    except IndexError:
        print("only %d run(s) in the file" % len(runs))
        return
    if len(runs) > 1:
        print("run %d of %d in the file (--run to choose)" % ((a.run % len(runs)) + 1 if a.run < 0 else a.run, len(runs)))
    t_all = np.array([int(r[1]) for r in rows]) / 1e9
    t0, t1 = t_all.min(), t_all.max()
    lo = t1 - a.last if a.last else t0 + (a.t_from or 0)
    hi = t0 + a.t_to if a.t_to is not None else t1
    rows = [r for r, t in zip(rows, t_all) if lo <= t <= hi]
    print("%s: %.1f s (%.1f .. %.1f s from the run's start)" % (a.csv, hi - lo, lo - t0, hi - t0))
    for side in ("left", "right"):
        F = [r for r in rows if r[0] == "F" and r[3] == side]
        if not F:
            continue
        valid = np.array([r[4] == "1" for r in F])
        t = np.array([int(r[1]) for r in F]) / 1e9
        dur = max(1e-6, t[-1] - t[0]) if len(t) > 1 else 1e-6
        P = {i: np.array([float(x) for x in F[i][6:9]]) for i in range(len(F)) if valid[i]}
        Q = {i: np.array([float(x) for x in F[i][9:13]]) for i in P}
        pairs = [i for i in range(1, len(F)) if i in P and i - 1 in P]
        steps = [np.linalg.norm(P[i] - P[i - 1]) * 1000 for i in pairs]
        rsteps = [np.degrees(2 * np.arccos(min(1.0, abs(float(Q[i] @ Q[i - 1]))))) for i in pairs]
        lat = [(int(F[i][2]) - int(F[i][1])) / 1e6 for i in P]
        gaps, run = [], 0
        for v in valid:
            if not v:
                run += 1
            elif run:
                gaps.append(run)
                run = 0
        if run:
            gaps.append(run)
        O = [r for r in rows if r[0] == "O" and r[2] == side]
        N = collections.Counter(r[3] for r in rows if r[0] == "N" and r[2] == side)
        ev = collections.Counter(r[0] for r in rows if r[0] in "ALCX" and r[2] == side)
        print("\n%s hand: enabled %.1f s, sent %.1f%% of frames, %d gaps (longest %.0f ms)" % (
            side, dur, 100 * valid.mean(), len(gaps), 1000 * max(gaps, default=0) / 60))
        print("  frame-to-frame step: position p50 %.1f p95 %.1f p99 %.1f mm, rotation p50 %.2f p95 %.2f p99 %.2f deg"
              " (with a still hand: jitter)" % (pct(steps, 50), pct(steps, 95), pct(steps, 99),
                                                pct(rsteps, 50), pct(rsteps, 95), pct(rsteps, 99)))
        print("  capture -> sent p50 %.0f p95 %.0f ms" % (pct(lat, 50), pct(lat, 95)))
        if O:
            res = np.array([float(r[6]) for r in O])
            hm = np.array([float(r[5]) for r in O])
            age = np.array([float(r[7]) for r in O])
            stereo = sum(len(r[3]) == 2 for r in O)
            print("  observations %.1f/s (stereo %d), age p50 %.0f p95 %.0f ms, residual p50 %.2f p95 %.2f deg, "
                  "heatmap p50 %.2f p5 %.2f" % (len(O) / dur, stereo, pct(age, 50), pct(age, 95), pct(res, 50),
                                               pct(res, 95), pct(hm, 50), pct(hm, 5)))
        print("  rejected crops: %s" % (dict(N) or "none"))
        print("  acquired %d, lost %d, duplicate dropped %d, restarted %d" % (ev["A"], ev["L"], ev["C"], ev["X"]))
    V = collections.Counter("%s/%s" % (r[2], r[4]) for r in rows if r[0] == "V")
    if V:
        print("\ncandidates rejected by the keypoint model (finder/why): %s" % dict(V))


if __name__ == "__main__":
    main()
