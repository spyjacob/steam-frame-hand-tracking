#!/usr/bin/env python3
"""Summarise handtrack telemetry (handtrack/live.py Telemetry CSV).

usage: telemetry_report.py [CSV] [--last SECONDS] [--from T_NS --to T_NS]

Separates what is FRESHLY OBSERVED from what is merely re-sent:
  position   : fresh = stereo/mono_ray optical-flow observation on this frame;
               held / mediapipe-fallback counted separately
  orientation: fresh = a new source capture time (q_age resets), i.e. a new MediaPipe
               result (or per-frame flow orientation when --fast-orient)
  curls      : fresh = a new MediaPipe result for that hand
"""
import argparse
import collections
import os

import numpy as np


def pct(a, ps=(50, 95, 99)):
    a = np.asarray(a, float)
    return " ".join("p%d %.1f" % (p, np.percentile(a, p)) for p in ps) if len(a) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=os.path.expanduser("~/handtrack/data/telemetry.csv"))
    ap.add_argument("--last", type=float, default=0, help="only the last N seconds")
    a = ap.parse_args()

    F, M = [], []
    for line in open(a.csv):
        p = line.rstrip("\n").split(",")
        if p[0] == "F" and len(p) >= 18:
            F.append(p)
        elif p[0] == "M" and len(p) >= 7:
            M.append(p)
    if not F and not M:
        print("no telemetry rows")
        return
    t_end = max([int(r[1]) for r in F] + [int(r[1]) for r in M])
    t_beg = t_end - a.last * 1e9 if a.last else min([int(r[1]) for r in F] + [int(r[1]) for r in M])
    F = [r for r in F if int(r[1]) >= t_beg]
    M = [r for r in M if int(r[1]) >= t_beg]
    dur = (t_end - t_beg) / 1e9
    print("window %.1f s" % dur)

    # ---- MediaPipe runs
    if M:
        runs = len({(r[1], r[3]) for r in M})
        d = collections.defaultdict(list)
        for r in M:
            d[int(r[6])].append(float(r[5]))
        print("MediaPipe: %d runs (%.1f/s), %d camera-detections (%.1f/s)" % (runs, runs / dur, len(M), len(M) / dur))
        for nh in sorted(d):
            print("   %d hand(s) found: n=%4d  duration ms %s" % (nh, len(d[nh]), pct(d[nh])))
        allm = [float(r[5]) for r in M]
        print("   runs > 35 ms (full-image search likely ran): %.0f%%" % (100 * np.mean(np.array(allm) > 35)))

    # ---- per hand
    for side in ("left", "right"):
        R = [r for r in F if r[3] == side]
        if not R:
            print("%s: never sent" % side)
            continue
        tc = np.array([int(r[1]) for r in R])
        ts = np.array([int(r[2]) for r in R])
        src = collections.Counter(r[4] for r in R)
        pos_age = np.array([float(r[5]) for r in R])
        q_src = collections.Counter(r[6] for r in R)
        q_age = np.array([float(r[7]) for r in R])
        c_age = np.array([float(r[8]) for r in R])
        P = np.array([[float(x) for x in r[11:14]] for r in R])
        Q = np.array([[float(x) for x in r[14:18]] for r in R])
        # fresh orientation / curl observations: the source capture time changes
        q_t = ts / 1e9 - q_age / 1e3
        c_t = ts / 1e9 - c_age / 1e3
        q_new = int(np.sum(np.abs(np.diff(q_t)) > 1e-3)) + 1
        c_new = int(np.sum(np.abs(np.diff(c_t)) > 1e-3)) + 1
        fresh_pos = sum(v for k, v in src.items() if k.startswith(("stereo", "mono_ray")))
        span = max((ts[-1] - ts[0]) / 1e9, 1e-3)
        print("%s: %d frames sent over %.1f s visible-ish (%.1f/s)" % (side, len(R), span, len(R) / span))
        print("   position sources: %s" % dict(src))
        print("   fresh position observations %.1f/s | age at send ms %s" % (fresh_pos / span, pct(pos_age)))
        print("   orientation sources: %s" % dict(q_src))
        print("   fresh orientation observations %.1f/s | age at send ms %s" % (q_new / span, pct(q_age)))
        print("   fresh finger-curl observations %.1f/s | age at send ms %s" % (c_new / span, pct(c_age)))
        print("   capture->send ms %s" % pct((ts - tc) / 1e6))
        # frame-to-frame steps (not accuracy): position mm, rotation deg
        if len(P) > 3:
            dp = np.linalg.norm(np.diff(P, axis=0), axis=1) * 1000
            dq = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(Q[1:] * Q[:-1], axis=1)), 0, 1)))
            print("   step per sent frame: position mm %s | rotation deg %s" % (pct(dp), pct(dq)))


if __name__ == "__main__":
    main()
