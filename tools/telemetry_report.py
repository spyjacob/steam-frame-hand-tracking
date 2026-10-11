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
    ap.add_argument("--seg", type=int, default=-1,
                    help="only profile segment N (between P rows written on each live profile change); "
                         "--seg -2 lists the segments")
    a = ap.parse_args()

    F, M, Rr, Pr = [], [], [], []
    for line in open(a.csv):
        p = line.rstrip("\n").split(",")
        if p[0] == "F" and len(p) >= 18:
            F.append(p)
        elif p[0] == "M" and len(p) >= 7:
            M.append(p)
        elif p[0] == "R" and len(p) >= 17:
            Rr.append(p)
        elif p[0] == "P" and len(p) >= 3:
            Pr.append((int(p[1]), ",".join(p[2:])))
    if not F and not M:
        print("no telemetry rows")
        return
    t_end = max([int(r[1]) for r in F] + [int(r[1]) for r in M])
    t_beg = t_end - a.last * 1e9 if a.last else min([int(r[1]) for r in F] + [int(r[1]) for r in M])
    if a.seg == -2:
        for i, (tp, desc) in enumerate(Pr):
            print("segment %d: %.1f s  %s" % (i, ((Pr[i + 1][0] if i + 1 < len(Pr) else t_end) - tp) / 1e9, desc))
        return
    if a.seg >= 0:
        t_beg = Pr[a.seg][0] + 3e9                     # skip 3 s after a switch (filters settle)
        t_end = Pr[a.seg + 1][0] if a.seg + 1 < len(Pr) else t_end
        print("profile segment %d: %s" % (a.seg, Pr[a.seg][1]))
        F = [r for r in F if int(r[1]) <= t_end]
        M = [r for r in M if int(r[1]) <= t_end]
        Rr = [r for r in Rr if int(r[8]) <= t_end]
    F = [r for r in F if int(r[1]) >= t_beg]
    M = [r for r in M if int(r[1]) >= t_beg]
    Rr = [r for r in Rr if int(r[8]) >= t_beg]
    dur = (t_end - t_beg) / 1e9
    print("window %.1f s" % dur)

    # ---- Mercury observations (R rows): metric 1, new-result processing latency, per stage
    # R,id,side,slot,t_cap,t_sub,t0,t1,t_col,cpu_ms,wait_ms,njob,outcome,qw,qx,qy,qz
    if Rr:
        print("Mercury observations: %d (%.1f/s), hands per job %s" % (
            len(Rr), len(Rr) / dur, dict(collections.Counter(r[11] for r in Rr))))
        print("   outcomes %s" % dict(collections.Counter(r[12] for r in Rr)))
        cap, sub, t0, t1, col = (np.array([int(r[k]) for r in Rr], float) for k in (4, 5, 6, 7, 8))
        ok = t0 > 0
        ms = lambda x: pct(x[ok] / 1e6)
        print("   frame age at submit (cap->submit) ms %s" % ms(sub - cap))
        print("   to worker start (submit->start: IPC, pickling, earlier crops in the job) ms %s" % ms(t0 - sub))
        print("   inference wall (start->end) ms %s" % ms(t1 - t0))
        print("   inference thread CPU ms %s | runqueue wait ms %s" % (
            pct([float(r[9]) for r in Rr]), pct([float(r[10]) for r in Rr])))
        print("   back to tracker (end->collect: IPC + loop cadence) ms %s" % ms(col - t1))
        nohint = [r for r in Rr if len(r) > 17 and r[17] == "0"]
        if nohint:
            print("   crops WITHOUT keypoint hint (recovery): %d, outcomes %s" % (
                len(nohint), dict(collections.Counter(r[12] for r in nohint))))
        acc = np.array([r[12] in ("mono", "stereo") for r in Rr])
        print("   NEW-RESULT LATENCY cap->collect, accepted only ms %s" % pct(((col - cap) / 1e6)[ok & acc]))
        for side in ("left", "right"):
            rs = [r for r in Rr if r[2] == side and r[12] in ("mono", "stereo")]
            if len(rs) > 1:
                tt = np.array([int(r[4]) for r in rs]) / 1e9
                print("   %s: %.1f accepted observations/s, gap between them ms %s" % (
                    side, len(rs) / dur, pct(np.diff(np.sort(tt)) * 1e3)))

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
            # tracking loss: time not sent (gaps > 100 ms) + time frozen ("hold"); depth/position excursions
            gaps = np.diff(ts) / 1e9
            lost_s = float(np.sum(gaps[gaps > 0.1]))
            hold_s = sum(1 for r in R if r[4] == "hold") / 60.0
            print("   LOST: not sent %.1f s (%d gaps >100 ms) + frozen %.1f s, of %.1f s | position jumps >5 cm/frame: %d" % (
                lost_s, int(np.sum(gaps > 0.1)), hold_s, span, int(np.sum(dp > 50))))
            still = (dp < 0.3) & (np.diff(ts) < 40e6)     # palm moving < ~18 mm/s, consecutive frames
            if still.sum() > 30:
                print("   STILL hand (%d frames): rotation jitter deg/frame %s" % (still.sum(), pct(dq[still])))
        # metric 3: motion-response lag of the sent orientation behind the raw (unfiltered) Mercury
        # observations, by the shift that best aligns them.  Only meaningful while the wrist turns.
        raw = sorted((int(r[4]) / 1e9, np.array([float(x) for x in r[13:17]])) for r in Rr
                     if r[2] == side and r[12] in ("mono", "stereo"))
        if len(raw) > 20 and len(R) > 60:
            rt = np.array([x[0] for x in raw])
            rq = np.array([x[1] for x in raw])
            send_t = ts / 1e9
            moving = np.r_[False, dq > 0.5] if len(P) > 3 else np.zeros(len(R), bool)
            if moving.sum() > 30:
                best = None
                for L in np.arange(0.0, 0.401, 0.005):
                    # raw orientation at (send - L), interpolated between the bracketing observations
                    # (comparing against the last-held observation would bias L low by ~half a gap)
                    tq = send_t[moving] - L
                    idx = np.searchsorted(rt, tq, side="right") - 1
                    okk = (idx >= 0) & (idx < len(rt) - 1)
                    if okk.sum() < 20:
                        continue
                    i0 = idx[okk]
                    a0, a1 = rq[i0], rq[i0 + 1]
                    a1 = a1 * np.sign(np.sum(a0 * a1, axis=1))[:, None]
                    f = ((tq[okk] - rt[i0]) / np.maximum(rt[i0 + 1] - rt[i0], 1e-6))[:, None]
                    qi = a0 + f * (a1 - a0)
                    qi /= np.linalg.norm(qi, axis=1)[:, None]
                    ang = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(Q[moving][okk] * qi, axis=1)), 0, 1)))
                    m = float(np.mean(ang))
                    if best is None or m < best[1]:
                        best = (L, m)
                    if L == 0.0:
                        m0 = m
                if best:
                    print("   orientation lag behind raw observations (capture time -> sent, while turning): "
                          "%.0f ms (mean error %.1f deg there, %.1f deg at 0 ms)" % (best[0] * 1e3, best[1], m0))


if __name__ == "__main__":
    main()
