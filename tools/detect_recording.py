#!/usr/bin/env python3
"""Run hand detection over a decoded dataset stream (occ_extract.py --dump output).

usage: detect_recording.py STREAMDIR [--out DIR] [--debug-every N]
Writes STREAMDIR/hands.csv (one row per detected hand) and debug overlays.
This is RECORDED Arcturus data, not a live test.
"""
import argparse
import csv
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from handtrack.detect import HandDetector, draw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("streamdir")
    ap.add_argument("--debug-every", type=int, default=25)
    a = ap.parse_args()

    ts = []
    with open(os.path.join(a.streamdir, "timestamps.csv")) as f:
        for row in csv.DictReader(f):
            ts.append(float(row["ts"]))
    cap = cv2.VideoCapture(os.path.join(a.streamdir, "stream.h264"))
    dbg = os.path.join(a.streamdir, "debug")
    os.makedirs(dbg, exist_ok=True)
    det = HandDetector()
    n = nd = 0
    counts = [0, 0, 0]
    tsum = 0.0
    with open(os.path.join(a.streamdir, "hands.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["frame", "ts", "side", "side_score"] + ["%s%d" % (c, i) for i in range(21) for c in "xy"])
        while True:
            r, img = cap.read()
            if not r or n >= len(ts):
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            t0 = time.perf_counter()
            hands = det.detect(gray, ts[n])
            tsum += time.perf_counter() - t0
            counts[min(len(hands), 2)] += 1
            for hd in hands:
                w.writerow([n, "%.6f" % ts[n], hd.side, "%.3f" % hd.side_score] +
                           ["%.1f" % v for v in hd.px.reshape(-1)])
            if hands and nd < 400 and n % a.debug_every == 0:
                cv2.imwrite(os.path.join(dbg, "%06d.jpg" % n), draw(gray, hands))
                nd += 1
            n += 1
    det.close()
    print("frames %d  0 hands: %d  1 hand: %d  2 hands: %d  mean detect %.1f ms"
          % (n, counts[0], counts[1], counts[2], 1e3 * tsum / max(n, 1)))


if __name__ == "__main__":
    main()
