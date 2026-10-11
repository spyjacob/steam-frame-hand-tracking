#!/usr/bin/env python3
"""Validate the Mercury keypoint backend against MediaPipe on identical live frames.

Needs ptgrab streaming (/dev/shm/mrhand-pt).  For each synced frame pair (<= --hz):
  MediaPipe on both cameras -> stereo fuse (reference joints P, head frame)
  Mercury keypoint model per camera, ROI from P ("oracle" ROI) -> rays -> triangulate
Reports per-joint 2D distance (detector pixels) and 3D distance between the two, Mercury's
is_hand probability, and inference time.  Saves a few crops (local only) to OUTDIR.
Refuses to start above 80 C.  Development measurement, not a tracking result.

usage: validate_mercury.py [--seconds 30] [--hz 5] [--out /tmp/merc]
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from handtrack.calib import Rig, triangulate  # noqa: E402
from handtrack.detect import HandDetector, draw  # noqa: E402
from handtrack.live import ROTATE, PtShm  # noqa: E402
from handtrack.mercury_kp import MercuryKeypoint  # noqa: E402
from handtrack.palm import SLOT_CAM, PalmTracker  # noqa: E402


def hottest():
    return max(int(open(z + "/temp").read()) / 1000 for z in glob.glob("/sys/class/thermal/thermal_zone*")
               if open(z + "/type").read().startswith(("cpu", "gpuss")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--hz", type=float, default=5)
    ap.add_argument("--out", default="/tmp/merc")
    a = ap.parse_args()
    T = hottest()
    if T > 88:
        print("too hot (%.0f C); not starting" % T)
        return
    os.makedirs(a.out, exist_ok=True)
    cv2.setNumThreads(1)
    os.nice(10)
    rig = Rig()
    palm = PalmTracker(640, rig)
    mk = MercuryKeypoint(os.path.expanduser("~/handtrack/models/mercury/grayscale_keypoint_jan18.onnx"), rig, SLOT_CAM, 640)
    dets = {s: HandDetector(mirrored=True) for s in (0, 1)}
    shm = PtShm()
    last = {0: 0, 1: 0}
    e2d, e3d, ish, tinf, saved, frames = [], [], [], [], 0, 0
    t0 = time.monotonic()
    nxt = t0
    while time.monotonic() - t0 < a.seconds:
        if hottest() > 91:
            print("hot (>91 C): stopping early")
            break
        r0, r1 = shm.read(0, last[0]), shm.read(1, last[1])
        if not r0 or not r1 or abs(r0[1] - r1[1]) > 8e6 or time.monotonic() < nxt:
            time.sleep(0.005)
            continue
        nxt = time.monotonic() + 1 / a.hz
        last[0], last[1] = r0[0], r1[0]
        imgs = {0: cv2.rotate(r0[2], ROTATE[0]), 1: cv2.rotate(r1[2], ROTATE[1])}
        t_s = r0[1] / 1e9
        hands = {s: dets[s].detect(imgs[s], t_s) for s in (0, 1)}
        poses = palm.fuse(hands[0], hands[1], t_s)
        frames += 1
        for side, p in poses.items():
            if p is None:
                continue
            P = p[3]
            rays = {}
            for slot in (0, 1):
                R, t = rig.head_from_cam(SLOT_CAM[slot])
                jc = (P - t) @ R
                tt = time.perf_counter()
                res = mk.run(slot, imgs[slot], jc, flip=(side == "right"))
                tinf.append((time.perf_counter() - tt) * 1e3)
                if res is None:
                    continue
                ish.append(res["is_hand"])
                ref = next((h for h in hands[slot] if h.side == side), None)
                if ref is not None:
                    e2d.append(np.linalg.norm(res["det_px"] - ref.px, axis=1))
                rays[slot] = (t, res["rays"] @ R.T)
                if saved < 12:
                    vis = draw(imgs[slot], [ref] if ref is not None else [])
                    for (x, y) in res["det_px"]:
                        cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
                    cv2.imwrite(os.path.join(a.out, "f%03d_%s_s%d.jpg" % (frames, side, slot)),
                                np.hstack([vis, cv2.cvtColor(cv2.resize(res["crop"], (640, 640), interpolation=cv2.INTER_NEAREST),
                                                             cv2.COLOR_GRAY2BGR)]))
                    saved += 1
            if 0 in rays and 1 in rays:
                (o1, d1), (o0, d0) = rays[1], rays[0]
                Pm, gap, _, _ = triangulate(np.broadcast_to(o1, d1.shape), d1, np.broadcast_to(o0, d0.shape), d0)
                e3d.append(np.linalg.norm(Pm - P, axis=1))
    print("frames %d, Mercury inferences %d, %.1f ms median (p95 %.1f)" %
          (frames, len(tinf), np.median(tinf) if tinf else 0, np.percentile(tinf, 95) if tinf else 0))
    if ish:
        ish = np.array(ish)
        print("is_hand prob: median %.3f, >= 0.97 in %.0f%% of views" % (np.median(ish), 100 * np.mean(ish >= 0.97)))
    if e2d:
        E = np.array(e2d)
        print("Mercury vs MediaPipe 2D (detector px, 640 image): median %.1f, p90 %.1f; per joint median: %s" %
              (np.median(E), np.percentile(E, 90), " ".join("%.0f" % v for v in np.median(E, 0))))
    if e3d:
        E = np.array(e3d) * 100
        print("Mercury stereo vs MediaPipe stereo 3D (cm): median %.1f, p90 %.1f; palm joints (0,5,9,13,17) median %.1f" %
              (np.median(E), np.percentile(E, 90), np.median(E[:, [0, 5, 9, 13, 17]])))
    print("crops saved to %s (local only); end temp %.0f C" % (a.out, hottest()))


if __name__ == "__main__":
    main()
