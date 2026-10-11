#!/usr/bin/env python3
"""Validate passthrough stereo calibration against live frames (needs `ptgrab stream`).

Matches ORB features between the two raw passthrough buffers, then for each
hypothesis (which shm slot is passthrough_left, whether intrinsics need the
calibration's crop offset) triangulates the matches in the head frame and
reports the ray-gap statistics.  The right hypothesis gives gaps of a few mm.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from handtrack.calib import Rig, triangulate  # noqa: E402
from handtrack.live import PtShm  # noqa: E402


def main():
    rig = Rig()
    shm = PtShm()
    a = shm.read(0)[2]
    b = shm.read(1)[2]
    orb = cv2.ORB_create(6000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    m = sorted(m, key=lambda x: x.distance)[:1500]
    pa = np.array([ka[x.queryIdx].pt for x in m])
    pb = np.array([kb[x.trainIdx].pt for x in m])
    print("%d matches" % len(m))
    crops = {"passthrough_left": 550, "passthrough_right": 250}
    for left_slot in (1, 0):
        for use_crop in (False, True):
            pl, pr = (pb, pa) if left_slot == 1 else (pa, pb)
            out = {}
            for name, px in (("passthrough_left", pl), ("passthrough_right", pr)):
                cam = rig.cams[name]
                q = px.copy()
                if use_crop:
                    q[:, 0] += crops[name]
                R, t = rig.head_from_cam(name)
                out[name] = (t, cam.unproject(q) @ R.T)
            (tl, dl), (tr, dr) = out["passthrough_left"], out["passthrough_right"]
            P, gap, s, u = triangulate(tl, dl, tr, dr)
            ok = (s > 0.1) & (u > 0.1)
            g = gap[ok] * 1000
            q25, q50, q75 = np.percentile(g, [25, 50, 75]) if len(g) else (0, 0, 0)
            print("left=slot%d crop=%-5s  in-front %4d/%d  gap mm p25 %.1f p50 %.1f p75 %.1f  depth p50 %.2f m"
                  % (left_slot, use_crop, ok.sum(), len(gap), q25, q50, q75,
                     np.median(-P[ok, 2]) if ok.any() else 0))


if __name__ == "__main__":
    main()
