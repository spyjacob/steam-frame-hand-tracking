#!/usr/bin/env python3
"""Time the v4 tracker's pieces on this machine (run on the headset, tracker stopped or running).

  python tools/bench_models.py [--seconds 4] [--models ~/handtrack/models/mercury]

Reports p50/p95 ms for: one Mercury keypoint crop (1 and 2 ONNX Runtime threads), two crops at once
on two worker threads (what the tracker does for a stereo update or two hands), the Mercury detector,
and the palm fit (one and two views).  Uses the current camera frame if ptgrab is running, else
noise (inference time does not depend on the content).  Affinity / priority follow HANDTRACK_CPUS
like the tracker, so run it as `chrt --idle 0 ...` to see what SCHED_IDLE costs.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from handtrack import mercury as M  # noqa: E402
from handtrack.handfit import PALM, PalmTemplate, ViewObs, fit_palm  # noqa: E402
from handtrack.headset import lower_priority  # noqa: E402


def pct(a):
    return "p50 %5.1f  p95 %5.1f ms" % (np.percentile(a, 50), np.percentile(a, 95))


def timed(fn, seconds):
    out = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--models", default=os.path.expanduser("~/handtrack/models/mercury"))
    a = ap.parse_args()
    lower_priority()
    try:
        from handtrack.calib import Rig
        from handtrack.main import build_views
        views = build_views(Rig(), 640)
    except Exception as e:                       # no calibration here: a generic fisheye
        print("(no rig calibration: %r; using a generic camera)" % e)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from sim_tracker import sim_views
        views = sim_views()
    img = None
    try:
        from handtrack.headset import PtShm
        r = PtShm().read(0)
        img = r[2] if r else None
    except Exception:
        pass
    if img is None or img.shape[0] != 640:
        img = np.random.default_rng(0).integers(0, 255, (640, 640), np.uint8)
        print("(no camera frame: noise image)")
    v = views[0]
    X = np.array([[0.10, -0.12, -0.40]]) + np.random.default_rng(1).normal(0, 0.03, (21, 3))
    ins = M.instr_from_points(v.to_cam(X), flip=False)
    path_kp = os.path.join(a.models, "grayscale_keypoint_jan18.onnx")
    path_det = os.path.join(a.models, "grayscale_detection_160x160.onnx")
    print("cpus %s, policy %s" % (sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "-",
                                  os.sched_getscheduler(0) if hasattr(os, "sched_getscheduler") else "-"))
    for th in (1, 2):
        kp = M.KeypointModel(path_kp, th)
        kp.run(v, img, ins)
        print("keypoint crop, %d ORT thread(s):        %s" % (th, pct(timed(lambda: kp.run(v, img, ins), a.seconds))))
    kp = M.KeypointModel(path_kp, 1)
    pool = ThreadPoolExecutor(2)
    pair = timed(lambda: [f.result() for f in [pool.submit(kp.run, v, img, ins), pool.submit(kp.run, v, img, ins)]],
                 a.seconds)
    print("two crops at once (2 workers x 1 thread): %s  -> %.0f crops/s max" % (pct(pair), 2000 / np.median(pair)))
    det = M.HandDetector160(path_det, 1)
    print("Mercury detector (160x160):              %s" % pct(timed(lambda: det.run(v, img), a.seconds)))
    tpl = PalmTemplate("left")
    obs = []
    for slot, vv in views.items():
        U = X[PALM] - vv.t
        obs.append(ViewObs(vv.t, U / np.linalg.norm(U, axis=1)[:, None], np.ones(5), np.zeros(5), 1.0, slot))
    c0 = X[PALM].mean(0)
    print("palm fit, one view:                      %s" % pct(timed(lambda: fit_palm(tpl.T, "left", obs[:1], [(c0, np.eye(3), "x")]), 1.0)))
    print("palm fit, two views:                     %s" % pct(timed(lambda: fit_palm(tpl.T, "left", obs, [(c0, np.eye(3), "x")]), 1.0)))
    pool.shutdown()


if __name__ == "__main__":
    main()
