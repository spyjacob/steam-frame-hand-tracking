#!/usr/bin/env python3
"""v4 hand tracker on the Steam Frame: one keypoint model, one filter per hand (handtrack/tracker.py).

  camera frames (ptgrab, /dev/shm/mrhand-pt, both cameras at 60 Hz)
    -> Tracker.on_frame: Mercury keypoint crops on the tracked hands, palm fit, Kalman filter;
       Mercury detector + MediaPipe (own process) only while an enabled hand is missing
    -> Bridge: poses predicted to the frame's capture time -> PC driver (UDP, ~150 bytes per frame)

Status line every 5 s (stderr, /tmp/live.log; the dashboard panel reads its "[T C] ... fast:" prefix).
Telemetry: ~/handtrack/data/telemetry_v4.csv (rotated at 20 MB, poses and events only).
Debug view: MJPEG on 127.0.0.1:8765 (ssh -L 8765:127.0.0.1:8765 frame -> http://localhost:8765).

usage: python -m handtrack.main [--no-http] [--set key=value ...] [--seconds N]
       tracker parameters (handtrack/tracker.py DEFAULTS) can also be set live in ~/.config/mrhand/v4.conf
       (key = value per line, re-read once a second; delete the file to go back to the defaults)
"""
import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Small per-frame work: thread pools inside numpy/OpenCV only add spinning cores (heat).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)

from handtrack import mercury as M  # noqa: E402
from handtrack.calib import Rig  # noqa: E402
from handtrack.headset import EnabledHands, PtShm, Telemetry, ThermalGuard, Viewer, lower_priority  # noqa: E402
from handtrack.tracker import DEFAULTS, SIDES, Tracker  # noqa: E402

SLOT_CAM = {1: "passthrough_left", 0: "passthrough_right"}     # validated by tools/check_stereo.py
MODELS = os.path.expanduser("~/handtrack/models/mercury")
PARAMS_FILE = os.path.expanduser("~/.config/mrhand/v4.conf")


class MPFinder:
    """MediaPipe as a hand finder in its own process (handtrack/mpworker.py).  At most one run in
    flight; results come back as camera-frame keypoint rays for the tracker to check."""

    def __init__(self, views, gpu=False):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from handtrack import mpworker
        self.views = views
        self.pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=mpworker.init, initargs=(gpu, True))
        self.fn = mpworker.find
        self.job = None
        self.paused = False
        self.ms = []
        self.errors = []

    def busy(self):
        return self.job is not None or self.paused

    def submit(self, slot, view, img, t):
        self.job = (self.pool.submit(self.fn, slot, img, t), slot, t)

    def poll(self):
        if self.job is None or not self.job[0].done():
            return 0, []
        fut, slot, t = self.job
        self.job = None
        try:
            sec, hands = fut.result()
        except Exception as e:              # a dead worker must not stop tracking
            self.errors.append("mediapipe: %r" % e)
            return 1, []
        self.ms.append(sec * 1e3)
        v = self.views[slot]
        out = []
        for side, score, px_up in hands:
            rays = v.rays_cam(v.shm_from_upright(np.asarray(px_up, float)))
            out.append({"kind": "mp", "t": t, "slot": slot, "side": side, "score": float(score), "rays": rays})
        return 1, out

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


class Params:
    """Tracker parameters editable while running: ~/.config/mrhand/v4.conf, `key = value` per line
    (# comments).  Missing file = defaults (plus --set overrides)."""

    def __init__(self, base, path=PARAMS_FILE):
        self.base, self.path, self.mtime, self.next = dict(base), path, -1.0, 0.0     # first poll always reports

    def poll(self):
        now = time.monotonic()
        if now < self.next:
            return None
        self.next = now + 1.0
        try:
            mt = os.stat(self.path).st_mtime
        except OSError:
            mt = None
        if mt == self.mtime:
            return None
        self.mtime = mt
        vals = dict(self.base)
        if mt is not None:
            try:
                for line in open(self.path).read().splitlines():
                    line = line.split("#", 1)[0].strip()
                    if "=" not in line:
                        continue
                    k, v = (s.strip() for s in line.split("=", 1))
                    if k not in DEFAULTS:
                        print("v4.conf: unknown key %r ignored" % k, file=sys.stderr, flush=True)
                        continue
                    vals[k] = _parse_value(k, v)
            except (OSError, ValueError) as e:
                print("v4.conf: ignored (%r)" % e, file=sys.stderr, flush=True)
                return None
        changed = {k: v for k, v in vals.items() if v != DEFAULTS.get(k)}
        print("params: %s" % (" ".join("%s=%s" % kv for kv in sorted(changed.items())) or "defaults"),
              file=sys.stderr, flush=True)
        return vals


def _parse_value(k, v):
    d = DEFAULTS[k]
    if isinstance(d, tuple):
        return tuple(float(x) for x in v.replace(",", " ").split())
    if isinstance(d, int) and not isinstance(d, bool):
        return int(float(v))
    return float(v)


def build_views(rig, size, raw_size=2464):
    """The two passthrough cameras.  Calibration intrinsics apply to the raw 2464x2464 buffer, which
    ptgrab downscales to `size` (validated on live frames by tools/check_stereo.py)."""
    views = {}
    for slot, name in SLOT_CAM.items():
        R, t = rig.head_from_cam(name)
        cam = rig.cams[name]
        if int(cam.w) != raw_size:
            print("WARNING: %s calibration is %dx%d, expected %d: check tools/check_stereo.py" % (
                name, cam.w, cam.h, raw_size), file=sys.stderr, flush=True)
        views[slot] = M.View(slot, cam, R, t, size=size, raw_size=raw_size)
    return views


def draw_debug(views, imgs, tracker, out):
    """Upright camera images side by side: tracked hands' predicted palm + last keypoints, boxes."""
    tiles = []
    for slot in (1, 0):                                        # left camera on the left
        v = views[slot]
        up = cv2.cvtColor(v.upright(imgs[slot]), cv2.COLOR_GRAY2BGR)
        for side, h in tracker.hands.items():
            col = (0, 200, 255) if side == "left" else (255, 120, 0)
            r = h.last.get(slot)
            if r is not None:
                for p in r.px:
                    q = v.upright_from_shm(p)
                    cv2.circle(up, (int(q[0]), int(q[1])), 3, col, -1, cv2.LINE_AA)
            o = out.get(side)
            if o is not None:
                c = v.upright_from_shm(v.project_head(np.asarray(o[0])[None])[0])
                cv2.drawMarker(up, (int(c[0]), int(c[1])), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
                cv2.putText(up, "%s %.0f %.0f %.0f cm" % (side, *(100 * np.asarray(o[0]))), (int(c[0]) + 8, int(c[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        for b in tracker.boxes:
            if b.slot == slot:
                c = v.upright_from_shm(np.asarray(b.centre))
                cv2.circle(up, (int(c[0]), int(c[1])), int(b.size / 2), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(up, SLOT_CAM[slot], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(up)
    return np.hstack(tiles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-http", action="store_true", help="no debug view")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="tracker parameter (see handtrack/tracker.py DEFAULTS), e.g. --set hz=30")
    ap.add_argument("--workers", type=int, default=2, help="threads running keypoint crops / detection")
    ap.add_argument("--ort-threads", type=int, default=1, help="ONNX Runtime threads per inference")
    ap.add_argument("--no-mediapipe", action="store_true", help="hand finding by Mercury's detector only")
    ap.add_argument("--no-detector", action="store_true", help="hand finding by MediaPipe only")
    ap.add_argument("--gpu", action="store_true", help="MediaPipe GPU delegate (shares the GPU with the compositor)")
    ap.add_argument("--no-bridge", action="store_true", help="do not serve the PC driver")
    ap.add_argument("--telemetry", default=os.path.expanduser("~/handtrack/data/telemetry_v4.csv"),
                    help="telemetry CSV (rotated at 20 MB); '' disables")
    ap.add_argument("--models", default=MODELS)
    ap.add_argument("--seconds", type=float, default=0, help="stop after N s (0 = run forever)")
    a = ap.parse_args()

    base = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in DEFAULTS:
            ap.error("unknown tracker parameter %r" % k)
        base[k] = _parse_value(k, v)
    params = Params(base)

    lower_priority()
    print("v4 tracker: pid %d, cpus %s, policy %s" % (
        os.getpid(), sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "-",
        {0: "OTHER", 3: "BATCH", 5: "IDLE"}.get(os.sched_getscheduler(0), "?") if hasattr(os, "sched_getscheduler") else "-"),
        file=sys.stderr, flush=True)
    thermal = ThermalGuard()
    tel = Telemetry(a.telemetry) if a.telemetry else None
    shm = PtShm()
    size = 640
    rig = Rig()
    views = build_views(rig, size)
    kp = M.KeypointModel(os.path.join(a.models, "grayscale_keypoint_jan18.onnx"), a.ort_threads)
    det = None if a.no_detector else M.HandDetector160(os.path.join(a.models, "grayscale_detection_160x160.onnx"),
                                                       a.ort_threads)
    mp = None if a.no_mediapipe else MPFinder(views, a.gpu)
    pool = ThreadPoolExecutor(a.workers)
    tracker = Tracker(views, kp, pool, det=det, mp=mp, prm=params.poll())
    bridge = None
    if not a.no_bridge:
        from handtrack.bridge import Bridge
        bridge = Bridge()
    hands_cfg = EnabledHands()
    viewer = None if a.no_http else Viewer()
    if viewer:
        viewer.serve()
        print("debug view: ssh -L 8765:127.0.0.1:8765 frame  ->  http://localhost:8765", file=sys.stderr, flush=True)
    print("models: keypoint + %s; %d workers; shm %dx%d" % (
        " + ".join(x for x, on in (("detector", det), ("MediaPipe", mp)) if on) or "no finder", a.workers, shm.w, shm.h),
        file=sys.stderr, flush=True)

    stop = threading.Event()
    if a.seconds:
        threading.Timer(a.seconds, stop.set).start()

    def new_stats():
        return {"pairs": 0, "sent": {sd: 0 for sd in SIDES}, "lat": [], "frame_ms": [], "step": {sd: [] for sd in SIDES},
                "unpaired": 0}

    st = new_stats()
    last_seq = {0: 0, 1: 0}
    grid_t, grid_period = None, 1e9 / 60.0
    prev_out = {}
    t_stat = time.monotonic()
    t_view = 0.0
    enabled = None
    n_err = 0
    try:
        while not stop.is_set():
            p = params.poll()
            if p is not None:
                tracker.prm.update(p)
            en = hands_cfg.get()
            if en != enabled:
                tracker.set_enabled(en)
                enabled = set(en)
            if bridge:
                bridge.enabled_mask = (1 if "left" in enabled else 0) | (2 if "right" in enabled else 0)
            r0, r1 = shm.read(0, last_seq[0]), shm.read(1, last_seq[1])
            if not r0 or not r1:
                tracker.collect()                       # results between frames: no waiting for the next one
                time.sleep(0.001)
                continue
            if abs(r0[1] - r1[1]) > 8e6:                # not the same capture: wait for the lagging camera
                if r0[1] < r1[1]:
                    last_seq[0] = r0[0]
                else:
                    last_seq[1] = r1[0]
                st["unpaired"] += 1
                continue
            last_seq[0], last_seq[1] = r0[0], r1[0]
            t_raw = min(r0[1], r1[1])
            # snap to the camera's 60 Hz grid (ptgrab's detection time jitters by a few ms)
            if grid_t is None or abs(t_raw - grid_t) > 1e9:
                grid_t = t_raw
            else:
                nfr = max(1, round((t_raw - grid_t) / grid_period))
                pred = grid_t + nfr * grid_period
                resid = t_raw - pred
                grid_t = pred + 0.05 * resid
                grid_period = min(max(grid_period + 0.002 * resid / nfr, 15.0e6), 18.5e6)
            t_cap = int(grid_t)
            imgs = {0: r0[2], 1: r1[2]}
            if r0[2].shape[0] != size:
                imgs = {k: cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA) for k, im in imgs.items()}
            tf = time.perf_counter()
            tracker.scale = thermal.scale()
            if mp is not None:
                mp.paused = thermal.mp_paused
            out = tracker.on_frame(t_cap / 1e9, imgs)
            if bridge:
                bridge.send(out, t_cap)
            t_send = time.monotonic_ns()
            st["frame_ms"].append((time.perf_counter() - tf) * 1e3)
            st["pairs"] += 1
            st["lat"].append((t_send - t_cap) / 1e6)
            for sd in SIDES:
                o = out.get(sd)
                if o is not None:
                    st["sent"][sd] += 1
                    if sd in prev_out:
                        st["step"][sd].append(float(np.linalg.norm(o[0] - prev_out[sd])) * 1e3)
                    prev_out[sd] = o[0]
                else:
                    prev_out.pop(sd, None)
                if tel and sd in enabled:
                    h = tracker.hands.get(sd)
                    age = (t_cap / 1e9 - h.t_obs) * 1e3 if h is not None else -1
                    if o is None:
                        tel.write("F,%d,%d,%s,0,%.0f" % (t_cap, t_send, sd, age))
                    else:
                        tel.write("F,%d,%d,%s,1,%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f" % (
                            t_cap, t_send, sd, age, *o[0], *o[1], o[2]))
            if tel:
                while tracker.events:
                    tel.write(tracker.events.popleft())
            else:
                tracker.events.clear()
            if viewer and viewer.clients and time.monotonic() - t_view > 0.1:
                t_view = time.monotonic()
                viewer.publish(draw_debug(views, imgs, tracker, out))

            now = time.monotonic()
            if now - t_stat > 5.0:
                dt = now - t_stat
                ts = tracker.stats
                lat = np.percentile(st["lat"], [50, 95]) if st["lat"] else (0, 0)
                fm = np.median(st["frame_ms"]) if st["frame_ms"] else 0
                cm = np.median(ts["crop_ms"]) if ts["crop_ms"] else 0
                age = np.median(ts["obs_age_ms"]) if ts["obs_age_ms"] else 0
                steps = " ".join("%s %.1f/%.1f" % (sd[0].upper(), *np.percentile(st["step"][sd], [50, 95]))
                                 for sd in SIDES if st["step"][sd])
                tracked = " ".join("%s(%.0f/s,scale %.2f)" % (sd, ts["upd"][sd] / dt, tracker.tpl[sd].scale)
                                   for sd in SIDES if sd in tracker.hands)
                line = ("[%.0f C] fast: %.1f frames/s, %.1f ms/frame, poses/s L %.1f R %.1f, latency p50 %.0f p95 %.0f ms"
                        " | crops %.1f/s %.1f ms, accepted %d (stereo %d), missed %d, outliers %d, obs age %.0f ms"
                        " | tracked: %s | step mm p50/p95: %s | search: det %d mp %d verify %d acquired %d lost %d"
                        " | scale %.1f" % (
                            thermal.temp, st["pairs"] / dt, fm, st["sent"]["left"] / dt, st["sent"]["right"] / dt,
                            lat[0], lat[1], ts["crops"] / dt, cm, ts["acc"], ts["acc_stereo"], ts["miss"],
                            ts["outlier"], age, tracked or "-", steps or "-", ts["det_runs"], ts["mp_runs"],
                            ts["verify"], ts["acquired"], ts["lost"], tracker.scale))
                print(line, file=sys.stderr, flush=True)
                if tel:
                    tel.write("S,%d,%s" % (time.monotonic_ns(), line.replace(",", ";")))
                    tel.flush()
                errs = tracker.errors + (mp.errors if mp else [])
                if errs and n_err < 20:
                    print("errors (%d): %s" % (len(errs), " | ".join(errs[:3])), file=sys.stderr, flush=True)
                    n_err += 1
                tracker.errors.clear()
                if mp:
                    mp.errors.clear()
                tracker.stats = tracker._new_stats()
                st = new_stats()
                t_stat = now
    except KeyboardInterrupt:
        pass
    finally:
        if mp:
            mp.close()
        pool.shutdown(wait=False, cancel_futures=True)
        if tel:
            tel.flush()


if __name__ == "__main__":
    main()
