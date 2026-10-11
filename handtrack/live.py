#!/usr/bin/env python3
"""Live optical hand detection on the Steam Frame passthrough cameras.

Frames come from `ptgrab stream` via /dev/shm/mrhand-pt (luma, 2464x2464 per
camera, sensors mounted rotated ±90°).  Each camera gets its own MediaPipe
HandLandmarker in VIDEO mode.

Debug view: MJPEG on 127.0.0.1:8765 (camera imagery never leaves the headset
unless you tunnel it):  ssh -L 8765:127.0.0.1:8765 frame   then open
http://localhost:8765 on the PC.

usage: python -m handtrack.live [--size 640] [--no-http] [--log hands.jsonl]
"""
import argparse
import json
import math
import mmap
import os
import struct
import sys
import threading
import types
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def cpus_from_env():
    """CPU cores for our workload: HANDTRACK_CPUS like "0-2" or "0,1,2" (default 0-6, never the prime core 7)."""
    spec = os.environ.get("HANDTRACK_CPUS", "0-6")
    out = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        elif part.strip():
            out.add(int(part))
    return out or set(range(7))


# Small per-frame work: thread pools only add spinning cores (heat), so one thread each.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)

from handtrack.calib import triangulate
from handtrack.detect import HandDetector, draw
from handtrack.fasttrack import FastTracker, plausible
from handtrack.palm import PALM_IDX, Bridge, OneEuro, PalmTracker

SHM = "/dev/shm/mrhand-pt"
# which hands to track, written by the "Hand Tracking" dashboard panel: "left", "right" or "left right"
# (missing file = both).  Re-read about once a second; a disabled hand gets no Mercury crops, no
# optical flow and is sent as invalid, so one-hand mode saves roughly half the per-hand work.
HANDS_FILE = os.path.expanduser("~/.config/mrhand/hands")


PROFILE_FILE = os.path.expanduser("~/.config/mrhand/profile")
# Latency/smoothing profile, editable live (re-read about once a second).  Missing file or missing key =
# the current smooth defaults below, so deleting the file is an instant rollback.
#   axis_filter   = <min_cutoff> <beta> | off   palm orientation axes One Euro (palm.py)
#   centre_filter = <min_cutoff> <beta> | off   palm centre One Euro inside the pose (palm.py)
#   curl_filter   = <min_cutoff> <beta> | off   finger curls One Euro
#   pos_filter    = <min_cutoff> <beta> | off   sent position One Euro (fast path)
#   rot_glide_ms  = <ms>  (0 = off)             exponential glide toward the newest orientation
# Default = "A" from the 2026-10-11 latency pass: smoother than before when the hand is still, faster when
# it moves (orientation lag ~130 vs ~160 ms, same still-hand jitter).  The previous smooth profile is
# tools/profile_smooth_v1.conf (copy it to ~/.config/mrhand/profile to go back).
PROFILE_DEFAULTS = {"axis_filter": (0.4, 1.2), "centre_filter": (1.5, 0.6), "curl_filter": (1.0, 1.0),
                    "pos_filter": (1.2, 1.5), "rot_glide_ms": 25.0}


class Profile:
    def __init__(self, path=PROFILE_FILE):
        self.path, self.mtime, self.next = path, None, 0.0
        self.values = dict(PROFILE_DEFAULTS)

    def _parse(self, text):
        v = dict(PROFILE_DEFAULTS)
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            k, val = (s.strip() for s in line.split("=", 1))
            if k not in v:
                continue
            if k == "rot_glide_ms":
                v[k] = float(val)
            elif val == "off":
                v[k] = None
            else:
                a, b = val.split()[:2]
                v[k] = (float(a), float(b))
        return v

    def poll(self):
        """Returns the new values dict when the file changed, else None."""
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
        try:
            self.values = self._parse(open(self.path).read()) if mt is not None else dict(PROFILE_DEFAULTS)
        except (OSError, ValueError) as e:
            print("profile: ignoring %s (%r)" % (self.path, e), file=sys.stderr, flush=True)
            return None
        print("profile: %s" % " ".join("%s=%s" % kv for kv in sorted(self.values.items())), file=sys.stderr, flush=True)
        return self.values


class EnabledHands:
    def __init__(self, path=HANDS_FILE):
        self.path, self.mtime, self.sides, self.next = path, None, {"left", "right"}, 0.0

    def get(self):
        now = time.monotonic()
        if now >= self.next:
            self.next = now + 1.0
            try:
                mt = os.stat(self.path).st_mtime
                if mt != self.mtime:
                    self.mtime = mt
                    self.sides = set(open(self.path).read().split()) & {"left", "right"}
                    print("hands enabled: %s" % (" ".join(sorted(self.sides)) or "none"), file=sys.stderr, flush=True)
            except OSError:
                if self.mtime is not None:
                    self.mtime, self.sides = None, {"left", "right"}
        return self.sides
HDR = struct.Struct("<4I" + "QQIIQ" * 2 + "2Q")   # MrhandPtHeader (capture/handcam_proto.h)
MAGIC = 0x4D525054
# sensor mounting: rotate each camera's image upright
ROTATE = {0: cv2.ROTATE_90_CLOCKWISE, 1: cv2.ROTATE_90_COUNTERCLOCKWISE}


class PtShm:
    def __init__(self, path=SHM):
        fd = os.open(path, os.O_RDONLY)
        self.m = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
        os.close(fd)
        magic, self.w, self.h, self.cams = struct.unpack_from("<4I", self.m, 0)
        if magic != MAGIC:
            raise RuntimeError("bad shm magic; is ptgrab stream running?")
        self.frame = self.w * self.h
        self.data0 = HDR.size

    def slot(self, cam):
        off = 16 + cam * 32
        return struct.unpack_from("<QQIIQ", self.m, off)   # seq, tDetectNs, bufferId, pad, frameCount

    def read(self, cam, last_seq=0):
        """Return (seq, t_detect_ns, gray HxW) for the newest frame, or None if unchanged."""
        for _ in range(5):
            seq, t, buf, _pad, n = self.slot(cam)
            if seq == last_seq or seq & 1 or seq == 0:
                if seq & 1:
                    time.sleep(0.0005)
                    continue
                return None
            img = np.frombuffer(self.m, np.uint8, self.frame, self.data0 + cam * self.frame).reshape(self.h, self.w).copy()
            if self.slot(cam)[0] == seq:
                return seq, t, img
        return None


class Telemetry:
    """Bounded per-frame / per-detection telemetry (CSV, rotated at max_bytes, kept on the headset).
    F rows (one per hand per sent frame):
      F,t_cap_ns,t_send_ns,side,pos_src,pos_obs_age_ms,q_src,q_age_ms,curl_age_ms,npts0,npts1,x,y,z,qw,qx,qy,qz
      pos_src: stereo | mono_ray0/1 (flow in one camera, depth held) | hold (frozen) | mediapipe
      *_age_ms: t_send minus the capture time of the image the component was measured on
    M rows (one per camera per MediaPipe run):
      M,t_det_ns,t_done_ns,kind,slot,dur_ms,nhands
    """

    def __init__(self, path, max_bytes=20_000_000):
        self.path, self.max_bytes = path, max_bytes
        self.f = open(path, "a", buffering=1 << 16)
        self.n = 0

    def write(self, line):
        self.f.write(line + "\n")
        self.n += 1
        if self.n % 2000 == 0 and self.f.tell() > self.max_bytes:
            self.f.close()
            os.replace(self.path, self.path + ".1")
            self.f = open(self.path, "a", buffering=1 << 16)


class ThermalGuard:
    """Throttles MediaPipe by temperature.  Conservative and NOT validated as safe limits:
    the headset once rebooted under high thermal load (cpu7_middle ~95-97 C, fan at max);
    the cause is unproven (kernel CPU trip points are 110 C), so we stay well below.

    Sensors: max over cpu*-thermal (incl. cpu7-middle, which the fan controller watches),
    cpuss*-thermal and gpuss-*-thermal (ptgrab uses the GPU).
      < SLOW_C          : max_hz
      SLOW_C..SLOWER_C  : max_hz / 2
      SLOWER_C..PAUSE_C : 2 Hz
      >= PAUSE_C        : MediaPipe paused (fast tracker continues) until < RESUME_C
    A separate watchdog (watchdog.py) stops the whole workload at WATCHDOG_C sustained."""
    SLOW_C, SLOWER_C, PAUSE_C, RESUME_C = 88.0, 90.0, 92.0, 88.0   # user-set 2026-10-11 (was 80/84/87/82)

    def __init__(self):
        import glob
        self.zones = [z for z in glob.glob("/sys/class/thermal/thermal_zone*")
                      if open(z + "/type").read().strip().startswith(("cpu", "gpuss"))]
        self.temp = 0.0
        self.paused = False
        self.t_read = 0.0

    def read(self):
        now = time.monotonic()
        if now - self.t_read > 1.0:
            self.t_read = now
            ts = []
            for z in self.zones:
                try:
                    ts.append(int(open(z + "/temp").read()) / 1000.0)
                except (OSError, ValueError):
                    pass
            new = max(ts) if ts else 0.0
            if self.paused and new < self.RESUME_C:
                self.paused = False
                print("thermal: resumed at %.0f C" % new, file=sys.stderr, flush=True)
            elif not self.paused and new >= self.PAUSE_C:
                self.paused = True
                print("thermal: MediaPipe PAUSED at %.0f C" % new, file=sys.stderr, flush=True)
            self.temp = new
        return self.temp

    def interval(self, max_hz):
        """Seconds between MediaPipe runs, or None while paused."""
        t = self.read()
        if self.paused:
            return None
        hz = max_hz
        if t >= self.SLOWER_C:
            hz = min(hz, 2.0)
        elif t >= self.SLOW_C:
            hz = hz / 2
        return 1.0 / hz


class Viewer:
    def __init__(self):
        self.jpeg = None
        self.cv = threading.Condition()
        self.clients = 0   # encode frames only while someone is watching

    def publish(self, bgr):
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.cv:
                self.jpeg = enc.tobytes()
                self.cv.notify_all()

    def serve(self, port=8765):
        viewer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path != "/":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                viewer.clients += 1
                try:
                    while True:
                        with viewer.cv:
                            viewer.cv.wait(1.0)
                            j = viewer.jpeg
                        if j:
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j))
                            self.wfile.write(j + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    viewer.clients -= 1

        srv = ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=640, help="detector input size (square)")
    ap.add_argument("--no-http", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="MediaPipe GPU delegate (shares the GPU with the compositor)")
    ap.add_argument("--max-hz", type=float, default=7.0,
                    help="cap on MediaPipe runs per second (position is tracked every frame regardless)")
    ap.add_argument("--mode", choices=("alt", "stereo"), default="alt",
                    help="alt: one camera per cycle (alternating) + periodic stereo; stereo: both cameras every cycle")
    ap.add_argument("--stereo-every", type=int, default=6, help="alt mode: run a stereo cycle every N cycles")
    ap.add_argument("--idle-hz", type=float, default=5.0, help="detection rate while no hand has been seen for 1 s")
    ap.add_argument("--backend", choices=("mediapipe", "mercury"), default="mediapipe",
                    help="mercury: MediaPipe only discovers hands; Mercury keypoints track them continuously")
    ap.add_argument("--curl-source", choices=("mercury", "geometric"), default="mercury",
                    help="finger curls from Mercury's curl output (default) or from the 3D joint angles")
    ap.add_argument("--merc-hz", type=float, default=30.0, help="Mercury tracking cycles per second (one crop per hand)")
    ap.add_argument("--max-speed", type=float, default=4.0,
                    help="fast tracker: apparent palm speed (m/s) above which a track is treated as an error")
    ap.add_argument("--lk-levels", type=int, default=3, help="fast tracker: optical-flow pyramid levels")
    ap.add_argument("--merc-predict-roi", action="store_true",
                    help="Mercury crop: extrapolate the palm with its recent velocity when the fast tracker has "
                         "no position, and enlarge the crop while the hand moves fast")
    ap.add_argument("--merc-stereo-ms", type=float, default=40.0,
                    help="max capture-time gap (ms) for triangulating two Mercury views; 0 = only the same capture "
                         "(views from different times describe different hand poses when the hand moves)")
    ap.add_argument("--curl-latest", action="store_true",
                    help="finger curls from the newest observation only (no blend with older views of other cameras)")
    ap.add_argument("--merc-hint-reset", action="store_true",
                    help="no keypoint hint for Mercury crops placed after a rejected observation or lost flow")
    ap.add_argument("--merc-early", action="store_true",
                    help="collect finished Mercury jobs and submit the next one between camera frames")
    ap.add_argument("--merc-sched", choices=("batch", "perhand"), default="batch",
                    help="batch: all tracked hands in one Mercury job, published together (original); "
                         "perhand: one hand per job on the newest frame, published as soon as it is done")
    ap.add_argument("--discover-hz", type=float, default=2.0,
                    help="with --backend mercury: MediaPipe rate while Mercury is tracking a hand")
    ap.add_argument("--mercury-shadow", action="store_true",
                    help="also run Mercury's keypoint model on each stereo MediaPipe frame and log the comparison")
    ap.add_argument("--fast-orient", action="store_true",
                    help="experimental: 60 Hz orientation from triangulated optical-flow landmarks (measured too noisy: "
                         "p95 14-23 deg frame-to-frame); default = MediaPipe orientation, smoothed")
    ap.add_argument("--rot-smooth-ms", type=float, default=40.0,
                    help="rotation glides toward each new orientation with this time constant (0 = step)")
    ap.add_argument("--blend-ms", type=float, default=120.0,
                    help="ease MediaPipe re-seed corrections in over this time constant (0 = snap)")
    ap.add_argument("--no-bridge", action="store_true", help="no stereo palm pose / UDP bridge to the PC")
    ap.add_argument("--telemetry", default=os.path.expanduser("~/handtrack/data/telemetry.csv"),
                    help="per-frame telemetry CSV (rotated at 20 MB); '' disables")
    ap.add_argument("--log", help="append per-frame results as JSON lines")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N s (0 = run forever)")
    a = ap.parse_args()
    import handtrack.fasttrack as _ft
    _ft.MAX_SPEED = a.max_speed
    _ft.LK["maxLevel"] = a.lk_levels

    # stay off the prime core (7, the one that overheated) and below normal priority
    try:
        os.sched_setaffinity(0, cpus_from_env())
        os.nice(10)
    except (AttributeError, OSError):
        pass
    thermal = ThermalGuard()
    tel = Telemetry(a.telemetry) if a.telemetry else None
    shm = PtShm()
    # MediaPipe's own handedness labels are correct for these cameras (user test 2026-10-10:
    # inverting them swapped the hands), so keep them as-is (mirrored=True = no inversion).
    # MediaPipe runs in its own process (handtrack/mpworker.py) so it can never stall the 60 Hz loop
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from handtrack import mpworker
    mp_pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                                  initializer=mpworker.init, initargs=(a.gpu, True))
    dets = None
    viewer = None if a.no_http else Viewer()
    if viewer:
        viewer.serve()
        print("debug view: ssh -L 8765:127.0.0.1:8765 frame  ->  http://localhost:8765", file=sys.stderr)
    logf = open(a.log, "a") if a.log else None

    palm = None if a.no_bridge else PalmTracker(a.size)
    mt = None
    if a.backend == "mercury":
        # continuous per-hand tracking with Mercury's keypoint model, in its own process
        from handtrack.merctrack import MercuryTracker
        merc_pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=mpworker.init_mercury_proc)
        mt = MercuryTracker(palm, merc_pool, mpworker)
        mt.curl_source = a.curl_source
        mt.sched = a.merc_sched
        mt.predict_roi = a.merc_predict_roi
        mt.stereo_ms = a.merc_stereo_ms
        mt.curl_latest = a.curl_latest
        mt.hint_reset = a.merc_hint_reset
        print("mercury: stereo<=%.0f ms, curls=%s, hint reset=%s, roi predict=%s, sched=%s" % (
            mt.stereo_ms, "latest" if mt.curl_latest else "blend", mt.hint_reset, mt.predict_roi, mt.sched),
            file=sys.stderr, flush=True)
    bridge = None if a.no_bridge else Bridge()

    def new_stats():
        return {"frames": 0, "hands": 0, "lat": [], "det": [], "pairs": 0, "palms": 0, "monoerr": [],
                "fast_ms": [], "sent": 0, "id_fix": 0, "merc_ms": [], "merc_ishand": [], "merc_e2d": [], "merc_e3d": []}

    lock = threading.Lock()
    stats = new_stats()
    stop = threading.Event()
    NAME = {0: "cam0 (right)", 1: "cam1 (left)"}

    pool = ThreadPoolExecutor(2)

    def prep(cam, img, t_ns):
        # ptgrab normally delivers frames already downscaled on the GPU; otherwise shrink here
        if img.shape[0] != a.size:
            img = cv2.resize(img, (a.size, a.size), interpolation=cv2.INTER_AREA)
        small = cv2.rotate(img, ROTATE[cam])
        td = time.perf_counter()
        hands = dets[cam].detect(small, t_ns / 1e9)
        return small, hands, time.perf_counter() - td

    def reassociate(poses, t, last_sent):
        """Re-label MediaPipe results by where the tracked hands physically are.
        Returns (poses, description-of-swap or '')."""
        tracked = {sd: v[1] for sd, v in last_sent.items() if t - v[0] < 0.5}
        dets = {sd: p for sd, p in poses.items() if p is not None}
        if not dets or (not tracked and len(dets) < 2):
            return poses, ""
        dist = lambda p, sd: float(np.linalg.norm(np.asarray(p[0]) - tracked[sd])) if sd in tracked else 9.0
        other = {"left": "right", "right": "left"}
        new = None
        if len(dets) == 2 and len(tracked) == 2:
            keep = dist(dets["left"], "left") + dist(dets["right"], "right")
            swap = dist(dets["left"], "right") + dist(dets["right"], "left")
            if swap < keep - 0.05:
                new = {"left": dets["right"], "right": dets["left"]}
        elif len(dets) == 2:
            # not both tracked yet: geometry beats labels when they clearly disagree (left hand further left)
            if dets["left"][0][0] > dets["right"][0][0] + 0.10:
                new = {"left": dets["right"], "right": dets["left"]}
        elif len(dets) == 1:
            sd, p = next(iter(dets.items()))
            od = other[sd]
            if od in tracked and dist(p, od) < 0.08 and dist(p, sd) > 0.15:
                new = {od: p, sd: None}
        if new is None:
            return poses, ""
        out = {"left": None, "right": None}
        for sd, p in new.items():
            if p is not None:     # rebuild with the correct hand's geometry (palm axes depend on handedness)
                out[sd] = palm._pose(sd, p[3], t, p[2]) + (p[4],)
        return out, "->".join(sorted(k for k, v in new.items() if v is not None))

    def mono_vs_stereo(poses, h1):
        """On stereo cycles: how far the left camera's single-camera palm estimate is from stereo (m)."""
        errs = []
        R, t = palm.pose[1]
        for side, p in poses.items():
            if p is None:
                continue
            for h in h1:
                if h.side != side:
                    continue
                Pc = palm.mono_points(1, h)
                if Pc is not None:
                    Pm = (Pc * palm.scale.get(side, 1.0)) @ R.T + t
                    errs.append(float(np.linalg.norm(Pm[PALM_IDX].mean(0) - p[3][PALM_IDX].mean(0))))
        return errs

    def coordinator():
        """Fast path every camera frame: optical-flow palm tracking in both cameras ->
        triangulated position -> send (orientation + finger curls from the latest
        MediaPipe result).  Slow path in the background: MediaPipe on alternating
        cameras (stereo every `stereo_every`-th run), which re-seeds the fast tracker."""
        nonlocal stats
        last = {0: 0, 1: 0}
        fast = FastTracker(palm, orient=a.fast_orient, blend_s=a.blend_ms / 1000.0)
        rot_tau = a.rot_smooth_ms / 1000.0
        # adaptive: strong smoothing when the hand is nearly still, little when it moves fast
        pos_filt = {sd: OneEuro(1.2, 1.5) for sd in ("left", "right")}
        q_filt = {sd: OneEuro(2.0, 0.6) for sd in ("left", "right")}
        last_q = {}
        slow = {}            # side -> latest MediaPipe pose tuple (pos, quat, conf, P, curls)
        slow_t = {}          # side -> its capture time (s)
        slow_cam_q = {"left": {}, "right": {}}   # side -> slot -> (t, quat, backend) latest orientation per camera
        slow_src = {}        # side -> backend of the latest articulated observation ("merc" / "mp")
        last_sent = {}       # side -> (t, position) last sent to the PC
        grid_t, grid_period = None, 1e9 / 60.0                       # camera frame grid (ns)
        rot_pending = {}     # side -> (t, quat) orientation outlier awaiting confirmation
        shown = {0: None, 1: None}
        job = None           # (future, kind, slot, imgs_at_detection, t_cap)
        shadow = None        # Mercury shadow comparison in flight
        next_merc = 0.0
        next_det = 0.0
        k = 0
        frame_no = 0
        last_hand = 0.0
        def merc_service():
            """Mercury continuous tracking: collect a finished job, then submit the next one on the
            newest frame (imgs / t_cap / fast_pos of the last processed camera frame)."""
            nonlocal next_merc, next_det, last_hand
            if mt is not None:
                for side, (pose_m, sl_m, px_m, t_m, img_m) in mt.collect().items():
                    if side not in enabled:                 # result for a hand disabled while in flight
                        continue
                    slow[side], slow_t[side], slow_src[side] = pose_m, t_m, "merc"
                    slow_cam_q[side][sl_m] = (t_m, np.asarray(pose_m[1], float), "merc")
                    fast.seed(sl_m, side, types.SimpleNamespace(px=px_m), img_m, imgs[sl_m], t_s, pos3d=pose_m[0])
                    last_hand = time.monotonic()
                    if tel is not None:
                        tel.write("K,%d,%s,%d" % (int(t_m * 1e9), side, sl_m))
                if tel is not None:
                    # one row per Mercury observation: capture -> submit -> worker start/end -> collect,
                    # thread CPU vs runqueue wait, hands in the job, outcome, raw (unfiltered) orientation
                    for rc in mt.records:
                        q_ = rc["rq"] if rc["rq"] is not None else (0, 0, 0, 0)
                        tel.write("R,%d,%s,%d,%d,%d,%d,%d,%d,%.2f,%.2f,%d,%s,%.4f,%.4f,%.4f,%.4f,%d" % (
                            rc["id"], rc["side"], rc["slot"], rc["t_cap"], rc["t_sub"], rc["t0"], rc["t1"], rc["t_col"],
                            rc["cpu_ns"] / 1e6, rc["wait_ns"] / 1e6, rc["njob"], rc["outcome"], *q_, rc["hint"]))
                mt.records.clear()
                if mt.collapsed:
                    # both trackers ended up on one physical hand: forget the dropped one everywhere and
                    # let discovery find the other hand right away
                    fast.hands.pop(mt.collapsed, None)
                    slow.pop(mt.collapsed, None)
                    slow_cam_q[mt.collapsed].clear()
                    next_det = 0.0
                    if tel is not None:
                        tel.write("C,%d,%s" % (t_cap, mt.collapsed))
                if not mt.busy() and time.monotonic() >= next_merc:
                    # perhand: one crop per job, so the same per-hand rate cap allows n jobs per period
                    n_ = max(1, len(mt.hands)) if a.merc_sched == "perhand" else 1
                    iv = thermal.interval(a.merc_hz * n_)
                    if iv is not None and mt.submit(imgs, t_cap, fast_pos):
                        next_merc = time.monotonic() + iv


        hands_cfg = EnabledHands()
        enabled = hands_cfg.get()
        have_frame = False
        prof = Profile()
        pv = dict(PROFILE_DEFAULTS)                 # apply the defaults once, then live changes
        while not stop.is_set():
            if pv is None:
                pv = prof.poll()
            if pv is not None:
                for sd in ("left", "right"):
                    if palm is not None:
                        fp_, fz_, fx_ = palm.filt[sd]
                        fp_.set(pv["centre_filter"])
                        fz_.set(pv["axis_filter"])
                        fx_.set(pv["axis_filter"])
                        palm.curl_filt[sd].set(pv["curl_filter"])
                    pos_filt[sd].set(pv["pos_filter"])
                rot_tau = (a.rot_smooth_ms if pv["rot_glide_ms"] is None else pv["rot_glide_ms"]) / 1000.0
                if tel is not None:   # profile switch marker: the report can split results by profile
                    tel.write("P,%d,%s" % (time.monotonic_ns(), ";".join(
                        "%s=%s" % (k, "off" if v is None else (v if not isinstance(v, tuple) else "%g/%g" % v))
                        for k, v in sorted(pv.items()))))
                pv = None
            en = hands_cfg.get()
            if en != enabled:
                for sd in ({"left", "right"} - en):          # just disabled: forget it everywhere
                    fast.hands.pop(sd, None)
                    slow.pop(sd, None)
                    slow_cam_q[sd].clear()
                    if mt is not None:
                        mt.hands.pop(sd, None)
                enabled = en
                next_det = 0.0
            if bridge:
                bridge.enabled_mask = (1 if "left" in enabled else 0) | (2 if "right" in enabled else 0)
            r0, r1 = shm.read(0, last[0]), shm.read(1, last[1])
            if not r0 or not r1:
                # --merc-early: between camera frames, collect a finished Mercury job and start the next
                # one right away instead of waiting up to a frame for the next loop pass
                if a.merc_early and have_frame and mt is not None and mt.job is not None and mt.job[0].done():
                    merc_service()
                time.sleep(0.002)
                continue
            if abs(r0[1] - r1[1]) > 8e6:          # not the same capture: wait for the lagging camera
                if r0[1] < r1[1]:
                    last[0] = r0[0]
                else:
                    last[1] = r1[0]
                continue
            last[0], last[1] = r0[0], r1[0]
            frame_no += 1
            have_frame = True
            t_raw = min(r0[1], r1[1])
            if grid_t is None or abs(t_raw - grid_t) > 1e9:
                grid_t = t_raw
            else:
                nfr = max(1, round((t_raw - grid_t) / grid_period))
                pred = grid_t + nfr * grid_period
                resid = t_raw - pred
                grid_t = pred + 0.05 * resid                       # follow phase slowly
                grid_period += 0.002 * resid / nfr                 # and the true frame period
                grid_period = min(max(grid_period, 15.0e6), 18.5e6)
            t_cap = int(grid_t)
            t_s = t_cap / 1e9
            imgs = {0: cv2.rotate(r0[2], ROTATE[0]), 1: cv2.rotate(r1[2], ROTATE[1])}

            # ---- fast path: position every camera frame (every other frame when hot)
            tf = time.perf_counter()
            skip_frame = thermal.temp >= thermal.SLOWER_C and frame_no % 2 == 1
            fast_pos = {} if skip_frame else fast.step(imgs, t_s)
            fast_ms = (time.perf_counter() - tf) * 1e3

            # ---- slow path: collect a finished MediaPipe run, re-seed the tracker
            if job and job[0].done():
                fut, kind, slot, det_imgs, t_det = job
                job = None
                res = fut.result()                 # {slot: (hands, det_seconds)}
                hands_by_slot = {sl: v[0] for sl, v in res.items()}
                if kind == "stereo":
                    poses_d = palm.fuse(hands_by_slot[0], hands_by_slot[1], t_det / 1e9)
                    merr = mono_vs_stereo(poses_d, hands_by_slot[1])
                    if a.mercury_shadow and shadow is None:
                        # identical-input comparison: Mercury keypoints on the very frames MediaPipe saw,
                        # ROI from MediaPipe's stereo joints (logged only; does not drive the hands)
                        sj = []
                        for sd, pp in poses_d.items():
                            if pp is None:
                                continue
                            for sl in (0, 1):
                                Rc, tc = palm.pose[sl]
                                sj.append((sl, sd, (pp[3] - tc) @ Rc))
                        if sj:
                            shadow = (mp_pool.submit(mpworker.run_mercury, sj, {sl: det_imgs[sl] for sl in (0, 1)}),
                                      t_det, hands_by_slot, {sd: pp[3] for sd, pp in poses_d.items() if pp is not None})
                else:
                    poses_d = palm.mono(slot, hands_by_slot[slot], t_det / 1e9)
                    merr = []
                # identity by position, not by MediaPipe's (single-view, error-prone) left/right label
                poses_d, swapped = reassociate(poses_d, t_det / 1e9, last_sent)
                # one-hand mode: a lone detection carries MediaPipe's single-view label, which is often
                # wrong and has nothing to be checked against.  If it is labelled as the disabled hand
                # but lies on the enabled hand's side of the head, it is the enabled hand.
                if len(enabled) == 1:
                    want = next(iter(enabled))
                    oth = "right" if want == "left" else "left"
                    po = poses_d.get(oth)
                    if poses_d.get(want) is None and po is not None:
                        x = float(po[0][0])
                        if (x < 0.05) if want == "left" else (x > -0.05):
                            poses_d = {want: palm._pose(want, po[3], t_det / 1e9, po[2]) + (po[4],), oth: None}
                            with lock:
                                stats["id_fix"] += 1
                            if tel is not None:
                                tel.write("I,%d,one-hand->%s" % (t_det, want))
                # identity is settled; only enabled hands go any further
                poses_d = {sd: (pp if sd in enabled else None) for sd, pp in poses_d.items()}
                if swapped:
                    with lock:
                        stats["id_fix"] += 1
                    if tel is not None:
                        tel.write("I,%d,%s" % (t_det, swapped))
                nh = sum(len(h) for h in hands_by_slot.values())
                if nh:
                    last_hand = time.monotonic()
                if mt is not None:
                    for sd_, p_ in poses_d.items():
                        if p_ is not None:
                            mt.seed(sd_, p_[3], t_det / 1e9)
                for side, p in poses_d.items():
                    if p is None:
                        continue
                    if mt is not None and side in mt.tracked() and mt.hands[side].t > t_det / 1e9:
                        continue          # Mercury already has a newer observation of this hand
                    slow[side], slow_t[side], slow_src[side] = p, t_det / 1e9, "mp"
                    qn = np.asarray(p[1], float)
                    lq = last_q.get(side)
                    accept = True
                    if lq is not None and t_s - lq[0] < 0.3:
                        dev = math.degrees(2 * math.acos(min(1.0, abs(float(qn @ lq[1])))))
                        if dev > 50:
                            pend = rot_pending.get(side)
                            if pend is not None and t_s - pend[0] < 0.4 and                                     math.degrees(2 * math.acos(min(1.0, abs(float(qn @ pend[1]))))) < 25:
                                rot_pending.pop(side, None)        # confirmed twice: a real turn
                            else:
                                rot_pending[side] = (t_s, qn)      # likely a bad result: wait for a second one
                                accept = False
                    if accept:
                        for sl in ((0, 1) if kind == "stereo" else (slot,)):
                            slow_cam_q[side][sl] = (t_det / 1e9, qn, "mp")
                    # seed the optical-flow tracker with the detection that is geometrically ON this hand
                    pc3 = p[3][[0, 5, 9, 17]].mean(0)
                    for sl, hs in hands_by_slot.items():
                        if not hs:
                            continue
                        target = palm.det_px(sl, pc3)[0]
                        dists = [np.linalg.norm(h.px[[0, 5, 9, 17]].mean(0) - target) for h in hs]
                        k_ = int(np.argmin(dists))
                        if dists[k_] < 60:
                            fast.seed(sl, side, hs[k_], det_imgs[sl], imgs[sl], t_s, pos3d=p[0])
                for sl, hs in hands_by_slot.items():
                    shown[sl] = hs
                with lock:
                    stats["det"] += [v[1] for v in res.values()]
                if tel is not None:
                    td_ns = time.monotonic_ns()
                    for sl, (hs_, dsec) in res.items():
                        tel.write("M,%d,%d,%s,%d,%.1f,%d" % (t_det, td_ns, kind, sl, dsec * 1e3, len(hs_)))
                    stats["frames"] += len(res)
                    stats["hands"] += nh
                    stats["palms"] += sum(p is not None for p in poses_d.values())
                    stats["monoerr"] += merr
            fast.drop_stale(t_s, max_age=2.0)

            # ---- Mercury shadow comparison results
            if shadow is not None and shadow[0].done():
                sfut, s_t, s_hands, s_P = shadow
                shadow = None
                try:
                    sres = sfut.result()
                except Exception as e:                     # never let the experiment break tracking
                    print("mercury shadow error: %r" % (e,), file=sys.stderr, flush=True)
                    sres = []
                rays_by = {}
                for sl, sd, r, dsec, *_ in sres:
                    if r is None:
                        continue
                    ref = next((h for h in s_hands.get(sl, []) if h.side == sd), None)
                    e2 = float(np.median(np.linalg.norm(r["det_px"] - ref.px, axis=1))) if ref is not None else -1
                    with lock:
                        stats["merc_ms"].append(dsec * 1e3)
                        stats["merc_ishand"].append(r["is_hand"])
                        if e2 >= 0:
                            stats["merc_e2d"].append(e2)
                    if tel is not None:
                        tel.write("S,%d,%s,%d,%.1f,%.3f,%.1f,%s" % (s_t, sd, sl, dsec * 1e3, r["is_hand"], e2,
                                                                     " ".join("%.2f" % c for c in r["curls"])))
                    Rc, tc = palm.pose[sl]
                    rays_by.setdefault(sd, {})[sl] = (tc, r["rays"] @ Rc.T)
                for sd, rv in rays_by.items():
                    if 0 in rv and 1 in rv and sd in s_P:
                        (o1, d1), (o0, d0) = rv[1], rv[0]
                        Pm, gap, _, _ = triangulate(np.broadcast_to(o1, d1.shape), d1, np.broadcast_to(o0, d0.shape), d0)
                        e3 = np.linalg.norm(Pm - s_P[sd], axis=1) * 100
                        with lock:
                            stats["merc_e3d"].append(float(np.median(e3)))
                        if tel is not None:
                            tel.write("S3,%d,%s,%.2f,%.2f,%.2f" % (s_t, sd, float(np.median(e3)),
                                                                  float(np.median(e3[[0, 5, 9, 13, 17]])),
                                                                  float(np.median(gap)) * 100))

            merc_service()

            # ---- schedule the next MediaPipe run (rate cap, thermal guard, idle mode)
            now_m = time.monotonic()
            lost = set(fast.lost)
            if job is None and (now_m >= next_det or lost):
                interval = thermal.interval(a.max_hz)
                if interval is not None:
                    if now_m - last_hand > 1.0:
                        interval = max(interval, 1.0 / a.idle_hz)
                    if mt is not None and enabled and len(mt.tracked() & enabled) >= len(enabled) and not lost:
                        interval = max(interval, 1.0 / a.discover_hz)   # both hands tracked: discovery only
                    next_det = now_m + interval
                    stereo = a.mode == "stereo" or k % a.stereo_every == 0
                    slot = 1 if k % 2 else 0
                    if lost and not stereo:        # a track just failed: re-detect in that camera now
                        slot = min(lost)
                    fast.lost.clear()
                    k += 1
                    det_imgs = dict(imgs)
                    use = (0, 1) if stereo else (slot,)

                    fut = mp_pool.submit(mpworker.run, use, {sl: det_imgs[sl] for sl in use}, t_cap / 1e9)
                    job = (fut, "stereo" if stereo else "mono", slot, det_imgs, t_cap)

            # ---- compose and send: fast position + latest orientation / curls
            # (a frame skipped for heat sends nothing: no old state relabelled as new)
            if skip_frame:
                continue
            poses = {}
            tel_rows = []
            for side in ("left", "right"):
                sp = slow.get(side)
                if side not in enabled or sp is None or t_s - slow_t[side] > 1.0:
                    poses[side] = None
                    continue
                fp = fast_pos.get(side)
                pos, q = (fp[0], fp[1]) if fp is not None else (None, None)
                finfo = fast.info.get(side, {}) if fp is not None else {}
                pos_src, pos_t = finfo.get("src", "none"), finfo.get("t_obs", t_s)
                q_src, q_t = ("flow", t_s) if q is not None else (slow_src.get(side, "mp"), slow_t[side])
                if q is None:
                    # average the two cameras' latest orientations: their single-view biases cancel
                    # instead of alternating (which looked like a few-times-a-second wobble)
                    # both cameras only if both are recent: an old estimate drags a turning hand backwards
                    qs_ = [qq for (tq, qq, _) in slow_cam_q[side].values() if t_s - tq < 0.15]
                    if len(qs_) == 2:
                        a0, a1 = qs_[0], qs_[1] if qs_[0] @ qs_[1] >= 0 else -qs_[1]
                        q = (a0 + a1) / np.linalg.norm(a0 + a1)
                        newest = max(slow_cam_q[side].values(), key=lambda e: e[0])
                        q_src, q_t = newest[2] + "_avg2", newest[0]
                    elif slow_cam_q[side]:
                        # newest accepted orientation (never a result the outlier gate rejected)
                        q_t, q, q_be = max(slow_cam_q[side].values(), key=lambda e: e[0])
                        q_src = q_be + "_newest"
                    else:
                        q = sp[1]
                if pos is None and t_s - slow_t[side] < 0.15 and plausible(sp[0]):
                    # fast tracker lost: use MediaPipe's own estimate only if it agrees with the
                    # last position we sent (single-camera depth can be several cm off)
                    ls = last_sent.get(side)
                    if ls is None or t_s - ls[0] > 0.3 or np.linalg.norm(sp[0] - ls[1]) < 0.10:
                        pos = sp[0]
                        pos_src, pos_t = "mediapipe", slow_t[side]
                if pos is None:
                    poses[side] = None
                    continue
                pos = pos_filt[side](pos, t_s)
                # rotation: glide toward the target (exponential, time constant rot_tau) instead of
                # stepping whenever a new MediaPipe orientation arrives
                q = np.asarray(q, float)
                lq = last_q.get(side)
                if lq is None or t_s - lq[0] > 0.3:
                    qs = q / np.linalg.norm(q)
                else:
                    prev = lq[1]
                    if q @ prev < 0:
                        q = -q
                    # adaptive time constant: ~2.5x rot_tau for tiny differences (still hand),
                    # ~0.6x for big ones (fast turn)
                    ang = math.degrees(2 * math.acos(min(1.0, abs(float(q @ prev)))))
                    k_ = min(1.0, max(0.0, (ang - 2.0) / 18.0))
                    tau = rot_tau * (2.5 - 1.9 * k_)
                    al = 1.0 - math.exp(-max(t_s - lq[0], 0.0) / tau) if rot_tau > 0 else 1.0
                    qs = (1 - al) * prev + al * q
                    qs = qs / np.linalg.norm(qs)
                last_q[side] = (t_s, qs)
                q = qs
                last_sent[side] = (t_s, pos)
                poses[side] = (pos, q, sp[2], sp[3], sp[4])
                tel_rows.append((side, pos_src, pos_t, q_src, q_t, slow_t[side], finfo.get("npts", {}), pos, q))
            if bridge:
                bridge.send(poses, t_cap)
            if tel is not None and tel_rows:
                t_send = time.monotonic_ns()
                for (sd, psrc, pt, qsrc, qt, ct, npts, pp, qq) in tel_rows:
                    tel.write("F,%d,%d,%s,%s,%.1f,%s,%.1f,%.1f,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (
                        t_cap, t_send, sd, psrc, (t_send / 1e9 - pt) * 1e3, qsrc, (t_send / 1e9 - qt) * 1e3,
                        (t_send / 1e9 - ct) * 1e3, npts.get(0, 0), npts.get(1, 0), *pp, *qq))
            tel_rows.clear()
            lat_ms = (time.monotonic_ns() - t_cap) / 1e6
            with lock:
                stats["lat"].append(lat_ms)
                stats["pairs"] += 1
                stats["fast_ms"].append(fast_ms)
                stats["sent"] += sum(p is not None for p in poses.values())
            if logf and any(p is not None for p in poses.values()):
                logf.write(json.dumps({"t_ns": t_cap, "lat_ms": round(lat_ms, 2),
                                       "palms": {kk: (None if p is None else {"pos": np.round(p[0], 4).tolist(),
                                                                              "quat": np.round(p[1], 4).tolist(),
                                                                              "curl": np.round(p[4], 3).tolist()})
                                                 for kk, p in poses.items()}}) + "\n")
            if viewer and viewer.clients and shown[0] is not None and shown[1] is not None and frame_no % 3 == 0:
                tiles = []
                for c in (1, 0):   # left camera on the left
                    hs = shown[c]
                    t = draw(imgs[c], hs)          # newest image, last MediaPipe skeleton
                    for side, ht in fast.hands.items():
                        ct = ht.cams.get(c)
                        if ct is not None:
                            cv2.circle(t, tuple(int(v) for v in ct.centre), 10, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.putText(t, "%s  %d hand(s)  %.0f ms" % (NAME[c], len(hs), lat_ms),
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    tiles.append(t)
                img = np.hstack(tiles)
                y = img.shape[0] - 14
                for side, p in poses.items():
                    if p is not None:
                        cv2.putText(img, "%s palm (head, cm): x %+.0f y %+.0f z %+.0f  curl %s" % (
                            side, *(100 * p[0]), " ".join("%.1f" % c for c in p[4])),
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                        y -= 26
                viewer.publish(img)

    threads = [threading.Thread(target=coordinator, daemon=True)]
    for t in threads:
        t.start()
    t0 = tstat = time.monotonic()
    try:
        while not a.seconds or time.monotonic() - t0 < a.seconds:
            time.sleep(0.25)
            now = time.monotonic()
            if now - tstat > 5:
                with lock:
                    st = stats
                    stats = new_stats()
                n = max(st["frames"], 1)
                lat = sorted(st["lat"]) or [0]
                me = sorted(st["monoerr"])
                if st["id_fix"]:
                    print("   identity corrections (MediaPipe label disagreed with position): %d" % st["id_fix"],
                          file=sys.stderr, flush=True)
                fm = sorted(st["fast_ms"]) or [0]
                dt = now - tstat
                print("[%.0f C] fast: %.1f frames/s, %.1f ms/frame, %.1f hand poses/s, latency p50 %.0f p95 %.0f ms"
                      "  | MediaPipe: %.1f runs/s, %.1f ms, hands/run %.2f  | mono-vs-stereo %s  hand scale %s"
                      % (thermal.temp, st["pairs"] / dt, fm[len(fm) // 2], st["sent"] / dt,
                         lat[len(lat) // 2], lat[int(len(lat) * 0.95)],
                         st["frames"] / dt, 1e3 * sum(st["det"]) / n, st["hands"] / n,
                         ("p50 %.1f cm (n=%d)" % (100 * me[len(me) // 2], len(me))) if me else "-",
                         {k: round(v, 2) for k, v in (palm.scale.items() if palm else [])}),
                      file=sys.stderr, flush=True)
                if mt is not None:
                    ms = mt.stats
                    print("   mercury tracking: %d crops (%.1f/s), %.1f ms median, accepted %d (stereo %d, single-view %d), "
                          "rejected %d, collapses %d, seeds blocked %d, tracked %s" % (
                              ms["runs"], ms["runs"] / dt, np.median(ms["ms"]) if ms["ms"] else 0, ms["accepted"],
                              ms["stereo"], ms["mono"], ms["rejected"], ms.get("collapses", 0), ms.get("seed_blocked", 0),
                              sorted(mt.tracked())), file=sys.stderr, flush=True)
                    mt.stats = {"runs": 0, "accepted": 0, "rejected": 0, "stereo": 0, "mono": 0, "ms": []}
                if st["merc_ms"]:
                    print("   mercury shadow: %d inferences, %.1f ms median | is_hand>=0.97 %.0f%% | vs MediaPipe: 2D median "
                          "%s px, 3D median %s cm" % (len(st["merc_ms"]), np.median(st["merc_ms"]),
                                                      100 * np.mean(np.array(st["merc_ishand"]) >= 0.97),
                                                      ("%.1f" % np.median(st["merc_e2d"])) if st["merc_e2d"] else "-",
                                                      ("%.1f" % np.median(st["merc_e3d"])) if st["merc_e3d"] else "-"),
                          file=sys.stderr, flush=True)
                tstat = now
    except KeyboardInterrupt:
        pass
    stop.set()
    for t in threads:
        t.join(1)


if __name__ == "__main__":
    main()
