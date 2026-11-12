"""Headset-side plumbing for the tracker (handtrack/main.py): camera frames from ptgrab's shared
memory, which hands are enabled, thermal scaling, telemetry and the local debug view.
"""
import glob
import mmap
import os
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

SHM = "/dev/shm/mrhand-pt"
# which hands to track, written by the "Hand Tracking" dashboard panel: "left", "right" or "left right"
# (missing file = both).  Re-read about once a second; a disabled hand gets no crops and is sent as
# invalid.
HANDS_FILE = os.path.expanduser("~/.config/mrhand/hands")


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


def lower_priority():
    """Stay off the prime core (7, the one that overheated) and below normal priority."""
    try:
        os.sched_setaffinity(0, cpus_from_env())
        os.nice(10)
    except (AttributeError, OSError):
        pass


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


class PtShm:
    """ptgrab's frames: per camera a seqlock-protected luma image in the sensor's orientation."""

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


class ThermalGuard:
    """Scales the tracker's work by temperature.  Conservative and NOT validated as safe limits:
    the headset once rebooted under high thermal load (cpu7_middle ~95-97 C, fan at max); the cause
    is unproven (kernel CPU trip points are 110 C), so we stay well below.

    Sensors: max over cpu*-thermal and gpuss*-thermal (ptgrab uses the GPU).
      < SLOW_C          : scale 1.0  (full crop budget and search rates)
      SLOW_C..SLOWER_C  : 0.7
      SLOWER_C..PAUSE_C : 0.5
      >= PAUSE_C        : 0.3 and no MediaPipe until < RESUME_C
    A separate watchdog (watchdog.py) stops the whole workload at WATCHDOG_C sustained."""
    SLOW_C, SLOWER_C, PAUSE_C, RESUME_C = 88.0, 90.0, 92.0, 88.0   # user-set 2026-10-11

    def __init__(self):
        self.zones = []
        for z in glob.glob("/sys/class/thermal/thermal_zone*"):
            try:
                if open(z + "/type").read().strip().startswith(("cpu", "gpuss")):
                    self.zones.append(z)
            except OSError:
                pass
        self.temp = 0.0
        self.mp_paused = False
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
            if self.mp_paused and new < self.RESUME_C:
                self.mp_paused = False
                print("thermal: MediaPipe resumed at %.0f C" % new, file=sys.stderr, flush=True)
            elif not self.mp_paused and new >= self.PAUSE_C:
                self.mp_paused = True
                print("thermal: MediaPipe PAUSED at %.0f C" % new, file=sys.stderr, flush=True)
            self.temp = new
        return self.temp

    def scale(self):
        t = self.read()
        if t >= self.PAUSE_C:
            return 0.3
        if t >= self.SLOWER_C:
            return 0.5
        if t >= self.SLOW_C:
            return 0.7
        return 1.0


class Telemetry:
    """Bounded telemetry (CSV, rotated at max_bytes, kept on the headset; poses only, no images).
    F rows (one per enabled hand per frame):
      F,t_cap_ns,t_send_ns,side,valid,obs_age_ms,x,y,z,qw,qx,qy,qz,conf
    Tracker event rows (handtrack/tracker.py):
      O,t,side,slots,is_hand,hm,res_deg,age_ms,init        accepted observation
      N,t,side,why,is_hand,res_deg                         rejected / failed crop
      X,t,side,nis_p,nis_r                                 track restarted where the hand really is
      A,t,side,finder,label,is_hand,hm_left,hm_right,res   hand acquired
      V,t,finder,side,why,is_hand,hm_left,hm_right         candidate rejected by the keypoint model
      L,t,side,age_ms                                      track dropped
      C,t,side                                             duplicate track dropped
    S rows every 5 s: S,t_ns,<key=value ...> (the status line)
    R,t_ns,start,pid when a tracker starts (separates runs; CLOCK_MONOTONIC restarts at a reboot)"""

    def __init__(self, path, max_bytes=20_000_000):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path, self.max_bytes = path, max_bytes
        self.f = open(path, "a", buffering=1 << 16)
        self.n = 0
        self.write("R,%d,start,%d" % (time.monotonic_ns(), os.getpid()))

    def write(self, line):
        self.f.write(line + "\n")
        self.n += 1
        if self.n % 2000 == 0 and self.f.tell() > self.max_bytes:
            self.f.close()
            os.replace(self.path, self.path + ".1")
            self.f = open(self.path, "a", buffering=1 << 16)

    def flush(self):
        self.f.flush()


class Viewer:
    """MJPEG debug view on 127.0.0.1 only (camera imagery never leaves the headset unless you tunnel
    it yourself: ssh -L 8765:127.0.0.1:8765 frame).  "/" streams, "/snap.jpg" returns the next frame.
    Frames are encoded only while someone watches."""

    def __init__(self):
        self.jpeg = None
        self.cv = threading.Condition()
        self.clients = 0

    def publish(self, bgr):
        import cv2
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
                if self.path == "/snap.jpg":
                    self.snap()
                    return
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

            def snap(self):
                viewer.clients += 1
                try:
                    with viewer.cv:
                        viewer.jpeg = None
                        viewer.cv.wait_for(lambda: viewer.jpeg is not None, 2.0)
                        j = viewer.jpeg
                finally:
                    viewer.clients -= 1
                if not j:
                    self.send_error(503, "no frame")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(j)))
                self.end_headers()
                self.wfile.write(j)

        srv = ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
