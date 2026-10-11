"""MediaPipe in its own process.

Running MediaPipe (45-55 ms per run) in a thread of the main process stalled the
60 Hz tracking loop (Python GIL), which showed up as bursty, uneven motion.  The
worker process owns one HandLandmarker per camera; the main process sends it
640x640 gray frames and gets back plain Hand records.
"""
import os

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


_dets = {}


def init(gpu=False, mirrored=True):
    """ProcessPoolExecutor initializer: lower priority, off the prime core, build the detectors."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    try:
        os.sched_setaffinity(0, cpus_from_env())
        os.nice(10)
    except (AttributeError, OSError):
        pass
    from handtrack.detect import HandDetector
    for slot in (0, 1):
        _dets[slot] = HandDetector(gpu=gpu, mirrored=mirrored)


_merc = None


def init_mercury():
    """Lazily load Monado Mercury's keypoint model (license UNVERIFIED: local measurement only)."""
    global _merc
    if _merc is None:
        from handtrack.calib import Rig
        from handtrack.mercury_kp import MercuryKeypoint
        from handtrack.palm import SLOT_CAM
        _merc = MercuryKeypoint(os.path.expanduser("~/handtrack/models/mercury/grayscale_keypoint_jan18.onnx"),
                                Rig(), SLOT_CAM, 640)
    return _merc


def init_mercury_proc():
    """Initializer for the dedicated Mercury worker process (no MediaPipe in it)."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    try:
        os.sched_setaffinity(0, cpus_from_env())
        os.nice(10)
    except (AttributeError, OSError):
        pass
    init_mercury()


def run_mercury(jobs, imgs):
    """jobs: [(slot, side, joints_cv (21,3))]; ROI from those joints, same images MediaPipe saw.
    Returns [(slot, side, result-without-crop or None, seconds)]."""
    import time
    mk = init_mercury()
    out = []
    for slot, side, jc in jobs:
        td = time.perf_counter()
        r = mk.run(slot, imgs[slot], jc, flip=(side == "right"))
        if r is not None:
            r.pop("crop", None)
        out.append((slot, side, r, time.perf_counter() - td))
    return out


def run(use, imgs, t_s):
    """Detect hands in imgs[slot] for each slot in `use`; returns {slot: (hands, seconds)}."""
    import time
    out = {}
    for sl in use:
        td = time.perf_counter()
        hs = _dets[sl].detect(imgs[sl], t_s)
        out[sl] = (hs, time.perf_counter() - td)
    return out
