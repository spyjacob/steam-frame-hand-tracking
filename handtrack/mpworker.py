"""MediaPipe in its own process: a hand finder for the v4 tracker (handtrack/main.py MPFinder).

MediaPipe runs only while an enabled hand is not being tracked (22-45 ms per run); in a thread of
the main process it would stall the 60 Hz loop (Python GIL).  The worker owns one HandLandmarker per
camera (VIDEO mode, its own timestamps per camera); the main process sends 640x640 gray frames in
the sensor's orientation and gets back plain arrays.
"""
import os

_dets = {}
_ROT = None


def init(gpu=False, mirrored=True):
    """ProcessPoolExecutor initializer: lower priority, off the prime core, build the detectors.
    MediaPipe's own handedness labels are correct for these cameras (user test 2026-10-10:
    inverting them swapped the hands), so mirrored=True = no inversion."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    from handtrack.headset import lower_priority
    lower_priority()
    global _ROT
    import cv2
    cv2.setNumThreads(1)
    _ROT = {0: cv2.ROTATE_90_CLOCKWISE, 1: cv2.ROTATE_90_COUNTERCLOCKWISE}
    from handtrack.detect import HandDetector
    for slot in (0, 1):
        _dets[slot] = HandDetector(gpu=gpu, mirrored=mirrored)


def find(slot, img, t_s):
    """img: one camera's 640x640 frame in sensor orientation.  Returns (seconds, [(side, score,
    px (21,2) in the upright image)])."""
    import time
    import cv2
    td = time.perf_counter()
    up = cv2.rotate(img, _ROT[slot])
    hands = _dets[slot].detect(up, t_s)
    return time.perf_counter() - td, [(h.side, h.side_score, h.px) for h in hands]
