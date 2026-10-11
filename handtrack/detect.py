"""MediaPipe hand landmark detection on single camera frames.

MediaPipe's handedness assumes a mirrored (selfie) image; our cameras face away
from the user, so labels are flipped unless mirrored=True.
"""
import os
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

DEFAULT_MODEL = os.path.expanduser("~/handtrack/models/hand_landmarker.task")

# 21-point topology (MediaPipe ordering)
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


@dataclass
class Hand:
    side: str                 # "left" / "right" (user's hand)
    side_score: float         # handedness classifier confidence
    px: np.ndarray            # (21, 2) pixel coords in the input image
    rel_z: np.ndarray         # (21,) MediaPipe relative depth (wrist-relative, image-width units)
    world: np.ndarray         # (21, 3) MediaPipe metric-ish hand-centred model coords (m)
    presence: float = 1.0     # not exposed per-hand by Tasks API; kept for gating API stability
    extra: dict = field(default_factory=dict)


class HandDetector:
    def __init__(self, model=DEFAULT_MODEL, num_hands=2, min_det=0.5, min_presence=0.5, min_track=0.5,
                 video=True, mirrored=False, gpu=False):
        delegate = mpt.BaseOptions.Delegate.GPU if gpu else mpt.BaseOptions.Delegate.CPU
        opts = vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=model, delegate=delegate),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_det,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_track)
        self.lm = vision.HandLandmarker.create_from_options(opts)
        self.video = video
        self.mirrored = mirrored
        self._last_ms = -1

    def close(self):
        self.lm.close()

    def detect(self, img, ts_s=None):
        """img: HxW gray or HxWx3 BGR uint8. ts_s: frame timestamp (s), required in video mode."""
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.video:
            ms = int(round(ts_s * 1000))
            if ms <= self._last_ms:
                ms = self._last_ms + 1
            self._last_ms = ms
            res = self.lm.detect_for_video(mpimg, ms)
        else:
            res = self.lm.detect(mpimg)
        hands = []
        for i, lms in enumerate(res.hand_landmarks):
            cat = res.handedness[i][0]
            side = cat.category_name.lower()
            if not self.mirrored:
                side = "left" if side == "right" else "right"
            px = np.array([[p.x * w, p.y * h] for p in lms], np.float32)
            rz = np.array([p.z for p in lms], np.float32)
            wl = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[i]], np.float32)
            hands.append(Hand(side, float(cat.score), px, rz, wl))
        return hands


def draw(img, hands, label=True):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    s = max(1, out.shape[0] // 500)
    for hd in hands:
        col = (0, 200, 255) if hd.side == "left" else (255, 120, 0)
        p = hd.px.astype(int)
        for a, b in HAND_EDGES:
            cv2.line(out, tuple(p[a]), tuple(p[b]), col, 2 * s, cv2.LINE_AA)
        for q in p:
            cv2.circle(out, tuple(q), 3 * s, (255, 255, 255), -1, cv2.LINE_AA)
        if label:
            cv2.putText(out, "%s %.2f" % (hd.side, hd.side_score), tuple(p[0] + [5, 20 * s]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, col, 2 * s, cv2.LINE_AA)
    return out
