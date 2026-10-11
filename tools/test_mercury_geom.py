#!/usr/bin/env python3
"""Geometry self-test for handtrack.mercury_kp (no model inference, synthetic joints):
crop mapping vs direct projection, and crop-position -> ray decoding round trip."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from handtrack.calib import Rig  # noqa: E402
from handtrack.mercury_kp import (MercuryKeypoint, WSIZE, make_instructions, project_unscaled,  # noqa: E402
                                  q_rot, sg_unproject)
from handtrack.palm import SLOT_CAM  # noqa: E402

rig = Rig()
mk = MercuryKeypoint(os.path.expanduser("~/handtrack/models/mercury/grayscale_keypoint_jan18.onnx"), rig, SLOT_CAM, 640)
rng = np.random.default_rng(0)
worst_px = worst_deg = 0
for case in range(6):
    centre = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.3, 0.0), -rng.uniform(0.3, 0.6)])
    joints_h = centre + rng.normal(0, 0.04, (21, 3))
    for slot in (0, 1):
        R, t = rig.head_from_cam(SLOT_CAM[slot])
        jc = (joints_h - t) @ R                      # head -> OpenCV camera frame
        for flip in (False, True):
            jm = jc * np.array([1, -1, -1])
            q, r, kp, _ = make_instructions(jm, flip)
            sg = project_unscaled(jm, q)
            # crop pixel of each joint (inverse of the crop's linear sg <-> pixel map)
            xpix = (sg[:, 0] + r) / (2 * r) * WSIZE if not flip else (r - sg[:, 0]) / (2 * r) * WSIZE
            ypix = (r - sg[:, 1]) / (2 * r) * WSIZE
            # 1) crop sampling position at that pixel vs direct projection of the joint
            sx = (-r + 2 * r * xpix / WSIZE) if not flip else (r - 2 * r * xpix / WSIZE)
            sy = r - 2 * r * ypix / WSIZE
            d = q_rot(q, np.stack([2 * sx, 2 * sy, sx * sx + sy * sy - 1], 1)) * np.array([1, -1, -1])
            det_crop = mk.raw_to_det(mk.cam[slot].project(d), slot)
            det_direct = mk.raw_to_det(mk.cam[slot].project(jc), slot)
            e_px = np.abs(det_crop - det_direct).max()
            # 2) decode: crop pixel -> heatmap location -> ray, vs the joint's true direction
            loc = np.stack([xpix, ypix], 1) * 22 / WSIZE
            px = (1 - loc[:, 0] / 11.0) if flip else (loc[:, 0] / 11.0 - 1)
            py = 1 - loc[:, 1] / 11.0
            rays = q_rot(q, sg_unproject(px * r, py * r)) * np.array([1, -1, -1])
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            truth = jc / np.linalg.norm(jc, axis=1, keepdims=True)
            e_deg = np.degrees(np.arccos(np.clip((rays * truth).sum(1), -1, 1))).max()
            # 3) the model's keypoint hint uses Mercury's own (different) sign convention: report only
            inside = np.mean((np.abs(kp) <= 1.0).all(1))
            worst_px, worst_deg = max(worst_px, e_px), max(worst_deg, e_deg)
            print("case %d slot %d %-5s crop-vs-direct %.4f px  decode %.5f deg  hint inside crop %.0f%%  radius %.3f" %
                  (case, slot, "right" if flip else "left", e_px, e_deg, 100 * inside, r))
print("worst: %.4f px, %.5f deg" % (worst_px, worst_deg))
