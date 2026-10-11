#!/usr/bin/env python3
"""Synthetic round trip for handtrack.palm: known hand pose (head frame) ->
project into both passthrough cameras -> detector-image pixels -> fuse ->
compare.  Development check only (synthetic data, not live tracking)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from handtrack.detect import Hand  # noqa: E402
from handtrack.palm import PalmTracker, SLOT_CAM  # noqa: E402

S = 640


def raw_to_det(raw, slot):
    k = 2464 / S
    x = (raw[:, 0] + 0.5) / k - 0.5
    y = (raw[:, 1] + 0.5) / k - 0.5
    if slot == 0:   # CW
        return np.stack([S - 1 - y, x], 1)
    return np.stack([y, S - 1 - x], 1)   # CCW


def synth_hand(side, centre, R):
    """21 points: wrist, MCPs laid out in the hand frame (x thumb-ward for left / pinky-ward for right)."""
    pts = np.zeros((21, 3))
    sgn = 1 if side == "left" else -1          # index on +x for left hand, -x for right (controller convention)
    pts[0] = [0, 0, 0.045]                      # wrist behind centre (+z)
    for j, off in zip((5, 9, 13, 17), (0.035, 0.012, -0.012, -0.032)):
        pts[j] = [sgn * off, 0, -0.045 + abs(off) * 0.2]
    for j in range(21):
        if not pts[j].any() and j != 0:
            pts[j] = pts[((j - 1) // 4) * 4 + 1] + [0, 0, -0.03]
    centre_pts = (pts[0] + pts[5] + pts[9] + pts[17]) / 4
    return (pts - centre_pts) @ R.T + centre


def rot(axis, deg):
    a = np.radians(deg)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def main():
    pt = PalmTracker(S)
    pt.roll_flip = False   # synthetic hand uses the unflipped convention
    worst_p = worst_a = 0
    for i, (side, c, R) in enumerate([
        ("right", [0.15, -0.20, -0.40], np.eye(3)),
        ("left", [-0.12, -0.10, -0.35], rot([0, 1, 0], 30)),
        ("right", [0.05, 0.05, -0.55], rot([1, 0, 0], -60)),
        ("left", [-0.20, -0.25, -0.30], rot([1, 1, 0], 45)),
    ]):
        P = synth_hand(side, np.array(c), R)
        hands = {}
        for slot, name in SLOT_CAM.items():
            Rc, tc = pt.pose[slot]
            Pc = (P - tc) @ Rc                      # head -> camera frame
            raw = pt.cam[slot].project(Pc)
            hands[slot] = [Hand(side, 0.9, raw_to_det(raw, slot).astype(np.float32), np.zeros(21), np.zeros((21, 3)))]
        pt.filt = {s: tuple(type(f)() for f in fs) for s, fs in pt.filt.items()}   # fresh filters
        out = pt.fuse(hands[0], hands[1], float(i))
        p = out[side]
        if p is None:
            print("case %d: %s hand NOT recovered" % (i, side))
            continue
        q = p[1]
        w, x, y, z = q
        Rq = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                       [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        dpos = np.linalg.norm(p[0] - np.array(c)) * 1000
        dang = np.degrees(np.arccos(np.clip((np.trace(Rq.T @ R) - 1) / 2, -1, 1)))
        worst_p, worst_a = max(worst_p, dpos), max(worst_a, dang)
        print("case %d %-5s pos err %.2f mm  rot err %.2f deg" % (i, side, dpos, dang))
        # single-camera path: metric model at true scale, arbitrary orientation (like MediaPipe world landmarks)
        Wm = (P - P.mean(0)) @ rot([0.3, 1, 0.2], 77).T
        for slot in (0, 1):
            hd = hands[slot][0]
            hd.world = Wm.astype(np.float32)
            pt.filt = {s: tuple(type(f)() for f in fs) for s, fs in pt.filt.items()}
            mo = pt.mono(slot, [hd], float(i) + 0.5)[side]
            if mo is None:
                print("   mono slot%d: NOT recovered" % slot)
                continue
            print("   mono slot%d pos err %.2f mm" % (slot, np.linalg.norm(mo[0] - np.array(c)) * 1000))
    print("worst: %.2f mm, %.2f deg" % (worst_p, worst_a))


if __name__ == "__main__":
    main()
