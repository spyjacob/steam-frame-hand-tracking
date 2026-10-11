"""Continuous per-hand tracking with Monado Mercury's keypoint model (opt-in: --backend mercury).

MediaPipe stays the DISCOVERY path (finds hands, initial 21-joint 3D).  Once a hand has a 3D
joint state, this tracker re-observes it every cycle with one Mercury crop per hand (cameras
alternate), using the hand's own previous joints as ROI and temporal hint.  A single view gives a
full articulated 3D hand: Mercury predicts each joint's ray AND its depth relative to the middle
knuckle in hand-size units; absolute depth is anchored to the 60 Hz stereo palm position from the
optical-flow tracker when available, else to the previous state.  When the other camera's result
for the same hand is fresh (< STEREO_MS), joints are triangulated instead.

The Mercury model license is UNVERIFIED (local evaluation only).
"""
import time

import numpy as np

from handtrack.calib import triangulate
from handtrack.palm import PALM_IDX, finger_curls

STEREO_MS = 40.0          # other view's rays this fresh -> triangulate joints
COLLAPSE_M = 0.07         # two tracked hands closer than this (palm centres) = one physical hand
CURL_SCALE = (2.0, 4.0, 4.0, 4.0, 4.0)   # Mercury curl output (negative = flexed) -> 0..1, thumb first
                                         # (provisional mapping, not calibrated against ground truth)
LOST_AFTER_S = 0.35       # no accepted Mercury observation for this long -> back to discovery


class HandState:
    def __init__(self, side, P, t):
        self.side = side
        self.P = np.asarray(P, float)             # (21,3) head frame
        self.t = t
        self.hand_size = float(np.linalg.norm(self.P[0] - self.P[9]))
        self.next_slot = 1
        self.last_submit = 0.0                    # monotonic s of the last job that included this hand
        self.vel = np.zeros(3)                    # palm velocity (m/s) from the last two accepted observations
        self.last_rays = {}                       # slot -> (t, origin, rays head frame)
        self.misses = 0
        self.created = t
        self.curls_by_view = {}                   # slot -> (t, mercury curls (5,), variances (5,))


class MercuryTracker:
    def __init__(self, palm, pool, worker):
        self.palm = palm                          # PalmTracker: pose[slot], _pose(), curl_filt
        self.pool = pool                          # dedicated ProcessPoolExecutor(1)
        self.worker = worker                      # module with run_mercury(jobs, imgs)
        self.hands = {}                           # side -> HandState
        self.job = None
        self.curl_source = "mercury"
        self.predict_roi = False
        self.stereo_ms = STEREO_MS                # max capture-time gap for triangulating two views (0 = same capture only)
        self.curl_latest = False                  # True: curls from this observation only (no blend with older views)
        self.hint_reset = False                   # True: no keypoint hint for a crop placed after a miss / lost flow
        self.sched = "batch"                      # "batch": all hands in one job (original);
                                                  # "perhand": one hand per job on the newest frame,
                                                  # least recently serviced hand first
        self.collapsed = None
        self.next_id = 0
        self.records = []                         # per-observation timing records, drained by the caller
        self.stats = {"runs": 0, "accepted": 0, "rejected": 0, "stereo": 0, "mono": 0, "ms": []}

    def tracked(self):
        return set(self.hands)

    @staticmethod
    def palm_c(P):
        return np.asarray(P)[[0, 5, 9, 17]].mean(0)

    def seed(self, side, P, t):
        """New 3D joints from MediaPipe discovery: start (or re-anchor) tracking this hand.
        Never start a hand on top of the other tracked hand."""
        if P is None:
            return
        other = self.hands.get("right" if side == "left" else "left")
        if other is not None and np.linalg.norm(self.palm_c(P) - self.palm_c(other.P)) < COLLAPSE_M:
            self.stats["seed_blocked"] = self.stats.get("seed_blocked", 0) + 1
            return
        hs = self.hands.get(side)
        if hs is None or t - hs.t > LOST_AFTER_S:
            self.hands[side] = HandState(side, P, t)

    def busy(self):
        return self.job is not None

    def submit(self, imgs, t_cap_ns, fast_pos, anchor_pos=None):
        """One Mercury crop per tracked hand on this frame (alternating cameras); in "perhand" mode
        only the least recently serviced hand.  fast_pos places the crop (its image location is right
        even when its depth is held); anchor_pos (default fast_pos) is the depth anchor for single-view
        joints and should only contain trustworthy (two-camera) positions."""
        if self.job is not None or not self.hands:
            return False
        jobs = []
        sides = list(self.hands)
        if self.sched == "perhand":
            sides = [min(sides, key=lambda sd: self.hands[sd].last_submit)]
        t_sub = time.monotonic()
        for side in sides:
            hs = self.hands[side]
            hs.last_submit = t_sub
            P = hs.P
            fp = fast_pos.get(side)
            expand = None
            if fp is not None:                    # move the ROI with the 60 Hz palm position
                P = P + (np.asarray(fp[0]) - P[PALM_IDX[:1] + [5, 9, 17]].mean(0))
            elif self.predict_roi:
                # fast tracker lost the hand (typically a quick slap): put the crop where the hand is
                # heading, not where it was (bounded extrapolation), and look wider
                dt = min(max(t_cap_ns / 1e9 - hs.t, 0.0), 0.15)
                P = P + hs.vel * dt
            if self.predict_roi:
                spd = float(np.linalg.norm(hs.vel))
                if fp is None or spd > 1.0:
                    expand = 1.7 * (1.35 if fp is None or spd > 2.5 else 1.15)
            slot = hs.next_slot
            hs.next_slot = 1 - slot
            R, t = self.palm.pose[slot]
            jc = (P - t) @ R
            if (jc[:, 2] < 0.08).any():          # behind / too close to this camera: try the other one
                slot = 1 - slot
                R, t = self.palm.pose[slot]
                jc = (P - t) @ R
                if (jc[:, 2] < 0.08).any():
                    continue
            # a crop after a rejected observation or without a fresh flow position is a recovery crop:
            # the previous keypoints describe a hand that is no longer there, so do not feed them as a hint
            hint = not (self.hint_reset and (hs.misses > 0 or fp is None))
            jobs.append((slot, side, jc, expand, hint))
        if not jobs:
            return False
        used = sorted({j[0] for j in jobs})
        self.job = (self.pool.submit(self.worker.run_mercury, jobs, {s: imgs[s] for s in used}), t_cap_ns, fast_pos,
                    {s: imgs[s] for s in used}, time.monotonic_ns(), len(jobs))
        if anchor_pos is not None:
            self.job = self.job[:2] + (anchor_pos,) + self.job[3:]
        return True

    def collect(self):
        """If the in-flight job finished: update hand states.
        Returns {side: (pose tuple, slot, det_px, t_capture_s, image_the_crop_came_from)}."""
        if self.job is None or not self.job[0].done():
            return {}
        fut, t_ns, fast_pos, job_imgs, t_sub_ns, njob = self.job
        self.job = None
        t = t_ns / 1e9
        t_col_ns = time.monotonic_ns()
        try:
            res = fut.result()
        except Exception:
            return {}
        out = {}
        for slot, side, r, dsec, *tim in res:
            self.stats["runs"] += 1
            self.stats["ms"].append(dsec * 1e3)
            tim = tim[0] if tim else {}
            rec = {"id": self.next_id, "side": side, "slot": slot, "t_cap": t_ns, "t_sub": t_sub_ns,
                   "t0": tim.get("t0", 0), "t1": tim.get("t1", 0), "cpu_ns": tim.get("cpu_ns", 0),
                   "wait_ns": tim.get("wait_ns", 0), "t_col": t_col_ns, "njob": njob, "outcome": "rej", "rq": None,
                   "hint": int(tim.get("hint", True))}
            self.next_id += 1
            self.records.append(rec)
            hs = self.hands.get(side)
            if hs is None:
                rec["outcome"] = "gone"
                continue
            if r is None or r["is_hand"] < 0.97:
                self.stats["rejected"] += 1
                hs.misses += 1
                continue
            R, o = self.palm.pose[slot]
            rays = r["rays"] @ R.T                       # head frame
            hs.last_rays[slot] = (t, o, rays)
            other = hs.last_rays.get(1 - slot)
            P = None
            if other is not None and abs(t - other[0]) * 1e3 <= self.stereo_ms and \
                    (self.stereo_ms > 0 or t == other[0]):
                (to, oo, ro) = other
                Pm, gap, s1, s2 = triangulate(np.broadcast_to(o, rays.shape), rays, np.broadcast_to(oo, ro.shape), ro)
                if np.median(gap) < 0.02 and (s1 > 0.05).all() and (s2 > 0.05).all():
                    P = Pm
                    hs.hand_size = 0.9 * hs.hand_size + 0.1 * float(np.linalg.norm(P[0] - P[9]))
                    self.stats["stereo"] += 1
                    rec["outcome"] = "stereo"
            if P is None:
                # single view: rays + Mercury's relative depths; absolute depth of the middle knuckle from
                # the 60 Hz stereo palm position if fresh, else from the previous state
                fp = fast_pos.get(side)
                anchor = np.asarray(fp[0]) if fp is not None else hs.P[[0, 5, 9, 17]].mean(0)
                prevc = hs.P[[0, 5, 9, 17]].mean(0)
                d9 = np.linalg.norm(hs.P[9] + (anchor - prevc) - o)
                dist = d9 + r["rel_depth"] * hs.hand_size
                P = o + rays * dist[:, None]
                # re-anchor so the palm centre sits on the stereo palm position
                if fp is not None:
                    P = P + (anchor - P[[0, 5, 9, 17]].mean(0))
                self.stats["mono"] += 1
                rec["outcome"] = "mono"
            jump = np.linalg.norm(P[[0, 5, 9, 17]].mean(0) - hs.P[[0, 5, 9, 17]].mean(0))
            if jump > 0.25 and t - hs.t < 0.2:         # implausible: keep the old state, count a miss
                self.stats["rejected"] += 1
                hs.misses += 1
                rec["outcome"] = "jump"
                continue
            dtv = t - hs.t
            if 0.005 < dtv < 0.2:
                v = (self.palm_c(P) - self.palm_c(hs.P)) / dtv
                sp = np.linalg.norm(v)
                hs.vel = v * min(1.0, 6.0 / sp) if sp > 0 else v      # bounded: 6 m/s
            else:
                hs.vel = np.zeros(3)
            hs.P, hs.t, hs.misses = P, t, 0
            self.stats["accepted"] += 1
            rec["rq"] = self.palm.raw_quat(side, P)
            hs.curls_by_view[slot] = (t, np.asarray(r["curls"], float), np.asarray(r["curl_var"], float))
            if self.curl_source == "mercury":
                # Mercury's own curl output (consistent across views): variance-weighted mean of fresh views
                vs = [(c, v) for (tc, c, v) in hs.curls_by_view.values() if t - tc < 0.2] if not self.curl_latest \
                    else [hs.curls_by_view[slot][1:]]
                wsum = sum(1 / v for _, v in vs)
                mc = sum(c / v for c, v in vs) / wsum
                raw_curls = np.clip(-mc / np.array(CURL_SCALE), 0, 1)
            else:
                raw_curls = finger_curls(P)
            curls = self.palm.curl_filt[side](raw_curls, t)
            pose = self.palm._pose(side, P, t, conf=r["is_hand"]) + (curls,)
            out[side] = (pose, slot, r["det_px"], t, job_imgs[slot])
        for side in list(self.hands):
            if t - self.hands[side].t > LOST_AFTER_S:
                del self.hands[side]
        # collapse guard: both trackers on the same physical hand -> drop the newer track, re-discover
        self.collapsed = None
        if len(self.hands) == 2:
            L, R = self.hands["left"], self.hands["right"]
            if np.linalg.norm(self.palm_c(L.P) - self.palm_c(R.P)) < COLLAPSE_M:
                drop = "left" if L.created > R.created else "right"
                del self.hands[drop]
                out.pop(drop, None)
                self.collapsed = drop
                self.stats["collapses"] = self.stats.get("collapses", 0) + 1
        return out
