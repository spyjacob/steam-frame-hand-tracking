"""Fast palm tracking between MediaPipe runs: position AND orientation at camera rate.

MediaPipe (palm search + 21-point landmark network) costs ~20-50 ms per camera
frame on the Frame's CPU, so it only runs ~5-10 times a second (heat).  Between
runs, the six palm landmarks MediaPipe found (wrist, thumb CMC, four knuckles)
are followed individually in both cameras with pyramidal Lucas-Kanade optical
flow, helped by up to 12 textured points inside the hand outline.  Triangulating
the six landmarks every frame gives the full palm frame (position + rotation)
at 60 Hz with the same construction palm.py uses.

When a new MediaPipe result re-seeds the points, the difference between the
old and new estimate is not applied as a jump: it is eased in over `blend_s`
(exponential), so re-seeds do not cause visible snaps.

Nothing here replaces MediaPipe: finger curls (and later full finger tracking)
still come from it.  Coordinates are the rotated detector images (det_size
square), the frame MediaPipe's landmarks are in; PalmTracker.rays() maps them
to head-frame rays.
"""
import collections
import math

import cv2
import numpy as np

from handtrack.calib import triangulate

LM_IDX = [0, 1, 5, 9, 13, 17]       # tracked landmark identities (MediaPipe indices)
CENTRE_ROWS = [0, 2, 3, 5]          # rows of LM_IDX forming the palm centre (0, 5, 9, 17) as in palm.py
LK_WIN, LK_LEVELS = (15, 15), 3
LK = dict(winSize=LK_WIN, maxLevel=LK_LEVELS,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.05))
N_EXTRA = 12                        # textured points per hand besides the 6 landmarks (was 30)
FEAT_MARGIN = 16                    # px of image around the hand outline used when picking texture points

# Nearly all of an LK call's cost is building both images' pyramids and the first image's derivatives:
# tracking 36 points costs about the same as tracking 1.  So all hands seen by a camera are tracked in
# one call pair (step_tracks).  Sharing one pyramid per image across calls would save most of the rest,
# but OpenCV's Python binding takes single images only (prebuilt pyramids are rejected, checked on
# OpenCV 5.0); that needs a C++ tracker.
MAX_SPEED = 4.0                     # m/s; faster apparent motion is a tracking error, not a hand
MAX_GAP = 0.025                     # m, median ray gap of the triangulated landmarks


def plausible(pos):
    """A palm position (head frame, m) a real hand can have: 10 cm - 1 m from the head, in front."""
    pos = np.asarray(pos, float)
    return 0.10 < np.linalg.norm(pos) < 1.0 and pos[2] < 0.05


# ---------------------------------------------------------------- quaternions (w, x, y, z)
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qangle(q):
    return 2 * math.acos(min(1.0, abs(q[0])))


def qpow(q, f):
    """q^f (fraction of the rotation), shortest path."""
    if q[0] < 0:
        q = -q
    ang = 2 * math.acos(min(1.0, q[0]))
    s = math.sin(ang / 2)
    if s < 1e-9:
        return np.array([1.0, 0, 0, 0])
    axis = q[1:] / s
    h = ang * f / 2
    return np.concatenate([[math.cos(h)], axis * math.sin(h)])


def palm_frame(side, wrist, idx, mid, ring, pinky, roll_flip=False):
    """Rotation (columns x, y, z) of the palm frame, same construction as palm.PalmTracker._pose."""
    fwd = (idx + mid + ring + pinky) / 4 - wrist
    z = -fwd / np.linalg.norm(fwd)
    across = idx - pinky
    x = -across if side == "right" else across
    if roll_flip:
        x = -x
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


# ---------------------------------------------------------------- per camera
class CamTrack:
    """Six identified palm landmarks + anonymous texture points of one hand in one camera."""

    def __init__(self, lm, extra):
        self.lm = lm.astype(np.float32).reshape(-1, 2)          # (6,2), identities = LM_IDX
        self.extra = extra.astype(np.float32).reshape(-1, 2)    # (N,2)
        self.ok = True

    @property
    def centre(self):
        return self.lm[CENTRE_ROWS].mean(0).astype(np.float64)

    def points(self):
        return np.vstack([self.lm, self.extra])

    def update(self, old, new, good):
        """Apply one flow result: old = points() before the step, new = their tracked positions."""
        if good.sum() < 4:
            self.ok = False
            return False
        shift = np.median(new[good] - old[good], axis=0)
        n = len(self.lm)
        # landmarks keep their identity: tracked if good, otherwise moved with the hand
        self.lm = np.where(good[:n, None], new[:n], old[:n] + shift).astype(np.float32)
        self.extra = new[n:][good[n:]].astype(np.float32)
        return True

    def step(self, prev, cur):
        return step_tracks([self], prev, cur)[0]


def step_tracks(tracks, prev, cur):
    """Follow CamTracks of one camera from image prev to cur, all their points in one forward and one
    backward LK call.  LK tracks every point on its own, so each track gets exactly the result it would
    get alone, for about the cost of one track (see the note at LK).  Returns each track's success."""
    pts = [ct.points() for ct in tracks]
    allp = np.vstack(pts).reshape(-1, 1, 2)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, allp, None, **LK)
    if nxt is None:
        for ct in tracks:
            ct.ok = False
        return [False] * len(tracks)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, **LK)   # forward-backward check
    fb = np.linalg.norm((back - allp).reshape(-1, 2), axis=1)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 2.5)
    new = nxt.reshape(-1, 2)
    out, i = [], 0
    for ct, p in zip(tracks, pts):
        out.append(ct.update(p, new[i:i + len(p)], good[i:i + len(p)]))
        i += len(p)
    return out


def texture_points(img, px):
    """Up to N_EXTRA corners inside the convex hull of the landmarks px (N,2), as (M,2) float32.
    Same points as goodFeaturesToTrack on the whole image with the hull as mask (several times the
    cost): the search runs on the hull's bounding box plus FEAT_MARGIN, which is wider than the corner
    filters reach, with x aligned to 16 px so each column gets the same SIMD arithmetic as in the whole
    image.  tools/test_fasttrack.py checks the equality."""
    hull = cv2.convexHull(px.astype(np.int32))
    h, w = img.shape[:2]
    lo, hi = hull.reshape(-1, 2).min(0), hull.reshape(-1, 2).max(0)
    if (lo < -4096).any() or (hi > 4096).any():          # wild landmarks: plain whole-image search
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0 = max(0, (int(lo[0]) - FEAT_MARGIN) // 16 * 16)
        y0 = max(0, int(lo[1]) - FEAT_MARGIN)
        x1 = min(w, (int(hi[0]) + FEAT_MARGIN + 16) // 16 * 16)
        y1 = min(h, int(hi[1]) + FEAT_MARGIN + 1)
        if x1 <= x0 or y1 <= y0:                       # hand entirely outside the image
            return np.zeros((0, 2), np.float32)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(mask, hull - np.array([x0, y0], np.int32), 255)
    f = cv2.goodFeaturesToTrack(img[y0:y1, x0:x1], maxCorners=N_EXTRA, qualityLevel=0.01, minDistance=8, mask=mask)
    if f is None:
        return np.zeros((0, 2), np.float32)
    return (f.reshape(-1, 2) + np.array([x0, y0], np.float32)).astype(np.float32)


class HandTrack:
    def __init__(self):
        self.cams = {}            # slot -> CamTrack
        self.t_confirm = 0.0      # last MediaPipe confirmation (s)
        self.last_pos = None      # raw (unblended) head-frame position
        self.last_pos_t = 0.0
        self.out_pos = None       # last output (blended)
        self.out_q = None
        self.reseeded = False
        self.corr_pos = np.zeros(3)
        self.corr_q = np.array([1.0, 0, 0, 0])
        self.corr_t = 0.0
        self.mono_since = None    # time the track fell back to one camera (v2: bounded)


class FastTracker:
    def __init__(self, palm, orient=True, blend_s=0.12):
        self.palm = palm          # PalmTracker: rays(), pose[]
        self.orient = orient
        self.blend_s = blend_s
        self.hands = {}           # side -> HandTrack
        self.prev = {}            # slot -> previous gray image (rotated det frame)
        self.lost = set()         # slots where a hand track just failed -> caller re-detects there now
        self.info = {}
        self._src = "none"
        # v2 (--flow-v2): frame history for frame-by-frame re-seeding, bounded single-camera depth,
        # rebuilding a lost camera from a 3D hand observation
        self.v2 = False
        self.mono_max_s = 0.2
        self.agree = False        # --flow-agree: a new/rebuilt camera view must agree with the other camera
        self.agree_jump = 0.06    # m: and with the current position (if fresh)
        self.hist = {}            # slot -> deque of (t, image), newest last
        self.stats = collections.Counter()

    # ------------------------------------------------------------ seeding
    def seed(self, slot, side, hand, det_img, cur_img, t, pos3d=None):
        """Re-seed from a MediaPipe result computed on det_img; carry the points forward
        to cur_img (the newest frame) with one optical-flow step.  pos3d: MediaPipe's
        head-frame palm centre, used only to start a track."""
        # the points must refer to the frame the next step() starts from: cur_img, unless the caller
        # skipped step() for this frame (heat), in which case the last frame it did process
        cur_img = self.prev.get(slot, cur_img)
        lm = hand.px[LM_IDX].astype(np.float32)
        extra = np.zeros((0, 2), np.float32)
        if det_img is not None:
            extra = texture_points(det_img, hand.px)
        ct = CamTrack(lm, extra)
        if det_img is not None and cur_img is not None and det_img is not cur_img:
            if not self._advance(ct, slot, det_img, cur_img):
                return
        ht = self.hands.setdefault(side, HandTrack())
        if not self._agrees(ht, slot, ct, t):
            return
        ht.cams[slot] = ct
        ht.t_confirm = t
        ht.reseeded = True
        if pos3d is not None and plausible(pos3d) and (ht.last_pos is None or t - ht.last_pos_t > 0.3):
            ht.last_pos, ht.last_pos_t = np.asarray(pos3d, float), t

    def _advance(self, ct, slot, src_img, cur_img):
        """Carry a CamTrack from src_img to cur_img.  v2: through every retained frame in between
        (a slap can reverse within the gap; one big LK step mis-tracks it).  Falls back to one step
        when src_img is not in the history."""
        if self.v2:
            try:
                return self._advance_hist(ct, slot, src_img, cur_img)
            except Exception as e:                    # never let the experiment stop tracking
                self.stats["advance_error"] += 1
                if self.stats["advance_error"] == 1:
                    print("flow v2 advance error: %r" % (e,), flush=True)
        return ct.step(src_img, cur_img)

    def _advance_hist(self, ct, slot, src_img, cur_img):
        if True:
            h = self.hist.get(slot, ())
            idx = next((i for i, (_, im) in enumerate(h) if im is src_img), None)
            if idx is not None:
                prev = src_img
                for _, im in list(h)[idx + 1:]:
                    if not ct.step(prev, im):
                        self.stats["reseed_fail"] += 1
                        return False
                    prev = im
                if prev is not cur_img and not ct.step(prev, cur_img):
                    return False
                self.stats["reseed_steps"] += len(h) - idx - 1
                return True
            self.stats["reseed_nohist"] += 1
        return ct.step(src_img, cur_img)

    def recover_view(self, side, P, t_obs):
        """v2: a hand tracked in only one camera gets its other camera rebuilt from a 3D observation
        (21 joints, head frame) made at capture time t_obs: project the palm landmarks into that
        camera's frame of the same capture, then follow them to the newest frame."""
        if not self.v2:
            return False
        ht = self.hands.get(side)
        if ht is None or len(ht.cams) != 1:
            return False
        slot = 1 - next(iter(ht.cams))
        h = self.hist.get(slot)
        if not h:
            return False
        src = next((im for (tt, im) in h if abs(tt - t_obs) < 1e-4), None)
        if src is None:
            self.stats["recover_noframe"] += 1
            return False
        R, tc = self.palm.pose[slot]
        if (((np.asarray(P) - tc) @ R)[:, 2] < 0.08).any():
            return False
        px = self.palm.det_px(slot, np.asarray(P))
        S = src.shape[0]
        if (px < 4).any() or (px > S - 5).any():
            return False
        lm = px[LM_IDX].astype(np.float32)
        ct = CamTrack(lm, texture_points(src, px))
        if not self._advance(ct, slot, src, h[-1][1]):
            return False
        if not self._agrees(ht, slot, ct, h[-1][0]):
            return False
        ht.cams[slot] = ct
        self.stats["recovered_view"] += 1
        return True

    def _agrees(self, ht, slot, ct, t):
        """--flow-agree: a (re)seeded or rebuilt view in `slot` is accepted only if, together with the
        other camera's current track, it triangulates consistently (median ray gap < MAX_GAP) AND lands
        within agree_jump of the hand's current position (when that is fresh).  Otherwise the new view
        would make the stereo position snap (the short in/out depth jumps)."""
        if not self.agree:
            return True
        other = ht.cams.get(1 - slot)
        if other is None:
            return True
        c = {slot: ct, 1 - slot: other}
        o1, d1 = self.palm.rays(1, c[1].lm)
        o0, d0 = self.palm.rays(0, c[0].lm)
        P, gap, s1, s0 = triangulate(np.broadcast_to(o1, d1.shape), d1, np.broadcast_to(o0, d0.shape), d0)
        ok = np.median(gap) < MAX_GAP and (s1 > 0.05).all() and (s0 > 0.05).all()
        if ok and ht.last_pos is not None and t - ht.last_pos_t < 0.1:
            ok = np.linalg.norm(P[CENTRE_ROWS].mean(0) - ht.last_pos) < self.agree_jump
        if ok:
            ht.agree_rejects = 0
            return True
        ht.agree_rejects = getattr(ht, "agree_rejects", 0) + 1
        self.stats["agree_reject"] += 1
        if ht.agree_rejects >= 3:
            # repeated disagreement: the existing track is more likely the stale one; take the new
            # detection and drop the other camera (it is rebuilt / re-seeded next)
            ht.agree_rejects = 0
            ht.cams.pop(1 - slot, None)
            self.stats["agree_override"] += 1
            return True
        return False

    def drop_stale(self, t, max_age=1.0):
        for side in list(self.hands):
            if t - self.hands[side].t_confirm > max_age:
                del self.hands[side]

    # ------------------------------------------------------------ per frame
    def _raw(self, side, ht, t=None):
        """Unblended estimate from the current tracks: (pos, quat or None)."""
        if 0 in ht.cams and 1 in ht.cams:
            o1, d1 = self.palm.rays(1, ht.cams[1].lm)
            o0, d0 = self.palm.rays(0, ht.cams[0].lm)
            P, gap, s1, s0 = triangulate(np.broadcast_to(o1, d1.shape), d1, np.broadcast_to(o0, d0.shape), d0)
            if np.median(gap) < MAX_GAP and (s1 > 0.05).all() and (s0 > 0.05).all():
                pos = P[CENTRE_ROWS].mean(0)
                q = None
                if self.orient:
                    from handtrack.palm import quat_from_R
                    R = palm_frame(side, P[0], P[2], P[3], P[4], P[5], self.palm.roll_flip)
                    q = quat_from_R(R)
                self._src = "stereo"
                ht.mono_since = None
                return pos, q
        if ht.cams and ht.last_pos is not None and ht.last_pos_t > 0:
            # one camera: keep the last distance along the new centre ray.  v2: only briefly; a held
            # depth during fast in/out motion is confidently wrong, so report nothing instead
            if self.v2 and t is not None:
                if ht.mono_since is None:
                    ht.mono_since = t
                elif t - ht.mono_since > self.mono_max_s:
                    self.stats["mono_expired"] += 1
                    self._src = "none"
                    return None, None
            slot, ct = next(iter(ht.cams.items()))
            o, d = self.palm.rays(slot, ct.centre.reshape(1, 2))
            self._src = "mono_ray%d" % slot
            return o.ravel() + d.ravel() * np.linalg.norm(ht.last_pos - o.ravel()), None
        self._src = "none"
        return None, None

    def step(self, imgs, t):
        """imgs: {slot: gray det-frame image} for this capture instant.
        Returns {side: (head-frame palm centre (3,), quat wxyz or None)} for tracked hands."""
        self.info = {}            # side -> telemetry: position source, observation time, live point counts
        out = {}
        for slot, cur in imgs.items():
            # every hand this camera sees, in one LK call pair
            if slot not in self.prev:
                continue
            seen = [ht for ht in self.hands.values() if slot in ht.cams]
            if not seen:
                continue
            for ht, ok in zip(seen, step_tracks([ht.cams[slot] for ht in seen], self.prev[slot], cur)):
                if not ok:
                    del ht.cams[slot]
                    self.lost.add(slot)
        for side, ht in self.hands.items():
            pos, q = self._raw(side, ht, t)
            # plausibility: near the face, and no faster than a real hand moves
            if pos is not None and ht.last_pos is not None and t - ht.last_pos_t < 0.5:
                if np.linalg.norm(pos - ht.last_pos) > MAX_SPEED * (t - ht.last_pos_t) + 0.03:
                    pos = None
            if pos is not None and not plausible(pos):
                pos = None
            if pos is None:
                if ht.cams:                       # points no longer trustworthy: drop, re-detect now
                    self.lost.update(ht.cams.keys())
                    ht.cams.clear()
                if ht.out_pos is not None and t - ht.last_pos_t < 0.3:
                    out[side] = (ht.out_pos, ht.out_q)    # hold the last output briefly
                    self.info[side] = {"src": "hold", "t_obs": ht.last_pos_t,
                                       "npts": {sl: 0 for sl in (0, 1)}}
                continue
            ht.last_pos, ht.last_pos_t = pos, t
            if q is None:
                q = ht.out_q                      # no fresh orientation: keep the last one
            # ease re-seed corrections in instead of jumping
            if ht.reseeded and ht.out_pos is not None and self.blend_s > 0:
                dp = ht.out_pos - pos
                dq = qmul(ht.out_q, qconj(q)) if (q is not None and ht.out_q is not None) else np.array([1.0, 0, 0, 0])
                if np.linalg.norm(dp) < 0.10 and qangle(dq) < math.radians(35):
                    ht.corr_pos, ht.corr_q, ht.corr_t = dp, dq, t
                else:                             # big disagreement = genuine recovery: snap
                    ht.corr_pos, ht.corr_q = np.zeros(3), np.array([1.0, 0, 0, 0])
            ht.reseeded = False
            f = math.exp(-(t - ht.corr_t) / self.blend_s) if self.blend_s > 0 else 0.0
            pos_out = pos + ht.corr_pos * f
            q_out = qmul(qpow(ht.corr_q, f), q) if q is not None else None
            ht.out_pos, ht.out_q = pos_out, q_out
            out[side] = (pos_out, q_out)
            self.info[side] = {"src": self._src, "t_obs": t,
                               "npts": {sl: (len(c.lm) + len(c.extra)) for sl, c in ht.cams.items()}}
        self.prev = dict(imgs)
        if self.v2:
            for sl, im in imgs.items():
                self.hist.setdefault(sl, collections.deque(maxlen=8)).append((t, im))
        return out
