"""Single-model hand tracker (v4): one keypoint model, one filter per hand, nothing else in the output.

Every camera frame (60 Hz):
  1. collect finished keypoint crops -> rigid palm fit (handfit.fit_palm) -> Kalman update
  2. for every tracked hand that is due (<= --hz per hand, shared --budget): predict its pose to this
     frame, place one 128x128 crop on it (Mercury keypoint model) in its primary camera; a new track and
     every few seconds afterwards: both cameras of the same frame (learns the user's palm size, fixes depth)
  3. if an enabled hand is missing: hand finders propose candidates (Mercury's detector in-process,
     MediaPipe in its own process); each is checked by the keypoint model in both mirror images, which
     also decides left vs right; a checked candidate starts a track
  4. send every enabled hand's pose predicted to this frame's capture time; a hand with no good
     observation glides to a stop, is held at most hold_s, then reported missing

Hands never take their position, orientation or fingers from different sources: one observation
updates all of them together, the filter carries them between observations.
"""
import math
import time
from collections import deque

import numpy as np

from handtrack import mercury as M
from handtrack.calib import triangulate
from handtrack.handfit import (MID, PALM, OneEuro, PalmTemplate, PosKF, RotKF, ViewObs, exp_so3, fit_palm,
                               fit_scale, init_from_rel, init_ippe, log_so3, quat_from_R)

SIDES = ("left", "right")
CURL_SCALE = np.array([2.0, 4.0, 4.0, 4.0, 4.0])   # Mercury curl (negative = flexed) -> 0..1, thumb first
                                                   # (provisional mapping carried over from v3)

DEFAULTS = dict(
    hz=30.0,             # max keypoint updates per second per hand
    budget=70.0,         # max keypoint crops per second for everything together (thermal scales it)
    is_hand=0.95,        # Mercury is-hand probability for an observation (Monado uses 0.97)
    hm_min=0.25,         # mean heatmap peak; wrong-chirality / garbage crops score lower
    gate_deg=4.0,        # max robust palm keypoint residual of an accepted fit
    hold_s=0.25,         # keep sending a hand this long after its last good observation (gliding)
    drop_s=0.40,         # forget the track after this long
    hold_occluded_s=0.6, # ... and these when the other hand is in front of it (it stays put meanwhile)
    drop_occluded_s=0.8,
    hm_drop=0.85,        # an observation whose heatmap confidence falls below this x the track's usual
                         # (wrong hand / mirror image in the crop) must agree with the prediction
    show_after=2,        # good observations before a new track is sent
    stereo_init=3,       # first N updates of a track use both cameras
    stereo_young_s=0.5,  # then both cameras every this often until the palm template is mature ...
    stereo_s=3.0,        # ... and every this often afterwards
    template_mature=12,  # palm-scale samples before the slow stereo cadence
    sa_lo=0.3, sa_hi=40.0,   # position process noise (m/s^2): floor (still hand) .. ceiling (fast moves)
    ra_lo=3.0, ra_hi=400.0,  # rotation process noise (rad/s^2)
    attack=9.0,          # NIS above which the process noise rises at once (the hand changed motion)
    attack_big=36.0,     # position NIS that opens the filter without waiting for a second surprise
    release=0.75,        # per update decay of the process noise back to the floor
    pos_floor=0.002,     # m, measurement noise floor
    rot_floor_deg=1.0,
    coast_after=0.05, coast_tau=0.05, vmax=4.0, wmax=20.0,
    nis_gate=30.0,       # an observation this far from the prediction is an outlier ...
    reinit_after=2,      # ... unless this many consecutive ones agree (then the hand really moved)
    blend_s=0.12,        # output blend after a hand is re-acquired somewhere else
    det_hz=30.0, det_idle_hz=10.0,    # Mercury detector runs/s while a hand is missing (recently / long)
    mp_hz=12.0, mp_idle_hz=4.0,       # MediaPipe runs/s while a hand is missing (recently / long)
    idle_after=2.0,      # s since every missing hand was last seen -> idle search rates
    det_min=0.6,         # Mercury detector probability to try a candidate (Monado: 0.92 averaged)
    chir_margin=0.03,    # heatmap-confidence margin that decides left/right over the finder's label
    collapse_m=0.02,     # two tracks closer than this ... (palms touching: centres ~3.5 cm)
    collapse_s=0.25,     # ... for this long are one hand (a clap's contact is shorter)
    same_hand_deg=9.0,   # a candidate this close to a tracked hand (angle in the camera) is that hand
    curl_filter=(1.5, 1.5),
)


class Job:
    __slots__ = ("kind", "side", "t", "slots", "futs", "extra", "t_sub")

    def __init__(self, kind, side, t, slots, futs, extra=None):
        self.kind, self.side, self.t, self.slots, self.futs, self.extra = kind, side, t, slots, futs, extra
        self.t_sub = time.monotonic()

    def done(self):
        return all(f.done() for f in self.futs)

    def results(self, errors=None):
        out = []
        for f in self.futs:
            try:
                out.append(f.result())
            except Exception as e:                 # never let one bad crop stop tracking
                if errors is not None:
                    errors.append(repr(e))
                out.append(None)
        return out


class Hand:
    def __init__(self, side, tpl, fit, joints, curls, t, prm, stereo):
        self.side, self.tpl, self.prm = side, tpl, prm
        self.pkf = PosKF(fit.c, t, sp=0.03, sv=1.0)
        self.rkf = RotKF(fit.R, t, sr=0.2, sw=6.0)
        self.t_obs = self.t_new = t
        self.n_ok, self.misses = 1, 0
        self.outliers = []
        self.joints_palm = (joints - fit.c) @ fit.R
        self.curl_f = OneEuro(*prm["curl_filter"])
        self.curls = self.curl_f(curls, t)
        self.job = None
        self.t_job = -1.0
        self.n_stereo = 1 if stereo else 0
        self.t_stereo = t if stereo else -1e9
        self.view = None
        self.blend = None                  # (t0, c, R) to blend the output from
        self.last = {}                     # slot -> KpResult (debug view)
        self.hm_base = None                # running heatmap confidence of this hand's good observations
        self.occluded = False              # the last misses saw the other hand in front of this one

    def predict(self, t):
        p = self.prm
        return (self.pkf.peek(t, p["coast_after"], p["coast_tau"], p["vmax"]),
                self.rkf.peek(t, p["coast_after"], p["coast_tau"], p["wmax"]))

    def joints_at(self, c, R):
        return c + self.joints_palm @ R.T


def _smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _plausible(c):
    d = float(np.linalg.norm(c))
    return 0.08 < d < 1.3 and c[2] < -0.03


class Tracker:
    def __init__(self, views, kp, pool, det=None, mp=None, prm=None, log=None):
        """views: {slot: mercury.View}; kp: KeypointModel; pool: thread pool for crops/detection;
        det: HandDetector160 or None; mp: MediaPipe finder (main.MPFinder) or None."""
        self.views, self.kp, self.pool, self.det, self.mp = views, kp, pool, det, mp
        self.prm = dict(DEFAULTS)
        self.prm.update(prm or {})
        self.log = log or (lambda *a: None)
        self.hands = {}
        self.tpl = {sd: PalmTemplate(sd) for sd in SIDES}
        self.enabled = set(SIDES)
        self.frames = deque(maxlen=12)               # recent (t, imgs): candidates are checked on their own frame
        self.tokens = 0.0
        self.t_prev = None
        self.scale = 1.0                             # thermal scale for budget and search rates (main sets it)
        self.det_job = None
        self.ver_job = None
        self.cands = []
        self.t_det = self.t_mp = -1e9
        self.det_slot = self.mp_slot = 0
        self.t_seen = {sd: -1e9 for sd in SIDES}     # last good observation of each hand
        self.t_close = None                          # since when the two tracks are on top of each other
        self.last_out = {}                           # side -> (t, c, R) last pose sent
        self.boxes = []                              # latest detector boxes (debug view)
        self.events = deque(maxlen=4000)             # telemetry rows, drained by the caller
        self.errors = []                             # exceptions from worker jobs (logged by the caller)
        self.stats = self._new_stats()

    @staticmethod
    def _new_stats():
        return {"crops": 0, "crop_ms": [], "acc": 0, "acc_stereo": 0, "miss": 0, "outlier": 0, "reinit": 0,
                "obs_age_ms": [], "det_runs": 0, "det_cands": 0, "mp_runs": 0, "mp_cands": 0, "verify": 0,
                "acquired": 0, "verify_fail": 0, "lost": 0, "collapse": 0, "chir_flip": 0, "upd": {sd: 0 for sd in SIDES},
                "res_deg": [], "budget_skip": 0}

    # ------------------------------------------------------------ main entry
    def set_enabled(self, sides):
        sides = set(sides) & set(SIDES)
        for sd in set(self.hands) - sides:
            del self.hands[sd]
        self.enabled = sides

    def on_frame(self, t, imgs):
        """t: capture time (s, CLOCK_MONOTONIC); imgs: {slot: 640x640 uint8, sensor orientation}.
        Returns {side: (pos(3), quat wxyz(4), conf, None, curls(5)) or None} for this capture time."""
        self.frames.append((t, imgs))
        if self.t_prev is not None:
            cap = self.prm["budget"] * self.scale
            self.tokens = min(cap * 0.12, self.tokens + cap * max(0.0, t - self.t_prev))
        self.t_prev = t
        self.collect()
        self._expire(t)
        self._schedule_tracking(t, imgs)
        self._search(t, imgs)
        return self._outputs(t)

    def collect(self):
        """Process every finished job (also called between frames to save latency)."""
        for side, h in list(self.hands.items()):
            if h.job is not None and h.job.done():
                job, h.job = h.job, None
                self._observe(h, job, job.results(self.errors))
        if self.det_job is not None and self.det_job.done():
            job, self.det_job = self.det_job, None
            res = job.results(self.errors)[0] or []
            self.stats["det_runs"] += 1
            self.boxes = res
            for b in res:
                if b.conf >= self.prm["det_min"]:
                    self.cands.append({"kind": "det", "t": job.t, "slot": b.slot, "side": b.side, "score": b.conf,
                                       "box": b})
                    self.stats["det_cands"] += 1
        if self.mp is not None:
            ran, cs = self.mp.poll()
            self.stats["mp_runs"] += ran
            self.cands += cs
            self.stats["mp_cands"] += len(cs)
        if self.ver_job is not None and self.ver_job.done():
            job, self.ver_job = self.ver_job, None
            self._verified(job, job.results(self.errors))

    # ------------------------------------------------------------ tracking
    def _expire(self, t):
        for side, h in list(self.hands.items()):
            if t - h.t_obs > self.prm["drop_occluded_s" if h.occluded else "drop_s"] and h.job is None:
                del self.hands[side]
                self.stats["lost"] += 1
                self.events.append("L,%d,%s,%.0f" % (int(t * 1e9), side, (t - h.t_obs) * 1e3))

    def _views_for(self, h, t):
        """Predicted joints and the cameras that can see them."""
        c, R = h.predict(t)
        J = h.joints_at(c, R)
        ok = {}
        for slot, v in self.views.items():
            Jc = v.to_cam(J)
            if (Jc[:, 2] > 0.05).all() and v.inside(v.project_cam(v.to_cam(c)))[0]:
                ok[slot] = Jc
        return c, R, ok

    def _schedule_tracking(self, t, imgs):
        p = self.prm
        for side in sorted(self.hands, key=lambda sd: self.hands[sd].t_job):
            h = self.hands[side]
            if h.job is not None or t - h.t_job < 1.0 / p["hz"] - 0.004:
                continue
            c, R, ok = self._views_for(h, t)
            if not ok:
                h.misses += 1
                h.t_job = t
                continue
            young = h.tpl.n < p["template_mature"]
            stereo = len(ok) == 2 and (h.n_stereo < p["stereo_init"] or
                                       t - h.t_stereo > (p["stereo_young_s"] if young else p["stereo_s"]))
            if stereo and self.tokens < 2:
                stereo = False
            if not stereo:
                slot = self._primary(h, ok, c)
                slots = [slot]
            else:
                slots = sorted(ok)
            if self.tokens < len(slots):
                self.stats["budget_skip"] += 1
                continue
            self.tokens -= len(slots)
            speed = h.pkf.speed()
            expand = M.EXPAND * (1 + min(0.5, 0.15 * speed)) * min(2.0, 1.25 ** h.misses)
            futs = []
            for slot in slots:
                ins = M.instr_from_points(ok[slot], flip=(side == "right"), expand=expand)
                futs.append(self.pool.submit(self.kp.run, self.views[slot], imgs[slot], ins, True))
            h.job = Job("track", side, t, slots, futs)
            h.t_job = t
            self.stats["crops"] += len(slots)

    def _primary(self, h, ok, c):
        """Same camera every update (consistent bias = no wobble) unless the other is clearly better."""
        ang = {s: self.views[s].axis_angle(c) for s in ok}
        if h.view not in ang:
            h.view = min(ang, key=ang.get)
        else:
            best = min(ang, key=ang.get)
            if best != h.view and ang[best] < ang[h.view] - math.radians(10):
                h.view = best
        return h.view

    def _sees_other(self, h, t, r, fit, c_p):
        """Did this hand's crop find the other (tracked) hand?  Compared where the observation is
        reliable: the palm's direction from the camera (a mirrored hand's depth is garbage), and
        the fitted position."""
        other = self.hands.get("right" if h.side == "left" else "left")
        if other is None or other.n_ok < 2 or t - other.t_obs > self.prm["hold_s"]:
            return False
        c_o = other.predict(t)[0]
        d_other = float(np.linalg.norm(fit.c - c_o))
        if d_other < self.prm["collapse_m"] * 1.6 and d_other < float(np.linalg.norm(fit.c - c_p)):
            return True
        v = self.views[r.slot]
        u = v.dirs_to_head(r.rays[PALM]).mean(0)
        u /= np.linalg.norm(u)

        def ang(c):
            d = c - v.t
            return math.degrees(math.acos(min(1.0, max(-1.0, float(u @ d) / float(np.linalg.norm(d))))))

        a_o = ang(c_o)
        return a_o < self.prm["same_hand_deg"] and a_o < ang(c_p)

    def _good(self, r):
        return r is not None and r.is_hand >= self.prm["is_hand"] and r.hm >= self.prm["hm_min"]

    def _obs(self, r):
        v = self.views[r.slot]
        return ViewObs(v.t, v.dirs_to_head(r.rays[PALM]), np.clip(r.conf[PALM] / 0.6, 0.25, 1.25), r.rel[PALM], 1.0, r.slot)

    def _joints(self, side, fit, good, tpl):
        """3D joints of this observation (for the next crop and the debug view): triangulated where
        both cameras agree, else the fitted palm depth plus Mercury's relative depths."""
        r0 = good[0]
        v0 = self.views[r0.slot]
        U0 = v0.dirs_to_head(r0.rays)
        X9 = fit.c + fit.R @ tpl.T[MID]
        rho = np.linalg.norm(X9 - v0.t) + (r0.rel - r0.rel[9]) * tpl.size
        J = v0.t + U0 * rho[:, None]
        if len(good) == 2:
            r1 = good[1]
            v1 = self.views[r1.slot]
            U1 = v1.dirs_to_head(r1.rays)
            P, gap, s1, s2 = triangulate(np.broadcast_to(v0.t, U0.shape), U0, np.broadcast_to(v1.t, U1.shape), U1)
            okj = (gap < 0.02) & (s1 > 0.05) & (s2 > 0.05)
            J[okj] = P[okj]
        return J

    def _observe(self, h, job, results):
        p = self.prm
        t = job.t
        for r in results:
            if r is not None:
                self.stats["crop_ms"].append(r.ms)
                h.last[r.slot] = r
        good = [r for r in results if self._good(r)]
        age_ms = (time.monotonic() - t) * 1e3
        if not good:
            self._miss(h, t, "notahand", results)
            return
        views = [self._obs(r) for r in good]
        c_p, R_p = h.predict(t)
        fresh = t - h.t_obs < 0.15
        inits = [(c_p, R_p, "pred")]
        g = init_from_rel(h.tpl.T, h.side, views[0], h.tpl.size)
        if g is not None:
            inits.append((g[0], g[1], "rel"))
        if h.misses > 0 or h.n_ok < 5 or not fresh:
            v = self.views[good[0].slot]
            inits += [(c, R, "ippe") for c, R in init_ippe(h.tpl.T, views[0], v.R, v.t)]
        fit = fit_palm(h.tpl.T, h.side, views, inits, prior=(c_p, R_p) if fresh else None)
        if fit is None or fit.res_deg > p["gate_deg"] or not _plausible(fit.c):
            self._miss(h, t, "fit", results, fit)
            return
        if self._sees_other(h, t, good[0], fit, c_p):
            self._miss(h, t, "other", results, fit)            # the crop saw the other hand
            h.occluded = True
            return
        Rz_p = fit.cov[:3, :3] + np.eye(3) * p["pos_floor"] ** 2
        Rz_r = fit.cov[3:, 3:] + np.eye(3) * math.radians(p["rot_floor_deg"]) ** 2
        hm_obs = min(r.hm for r in good)
        suspicious = h.hm_base is not None and hm_obs < h.hm_base * p["hm_drop"]
        if suspicious and (not fresh or h.pkf.nis(fit.c, Rz_p, t, h.pkf.sa) > p["attack"] or
                           h.rkf.nis(fit.R, Rz_r, t, h.rkf.sa) > p["attack"]):
            self._miss(h, t, "lowhm", results, fit)          # a stale track re-finds a hand only
            return                                            # through acquisition (checks both mirrors)
        reinit = False
        if h.n_ok >= 3 and fresh:
            nis_p = h.pkf.nis(fit.c, Rz_p, t, p["sa_hi"])
            nis_r = h.rkf.nis(fit.R, Rz_r, t, p["ra_hi"])
            if nis_p > p["nis_gate"] or nis_r > p["nis_gate"]:
                agree = [o for o in h.outliers if np.linalg.norm(o[1] - fit.c) < 0.05 and
                         abs(log_so3(o[2].T @ fit.R)).max() < 0.6]
                h.outliers.append((t, fit.c, fit.R))
                if len(agree) + 1 >= p["reinit_after"]:
                    self.stats["reinit"] += 1
                    self.events.append("X,%d,%s,%.1f,%.1f" % (int(t * 1e9), h.side, nis_p, nis_r))
                    self._reinit(h, t, fit)
                    reinit = True
                else:
                    self.stats["outlier"] += 1
                    self._miss(h, t, "outlier", results, fit)
                    return
        J = self._joints(h.side, fit, good, h.tpl)
        if not reinit:                                    # a restarted filter already sits on this fit
            # a big position surprise is real motion (single-view fits are not wrong sideways);
            # orientation opens on its own only when persistent, or together with the position
            h.pkf.update(fit.c, Rz_p, t, p["sa_lo"], p["sa_hi"], p["attack"], p["release"], big=p["attack_big"])
            h.rkf.update(fit.R, Rz_r, t, p["ra_lo"], p["ra_hi"], p["attack"], p["release"], force=h.pkf.attacked)
        h.joints_palm = (J - fit.c) @ fit.R
        curls = np.clip(-np.mean([r.curls for r in good], 0) / CURL_SCALE, 0.0, 1.0)
        h.curls = h.curl_f(curls, t)
        h.t_obs, h.misses = t, 0
        h.n_ok += 1
        h.outliers.clear()
        h.occluded = False
        if not suspicious:
            h.hm_base = hm_obs if h.hm_base is None else 0.9 * h.hm_base + 0.1 * hm_obs
        if len(good) == 2:
            h.n_stereo += 1
            h.t_stereo = t
            sc = fit_scale(h.tpl.T, views, fit.c, fit.R)
            if sc is not None:
                h.tpl.learn_scale(h.tpl.scale * sc[0], sc[1])
        self.t_seen[h.side] = t
        st = self.stats
        st["acc"] += 1
        st["acc_stereo"] += len(good) == 2
        st["upd"][h.side] += 1
        st["obs_age_ms"].append(age_ms)
        st["res_deg"].append(fit.res_deg)
        self.events.append("O,%d,%s,%s,%.3f,%.2f,%.2f,%.1f,%s" % (
            int(t * 1e9), h.side, "".join(str(r.slot) for r in good), min(r.is_hand for r in good),
            min(r.hm for r in good), fit.res_deg, age_ms, fit.init))

    def _miss(self, h, t, why, results, fit=None):
        h.misses += 1
        self.stats["miss"] += 1
        ih = max([r.is_hand for r in results if r is not None], default=0.0)
        self.events.append("N,%d,%s,%s,%.3f,%.2f" % (int(t * 1e9), h.side, why, ih,
                                                     fit.res_deg if fit is not None else -1))

    def _reinit(self, h, t, fit):
        """The hand is somewhere else than the filter thought: restart the filter there and blend the
        output over blend_s instead of jumping."""
        c0, R0 = h.predict(t)
        h.pkf = PosKF(fit.c, t, sp=0.03, sv=1.0)
        h.rkf = RotKF(fit.R, t, sr=0.2, sw=6.0)
        h.blend = (t, c0, R0)
        h.outliers.clear()

    # ------------------------------------------------------------ acquisition
    def _missing(self, t):
        """Enabled sides with no confident track (absent, or failing for a while)."""
        out = []
        for sd in self.enabled:
            h = self.hands.get(sd)
            if h is None or h.misses >= 2 or t - h.t_obs > 0.1:
                out.append(sd)
        return out

    def _search(self, t, imgs):
        p = self.prm
        missing = self._missing(t)
        if not missing:
            self.cands.clear()
            return
        idle = all(t - self.t_seen[sd] > p["idle_after"] for sd in missing)
        sc = max(self.scale, 0.2)
        if self.det is not None and self.det_job is None and \
                t - self.t_det >= 1.0 / ((p["det_idle_hz"] if idle else p["det_hz"]) * sc) - 0.004:
            slot = self.det_slot
            self.det_slot = 1 - slot
            self.det_job = Job("det", None, t, [slot], [self.pool.submit(self.det.run, self.views[slot], imgs[slot])])
            self.t_det = t
        if self.mp is not None and not self.mp.busy() and \
                t - self.t_mp >= 1.0 / ((p["mp_idle_hz"] if idle else p["mp_hz"]) * sc) - 0.004:
            slot = self.mp_slot
            self.mp_slot = 1 - slot
            self.mp.submit(slot, self.views[slot], imgs[slot], t)
            self.t_mp = t
        if self.ver_job is None and self.cands and self.tokens >= 2:
            self._verify_next(t, missing)

    def _same_as_tracked(self, slot, ray_cam, t):
        v = self.views[slot]
        for h in self.hands.values():
            if h.misses >= 2:
                continue
            c, _ = h.predict(t)
            d = v.to_cam(c)[0]
            d /= np.linalg.norm(d)
            if math.degrees(math.acos(min(1.0, max(-1.0, float(d @ ray_cam))))) < self.prm["same_hand_deg"]:
                return True
        return False

    def _verify_next(self, t, missing):
        frames = {ft: im for ft, im in self.frames}
        self.cands.sort(key=lambda c: -c["score"])
        while self.cands:
            c = self.cands.pop(0)
            imgs = frames.get(c["t"])
            if imgs is None or t - c["t"] > 0.2:
                continue
            v = self.views[c["slot"]]
            if c["kind"] == "det":
                ray = v.rays_cam(np.asarray(c["box"].centre)[None])[0]
            else:
                ray = c["rays"][[0, 5, 9, 13, 17]].mean(0)
                ray /= np.linalg.norm(ray)
            if self._same_as_tracked(c["slot"], ray, t):
                continue
            # one-hand mode: a finder's left/right label is only a hint; any side may be checked
            futs = []
            for flip in (False, True):
                if c["kind"] == "det":
                    ins = c["box"].instr(v, flip)
                else:
                    ins = M.instr_from_points(c["rays"], flip=flip, expand=M.EXPAND)
                futs.append(self.pool.submit(self.kp.run, v, imgs[c["slot"]], ins, False))
            self.tokens -= 2
            self.ver_job = Job("verify", None, c["t"], [c["slot"], c["slot"]], futs, extra=c)
            self.stats["verify"] += 1
            return

    def _verified(self, job, results):
        p = self.prm
        c = job.extra
        t = job.t
        rl, rr = results                                  # unmirrored = left-hand view, mirrored = right
        if rl is None and rr is None:
            self.stats["verify_fail"] += 1
            return
        hl = rl.hm if rl is not None else 0.0
        hr = rr.hm if rr is not None else 0.0
        if abs(hl - hr) >= p["chir_margin"]:
            side = "left" if hl > hr else "right"
            sure = True
        else:
            side = c["side"] if c.get("side") in SIDES else ("left" if hl >= hr else "right")
            sure = False
        r = rl if side == "left" else rr
        if not self._good(r):
            self.stats["verify_fail"] += 1
            self.events.append("V,%d,%s,%s,fail,%.3f,%.2f,%.2f" % (int(t * 1e9), c["kind"], side,
                                                                    r.is_hand if r else 0, hl, hr))
            return
        if side not in self.enabled:
            other = next(iter(self.enabled), None) if len(self.enabled) == 1 else None
            if other is None or sure:
                return
            side = other                                  # unsure label in one-hand mode: the enabled hand
            r = rl if side == "left" else rr
            if not self._good(r):
                return
        if c.get("side") in SIDES and c["side"] != side:
            self.stats["chir_flip"] += 1
        tpl = self.tpl[side]
        obs = self._obs(r)
        v = self.views[r.slot]
        inits = []
        g = init_from_rel(tpl.T, side, obs, tpl.size)
        if g is not None:
            inits.append((g[0], g[1], "rel"))
        inits += [(cc, RR, "ippe") for cc, RR in init_ippe(tpl.T, obs, v.R, v.t)]
        fit = fit_palm(tpl.T, side, [obs], inits) if inits else None
        if fit is None or fit.res_deg > p["gate_deg"] * 1.25 or not _plausible(fit.c):
            self.stats["verify_fail"] += 1
            self.events.append("V,%d,%s,%s,fit,%.3f,%.2f,%.2f" % (int(t * 1e9), c["kind"], side, r.is_hand, hl, hr))
            return
        old = self.hands.get(side)
        if old is not None and old.misses < 2 and t - old.t_obs < 0.1:
            return                                        # tracked meanwhile
        other = self.hands.get("right" if side == "left" else "left")
        if other is not None and np.linalg.norm(other.predict(t)[0] - fit.c) < p["collapse_m"] * 1.6:
            return                                        # that is the other hand
        J = self._joints(side, fit, [r], tpl)
        curls = np.clip(-r.curls / CURL_SCALE, 0.0, 1.0)
        h = Hand(side, tpl, fit, J, curls, t, p, stereo=False)
        h.last[r.slot] = r
        h.hm_base = r.hm
        h.view = r.slot
        lo = self.last_out.get(side)
        if old is not None:
            c0, R0 = old.predict(t)
            h.blend = (t, c0, R0)
            h.n_ok = max(h.n_ok, p["show_after"])         # it was being shown: keep showing it
        elif lo is not None and t - lo[0] < 0.6:
            h.blend = (t, lo[1], lo[2])
        self.hands[side] = h
        self.t_seen[side] = t
        self.stats["acquired"] += 1
        self.events.append("A,%d,%s,%s,%s,%.3f,%.2f,%.2f,%.2f" % (int(t * 1e9), side, c["kind"], c.get("side"),
                                                                   r.is_hand, hl, hr, fit.res_deg))

    # ------------------------------------------------------------ output
    def _outputs(self, t):
        p = self.prm
        close = False
        if len(self.hands) == 2:
            a, b = self.hands["left"], self.hands["right"]
            close = np.linalg.norm(a.predict(t)[0] - b.predict(t)[0]) < p["collapse_m"]
            if close and self.t_close is not None and t - self.t_close >= p["collapse_s"]:
                drop = "left" if a.t_new > b.t_new else "right"
                del self.hands[drop]
                self.stats["collapse"] += 1
                self.events.append("C,%d,%s" % (int(t * 1e9), drop))
                close = False
        if not close:
            self.t_close = None
        elif self.t_close is None:
            self.t_close = t
        out = {}
        for side in SIDES:
            h = self.hands.get(side)
            hold = p["hold_occluded_s"] if (h is not None and h.occluded) else p["hold_s"]
            if side not in self.enabled or h is None or h.n_ok < p["show_after"] or t - h.t_obs > hold:
                out[side] = None
                continue
            c, R = h.predict(t)
            if h.blend is not None:
                t0, c0, R0 = h.blend
                a = _smoothstep((t - t0) / p["blend_s"])
                if a >= 1.0:
                    h.blend = None
                else:
                    c = c0 + (c - c0) * a
                    R = exp_so3(log_so3(R @ R0.T) * a) @ R0
            age = t - h.t_obs
            conf = max(0.1, 1.0 - max(0.0, age - 0.05) / hold)
            out[side] = (c, quat_from_R(R), conf, None, h.curls.copy())
            self.last_out[side] = (t, c, R)
        return out
