#!/usr/bin/env python3
"""Closed-loop simulation of the v4 tracker (handtrack/tracker.py) against synthetic hands with known
ground truth: no headset, no cameras, no models needed.

What is real: the whole tracker (scheduling, crop placement, Monado's heatmap decode, palm fit,
Kalman filters, acquisition, outputs), threads and real-time pacing at 60 Hz.
What is simulated: a Frame-like camera rig; a kinematic hand; the keypoint model is an oracle that
renders heatmaps from the true joints with noise (and fails like the real one: hand outside the crop,
motion blur, wrong mirror image), the Mercury detector / MediaPipe finders return the true hands late.

  python tools/sim_tracker.py                 all scenarios, summary table
  python tools/sim_tracker.py fast slap -v    some scenarios, with event counts
  python tools/sim_tracker.py --kp-deg 1.2 --rel 0.2 --infer-ms 20     harsher model / slower CPU

Metrics (frames where the hand is in view): pos/rot error vs truth, still-hand jitter (frame-to-frame
output change), lag (time shift that best aligns output with truth while moving), valid %, time to
first output, largest frame-to-frame jump.
"""
import argparse
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from handtrack import mercury as M  # noqa: E402
from handtrack.calib import KBCamera  # noqa: E402
from handtrack.handfit import PALM, exp_so3, palm_frame5, rot_angle  # noqa: E402
from handtrack.tracker import CURL_SCALE, Tracker  # noqa: E402


# ---------------------------------------------------------------- Frame-like camera rig
def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def sim_views():
    """Two 2464x2464 fisheye cameras 64 mm apart, ~11 deg outward, ~12 deg down, sensors mounted
    rotated +-90 deg (slot 0 = right camera, upright = rotate clockwise; slot 1 = left, counter-clockwise)."""
    views = {}
    for slot, sx, up in ((0, +1, np.array([0, -1.0, 0])), (1, -1, np.array([0, 1.0, 0]))):
        cam = KBCamera({"sourceCamera": "sim%d" % slot, "width": 2464, "height": 2464,
                        "intrinsics": {"fx": 1475.0, "fy": 1475.0, "cx": 1232.0, "cy": 1232.0,
                                       "k1": 0.02, "k2": -0.005, "k3": 0.0, "k4": 0.0},
                        "extrinsics": {"plus_x": [1, 0, 0], "plus_z": [0, 0, 1], "position": [0, 0, 0]}})
        z = _rx(math.radians(-12)) @ _ry(math.radians(-11 * sx)) @ np.array([0, 0, -1.0])
        x = up - z * (up @ z)
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z], 1)
        views[slot] = M.View(slot, cam, R, np.array([0.032 * sx, -0.005, -0.057]))
    return views


# ---------------------------------------------------------------- kinematic hand (left-hand local frame)
# x towards the thumb, y along the fingers, z = back of the hand; right hand = mirrored x
_PALM_L = np.array([[0, 0, 0], [0.027, 0.083, -0.004], [0.006, 0.088, 0.0], [-0.014, 0.083, -0.004],
                    [-0.032, 0.074, -0.010]])
_PHAL = [(0.045, 0.026, 0.022), (0.050, 0.030, 0.024), (0.046, 0.028, 0.023), (0.036, 0.021, 0.020)]


def hand_points(side, curls, scale=1.0):
    """21 joints (Mercury/MediaPipe order) in the hand's local frame for curls (5, thumb first)."""
    P = np.zeros((21, 3))
    P[0] = _PALM_L[0]
    for f in range(4):
        base = _PALM_L[f + 1]
        P[5 + 4 * f] = base
        th = 0.0
        p = base.copy()
        for k, L in enumerate(_PHAL[f]):
            th += math.radians((90, 80, 80)[k]) * curls[f + 1]
            p = p + L * np.array([0, math.cos(th), -math.sin(th)])
            P[5 + 4 * f + k + 1] = p
    cmc = np.array([0.020, 0.022, -0.012])
    P[1] = cmc
    d0 = np.array([0.65, 0.72, -0.25])
    d0 /= np.linalg.norm(d0)
    ax = np.array([0.35, -0.45, 0.82])
    ax /= np.linalg.norm(ax)
    p, th = cmc, 0.0
    for k, L in enumerate((0.040, 0.032, 0.028)):
        th += math.radians((40, 50, 60)[k]) * curls[0]
        p = p + L * (exp_so3(-ax * th) @ d0)
        P[2 + k] = p
    P *= scale
    if side == "right":
        P[:, 0] *= -1
    return P


R_BASE = _rx(math.radians(-30))          # palm facing forward-down, fingers up-forward, back of hand to the user
HOME = {"left": np.array([-0.12, -0.12, -0.40]), "right": np.array([0.12, -0.12, -0.40])}


class Pose:
    def __init__(self, c, R, curls, visible=True):
        self.c, self.R, self.curls, self.visible = np.asarray(c, float), R, np.asarray(curls, float), visible


def S(f):
    return math.sin(2 * math.pi * f)


SCENARIOS = {
    "still": (6.0, {"left": lambda t: Pose(HOME["left"], R_BASE, [0.1] * 5)}),
    "reach": (8.0, {"left": lambda t: Pose(HOME["left"] + [0.08 * S(0.4 * t), 0.05 * S(0.3 * t), 0.12 * S(0.25 * t)],
                                           R_BASE @ _ry(0.4 * S(0.3 * t)) @ _rx(0.3 * S(0.2 * t)), [0.2] * 5)}),
    "fast": (6.0, {"left": lambda t: Pose(HOME["left"] + [0.20 * S(1.2 * t), 0.0, 0.0],
                                          R_BASE @ _ry(1.2 * S(1.2 * t)), [0.1] * 5)}),
    "slap": (6.0, {"right": lambda t: Pose(HOME["right"] + [0.0, 0.03 * S(2.5 * t), 0.12 * S(2.5 * t)],
                                           R_BASE @ _rx(0.4 * S(2.5 * t)), [0.0] * 5)}),
    "stop": (6.0, {"left": lambda t: Pose(HOME["left"] + [0.35 * _stepper(t), 0.0, 0.0], R_BASE, [0.1] * 5)}),
    "outback": (7.0, {"left": lambda t: _outback(t)}),
    "two": (8.0, {"left": lambda t: Pose([-0.06 + 0.11 * S(0.4 * t), -0.12 + 0.02 * S(0.8 * t), -0.43],
                                         R_BASE @ _ry(0.3 * S(0.4 * t)), [0.3 + 0.3 * S(0.5 * t)] * 5),
                  "right": lambda t: Pose([0.06 - 0.11 * S(0.4 * t), -0.11, -0.33],
                                          R_BASE @ _ry(-0.3 * S(0.4 * t)), [0.2] * 5)}),
    "clap": (6.0, {"left": lambda t: Pose([-0.018 - 0.10 * _clap(t), -0.10, -0.38], _clapR("left", t), [0.0] * 5),
                   "right": lambda t: Pose([0.018 + 0.10 * _clap(t), -0.10, -0.38], _clapR("right", t), [0.0] * 5)}),
    "fingers": (6.0, {"left": lambda t: Pose(HOME["left"], R_BASE,
                                             [0.1, 1.0 if (t % 1.0) < 0.5 else 0.0, 0.1, 0.1, 0.1])}),
}


def _clap(t):
    """0 = palms together, 1 = apart; a clap every second."""
    return 0.5 + 0.5 * math.cos(2 * math.pi * t)


def _clapR(side, t):
    """Palms facing each other (fingers up/forward): left palm faces +x, right palm faces -x."""
    base = _rx(math.radians(-60))                             # fingers up and forward
    return base @ _ry(math.radians(-90 if side == "left" else 90))


def _stepper(t):
    """0 -> 1 in 0.3 s (peak ~2 m/s for 0.35 m), hold, back, hold: sudden stops."""
    ph = t % 2.0
    ramp = lambda x: 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, x)))
    return ramp(ph / 0.3) if ph < 1.0 else 1.0 - ramp((ph - 1.0) / 0.3)


def _outback(t):
    if t < 2.0 or t > 4.6:
        y = 0.0
    elif t < 2.3:
        y = -0.6 * (t - 2.0) / 0.3
    elif t < 4.3:
        y = -0.6
    else:
        y = -0.6 * (1 - (t - 4.3) / 0.3)
    c = HOME["left"] + [0, y, 0]
    return Pose(c, R_BASE, [0.1] * 5, visible=y > -0.35)


class World:
    """Ground truth over time: joints of each hand in the head frame."""

    def __init__(self, scen, user_scale=1.08):
        self.dur, self.fns = SCENARIOS[scen]
        self.scale = user_scale

    def at(self, t):
        out = {}
        for side, fn in self.fns.items():
            p = fn(t)
            Pl = hand_points(side, p.curls, self.scale)
            cl, _ = palm_frame5("left", hand_points("left", p.curls, self.scale)[PALM])
            cl = cl * (np.array([-1, 1, 1]) if side == "right" else 1)
            X = p.c + (Pl - cl) @ p.R.T
            out[side] = (X, p.visible, p.curls)
        return out

    def speed(self, side, t, dt=0.01):
        a, b = self.at(t - dt).get(side), self.at(t + dt).get(side)
        if a is None or b is None:
            return 0.0
        return float(np.linalg.norm(b[0][PALM].mean(0) - a[0][PALM].mean(0)) / (2 * dt))


# ---------------------------------------------------------------- oracle models
def visible_hands(truth, view, near=0.02):
    """Hands this camera can see: in front of it and not hidden behind the other hand (palm discs
    overlapping in the image, the other one closer).  -> {side: (X, Xc, curls)}"""
    info = {}
    for side, (X, vis, curls) in truth.items():
        if not vis:
            continue
        Xc = view.to_cam(X)
        if (Xc[:, 2] < near).any():
            continue
        pc = Xc[PALM].mean(0)
        dist = float(np.linalg.norm(pc))
        rad = 0.045 / dist                                   # palm disc, angular radius
        info[side] = (X, Xc, curls, pc / dist, dist, rad)
    out = {}
    for side, (X, Xc, curls, d, dist, rad) in info.items():
        hidden = False
        for o, (_, _, _, d2, dist2, rad2) in info.items():
            if o != side and dist2 < dist - 0.02 and math.acos(min(1.0, float(d @ d2))) < 0.8 * (rad + rad2):
                hidden = True
        if not hidden:
            out[side] = (X, Xc, curls)
    return out


class Oracle:
    """Stands in for KeypointModel / HandDetector160 / MediaPipe.  Images are dummy arrays; the oracle
    looks the truth up by image identity."""

    def __init__(self, world, views, rng, kp_deg=0.7, rel_sig=0.10, infer_ms=11.0, det_ms=6.0, mp_ms=45.0,
                 blur_mps=1.6, blur_p=0.7):
        self.world, self.views, self.rng = world, views, rng
        self.kp_rad, self.rel_sig = math.radians(kp_deg), rel_sig
        self.infer_ms, self.det_ms, self.mp_ms = infer_ms, det_ms, mp_ms
        self.blur_mps, self.blur_p = blur_mps, blur_p
        self.img_t = {}
        self.lock = threading.Lock()
        self.mp_pending = None

    def register(self, imgs, t):
        for im in imgs.values():
            self.img_t[id(im)] = t
        if len(self.img_t) > 400:
            for k in list(self.img_t)[:200]:
                del self.img_t[k]

    def _truth(self, img):
        t = self.img_t.get(id(img))
        return t, (self.world.at(t) if t is not None else {})

    # keypoint model ---------------------------------------------------
    def run(self, view, img, ins, use_hint=True, keep_crop=False):
        t0 = time.perf_counter()
        t, truth = self._truth(img)
        with self.lock:
            noise = self.rng.normal(size=(21, 3)) * self.kp_rad
            nrel = self.rng.normal(size=21) * self.rel_sig
            ncurl = self.rng.normal(size=5) * 0.15
            u_blur = self.rng.random()
        best = None
        for side, (X, Xc, curls) in visible_hands(truth, view).items():
            dm = Xc / np.linalg.norm(Xc, axis=1)[:, None] * M.FLIP_YZ
            d = M.q_rot(M.q_conj(ins.q), dm)
            sg = d[:, :2] / np.maximum(1 - d[:, 2:3], 1e-9)
            pn = sg / ins.r
            dist = float(np.linalg.norm(pn[PALM].mean(0)))
            # Mercury is trained on left hands (right-hand crops are mirrored): with two hands in a
            # crop it latches on the one that looks like a left hand in this mirror
            score = dist + (0.0 if (side == "right") == ins.flip else 0.5)
            if dist < 0.8 and (best is None or score < best[0]):
                best = (score, side, X, Xc, curls)
        hm = np.zeros((21, M.GRID, M.GRID))
        hd = np.zeros((21, M.GRID))
        cu = np.zeros(10)
        logit = -6.0
        if best is not None:
            _, side, X, Xc, curls = best
            right_mirror = (side == "right") == ins.flip
            spd = self.world.speed(side, t)
            blurred = spd > self.blur_mps and u_blur < self.blur_p
            logit = -3.0 if blurred else 6.0
            Xn = Xc / np.linalg.norm(Xc, axis=1)[:, None]
            Xn = Xn + noise * (1.0 if right_mirror else 6.0)
            Xn /= np.linalg.norm(Xn, axis=1)[:, None]
            d = M.q_rot(M.q_conj(ins.q), Xn * M.FLIP_YZ)
            pn = d[:, :2] / np.maximum(1 - d[:, 2:3], 1e-9) / ins.r
            lx = (1 - pn[:, 0]) * 11 if ins.flip else (pn[:, 0] + 1) * 11
            ly = (1 - pn[:, 1]) * 11
            peak = 0.68 if right_mirror else 0.45
            g = np.arange(M.GRID)
            for j in range(21):
                cx, cy = np.clip(lx[j], 0.3, 21.7) - 0.5, np.clip(ly[j], 0.3, 21.7) - 0.5
                hm[j] = peak * np.exp(-((g[None, :] - cx) ** 2 + (g[:, None] - cy) ** 2) / (2 * 0.8 ** 2))
            rho = np.linalg.norm(Xc, axis=1)
            rel = (rho - rho[9]) / np.linalg.norm(X[0] - X[9]) + nrel
            com = (rel / 3.0 + 0.5) * M.GRID - 0.5
            for j in range(21):
                hd[j] = np.exp(-(g - com[j]) ** 2 / (2 * 0.7 ** 2))
            cu[:5] = -np.asarray(curls) * CURL_SCALE + ncurl
            cu[5:] = 0.1
        out = M.decode(view, ins, hm, hd, np.array([logit]), cu)
        rest = self.infer_ms / 1e3 - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)
        out.ms = (time.perf_counter() - t0) * 1e3
        return out

    # Mercury detector ---------------------------------------------------
    def det_run(self, view, img, min_conf=0.3):
        t0 = time.perf_counter()
        t, truth = self._truth(img)
        out = []
        for side, (X, Xc, _) in visible_hands(truth, view, 0.05).items():
            px = view.project_cam(Xc)
            if not view.inside(px[PALM].mean(0)[None])[0]:
                continue
            ctr = px[PALM].mean(0)
            size = float(np.ptp(px, axis=0).max()) * 1.25
            with self.lock:
                wrong = self.rng.random() < 0.1
            lab = side if not wrong else ("left" if side == "right" else "right")
            out.append(M.DetBox(view.slot, lab, 0.9, ctr, size))
        rest = self.det_ms / 1e3 - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)
        return out


class OracleDetector:
    def __init__(self, oracle):
        self.o = oracle

    def run(self, view, img, min_conf=0.3):
        return self.o.det_run(view, img, min_conf)


class OracleMP:
    """MediaPipe stand-in: results arrive mp_ms later (separate process in the real tracker)."""

    def __init__(self, oracle):
        self.o = oracle
        self.pending = None

    def busy(self):
        return self.pending is not None

    def submit(self, slot, view, img, t):
        self.pending = (time.monotonic() + self.o.mp_ms / 1e3, slot, view, img, t)

    def poll(self):
        if self.pending is None or time.monotonic() < self.pending[0]:
            return 0, []
        _, slot, view, img, t = self.pending
        self.pending = None
        _, truth = self.o._truth(img)
        out = []
        for side, (X, Xc, _) in visible_hands(truth, view, 0.05).items():
            if not view.inside(view.project_cam(Xc)[PALM].mean(0)[None])[0]:
                continue
            rays = Xc / np.linalg.norm(Xc, axis=1)[:, None]
            with self.o.lock:
                rays = rays + self.o.rng.normal(size=rays.shape) * math.radians(1.5)
            out.append({"kind": "mp", "t": t, "slot": slot, "side": side, "score": 0.9,
                        "rays": rays / np.linalg.norm(rays, axis=1)[:, None]})
        return 1, out


# ---------------------------------------------------------------- run one scenario
def run(scen, args, prm=None, seed=0):
    rng = np.random.default_rng(seed)
    views = sim_views()
    world = World(scen, args.user_scale)
    oracle = Oracle(world, views, rng, args.kp_deg, args.rel, args.infer_ms, args.det_ms, args.mp_ms,
                    args.blur_mps)
    pool = ThreadPoolExecutor(args.workers)
    tr = Tracker(views, oracle, pool, det=OracleDetector(oracle), mp=OracleMP(oracle), prm=prm)
    tr.set_enabled(set(world.fns))
    dur = world.dur
    t0 = time.monotonic() + 0.05
    rows = []
    k = 0
    while True:
        t = t0 + k / 60.0
        if t - t0 > dur:
            break
        while time.monotonic() < t + 0.006:            # ptgrab publishes ~6 ms after capture
            tr.collect()
            time.sleep(0.001)
        imgs = {0: np.zeros((2, 2), np.uint8), 1: np.zeros((2, 2), np.uint8)}
        oracle.register(imgs, t)
        out = tr.on_frame(t, imgs)
        truth = world.at(t)
        for side, (X, vis, curls) in truth.items():
            cg, Rg = palm_frame5(side, X[PALM])
            o = out.get(side)
            rows.append((t - t0, side, vis, cg, Rg, None if o is None else (o[0], o[1], o[4]), curls))
        k += 1
    pool.shutdown(wait=True)
    return rows, tr


def metrics(rows, side, world_scen):
    R = [r for r in rows if r[1] == side]
    vis = np.array([r[2] for r in R])
    valid = np.array([r[5] is not None for r in R])
    from handtrack.handfit import R_from_quat
    pe, re, out_c, out_R, gt_c, gt_R = [], [], [], [], [], []
    for r in R:
        gt_c.append(r[3])
        gt_R.append(r[4])
        if r[5] is not None:
            oc, oq = r[5][0], R_from_quat(r[5][1])
            out_c.append(oc)
            out_R.append(oq)
            if r[2]:
                pe.append(np.linalg.norm(oc - r[3]) * 100)
                re.append(math.degrees(rot_angle(oq, r[4])))
        else:
            out_c.append(None)
            out_R.append(None)
    first = next((r[0] for r in R if r[5] is not None), float("nan"))
    # still-hand jitter / largest jump: consecutive valid frames
    jit_p, jit_r, jump = [], [], 0.0
    gt_speed = []
    for i in range(1, len(R)):
        if out_c[i] is None or out_c[i - 1] is None:
            continue
        dp = np.linalg.norm(out_c[i] - out_c[i - 1]) * 1000
        dr = math.degrees(rot_angle(out_R[i], out_R[i - 1]))
        gp = np.linalg.norm(gt_c[i] - gt_c[i - 1]) * 1000
        gr = math.degrees(rot_angle(gt_R[i], gt_R[i - 1]))
        gt_speed.append(gp)
        if gp < 1e-3 and gr < 1e-3:
            jit_p.append(dp)
            jit_r.append(dr)
        jump = max(jump, dp - gp)
    # lag: shift (ms) of the output that best matches the truth (position, moving frames)
    lag = float("nan")
    idx = [i for i in range(len(R)) if out_c[i] is not None and R[i][2]]
    moving = np.array([np.linalg.norm(gt_c[min(i + 1, len(R) - 1)] - gt_c[max(i - 1, 0)]) > 1e-4 for i in range(len(R))])
    if moving.sum() > 30:
        best = None
        for sh in range(0, 13):                                   # 0..200 ms in frames
            e = [np.linalg.norm(out_c[i] - gt_c[i - sh]) for i in idx if i - sh >= 0 and moving[i]]
            if e:
                m = float(np.mean(e))
                if best is None or m < best[0]:
                    best = (m, sh)
        lag = best[1] * 1000 / 60 if best else float("nan")
    curl_err = [np.abs(np.asarray(r[5][2]) - np.asarray(r[6])).mean() for r in R if r[5] is not None and r[2]]
    # identity swaps: output far from its own hand, on the other hand, and moving (a hand frozen while
    # hidden behind the other one can be passed by it: that is not a swap)
    other = [r for r in rows if r[1] != side]
    swap = 0
    if other:
        oc = {round(r[0], 4): r[3] for r in other}
        prev = None
        for r in R:
            if r[5] is not None and round(r[0], 4) in oc:
                moving = prev is not None and np.linalg.norm(r[5][0] - prev) > 0.001
                if moving and np.linalg.norm(r[5][0] - r[3]) > 0.08 and np.linalg.norm(r[5][0] - oc[round(r[0], 4)]) < 0.05:
                    swap += 1
            prev = None if r[5] is None else r[5][0]
    pct = lambda a, q: float(np.percentile(a, q)) if len(a) else float("nan")
    return {"valid%": 100.0 * (valid & vis).sum() / max(1, vis.sum()), "first_s": first,
            "pos50": pct(pe, 50), "pos95": pct(pe, 95), "rot50": pct(re, 50), "rot95": pct(re, 95),
            "jit_mm95": pct(jit_p, 95), "jit_deg95": pct(jit_r, 95), "lag_ms": lag, "jump_mm": jump,
            "curl_err": float(np.mean(curl_err)) if curl_err else float("nan"), "swap": swap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scen", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--kp-deg", type=float, default=0.7, help="keypoint direction noise (deg)")
    ap.add_argument("--rel", type=float, default=0.10, help="relative depth noise (hand-size units)")
    ap.add_argument("--infer-ms", type=float, default=11.0, help="simulated keypoint inference time")
    ap.add_argument("--det-ms", type=float, default=6.0)
    ap.add_argument("--mp-ms", type=float, default=45.0)
    ap.add_argument("--blur-mps", type=float, default=1.6, help="palm speed above which crops often fail (blur)")
    ap.add_argument("--user-scale", type=float, default=1.08, help="true palm size / default template size")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--set", action="append", default=[], help="tracker parameter override key=value")
    ap.add_argument("-v", action="store_true", help="print tracker event counts")
    a = ap.parse_args()
    prm = {}
    for kv in a.set:
        k, v = kv.split("=")
        prm[k] = float(v)
    print("%-8s %-5s %6s %6s %6s %6s %6s %6s %7s %8s %6s %7s %6s %5s" % (
        "scenario", "hand", "valid%", "first", "pos50", "pos95", "rot50", "rot95", "jit_mm", "jit_deg", "lag",
        "jump_mm", "curl", "swap"))
    for scen in a.scen:
        rows, tr = run(scen, a, prm)
        for side in SCENARIOS[scen][1]:
            m = metrics(rows, side, scen)
            print("%-8s %-5s %6.1f %6.2f %6.2f %6.2f %6.1f %6.1f %7.2f %8.2f %6.0f %7.1f %6.2f %5d" % (
                scen, side, m["valid%"], m["first_s"], m["pos50"], m["pos95"], m["rot50"], m["rot95"],
                m["jit_mm95"], m["jit_deg95"], m["lag_ms"], m["jump_mm"], m["curl_err"], m["swap"]), flush=True)
        if a.v:
            st = tr.stats
            print("         stats: " + ", ".join("%s=%s" % (k, v) for k, v in st.items()
                                                 if not isinstance(v, (list, dict))) +
                  ", upd=%s, crop_ms p50 %.1f, obs age p50 %.1f ms, palm tpl n=%s size=%s" % (
                      st["upd"], np.median(st["crop_ms"]) if st["crop_ms"] else 0,
                      np.median(st["obs_age_ms"]) if st["obs_age_ms"] else 0,
                      {sd: tr.tpl[sd].n for sd in tr.tpl}, {sd: round(tr.tpl[sd].size, 4) for sd in tr.tpl}))
            if tr.errors:
                print("         ERRORS:", tr.errors[:3])
    print("units: valid% of in-view frames; first = s to first output; pos cm; rot deg; jitter = p95 frame-to-frame "
          "change while the true hand is still; lag ms; jump = largest frame-to-frame output change beyond the true "
          "change (mm); curl = mean |error| (0..1); swap = frames on the other hand")


if __name__ == "__main__":
    main()
