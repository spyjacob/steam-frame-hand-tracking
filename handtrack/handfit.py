"""Hand pose math for the single-model tracker (handtrack/tracker.py).

- rotations (SO(3) exp/log, quaternions)
- the palm frame sent to the PC (same definition the PC driver was built against)
- PalmTemplate: a canonical palm (wrist + four knuckles) scaled to the user; the scale is learned
  from synchronized stereo views (fit_scale), and a single camera view plus this known size gives depth
- fit_palm: rigid fit of the template to keypoint rays from one or two views
  (Levenberg-Marquardt, analytic Jacobian, Huber-weighted joints)
- PosKF / RotKF: constant-velocity Kalman filters whose process noise follows the innovations, so
  a still hand is steady and a moving hand is followed without lag

Frames: head frame = OpenVR HMD frame (+x right, +y up, -z forward, metres).
"""
import math
from collections import deque

import cv2
import numpy as np

PALM = np.array([0, 5, 9, 13, 17])     # wrist + index/middle/ring/pinky knuckles (MCPs): the rigid part
MID = 2                                # index of the middle knuckle inside PALM


# ---------------------------------------------------------------- rotations
def hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp_so3(w):
    th = math.sqrt(float(w @ w))
    if th < 1e-9:
        return np.eye(3) + hat(w)
    K = hat(w / th)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def log_so3(R):
    c = min(1.0, max(-1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return v / 2.0
    if th > math.pi - 1e-4:
        A = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(A)))
        k = A[:, i] / math.sqrt(max(A[i, i], 1e-12))
        if k @ v < 0:
            k = -k
        return k * th
    return th / (2.0 * math.sin(th)) * v


def rot_angle(Ra, Rb):
    """Angle (rad) between two rotations."""
    return math.acos(min(1.0, max(-1.0, (np.trace(Ra.T @ Rb) - 1.0) / 2.0)))


def quat_from_R(R):
    """(w, x, y, z)"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def R_from_quat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def kabsch(A, B):
    """R, t minimising |R A + t - B| (rows are points)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


# ---------------------------------------------------------------- palm frame (the pose sent to the PC)
def palm_frame5(side, P5):
    """Palm centre and head_from_palm rotation from the 5 palm points (PALM order).
    centre = mean(wrist, index, middle, pinky knuckles); -z along the fingers; x towards the thumb for
    the left hand and away from it for the right hand; y = z x x = back of the hand.  This is the frame
    the PC driver places its controller with (pcdriver/src/driver_framehands.cpp)."""
    wrist, idx, mid, ring, pinky = P5
    centre = (wrist + idx + mid + pinky) / 4.0
    fwd = (idx + mid + ring + pinky) / 4.0 - wrist
    z = -fwd / max(np.linalg.norm(fwd), 1e-9)
    across = idx - pinky
    x = -across if side == "right" else across
    x = x - z * (x @ z)
    x = x / max(np.linalg.norm(x), 1e-9)
    return centre, np.stack([x, np.cross(z, x), z], 1)


# canonical adult palm, metres: (towards the thumb, along the fingers); flat.  Only the starting
# guess: the template is replaced by the user's own palm as soon as both cameras see the hand.
_CANON = np.array([[0.000, 0.000],     # wrist
                   [0.027, 0.083],     # index knuckle
                   [0.006, 0.088],     # middle knuckle
                   [-0.014, 0.083],    # ring knuckle
                   [-0.032, 0.074]])   # pinky knuckle
DEFAULT_SIZE = float(np.hypot(*(_CANON[2] - _CANON[0])))   # wrist -> middle knuckle, m


class PalmTemplate:
    """The palm's 5 points in the palm frame (centre at 0, axes = palm_frame5): the canonical palm
    scaled to the user.  Only the scale is learned: one stereo frame triangulates each knuckle to
    +-3 cm in depth at a 64 mm baseline, far too noisy to learn a shape, but the whole palm's size
    comes out to a few percent per frame and a running median of those is good to ~1%."""

    SIZE_RANGE = (0.065, 0.115)          # plausible wrist -> middle knuckle, m
    KEEP = 25                            # scale samples in the running median
    STEP = 0.02                          # max relative scale change per sample: depth glides, never jumps

    def __init__(self, side, scale=1.0):
        self.side = side
        P = np.zeros((5, 3))
        P[:, 0] = _CANON[:, 0]                    # across (towards the thumb)
        P[:, 1] = _CANON[:, 1]                    # along the fingers
        c, R = palm_frame5(side, P)
        self.T1 = (P - c) @ R                     # unit scale
        self.scale = float(scale)
        self.T = self.T1 * self.scale
        self.samples = deque(maxlen=self.KEEP)
        self.n = 0                                # samples accepted so far

    @property
    def size(self):
        return DEFAULT_SIZE * self.scale

    def learn_scale(self, k, sd):
        """k: the palm's scale relative to the canonical palm measured on one stereo frame, sd: its
        standard deviation (relative)."""
        if not (self.SIZE_RANGE[0] < k * DEFAULT_SIZE < self.SIZE_RANGE[1]) or not (sd < 0.25):
            return False
        self.samples.append(float(k))
        self.n += 1
        if self.n >= 3:
            step = float(np.median(self.samples)) / self.scale - 1.0
            self.scale *= 1.0 + min(self.STEP, max(-self.STEP, step))
            self.T = self.T1 * self.scale
        return True


# ---------------------------------------------------------------- multi-view rigid palm fit
SIG_RAY = math.radians(1.0)      # keypoint direction noise (CNN on 640x640 crops: ~3-6 px)
SIG_REL = 0.15                   # Mercury relative depth noise (hand-size units)
HUBER_DEG = 2.5                  # joints worse than this are down-weighted


class ViewObs:
    """One camera's palm keypoints: origin o (3), unit rays U (5,3) and weights w (5,), all head frame;
    rel (5,) = Mercury depth relative to the middle knuckle in hand-size units (or None)."""
    __slots__ = ("o", "U", "w", "rel", "w_rel", "slot")

    def __init__(self, o, U, w, rel=None, w_rel=1.0, slot=-1):
        self.o, self.U, self.w, self.rel, self.w_rel, self.slot = o, U, w, rel, w_rel, slot


class Fit:
    __slots__ = ("c", "R", "cost", "res_deg", "cov", "err_deg", "init")

    def __init__(self, c, R, cost, res_deg, cov, err_deg, init):
        self.c, self.R, self.cost, self.res_deg, self.cov, self.err_deg, self.init = c, R, cost, res_deg, cov, err_deg, init


def _res_jac(c, R, T, views, s, hub, want_jac=True):
    RT = T @ R.T                                   # (5,3) palm points relative to the centre
    X = c + RT
    res, jac = [], []
    for vi, v in enumerate(views):
        d = X - v.o
        rho = np.linalg.norm(d, axis=1)
        n = d / rho[:, None]
        wv = (v.w * hub[vi]) / SIG_RAY
        res.append(((n - v.U) * wv[:, None]).ravel())
        if want_jac:
            P = (np.eye(3)[None] - n[:, :, None] * n[:, None, :]) / rho[:, None, None]      # d n / d X
            Jc = P * wv[:, None, None]
            H = np.stack([hat(a) for a in RT])                                                # (5,3,3)
            Jt = -np.einsum("jab,jbc->jac", P, H) * wv[:, None, None]
            jac.append(np.concatenate([Jc, Jt], 2).reshape(-1, 6))
        if v.rel is not None and v.w_rel > 0:
            wr = v.w_rel / SIG_REL
            res.append(((rho - rho[MID]) / s - (v.rel - v.rel[MID])) * wr)
            if want_jac:
                drc = n                                            # d rho / d c
                drt = np.cross(RT, n)                              # d rho / d theta (left perturbation)
                Jr = np.concatenate([drc - drc[MID], drt - drt[MID]], 1) / s * wr
                jac.append(Jr)
    r = np.concatenate(res)
    return (r, np.concatenate(jac, 0)) if want_jac else (r, None)


def _ray_err_deg(c, R, T, views):
    X = c + T @ R.T
    out = []
    for v in views:
        d = X - v.o
        n = d / np.linalg.norm(d, axis=1)[:, None]
        out.append(np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(n - v.U, axis=1) / 2, 0, 1))))
    return out


def _lm(c, R, T, views, s, iters=10):
    hub = [np.ones(5) for _ in views]
    lam = 1e-3
    r, J = _res_jac(c, R, T, views, s, hub)
    cost = float(r @ r)
    for it in range(iters):
        if it in (1, 3):                             # Huber re-weighting (IRLS) once the fit is close
            errs = _ray_err_deg(c, R, T, views)
            hub = [np.where(e > HUBER_DEG, np.sqrt(HUBER_DEG / np.maximum(e, 1e-6)), 1.0) for e in errs]
            r, J = _res_jac(c, R, T, views, s, hub)
            cost = float(r @ r)
        H = J.T @ J
        g = J.T @ r
        step = None
        while lam < 1e6:
            try:
                dx = -np.linalg.solve(H + lam * (np.diag(np.diag(H)) + 1e-9 * np.eye(6)), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            c2, R2 = c + dx[:3], exp_so3(dx[3:]) @ R
            r2, J2 = _res_jac(c2, R2, T, views, s, hub)
            cost2 = float(r2 @ r2)
            if cost2 <= cost:
                c, R, r, J, cost, step = c2, R2, r2, J2, cost2, dx
                lam = max(lam / 4, 1e-7)
                break
            lam *= 6
        if step is None or np.abs(step).max() < 1e-6:
            break
    return c, R, cost, J, hub


def init_from_rel(T, side, view, s):
    """Closed-form single-view guess: distance from the template's known lengths, shape from the
    relative depths, rigid alignment (Kabsch)."""
    U, rel = view.U, (view.rel if view.rel is not None else np.zeros(5))
    Ds = []
    for a, b in ((0, 2), (0, 1), (0, 4), (1, 4), (2, 4), (0, 3)):
        L = np.linalg.norm(T[a] - T[b])
        al, be = rel[a] * s, rel[b] * s
        w = U[a] - U[b]
        k = al * U[a] - be * U[b]
        A, B, C = w @ w, 2 * (w @ k), k @ k - L * L
        disc = B * B - 4 * A * C
        if A > 1e-12 and disc >= 0:
            D = (-B + math.sqrt(disc)) / (2 * A)
            if 0.05 < D < 2.0:
                Ds.append(D)
    if not Ds:
        return None
    D = float(np.median(Ds))
    P = view.o + U * (D + rel * s)[:, None]
    R, t = kabsch(T, P)
    return t, R


def init_ippe(T, view, R_cam, t_cam):
    """The two planar-pose solutions (IPPE) for the flattened palm in one view."""
    Uc = view.U @ R_cam                        # rays in the camera frame (OpenCV)
    if (Uc[:, 2] < 0.2).any():
        return []
    img = (Uc[:, :2] / Uc[:, 2:3]).astype(np.float64)
    obj = np.stack([T[:, 0], T[:, 2], np.zeros(5)], 1).astype(np.float64)     # palm (x, z) plane
    A = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])                   # palm -> object coords
    try:
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return []
    out = []
    for rv, tv in zip(rvecs, tvecs):
        Rc, _ = cv2.Rodrigues(rv)
        R = R_cam @ Rc @ A
        c = R_cam @ tv.ravel() + t_cam
        out.append((c, R))
    return out


def fit_palm(T, side, views, inits, prior=None, prior_sig=(0.06, 0.6)):
    """Fit the template T (5,3) to the views.  inits: list of (c, R, name).  prior: (c, R) of the
    track's prediction, used only to choose between candidate solutions (a nearly flat palm seen by
    one camera has two mirror-image tilts that fit almost equally well), never inside the
    measurement; prior_sig = (m, rad), the prediction's spread.  Returns the best Fit or None."""
    s = float(np.linalg.norm(T[0] - T[MID]))
    best, best_score = None, None
    for c0, R0, name in inits:
        c, R, cost, J, hub = _lm(np.asarray(c0, float), np.asarray(R0, float), T, views, s)
        score = cost
        if prior is not None:
            score += float(np.sum((c - prior[0]) ** 2)) / prior_sig[0] ** 2 + \
                     (rot_angle(R, prior[1]) / prior_sig[1]) ** 2
        if best_score is None or score < best_score:
            best_score, best = score, (c, R, cost, J, name)
    if best is None:
        return None
    c, R, cost, J, name = best
    errs = _ray_err_deg(c, R, T, views)
    # robust residual: RMS of the best 4 of 5 joints per view (one occluded knuckle is normal)
    kept = np.concatenate([np.sort(e)[:4] for e in errs])
    res_deg = float(np.sqrt(np.mean(kept ** 2)))
    dof = max(1, J.shape[0] - 6)
    H = J.T @ J + np.diag([1 / 0.5 ** 2] * 3 + [1 / 2.0 ** 2] * 3)     # weak bound: 0.5 m, 2 rad
    try:
        cov = np.linalg.inv(H) * max(1.0, cost / dof)
    except np.linalg.LinAlgError:
        return None
    return Fit(c, R, cost, res_deg, cov, errs, name)


def fit_scale(T, views, c, R, iters=8):
    """Two views of one instant: refine (c, R) and a scale factor k of the template together, from
    the rays only (Mercury's relative depths are in hand-size units: they carry no absolute size).
    The size is observable only through the disparity between the cameras.  Starts from a fixed-size
    fit (c, R).  Returns (k, sd of log k) or None."""
    if len(views) < 2:
        return None
    errs = _ray_err_deg(c, R, T, views)
    hub = [np.where(e > HUBER_DEG, np.sqrt(HUBER_DEG / np.maximum(e, 1e-6)), 1.0) for e in errs]

    def rj(c, R, k):
        RT = k * (T @ R.T)
        X = c + RT
        res, jac = [], []
        for vi, v in enumerate(views):
            d = X - v.o
            rho = np.linalg.norm(d, axis=1)
            n = d / rho[:, None]
            wv = (v.w * hub[vi]) / SIG_RAY
            res.append(((n - v.U) * wv[:, None]).ravel())
            P = (np.eye(3)[None] - n[:, :, None] * n[:, None, :]) / rho[:, None, None]
            Jc = P * wv[:, None, None]
            H = np.stack([hat(a) for a in RT])
            Jt = -np.einsum("jab,jbc->jac", P, H) * wv[:, None, None]
            Jk = np.einsum("jab,jb->ja", P, RT) * wv[:, None]
            jac.append(np.concatenate([Jc, Jt, Jk[:, :, None]], 2).reshape(-1, 7))
        return np.concatenate(res), np.concatenate(jac, 0)

    c, R, k = np.asarray(c, float), np.asarray(R, float), 1.0
    r, J = rj(c, R, k)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(iters):
        H, g = J.T @ J, J.T @ r
        step = None
        while lam < 1e6:
            try:
                dx = -np.linalg.solve(H + lam * (np.diag(np.diag(H)) + 1e-9 * np.eye(7)), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            c2, R2, k2 = c + dx[:3], exp_so3(dx[3:6]) @ R, k * math.exp(max(-0.5, min(0.5, dx[6])))
            r2, J2 = rj(c2, R2, k2)
            cost2 = float(r2 @ r2)
            if cost2 <= cost:
                c, R, k, r, J, cost, step = c2, R2, k2, r2, J2, cost2, dx
                lam = max(lam / 4, 1e-7)
                break
            lam *= 6
        if step is None or np.abs(step).max() < 1e-6:
            break
    try:
        cov = np.linalg.inv(J.T @ J + np.diag([1 / 0.5 ** 2] * 3 + [1 / 2.0 ** 2] * 3 + [1 / 0.5 ** 2]))
    except np.linalg.LinAlgError:
        return None
    sd = math.sqrt(max(0.0, cov[6, 6]) * max(1.0, cost / max(1, len(r) - 7)))
    return k, sd


# ---------------------------------------------------------------- filters
def _q_dwna(dt, sa):
    """Discrete white-noise-acceleration process noise for one axis (2x2)."""
    return sa * sa * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt * dt]])


class _Adaptive:
    """Process-noise level driven by the innovations (not by the estimated speed, which is itself
    noise when the hand is still and would feed back into the gain).

    An observation that disagrees with the prediction more than its covariance allows (NIS above
    `attack`) is either a bad fit (single-view palm fits have heavy-tailed tilt errors) or the hand
    changing motion.  The first surprise is only down-weighted (Huber); a second one in the same
    direction - or a big surprise, or the caller saying the hand is moving - raises the process
    noise at once until the observation fits.  Otherwise the level decays back to the floor.
    Still hand: heavy smoothing.  Moving hand: the filter opens within one or two updates."""

    sa = 1e9
    y_prev = None
    attacked = False

    def _adapt(self, dt, lo, hi, attack, release, innov, Rz, force=False, big=None):
        sa = min(hi, max(lo, self.sa))
        pred = self._predicted(dt, sa)
        y, S = innov(pred, Rz)
        nis0 = nis = float(y @ np.linalg.solve(S, y))
        self.attacked = False
        if nis > attack:
            persistent = self.y_prev is not None and float(y @ self.y_prev) > 0
            if persistent or force or (big is not None and nis > big):
                while nis > attack and sa < hi:
                    sa = min(hi, sa * 2.5)
                    pred = self._predicted(dt, sa)
                    y, S = innov(pred, Rz)
                    nis = float(y @ np.linalg.solve(S, y))
                self.attacked = True
                self.y_prev = None
            else:
                y, S = innov(pred, Rz * (nis / attack))      # maybe a bad fit: trust it less
                self.y_prev = y
        else:
            self.y_prev = None
        self.sa = max(lo, sa * release)
        return pred, y, S, nis0

    def pred_sd(self, t):
        """Largest standard deviation of the predicted state (position or rotation part) at t."""
        P = self._predicted(max(t - self.t, 1e-4), self.sa)[1]
        return math.sqrt(max(0.0, float(np.linalg.eigvalsh(P[:3, :3]).max())))


class PosKF(_Adaptive):
    """Palm position, constant velocity.  State [p(3), v(3)]."""

    def __init__(self, p, t, sp=0.02, sv=1.0, sa=None):
        self.x = np.r_[np.asarray(p, float), np.zeros(3)]
        self.P = np.diag([sp * sp] * 3 + [sv * sv] * 3)
        self.t = t
        self.sa = 1e9 if sa is None else sa            # starts at the ceiling: a new track converges fast

    def _predicted(self, dt, sa):
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = _q_dwna(dt, sa)
        Q = np.zeros((6, 6))
        for a in range(3):
            Q[np.ix_([a, a + 3], [a, a + 3])] = q
        return F @ self.x, F @ self.P @ F.T + Q

    def update(self, z, Rz, t, sa_lo, sa_hi, attack=9.0, release=0.75, force=False, big=None):
        """Predict to t, then fuse measurement z (3) with covariance Rz (3x3).  Returns the NIS at
        the process-noise level before this update (how surprising the observation was)."""
        dt = max(t - self.t, 1e-4)
        (x, P), y, S, nis = self._adapt(dt, sa_lo, sa_hi, attack, release,
                                        lambda pr, Rz: (z - pr[0][:3], pr[1][:3, :3] + Rz), Rz, force, big)
        K = P[:, :3] @ np.linalg.inv(S)
        self.x = x + K @ y
        self.P = (np.eye(6) - K[:, :3] @ np.eye(3, 6)) @ P
        self.P = (self.P + self.P.T) / 2
        self.t = t
        return nis

    def nis(self, z, Rz, t, sa):
        x, P = self._predicted(max(t - self.t, 1e-4), sa)
        y = z - x[:3]
        return float(y @ np.linalg.solve(P[:3, :3] + Rz, y))

    def peek(self, t, coast_after, tau, vmax):
        """Position at time t without changing the state.  Beyond coast_after s since the last
        update the velocity decays (time constant tau): a lost hand glides to a stop, never flies."""
        h = max(t - self.t, 0.0)
        v = self.x[3:]
        sp = float(np.linalg.norm(v))
        if sp > vmax:
            v = v * (vmax / sp)
        if h > coast_after:
            h = coast_after + tau * (1 - math.exp(-(h - coast_after) / tau))
        return self.x[:3] + v * h

    def speed(self):
        return float(np.linalg.norm(self.x[3:]))


class RotKF(_Adaptive):
    """Palm orientation, constant angular velocity (head frame).  State R (3x3) + w (3);
    covariance on (rotation error, w), errors applied on the left: R = exp(e) R."""

    def __init__(self, R, t, sr=0.15, sw=6.0, sa=None):
        self.R = np.asarray(R, float).copy()
        self.w = np.zeros(3)
        self.P = np.diag([sr * sr] * 3 + [sw * sw] * 3)
        self.t = t
        self.sa = 1e9 if sa is None else sa

    def _predicted(self, dt, sa):
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = _q_dwna(dt, sa)
        Q = np.zeros((6, 6))
        for a in range(3):
            Q[np.ix_([a, a + 3], [a, a + 3])] = q
        return exp_so3(self.w * dt) @ self.R, F @ self.P @ F.T + Q

    def update(self, Rm, Rz, t, sa_lo, sa_hi, attack=9.0, release=0.75, force=False, big=None):
        dt = max(t - self.t, 1e-4)
        (Rp, P), y, S, nis = self._adapt(dt, sa_lo, sa_hi, attack, release,
                                         lambda pr, Rz: (log_so3(Rm @ pr[0].T), pr[1][:3, :3] + Rz), Rz, force, big)
        K = P[:, :3] @ np.linalg.inv(S)
        dx = K @ y
        self.R = exp_so3(dx[:3]) @ Rp
        U, _, Vt = np.linalg.svd(self.R)                 # keep it a rotation
        self.R = U @ Vt
        self.w = self.w + dx[3:]
        self.P = (np.eye(6) - K[:, :3] @ np.eye(3, 6)) @ P
        self.P = (self.P + self.P.T) / 2
        self.t = t
        return nis

    def nis(self, Rm, Rz, t, sa):
        Rp, P = self._predicted(max(t - self.t, 1e-4), sa)
        y = log_so3(Rm @ Rp.T)
        return float(y @ np.linalg.solve(P[:3, :3] + Rz, y))

    def peek(self, t, coast_after, tau, wmax):
        h = max(t - self.t, 0.0)
        w = self.w
        sp = float(np.linalg.norm(w))
        if sp > wmax:
            w = w * (wmax / sp)
        if h > coast_after:
            h = coast_after + tau * (1 - math.exp(-(h - coast_after) / tau))
        return exp_so3(w * h) @ self.R

    def speed(self):
        return float(np.linalg.norm(self.w))


class OneEuro:
    """One Euro filter (Casiez et al.) for vectors; used for the finger curls only."""

    def __init__(self, min_cutoff=1.5, beta=1.5, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _a(cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self.x is not None and t <= self.t:
            return self.x
        if self.x is None or t - self.t > 0.3:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = t - self.t
        ad = self._a(self.d_cutoff, dt)
        self.dx = ad * (x - self.x) / dt + (1 - ad) * self.dx
        a = self._a(self.min_cutoff + self.beta * np.linalg.norm(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x
