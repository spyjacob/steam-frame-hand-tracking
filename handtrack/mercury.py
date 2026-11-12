"""Monado Mercury's models on the Steam Frame passthrough cameras.

Python port of the crop / decode stages of Monado's src/xrt/tracking/hand/mercury/hg_model.cpp and
hg_image_distorter.cpp (commit 13b5c134b16c86ff3e3be52e8fa6dc720a8247e0, BSL-1.0):

- KeypointModel (grayscale_keypoint_jan18.onnx): one 128x128 stereographic crop of one hand in one
  camera -> 21 keypoint rays, relative depths, is-hand probability, 5 finger curls.  Right hands are
  mirrored into the crop (the model sees left hands).  Optional temporal hint = predicted keypoints.
- HandDetector160 (grayscale_detection_160x160.onnx): whole upright camera image at 160x160 ->
  for the left and the right hand: probability, centre, size.

Models from gitlab.freedesktop.org/monado/utilities/hand-tracking-models @37a8a81b (fetched by
tools/fetch_models.sh).  Their license is UNVERIFIED (no license file in that repo): local use only.

Images: ptgrab's 640x640 luma per camera in the sensor's own orientation (no rotation needed for
crops).  Pixel spaces: raw = calibrated 2464x2464 sensor pixels; shm = the 640x640 image;
upright = shm rotated so the world is upright (slot 0 clockwise, slot 1 counter-clockwise), which is
what the detectors and the debug view see.
Frames: camera = OpenCV (x right, y down, z forward, raw image orientation); Mercury's crop math is
OpenXR-like (x right, y up, -z forward): flip y and z when crossing.
"""
import math
import time

import cv2
import numpy as np

WSIZE = 128
GRID = 22
EXPAND = 1.7          # Monado dyn_radii_fac: crop radius around the predicted keypoints
AFTER_DET = 0.65      # Monado after_detection_fac: crop radius around a detector box
ROTATE = {0: cv2.ROTATE_90_CLOCKWISE, 1: cv2.ROTATE_90_COUNTERCLOCKWISE}
FLIP_YZ = np.array([1.0, -1.0, -1.0])


# ---------------------------------------------------------------- quaternion helpers (w, x, y, z)
def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def q_rot(q, v):
    """Rotate vectors v (N,3) by quaternion q."""
    w, x, y, z = q
    u = np.array([x, y, z])
    v = np.atleast_2d(v)
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_from_two_vectors(a, b):
    """Minimal rotation taking direction a to direction b (Eigen setFromTwoVectors)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = float(a @ b)
    if c < -1 + 1e-6:
        ax = np.cross(a, [1, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0, 1, 0])
        ax /= np.linalg.norm(ax)
        return np.array([0.0, *ax])
    ax = np.cross(a, b)
    s = math.sqrt((1 + c) * 2)
    return np.array([s / 2, *(ax / s)])


def direction(dir3, twist=0.0):
    one = q_from_two_vectors(np.array([0.0, 0, -1]), dir3)
    h = twist / 2
    return q_mul(one, np.array([math.cos(h), 0, 0, -math.sin(h)]))


def sg_unproject(x, y):
    d = 1 + x * x + y * y
    return np.stack([2 * x / d, 2 * y / d, (-1 + x * x + y * y) / d], -1)


def _project_unscaled(dirs_m, q):
    d = dirs_m / np.linalg.norm(dirs_m, axis=1, keepdims=True)
    d = q_rot(q_conj(q), d)
    den = np.maximum(1 - d[:, 2], 1e-9)
    return np.stack([d[:, 0] / den, d[:, 1] / den], 1)


class Instr:
    """Where a crop looks: rotation q (Mercury frame), stereographic radius r, mirror flag, and the
    predicted keypoints in the model's input convention (hint) when available."""
    __slots__ = ("q", "r", "flip", "hint")

    def __init__(self, q, r, flip, hint=None):
        self.q, self.r, self.flip, self.hint = q, r, flip, hint


def instr_from_points(pts_cam, flip, expand=EXPAND):
    """Crop around 21 points (camera frame; only their directions matter).  Monado
    make_projection_instructions, incl. the minimum radius around the palm (closed fist)."""
    jm = np.asarray(pts_cam, float) * FLIP_YZ
    dir3 = jm[9] / np.linalg.norm(jm[9])
    q = direction(dir3)
    old = dir3
    r = 0.1
    for _ in range(8):
        sg = _project_unscaled(jm, q)
        mn, mx = sg.min(0), sg.max(0)
        c = (mn + mx) / 2
        r = max(c[0] - mn[0], c[1] - mn[1])
        new = q_rot(q, sg_unproject(c[0], c[1]))[0]
        q = direction(new)
        if np.linalg.norm(old - new) < 1e-4:
            break
        old = new
    sg = _project_unscaled(jm, q)
    palm_l = max(np.linalg.norm(sg[0] - sg[9]), np.linalg.norm(sg[5] - sg[17]))
    r = max(r * expand, palm_l * 0.5 * (2.2 / 1.65) * expand)
    hint = np.stack([sg[:, 0] / r * (-1 if flip else 1), -sg[:, 1] / r], 1).astype(np.float32)
    return Instr(q, r, flip, hint.reshape(1, 42))


def instr_angular(dir_cam, angle, flip, expand=AFTER_DET):
    """Crop around a direction with an angular radius (after a detection: no hint)."""
    d = np.asarray(dir_cam, float) * FLIP_YZ
    r = math.sin(angle) / (1 + math.cos(angle)) * expand
    return Instr(direction(d / np.linalg.norm(d)), r, flip, None)


# ---------------------------------------------------------------- one camera
class View:
    """One passthrough camera: calibration and the pixel mappings between raw, shm and upright."""

    def __init__(self, slot, cam, R, t, size=640, raw_size=2464):
        self.slot, self.cam, self.R, self.t = slot, cam, np.asarray(R, float), np.asarray(t, float)
        self.size, self.raw_size = size, raw_size
        self.k = raw_size / size
        g = np.arange(WSIZE, dtype=np.float64)
        self.gx, self.gy = np.meshgrid(g, g)

    # pixel spaces
    def shm_from_raw(self, raw):
        return (np.asarray(raw, float) + 0.5) / self.k - 0.5

    def raw_from_shm(self, px):
        return (np.asarray(px, float) + 0.5) * self.k - 0.5

    def upright_from_shm(self, px):
        px = np.asarray(px, float)
        S = self.size
        if self.slot == 0:                          # clockwise
            return np.stack([S - 1 - px[..., 1], px[..., 0]], -1)
        return np.stack([px[..., 1], S - 1 - px[..., 0]], -1)

    def shm_from_upright(self, up):
        up = np.asarray(up, float)
        S = self.size
        if self.slot == 0:
            return np.stack([up[..., 1], S - 1 - up[..., 0]], -1)
        return np.stack([S - 1 - up[..., 1], up[..., 0]], -1)

    def upright(self, img):
        return cv2.rotate(img, ROTATE[self.slot])

    # geometry
    def rays_cam(self, px_shm):
        return self.cam.unproject(self.raw_from_shm(px_shm))

    def to_cam(self, X_head):
        return (np.atleast_2d(X_head) - self.t) @ self.R

    def dirs_to_head(self, U_cam):
        return np.atleast_2d(U_cam) @ self.R.T

    def project_cam(self, X_cam):
        return self.shm_from_raw(self.cam.project(X_cam))

    def project_head(self, X_head):
        return self.project_cam(self.to_cam(X_head))

    def inside(self, px, margin=0.04):
        m = self.size * margin
        px = np.atleast_2d(px)
        return (px[:, 0] >= m) & (px[:, 0] < self.size - m) & (px[:, 1] >= m) & (px[:, 1] < self.size - m)

    def axis_angle(self, X_head):
        """Angle (rad) between this camera's optical axis and the direction to a point."""
        d = self.to_cam(X_head)[0]
        return math.acos(min(1.0, max(-1.0, d[2] / max(np.linalg.norm(d), 1e-9))))

    # crop
    def crop(self, img, ins):
        r = ins.r
        sx = (-r + 2 * r * self.gx / WSIZE) if not ins.flip else (r - 2 * r * self.gx / WSIZE)
        sy = r - 2 * r * self.gy / WSIZE
        d = np.stack([2 * sx, 2 * sy, sx * sx + sy * sy - 1], -1).reshape(-1, 3)
        d = q_rot(ins.q, d) * FLIP_YZ                         # Mercury -> camera frame
        px = self.project_cam(d).reshape(WSIZE, WSIZE, 2).astype(np.float32)
        behind = (d[:, 2] <= 1e-3).reshape(WSIZE, WSIZE)
        px[behind] = -10
        return cv2.remap(img, px[..., 0], px[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def normalise(img):
    f = img.astype(np.float32) / 255.0
    sd = float(f.std())
    if sd < 1e-6:
        return None
    f *= 0.25 / sd
    f += 0.5 - float(f.mean())
    return f


class KpResult:
    __slots__ = ("slot", "rays", "px", "conf", "rel", "is_hand", "hm", "curls", "curl_var", "instr", "ms", "crop")


class KeypointModel:
    def __init__(self, path, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    def run(self, view, img, ins, use_hint=True, keep_crop=False):
        """Thread-safe (ORT sessions allow concurrent run).  Returns KpResult or None."""
        t0 = time.perf_counter()
        crop = view.crop(img, ins)
        x = normalise(crop)
        if x is None:
            return None
        hint = ins.hint is not None and use_hint
        feeds = {"inputImg": x[None, None],
                 "lastKeypoints": ins.hint if hint else np.zeros((1, 42), np.float32),
                 "useLastKeypoints": np.array([1.0 if hint else 0.0], np.float32)}
        hm, hd, extras, curls = self.sess.run(["heatmap_xy", "heatmap_depth", "scalar_extras", "curls"], feeds)
        out = decode(view, ins, hm[0], hd[0], extras[0], curls[0])
        out.ms = (time.perf_counter() - t0) * 1e3
        out.crop = crop if keep_crop else None
        return out


def decode(view, ins, hm, hd, extras, curls):
    """Model outputs (one crop) -> KpResult: keypoint rays in the camera frame (Monado's decode)."""
    hm = np.asarray(hm).reshape(21, GRID, GRID)
    locs, conf = refine_peaks(hm)
    px = (1 - locs[:, 0] / 11.0) if ins.flip else (locs[:, 0] / 11.0 - 1)
    py = 1 - locs[:, 1] / 11.0
    dm = q_rot(ins.q, sg_unproject(px * ins.r, py * ins.r))
    rays = dm * FLIP_YZ
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    dep = np.clip(np.asarray(hd).reshape(21, GRID), 0, 1)
    s = dep.sum(1)
    com = (dep * np.arange(GRID)).sum(1) / np.maximum(s, 1e-5)
    out = KpResult()
    out.slot = view.slot
    out.rays = rays
    out.px = view.project_cam(rays)
    out.conf = conf
    out.rel = np.where(s > 1e-5, ((com + 0.5) / GRID - 0.5) * 3.0, 0.0)
    out.is_hand = float(1 / (1 + math.exp(-float(np.ravel(extras)[0]))))
    out.hm = float(conf.mean())
    out.curls = np.ravel(curls)[:5].astype(float)
    out.curl_var = np.abs(np.ravel(curls)[5:10]).astype(float) + 0.01
    out.instr = ins
    out.ms = 0.0
    out.crop = None
    return out


def refine_peaks(hm, kmax=10):
    """Monado refine_center_of_distribution for all 21 heatmaps: argmax, then centre of mass in a
    window around the peak (window clipped at the borders exactly as Monado does)."""
    n = hm.shape[0]
    flat = hm.reshape(n, -1)
    idx = flat.argmax(1)
    conf = flat[np.arange(n), idx]
    cy, cx = np.divmod(idx, GRID)
    locs = np.empty((n, 2))
    for i in range(n):
        x, y = int(cx[i]), int(cy[i])
        kx = max(0, min(x, min(kmax, abs(x - GRID) - 1)))
        ky = max(0, min(y, min(kmax, abs(y - GRID) - 1)))
        win = hm[i, y - ky:y + ky + 1, x - kx:x + kx + 1]
        sm = float(win.sum())
        if sm <= 0:
            locs[i] = (x, y)
            continue
        xs = np.arange(x - kx, x + kx + 1) + 0.5
        ys = np.arange(y - ky, y + ky + 1) + 0.5
        locs[i] = ((win.sum(0) * xs).sum() / sm, (win.sum(1) * ys).sum() / sm)
    return locs, conf


class DetBox:
    __slots__ = ("slot", "side", "conf", "centre", "size")

    def __init__(self, slot, side, conf, centre, size):
        self.slot, self.side, self.conf, self.centre, self.size = slot, side, conf, centre, size

    def instr(self, view, flip, expand=AFTER_DET):
        """Crop instructions for this box (Monado: angle to the box corners from its centre)."""
        c = np.asarray(self.centre, float)
        h = self.size / 2
        corners = c + np.array([[h, h], [-h, h], [-h, -h], [h, -h]])
        rc = view.rays_cam(c[None])[0]
        rr = view.rays_cam(corners)
        angle = float(np.max(np.arccos(np.clip(rr @ rc, -1, 1))))
        return instr_angular(rc, angle, flip, expand)


class HandDetector160:
    """Mercury's whole-image hand detector.  Output per hand (left, right): probability, centre,
    size; decoded exactly as Monado run_hand_detection, with our square 640 image (no black bars)."""

    SIDES = ("left", "right")

    def __init__(self, path, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    def run(self, view, img, min_conf=0.3):
        up = cv2.resize(view.upright(img), (160, 160), interpolation=cv2.INTER_AREA)
        x = normalise(up)
        if x is None:
            return []
        he, cx, cy, sz = self.sess.run(["hand_exists", "cx", "cy", "size"], {"inputImg": x[None, None]})
        k = view.size / 160.0
        out = []
        for i, side in enumerate(self.SIDES):
            conf = float(he.ravel()[i])
            if conf < min_conf:
                continue
            u = (float(cx.ravel()[i]) + 1) / 2 * 160 * k
            v = (float(cy.ravel()[i]) + 1) / 2 * 160 * k
            size = float(sz.ravel()[i]) * 160 * 2 * k
            out.append(DetBox(view.slot, side, conf, view.shm_from_upright(np.array([u, v])), size))
        return out
