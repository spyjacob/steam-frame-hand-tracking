"""Monado Mercury's keypoint model as a per-hand ROI backend (Python port of the crop /
decode stage of src/xrt/tracking/hand/mercury/hg_model.cpp + hg_image_distorter.cpp,
Monado commit 13b5c134b16c86ff3e3be52e8fa6dc720a8247e0, BSL-1.0).

Model: grayscale_keypoint_jan18.onnx (sha256 40c0daa5...d1d5), from
gitlab.freedesktop.org/monado/utilities/hand-tracking-models @37a8a81b.  Its license is
UNVERIFIED (no license file in that repo): local benchmarking only, not a shipping dependency.

Per hand and camera view: a 128x128 stereographic reprojection of the fisheye image, centred on
the hand's predicted 3D keypoints (right hands mirrored), normalised, run through the model with
the predicted keypoints as a temporal hint; heatmaps are decoded back to 3D ray directions.

Frames: our camera frame is OpenCV (x right, y down, z forward); Mercury's is OpenXR-like
(x right, y up, -z forward) -> flip y and z when crossing.
"""
import math

import cv2
import numpy as np

WSIZE = 128
GRID = 22
EXPAND = 1.7          # Mercury tuneable dyn_radii_fac default
IS_HAND_MIN = 0.97    # Mercury: when the model is sure, it's really sure


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
    if c < -1 + 1e-6:                      # opposite: rotate pi about any orthogonal axis
        ax = np.cross(a, [1, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0, 1, 0])
        ax /= np.linalg.norm(ax)
        return np.array([0.0, *ax])
    ax = np.cross(a, b)
    s = math.sqrt((1 + c) * 2)
    return np.array([s / 2, *(ax / s)])


def direction(dir3, twist):
    one = q_from_two_vectors(np.array([0.0, 0, -1]), dir3)
    h = twist / 2
    two = np.array([math.cos(h), 0, 0, -math.sin(h)])     # AngleAxis(twist, -UnitZ)
    return q_mul(one, two)


def sg_unproject(x, y):
    d = 1 + x * x + y * y
    return np.stack([2 * x / d, 2 * y / d, (-1 + x * x + y * y) / d], -1)


# ---------------------------------------------------------------- projection instructions
def project_unscaled(joints_m, q):
    d = joints_m / np.linalg.norm(joints_m, axis=1, keepdims=True)
    d = q_rot(q_conj(q), d)
    den = 1 - d[:, 2]
    return np.stack([d[:, 0] / den, d[:, 1] / den], 1)


def make_instructions(joints_m, flip, expand=EXPAND, twist=0.0):
    """joints_m: (21,3) in Mercury camera frame.  Returns rot_quat, radius, predicted keypoints
    (21,2) in the model's normalised input convention, relative depths (21,)."""
    dir3 = joints_m[9] / np.linalg.norm(joints_m[9])
    q = direction(dir3, twist)
    old = dir3
    for _ in range(8):
        sg = project_unscaled(joints_m, q)
        mn, mx = sg.min(0), sg.max(0)
        c = (mn + mx) / 2
        r = max(c[0] - mn[0], c[1] - mn[1])
        new = q_rot(q, sg_unproject(c[0], c[1]))[0]
        q = direction(new, twist)
        if np.linalg.norm(old - new) < 1e-4:
            break
        old = new
    sg = project_unscaled(joints_m, q)
    fwd = np.linalg.norm(sg[0] - sg[9])
    side = np.linalg.norm(sg[5] - sg[17])
    palm_l = max(fwd, side)
    r = max(r * expand, palm_l * 0.5 * (2.2 / 1.65) * expand)
    kp = np.stack([sg[:, 0] / r * (-1 if flip else 1), -sg[:, 1] / r], 1)   # exactly as Mercury feeds it
    norms = np.linalg.norm(joints_m, axis=1)
    hand_size = np.linalg.norm(joints_m[0] - joints_m[9])
    rel = (norms - norms[9]) / max(hand_size, 1e-6)
    return q, r, kp.astype(np.float32), rel


# ---------------------------------------------------------------- model wrapper
class MercuryKeypoint:
    def __init__(self, model_path, rig, slot_cams, det_size, raw_size=2464, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.cam = {s: rig.cams[n] for s, n in slot_cams.items()}
        self.det_size, self.raw_size = det_size, raw_size
        g = (np.arange(WSIZE, dtype=np.float32))
        self.gx, self.gy = np.meshgrid(g, g)

    # det image (rotated, det_size square) <-> raw calibrated pixels
    def raw_to_det(self, raw, slot):
        k = self.raw_size / self.det_size
        x = (raw[..., 0] + 0.5) / k - 0.5
        y = (raw[..., 1] + 0.5) / k - 0.5
        S = self.det_size
        if slot == 0:            # rotated CW
            return np.stack([S - 1 - y, x], -1)
        return np.stack([y, S - 1 - x], -1)   # rotated CCW

    def crop(self, slot, det_img, q, r, flip):
        sx = (-r + 2 * r * self.gx / WSIZE) if not flip else (r - 2 * r * self.gx / WSIZE)
        sy = r - 2 * r * self.gy / WSIZE
        d = np.stack([2 * sx, 2 * sy, sx * sx + sy * sy - 1], -1).reshape(-1, 3)
        d = q_rot(q, d)
        d[:, 1] *= -1           # Mercury -> OpenCV camera frame
        d[:, 2] *= -1
        raw = self.cam[slot].project(d).reshape(WSIZE, WSIZE, 2)
        det = self.raw_to_det(raw, slot).astype(np.float32)
        return cv2.remap(det_img, det[..., 0], det[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    @staticmethod
    def normalise(img):
        f = img.astype(np.float32) / 255.0
        sd = f.std()
        if sd == 0:
            return None
        f *= 0.25 / sd
        f += 0.5 - f.mean()
        return f

    def run(self, slot, det_img, joints_cv, flip, use_hint=True, expand=None):
        """joints_cv: (21,3) predicted joints in this camera's OpenCV frame (metres).
        Returns dict: rays (21,3) unit directions in the OpenCV camera frame, det_px (21,2),
        conf (21,) heatmap peak values, rel_depth (21,), is_hand prob, curls (5,), curl_var (5,)."""
        jm = joints_cv * np.array([1, -1, -1])
        q, r, kp, _ = make_instructions(jm, flip, expand=EXPAND if expand is None else expand)
        img = self.crop(slot, det_img, q, r, flip)
        x = self.normalise(img)
        if x is None:
            return None
        feeds = {"inputImg": x[None, None], "lastKeypoints": kp.reshape(1, 42) if use_hint else np.zeros((1, 42), np.float32),
                 "useLastKeypoints": np.array([1.0 if use_hint else 0.0], np.float32)}
        hm, hd, extras, curls = self.sess.run(["heatmap_xy", "heatmap_depth", "scalar_extras", "curls"], feeds)
        hm = hm[0].reshape(21, GRID * GRID)
        locs = np.zeros((21, 2))
        conf = hm.max(1)
        for i in range(21):
            row, col = divmod(int(hm[i].argmax()), GRID)
            locs[i] = self._refine(hm[i].reshape(GRID, GRID), col, row)
        px = (1 - locs[:, 0] / 11.0) if flip else (locs[:, 0] / 11.0 - 1)
        py = 1 - locs[:, 1] / 11.0
        dl = sg_unproject(px * r, py * r)
        dm = q_rot(q, dl)
        rays = dm * np.array([1, -1, -1])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        raw = self.cam[slot].project(rays)
        dep = np.clip(hd[0], 0, 1)
        s = dep.sum(1)
        com = (dep * np.arange(GRID)).sum(1) / np.maximum(s, 1e-5)
        rel = ((com + 0.5) / GRID - 0.5) * 3.0
        return {"rays": rays, "det_px": self.raw_to_det(raw, slot), "conf": conf, "rel_depth": rel,
                "is_hand": float(1 / (1 + math.exp(-float(extras[0][0])))),
                "curls": curls[0][:5].copy(), "curl_var": np.abs(curls[0][5:]) + 0.01, "crop": img}

    @staticmethod
    def _refine(h, cx, cy, kmax=10):
        """Mercury's refine_center_of_distribution (centre of mass in a window around the peak)."""
        w = hgt = GRID
        kx = max(0, min(cx, min(kmax, abs(cx - w) - 1)))
        ky = max(0, min(cy, min(kmax, abs(cy - hgt) - 1)))
        win = h[cy - ky:cy + ky + 1, cx - kx:cx + kx + 1]
        s = win.sum()
        if s == 0:
            return np.array([cx, cy], float)
        ys, xs = np.mgrid[cy - ky:cy + ky + 1, cx - kx:cx + kx + 1]
        return np.array([(win * (xs + 0.5)).sum() / s, (win * (ys + 0.5)).sum() / s])
