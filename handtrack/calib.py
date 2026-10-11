"""Steam Frame camera calibration (from XRService's sensorRigCalibration.json).

Conventions
- Rig frame: XRService's, origin at slam_right, millimetres in the file (metres here).
- Camera extrinsics: plus_x / plus_z are the camera axes expressed in the rig
  frame, position is the camera centre -> rig_from_cam = [x y z | p].
- Camera model "kb" (Kannala-Brandt, 4 coefficients) on raw buffer pixels.
- Head frame: OpenVR HMD frame (+x right, +y up, -z forward, metres).
  head_from_rig is solved by aligning the rig's tracking-camera positions with
  the CAD positions in `cad.cameras` (the same "FrontAndUpperCamPositions"
  alignment driver_cv uses); CAD -> head is x->-x, z->-z per cad.__head.
"""
import itertools
import json
import os

import numpy as np

DEFAULT = os.path.expanduser("~/handtrack/data/calib/sensorRigCalibration.json")


class KBCamera:
    def __init__(self, cam):
        k = cam["intrinsics"]
        self.name = cam.get("sourceCamera")
        self.w, self.h = cam["width"], cam["height"]
        self.fx, self.fy, self.cx, self.cy = k["fx"], k["fy"], k["cx"], k["cy"]
        self.k = np.array([k.get("k1", 0), k.get("k2", 0), k.get("k3", 0), k.get("k4", 0)])
        e = cam["extrinsics"]
        x = np.array(e["plus_x"], float)
        z = np.array(e["plus_z"], float)
        y = np.cross(z, x)
        self.R_rig = np.stack([x, y, z], 1)              # rig_from_cam rotation
        self.t_rig = np.array(e["position"], float) / 1000.0

    def _d(self, th):
        t2 = th * th
        return th * (1 + t2 * (self.k[0] + t2 * (self.k[1] + t2 * (self.k[2] + t2 * self.k[3]))))

    def project(self, P):
        """camera-frame points (N,3) -> pixels (N,2)."""
        P = np.atleast_2d(P)
        r = np.hypot(P[:, 0], P[:, 1])
        th = np.arctan2(r, P[:, 2])
        d = self._d(th)
        s = np.where(r > 1e-12, d / np.maximum(r, 1e-12), 0)
        return np.stack([self.fx * P[:, 0] * s + self.cx, self.fy * P[:, 1] * s + self.cy], 1)

    def unproject(self, uv):
        """pixels (N,2) -> unit rays (N,3) in camera frame."""
        uv = np.atleast_2d(np.asarray(uv, float))
        mx = (uv[:, 0] - self.cx) / self.fx
        my = (uv[:, 1] - self.cy) / self.fy
        d = np.hypot(mx, my)
        th = d.copy()
        for _ in range(10):   # Newton on d(th) = d
            t2 = th * th
            f = self._d(th) - d
            df = 1 + t2 * (3 * self.k[0] + t2 * (5 * self.k[1] + t2 * (7 * self.k[2] + t2 * 9 * self.k[3])))
            th -= f / df
        s = np.where(d > 1e-12, np.sin(th) / np.maximum(d, 1e-12), 1)
        rays = np.stack([mx * s, my * s, np.cos(th)], 1)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _kabsch(A, B):
    """R, t minimising |R A + t - B| (rows are points)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


class Rig:
    def __init__(self, path=DEFAULT):
        c = json.load(open(path))
        self.cams = {cam["sourceCamera"]: KBCamera(cam) for cam in c["cameras"]}
        # head_from_rig via CAD tracking-camera positions
        cad = c["cad"]
        cad_pos = np.array([cm["position"] for cm in cad["cameras"]], float) / 1000.0
        hx = np.array(cad["__head"]["plus_x"], float)
        hz = np.array(cad["__head"]["plus_z"], float)
        cad_from_head = np.stack([hx, np.cross(hz, hx), hz], 1)
        head_pos = cad_pos @ cad_from_head          # rows: head-frame coords (R^T p)
        track = ["slam_right", "slam_left", "upper_left", "upper_right"]
        rig_pos = np.array([self.cams[n].t_rig for n in track])
        best = None
        for perm in itertools.permutations(range(4)):
            R, t = _kabsch(rig_pos, head_pos[list(perm)])
            err = np.linalg.norm(rig_pos @ R.T + t - head_pos[list(perm)], axis=1).max()
            if best is None or err < best[0]:
                best = (err, R, t, perm)
        self.align_err_m, self.R_head_rig, self.t_head_rig, self.cad_perm = best

    def head_from_cam(self, name):
        cam = self.cams[name]
        return self.R_head_rig @ cam.R_rig, self.R_head_rig @ cam.t_rig + self.t_head_rig


def triangulate(o1, d1, o2, d2):
    """Midpoint of closest approach of rays o+s*d (all (3,) or (N,3)); returns points and gap."""
    o1, d1, o2, d2 = map(np.atleast_2d, (o1, d1, o2, d2))
    w = o1 - o2
    a = (d1 * d1).sum(1)
    b = (d1 * d2).sum(1)
    c = (d2 * d2).sum(1)
    d = (d1 * w).sum(1)
    e = (d2 * w).sum(1)
    den = np.maximum(a * c - b * b, 1e-12)
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    p1 = o1 + s[:, None] * d1
    p2 = o2 + t[:, None] * d2
    return (p1 + p2) / 2, np.linalg.norm(p1 - p2, axis=1), s, t


if __name__ == "__main__":
    rig = Rig()
    print("rig->head alignment: max residual %.1f mm, CAD order %s" % (rig.align_err_m * 1000, rig.cad_perm))
    for n in ("passthrough_left", "passthrough_right", "slam_left", "slam_right"):
        R, t = rig.head_from_cam(n)
        print("%-18s head pos (m) %s  optical axis (head) %s" % (n, np.round(t, 4), np.round(R[:, 2], 3)))
    L, Rr = rig.head_from_cam("passthrough_left"), rig.head_from_cam("passthrough_right")
    print("passthrough baseline %.1f mm" % (1000 * np.linalg.norm(L[1] - Rr[1])))
    cam = rig.cams["passthrough_left"]
    uv = np.array([[100.0, 200.0], [1232, 1232], [2300, 2400]])
    print("unproject/project roundtrip error px:", np.abs(cam.project(cam.unproject(uv)) - uv).max())
