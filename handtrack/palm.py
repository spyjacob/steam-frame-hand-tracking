"""Stereo palm pose (head frame) from per-camera MediaPipe landmarks, plus the
UDP bridge that streams it to the PC driver (pcdriver/, FhHandPacket).

Camera slots (validated against live frames, tools/check_stereo.py):
  shm slot 1 = passthrough_left, slot 0 = passthrough_right; calibration
  intrinsics apply directly to the raw 2464x2464 buffer pixels.
"""
import math
import socket
import struct
import threading
import time

import cv2
import numpy as np

from handtrack.calib import Rig, triangulate

SLOT_CAM = {1: "passthrough_left", 0: "passthrough_right"}
PALM_IDX = [0, 5, 9, 13, 17]   # wrist + MCPs: rigid-ish palm points


# ---------------------------------------------------------------- pixel mapping
def det_to_raw(px, slot, det_size, raw_size=2464):
    """Landmark pixels in the rotated det_size x det_size detector image -> raw buffer pixels.
    live.py: small = rotate(resize(raw, det_size), ROTATE[slot]); slot0 rotated CW, slot1 CCW."""
    u, v = px[:, 0], px[:, 1]
    S = det_size
    if slot == 0:            # CW: raw(x,y) -> (S-1-y, x)
        x, y = v, S - 1 - u
    else:                    # CCW: raw(x,y) -> (y, S-1-x)
        x, y = S - 1 - v, u
    k = raw_size / S
    return np.stack([(x + 0.5) * k - 0.5, (y + 0.5) * k - 0.5], 1)


# ---------------------------------------------------------------- filtering
class OneEuro:
    """One Euro filter (Casiez et al.) for vectors."""

    def __init__(self, min_cutoff=1.5, beta=0.6, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _a(cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self.x is None or t - self.t > 0.3:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        ad = self._a(self.d_cutoff, dt)
        self.dx = ad * dx + (1 - ad) * self.dx
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        a = self._a(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


def quat_from_R(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


# ---------------------------------------------------------------- finger curls
# Scaled to OpenVR's handskeletonsimulation sample (pcdriver/third_party/openvr_handsim):
# finger curl 1.0 = 90 deg MCP + 80 deg PIP + 80 deg DIP; thumb ~140 deg over MCP + IP.
_FINGERS = [(0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]


def _ang(a, b):
    c = (a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)
    return math.acos(max(-1.0, min(1.0, c)))


def finger_curls(W):
    """W: (21,3) MediaPipe world landmarks -> curls [thumb, index, middle, ring, pinky] in 0..1."""
    out = []
    t = [W[2] - W[1], W[3] - W[2], W[4] - W[3]]
    out.append(min(1.0, max(0.0, (_ang(t[0], t[1]) + _ang(t[1], t[2])) / math.radians(140))))
    for ch in _FINGERS:
        b = [W[ch[i + 1]] - W[ch[i]] for i in range(4)]
        tot = _ang(b[0], b[1]) + _ang(b[1], b[2]) + _ang(b[2], b[3])
        out.append(min(1.0, max(0.0, tot / math.radians(250))))
    return np.array(out)


# ---------------------------------------------------------------- stereo palm
class PalmTracker:
    """Fuses the two cameras' detections of one capture instant into palm poses."""

    MAX_GAP = 0.025          # m, ray gap of a palm point
    DEPTH = (0.10, 1.30)     # m in front of the head
    PALM_LEN = (0.05, 0.14)  # m, wrist -> middle MCP

    def __init__(self, det_size, rig=None):
        self.rig = rig or Rig()
        self.det_size = det_size
        self.cam = {s: self.rig.cams[n] for s, n in SLOT_CAM.items()}
        self.pose = {s: self.rig.head_from_cam(n) for s, n in SLOT_CAM.items()}
        self.filt = {side: (OneEuro(), OneEuro(1.0, 0.3), OneEuro(1.0, 0.3)) for side in ("left", "right")}
        self.curl_filt = {side: OneEuro(1.0, 1.0) for side in ("left", "right")}   # steadier when still
        self.roll_flip = False   # was a workaround for swapped hand labels (fixed in live.py: mirrored=True)
        self.scale = {}      # per hand: true hand size / MediaPipe model size (learned from stereo frames)
        self.scale_n = {}
        self.last_good = {"left": 0.0, "right": 0.0}

    def det_px(self, slot, P):
        """Head-frame points (N,3) -> pixels in this camera's rotated detector image."""
        R, t = self.pose[slot]
        raw = self.cam[slot].project((np.atleast_2d(P) - t) @ R)
        k = 2464 / self.det_size
        x = (raw[:, 0] + 0.5) / k - 0.5
        y = (raw[:, 1] + 0.5) / k - 0.5
        S = self.det_size
        return np.stack([S - 1 - y, x], 1) if slot == 0 else np.stack([y, S - 1 - x], 1)

    def rays(self, slot, px):
        R, t = self.pose[slot]
        raw = det_to_raw(px, slot, self.det_size)
        return t, self.cam[slot].unproject(raw) @ R.T

    def fuse(self, hands0, hands1, t_cap):
        """hands0/1: Hand lists from slot 0 (right cam) / slot 1 (left cam).
        Returns {"left": pose|None, "right": pose|None}; pose = (pos(3), quat wxyz(4), conf, pts(21,3))."""
        out = {"left": None, "right": None}
        cands = []
        for h1 in hands1:
            o1, d1 = self.rays(1, h1.px)
            for h0 in hands0:
                o0, d0 = self.rays(0, h0.px)
                P, gap, s, u = triangulate(o1, d1, o0, d0)
                pg = gap[PALM_IDX]
                score = float(np.median(pg))
                same = h1.side == h0.side
                cands.append((score + (0 if same else 0.05), h1, h0, P, gap, s, u))
        cands.sort(key=lambda c: c[0])
        used0, used1 = set(), set()
        for score, h1, h0, P, gap, s, u in cands:
            if id(h0) in used0 or id(h1) in used1:
                continue
            palm = P[PALM_IDX]
            if np.median(gap[PALM_IDX]) > self.MAX_GAP or (s[PALM_IDX] <= 0).any() or (u[PALM_IDX] <= 0).any():
                continue
            depth = -palm[:, 2].mean()
            plen = np.linalg.norm(P[9] - P[0])
            if not (self.DEPTH[0] < depth < self.DEPTH[1] and self.PALM_LEN[0] < plen < self.PALM_LEN[1]):
                continue
            # side: both cameras' votes weighted by their confidence
            vote = (h1.side_score if h1.side == "left" else -h1.side_score) + \
                   (h0.side_score if h0.side == "left" else -h0.side_score)
            side = "left" if vote > 0 else "right"
            if out[side] is not None:
                continue
            used0.add(id(h0))
            used1.add(id(h1))
            curls = self.curl_filt[side]((finger_curls(h1.world) + finger_curls(h0.world)) / 2, t_cap)
            out[side] = self._pose(side, P, t_cap, conf=min(1.0, (h1.side_score + h0.side_score) / 2)) + (curls,)
            self._learn_scale(side, P, ((1, h1), (0, h0)))
        return out

    # ------------------------------------------------------------ single camera
    def mono_points(self, slot, h):
        """Fit MediaPipe's metric hand model (h.world) to the 2D landmarks with PnP.
        Returns (21,3) points in the camera frame at MediaPipe's default hand scale, or None."""
        raw = det_to_raw(h.px, slot, self.det_size)
        rays = self.cam[slot].unproject(raw)
        if (rays[:, 2] < 0.2).any():
            return None
        img = (rays[:, :2] / rays[:, 2:3]).astype(np.float64)
        W = np.asarray(h.world, np.float64)
        if np.ptp(W, axis=0).max() < 1e-3:       # degenerate / missing model
            return None
        try:
            ok, rvec, tvec = cv2.solvePnP(W, img, np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            return None
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        Pc = W @ R.T + tvec.reshape(1, 3)
        if Pc[:, 2].min() <= 0.05:
            return None
        return Pc

    def _learn_scale(self, side, P_head, cam_hands):
        """Stereo frame: learn true-hand-size / MediaPipe-model-size per camera view (EMA)."""
        for slot, h in cam_hands:
            Pc = self.mono_points(slot, h)
            if Pc is None:
                continue
            R, t = self.pose[slot]
            stereo_c = (P_head[PALM_IDX] - t) @ R          # stereo palm points in this camera's frame
            ratio = np.linalg.norm(stereo_c.mean(0)) / np.linalg.norm(Pc[PALM_IDX].mean(0))
            if 0.6 < ratio < 1.6:
                old = self.scale.get(side)
                self.scale[side] = ratio if old is None else 0.8 * old + 0.2 * ratio
                self.scale_n[side] = self.scale_n.get(side, 0) + 1

    def mono(self, slot, hands, t_cap):
        """Single-camera poses for one frame (same output format as fuse)."""
        out = {"left": None, "right": None}
        R, t = self.pose[slot]
        for h in sorted(hands, key=lambda h: -h.side_score):
            side = h.side
            if out[side] is not None:
                continue
            Pc = self.mono_points(slot, h)
            if Pc is None:
                continue
            P = (Pc * self.scale.get(side, 1.0)) @ R.T + t
            depth = -P[PALM_IDX][:, 2].mean()
            plen = np.linalg.norm(P[9] - P[0])
            if not (self.DEPTH[0] < depth < self.DEPTH[1] and self.PALM_LEN[0] < plen < self.PALM_LEN[1]):
                continue
            curls = self.curl_filt[side](finger_curls(h.world), t_cap)
            out[side] = self._pose(side, P, t_cap, conf=h.side_score * 0.9) + (curls,)
        return out

    def _pose(self, side, P, t, conf):
        wrist, idx, mid, ring, pinky = P[0], P[5], P[9], P[13], P[17]
        centre = (wrist + idx + mid + pinky) / 4
        fwd = (idx + mid + ring + pinky) / 4 - wrist   # wrist -> knuckle line: along the fingers
        across = idx - pinky                      # towards the thumb side
        z = -fwd / np.linalg.norm(fwd)            # controller convention: -z points along the fingers
        x = (-across if side == "right" else across)
        if self.roll_flip:   # user test: palm-down showed palm-up -> 180 deg about the finger axis
            x = -x
        x = x - z * (x @ z)
        x /= np.linalg.norm(x)
        fp, fz, fx = self.filt[side]
        centre = fp(centre, t)
        z = fz(z, t)
        z /= np.linalg.norm(z)
        x = fx(x, t)
        x = x - z * (x @ z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)                        # back of the hand
        R = np.stack([x, y, z], 1)
        self.last_good[side] = t
        return centre, quat_from_R(R), conf, P


# ---------------------------------------------------------------- UDP bridge
FH_MAGIC, FH_VERSION, FH_PORT, FH_SUBSCRIBE = 0x444E4846, 3, 47811, 1
FH_VERSIONS = (2, 3)                     # served per subscriber (old driver = v2)
_HAND = struct.Struct("<If3f4ff5f5f")   # FhHand: valid, conf, pos[3], quat[4], ageMs, curl[5], splay[5]
_PKT_HEAD = {2: struct.Struct("<IIQII"),     # magic, version, seq, mode, pad
             3: struct.Struct("<IIQQII")}    # magic, version, seq, tCapNs (headset CLOCK_MONOTONIC), mode, pad


class Bridge:
    """Serves FhHandPacket to subscribers (the PC driver sends FhSubscribe every second,
    with the protocol version it speaks; v3 adds the exact camera capture time)."""

    enabled_mask = 3                              # set by the tracker from ~/.config/mrhand/hands

    def __init__(self, port=FH_PORT, private_only=True):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.subs = {}                   # addr -> (last subscribe time, version)
        self.seq = 0
        self.private_only = private_only
        self.lock = threading.Lock()
        threading.Thread(target=self._rx, daemon=True).start()

    @staticmethod
    def _private(ip):
        a = [int(x) for x in ip.split(".")]
        return a[0] == 10 or (a[0] == 172 and 16 <= a[1] < 32) or (a[0] == 192 and a[1] == 168) or a[0] == 127

    def _rx(self):
        while True:
            data, addr = self.sock.recvfrom(64)
            t_rx = time.monotonic_ns()
            if len(data) < 12:
                continue
            magic, ver, kind = struct.unpack_from("<III", data)
            if magic != FH_MAGIC or kind != FH_SUBSCRIBE or ver not in FH_VERSIONS:
                continue
            if self.private_only and not self._private(addr[0]):
                continue
            if ver >= 3 and len(data) >= 24:
                # clock ping: answer immediately with our CLOCK_MONOTONIC receive time (FhPong)
                pc_send = struct.unpack_from("<Q", data, 16)[0]
                try:
                    self.sock.sendto(struct.pack("<IIIIQQ", FH_MAGIC, 3, 2, 0, pc_send, t_rx), addr)
                except OSError:
                    pass
            with self.lock:
                old = self.subs.get(addr)
                self.subs[addr] = (time.monotonic(), ver)
            if old is None or old[1] != ver:
                print("bridge: subscriber %s:%d (protocol v%d)" % (addr[0], addr[1], ver), flush=True)

    def send(self, hands, t_cap_ns):
        """hands: {"left": pose|None, "right": pose|None}; t_cap_ns: capture time (headset monotonic ns)."""
        now = time.monotonic()
        with self.lock:
            self.subs = {a: v for a, v in self.subs.items() if now - v[0] < 5}
            subs = list(self.subs.items())
        if not subs:
            return
        age = (time.monotonic_ns() - t_cap_ns) / 1e6
        self.seq += 1
        body = b""
        for side in ("left", "right"):
            p = hands.get(side)
            if p is None:
                body += _HAND.pack(0, 0, 0, 0, 0, 1, 0, 0, 0, age, *([0.0] * 10))
            else:
                pos, q, conf = p[0], p[1], p[2]
                curls = p[4] if len(p) > 4 else np.zeros(5)
                body += _HAND.pack(1, conf, *map(float, pos), *map(float, q), age, *map(float, curls), *([0.0] * 5))
        # v3 "pad" carries the enabled hands: bit 31 = present, bit 0 = left, bit 1 = right
        flags = 0x80000000 | self.enabled_mask
        bufs = {2: _PKT_HEAD[2].pack(FH_MAGIC, 2, self.seq, 1, 0) + body,
                3: _PKT_HEAD[3].pack(FH_MAGIC, 3, self.seq, int(t_cap_ns), 1, flags) + body}
        for a, (_, ver) in subs:
            try:
                self.sock.sendto(bufs[ver], a)
            except OSError:
                pass
