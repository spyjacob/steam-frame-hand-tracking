"""UDP bridge to the PC driver (pcdriver/, FhHandPacket protocol v2/v3).

The PC driver (driver_framehands) sends FhSubscribe about once a second with the protocol version it
speaks; every tracked frame is answered with one FhHandPacket per subscriber.  v3 carries the exact
camera capture time (headset CLOCK_MONOTONIC) and the enabled-hands mask, and answers clock pings
(FhPong) so the driver can map headset time to its own.  Only hand poses (~150 bytes) leave the
headset - never images.  Subscribers are accepted from private addresses only.
"""
import socket
import struct
import threading
import time

import numpy as np

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
