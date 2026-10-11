#!/usr/bin/env python3
"""Synthetic headset for testing the PC driver in isolation (no cameras, no tracking).

Listens on UDP 47811 like the headset bridge, answers FhPing with FhPong, and streams
FhHandPacket v3 at 60 Hz with a KNOWN hand trajectory (head frame), optionally with
injected network jitter, loss and reordering.  Point the driver at it with
  headsetHost = 127.0.0.1      (in pcdriver\\framehands\\framehands_live.ini)
and use forceHandMode = 1 so the virtual controllers connect without the set-down gesture.
Development/measurement only: this is synthetic data, not hand tracking.

usage: replay_sender.py [--pattern still|sine|rotate|reversal|steps] [--seconds 30]
                        [--jitter-ms 0] [--loss 0] [--reorder 0] [--seed 1]
"""
import argparse
import math
import random
import socket
import struct
import threading
import time

MAGIC, VER = 0x444E4846, 3
HAND = struct.Struct("<If3f4ff5f5f")
HEAD = struct.Struct("<IIQQII")


def quat_axis(ax, deg):
    n = math.sqrt(sum(a * a for a in ax))
    h = math.radians(deg) / 2
    return (math.cos(h), *(a / n * math.sin(h) for a in ax))


def pose(pattern, t, side):
    """Head-frame palm pose (pos, quat wxyz) for the given pattern at time t (s)."""
    sx = -0.15 if side == 0 else 0.15
    base = [sx, -0.20, -0.40]
    q = (1.0, 0.0, 0.0, 0.0)
    if pattern == "sine":            # 0.5 Hz, +-10 cm sideways
        base[0] += 0.10 * math.sin(2 * math.pi * 0.5 * t)
    elif pattern == "rotate":        # wrist roll +-60 deg at 0.3 Hz, position fixed
        q = quat_axis((0, 0, 1), 60 * math.sin(2 * math.pi * 0.3 * t))
    elif pattern == "reversal":      # constant 0.6 m/s sweeps, reversing every 0.5 s
        ph = (t % 1.0)
        base[0] += 0.3 * (ph if ph < 0.5 else 1.0 - ph) - 0.075
    elif pattern == "steps":         # 5 cm step every 2 s (tests jump handling)
        base[1] += 0.05 * (int(t / 2) % 2)
    return base, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="sine", choices=["still", "sine", "rotate", "reversal", "steps"])
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--jitter-ms", type=float, default=0, help="uniform 0..N ms extra send delay")
    ap.add_argument("--loss", type=float, default=0, help="packet loss probability")
    ap.add_argument("--reorder", type=float, default=0, help="probability a packet is held one slot")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 47811))
    subs = {}

    def rx():
        while True:
            d, addr = sock.recvfrom(64)
            t_rx = time.perf_counter_ns()
            if len(d) >= 24 and struct.unpack_from("<III", d) == (MAGIC, VER, 1):
                subs[addr] = time.monotonic()
                pc_send = struct.unpack_from("<Q", d, 16)[0]
                # our clock = perf_counter (not the driver's steady_clock epoch): the driver must
                # estimate the offset, exactly as with the real headset
                sock.sendto(struct.pack("<IIIIQQ", MAGIC, VER, 2, 0, pc_send, t_rx), addr)
    threading.Thread(target=rx, daemon=True).start()

    t0 = time.perf_counter()
    seq = 0
    held = None
    period = 1 / 60
    nxt = t0
    sent = lost = reordered = 0
    while time.perf_counter() - t0 < a.seconds:
        nxt += period
        time.sleep(max(0.0, nxt - time.perf_counter()))
        t_cap = time.perf_counter_ns()
        t = (t_cap / 1e9) - t0
        seq += 1
        body = b""
        for side in (0, 1):
            p, q = pose(a.pattern, t, side)
            body += HAND.pack(1, 1.0, *p, *q, 0.0, *([0.3] * 5), *([0.0] * 5))
        pkt = HEAD.pack(MAGIC, VER, seq, t_cap, 1, 0) + body
        if a.jitter_ms > 0:
            time.sleep(rnd.uniform(0, a.jitter_ms) / 1000)
        if rnd.random() < a.loss:
            lost += 1
            continue
        out = [pkt]
        if held is not None:
            out.append(held)
            held = None
        elif rnd.random() < a.reorder:
            held, out = pkt, []
            reordered += 1
        for addr in list(subs):
            for o in out:
                sock.sendto(o, addr)
        sent += len(out)
    print("sent %d packets (%s), lost %d, reordered %d, subscribers %d" % (sent, a.pattern, lost, reordered, len(subs)))


if __name__ == "__main__":
    main()
