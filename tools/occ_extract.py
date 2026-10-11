#!/usr/bin/env python3
"""Read XRService dataset recordings (.occ, Occipital chunk format).

Chunk: b"occipitl" | u32 type | u32 size | payload (TLV: i32 field, u32 len, bytes).
Image chunks (observed types 32, 38) carry an H.264 stream:
  field 1  (first, 4 B)  camera stream id
  field 1  (44 B)        u32 width, u32 height, ...
  field 3  (f64)         timestamp seconds (CLOCK_MONOTONIC-like; matches cameraMetadata.json)
  field 6  (u64)         frame counter
  field 14 (repeated)    SPS / PPS NAL units
  field 13               4-byte big-endian length-prefixed NAL units
Everything else is undocumented; treated as opaque.

usage: occ_extract.py FILE.occ [--stats] [--dump OUTDIR [--every N]]
"""
import argparse
import mmap
import os
import struct
import subprocess
import sys
from collections import defaultdict

MAGIC = b"occipitl"
IMAGE_TYPES = (32, 38)


def chunks(m):
    o = 0
    while o + 16 <= len(m):
        if m[o:o + 8] != MAGIC:
            raise ValueError("bad magic at %d" % o)
        t, n = struct.unpack_from("<II", m, o + 8)
        yield t, o + 16, n
        o += 16 + n


def tlv(m, o, n):
    p, end = o, o + n
    while p + 8 <= end:
        fid, ln = struct.unpack_from("<iI", m, p)
        yield fid, p + 8, ln
        p += 8 + ln


def parse_image(m, o, n):
    f = {"sps_pps": [], "nal": None, "stream": None}
    for fid, p, ln in tlv(m, o, n):
        if fid == 1 and ln == 4 and f["stream"] is None:
            f["stream"] = struct.unpack_from("<I", m, p)[0]
        elif fid == 1 and ln == 44:
            f["w"], f["h"] = struct.unpack_from("<II", m, p)
        elif fid == 3:
            f["ts"] = struct.unpack_from("<d", m, p)[0]
        elif fid == 6:
            f["counter"] = struct.unpack_from("<Q", m, p)[0]
        elif fid == 14:
            f["sps_pps"].append(bytes(m[p:p + ln]))
        elif fid == 13:
            f["nal"] = (p, ln)
    return f


def annexb(m, f):
    out = bytearray()
    for ps in f["sps_pps"]:
        out += b"\0\0\0\1" + ps
    p, ln = f["nal"]
    end = p + ln
    while p + 4 <= end:
        L = struct.unpack_from(">I", m, p)[0]
        out += b"\0\0\0\1" + m[p + 4:p + 4 + L]
        p += 4 + L
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("occ")
    ap.add_argument("--dump", help="decode streams into OUTDIR/stream<id>/NNNNNN.png")
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()

    fd = open(a.occ, "rb")
    m = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
    frames = defaultdict(list)
    for t, o, n in chunks(m):
        if t in IMAGE_TYPES:
            f = parse_image(m, o, n)
            f["type"] = t
            frames[f["stream"]].append(f)

    for sid, fl in sorted(frames.items()):
        ts = [f["ts"] for f in fl]
        dts = sorted(b - a_ for a_, b in zip(ts, ts[1:]))
        cnt = [f.get("counter") for f in fl]
        gaps = sum(1 for a_, b in zip(cnt, cnt[1:]) if b != a_ + 1)
        print("stream %d (chunk type %d): %d frames %dx%d  t=[%.3f .. %.3f]  fps=%.2f  dt median=%.2fms min=%.2f max=%.2f  counter gaps=%d"
              % (sid, fl[0]["type"], len(fl), fl[0]["w"], fl[0]["h"], ts[0], ts[-1],
                 (len(ts) - 1) / (ts[-1] - ts[0]), dts[len(dts) // 2] * 1e3, dts[0] * 1e3, dts[-1] * 1e3, gaps))

    sids = sorted(frames)
    if len(sids) >= 2:
        A, B = frames[sids[0]], frames[sids[1]]
        tb = {round(f["ts"], 6): f for f in B}
        exact = sum(1 for f in A if round(f["ts"], 6) in tb)
        print("stereo sync: %d/%d frames of stream %d have an identical timestamp in stream %d"
              % (exact, len(A), sids[0], sids[1]))

    if a.dump:
        import cv2  # noqa: deferred so --stats works without OpenCV
        for sid, fl in frames.items():
            d = os.path.join(a.dump, "stream%d" % sid)
            os.makedirs(d, exist_ok=True)
            es = os.path.join(d, "stream.h264")
            with open(es, "wb") as w:
                for f in fl:
                    w.write(annexb(m, f))
            cap = cv2.VideoCapture(es)
            i = ok = 0
            with open(os.path.join(d, "timestamps.csv"), "w") as tsf:
                tsf.write("index,counter,ts\n")
                while True:
                    r, img = cap.read()
                    if not r:
                        break
                    if i < len(fl):
                        tsf.write("%d,%s,%.6f\n" % (i, fl[i].get("counter"), fl[i]["ts"]))
                    if i % a.every == 0:
                        cv2.imwrite(os.path.join(d, "%06d.png" % i), img)
                    i += 1
                    ok += 1
            print("stream %d: decoded %d/%d frames -> %s" % (sid, ok, len(fl), d))


if __name__ == "__main__":
    main()
