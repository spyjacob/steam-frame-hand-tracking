"""Consumer side of VK_LAYER_MRHAND: receive the camera dma-bufs vrcompositor
imported, map them read-only, and read NV12 frames.

Nothing here talks to XRService or the compositor's GPU work; we only read
memory the compositor already holds, after the layer handed us a dup'd fd.
"""
import fcntl
import mmap
import os
import socket
import struct

import numpy as np

SOCK = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "mrhand", "cam.sock")
MAGIC = 0x4D52484E
_FMT = "<4I2Q8IQ2I4Q4Q"          # mirrors MrhandBufInfo (capture/handcam_proto.h)
_SIZE = struct.calcsize(_FMT)
DMA_BUF_IOCTL_SYNC = 0x40086200  # _IOW('b', 0, struct dma_buf_sync)
DMA_BUF_SYNC_READ, DMA_BUF_SYNC_START, DMA_BUF_SYNC_END = 1, 0, 4
VK_FORMAT_G8_B8R8_2PLANE_420_UNORM = 1000156003


class Buffer:
    def __init__(self, fields, fd):
        (self.magic, self.generation, self.id, self.handle_type, self.size, self.bind_offset,
         self.has_image, self.format, self.width, self.height, self.array_layers, self.tiling, self.usage,
         self.has_modifier, self.modifier, self.plane_count, _pad, *rest) = fields
        self.plane_offset = rest[:4]
        self.plane_pitch = rest[4:8]
        self.fd = fd
        self._map = None

    def __repr__(self):
        return ("Buffer#%d %dB fmt=%d %dx%d layers=%d tiling=%d mod=%s planes=%d off=%s pitch=%s"
                % (self.id, self.size, self.format, self.width, self.height, self.array_layers, self.tiling,
                   hex(self.modifier) if self.has_modifier else "-", self.plane_count,
                   list(self.plane_offset[:self.plane_count]), list(self.plane_pitch[:self.plane_count])))

    def map(self):
        if self._map is None:
            self._map = mmap.mmap(self.fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ)
        return self._map

    def _sync(self, flags):
        try:
            fcntl.ioctl(self.fd, DMA_BUF_IOCTL_SYNC, struct.pack("<Q", flags))
        except OSError:
            pass

    def read_y(self, layer=0):
        """Copy of the luma plane as HxW uint8 (assumes linear NV12)."""
        m = self.map()
        pitch = self.plane_pitch[0] or self.width
        off = self.bind_offset + self.plane_offset[0] + layer * pitch * self.height
        self._sync(DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ)
        try:
            raw = np.frombuffer(m, np.uint8, count=pitch * self.height, offset=off)
            return raw.reshape(self.height, pitch)[:, :self.width].copy()
        finally:
            self._sync(DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ)

    def read_raw(self, nbytes=None, offset=0):
        m = self.map()
        self._sync(DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ)
        try:
            return np.frombuffer(m, np.uint8, count=nbytes or (self.size - offset), offset=offset).copy()
        finally:
            self._sync(DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ)

    def close(self):
        if self._map is not None:
            self._map.close()
        os.close(self.fd)


def fetch(path=SOCK, timeout=5.0):
    """Connect to the layer and return the list of Buffers it currently knows."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(path)
    out = []
    try:
        while True:
            msg, fds, _flags, _addr = socket.recv_fds(s, _SIZE, 1)
            if len(msg) < _SIZE:
                for fd in fds:
                    os.close(fd)
                break
            fields = struct.unpack(_FMT, msg)
            if fields[0] != MAGIC:
                raise ValueError("bad magic from layer")
            if fields[2] == 0:
                break
            if fds:
                out.append(Buffer(fields, fds[0]))
    finally:
        s.close()
    return out
