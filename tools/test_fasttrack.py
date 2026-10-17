#!/usr/bin/env python3
"""Self-test for the two shortcuts in handtrack.fasttrack (synthetic images, no camera or calibration):
- step_tracks (all hands seen by a camera in one forward + one backward LK call) gives every track
  exactly what stepping it alone gives;
- texture_points (corner search on the hand's bounding box) returns exactly the points of
  goodFeaturesToTrack on the whole image with the hand outline as mask.
Both must hold bit for bit on any OpenCV build (run it on the headset after an OpenCV update).
Also prints what each shortcut saves on this machine."""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from handtrack.fasttrack import LK, N_EXTRA, CamTrack, step_tracks, texture_points  # noqa: E402

S = 640


def step_alone(lm, extra, prev, cur):
    """One track stepped on its own, as before step_tracks: (ok, lm, extra)."""
    pts = np.vstack([lm, extra]).astype(np.float32).reshape(-1, 1, 2)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, **LK)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, **LK)
    fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 2.5)
    if good.sum() < 4:
        return False, None, None
    old, new = pts.reshape(-1, 2), nxt.reshape(-1, 2)
    shift = np.median(new[good] - old[good], axis=0)
    n = len(lm)
    return (True, np.where(good[:n, None], new[:n], old[:n] + shift).astype(np.float32),
            new[n:][good[n:]].astype(np.float32))


def corners_whole(img, px):
    """Texture points as searched before texture_points: whole image, hand outline as mask."""
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(px.astype(np.int32)), 255)
    f = cv2.goodFeaturesToTrack(img, maxCorners=N_EXTRA, qualityLevel=0.01, minDistance=8, mask=mask)
    return f.reshape(-1, 2).astype(np.float32) if f is not None else np.zeros((0, 2), np.float32)


def textured(rng, n):
    img = cv2.GaussianBlur(rng.integers(0, 256, (n, n), dtype=np.uint8), (0, 0), 2.0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    for _ in range(12):
        c = tuple(int(v) for v in rng.integers(0, n, 2))
        cv2.circle(img, c, int(rng.integers(5, 40)), int(rng.integers(0, 256)), -1)
    return img


def frame_pair(rng):
    a = textured(rng, S)
    M = cv2.getRotationMatrix2D((S / 2, S / 2), rng.normal(0, 2), 1 + rng.normal(0, 0.02))
    M[:, 2] += rng.normal(0, 5, 2)
    b = cv2.warpAffine(a, M, (S, S), borderMode=cv2.BORDER_REFLECT)
    b = cv2.add(b, rng.integers(0, 12, b.shape, dtype=np.uint8))
    return a, b


def test_step_tracks(rng, n=300):
    bad = 0
    for _ in range(n):
        prev, cur = frame_pair(rng)
        tracks = []
        for _ in range(int(rng.integers(2, 4))):
            c = rng.uniform(-20, S + 20, 2)
            tracks.append((c + rng.normal(0, 30, (6, 2)), c + rng.normal(0, 40, (int(rng.integers(0, N_EXTRA + 1)), 2))))
        cts = [CamTrack(lm, ex) for lm, ex in tracks]
        before = [(ct.lm.copy(), ct.extra.copy()) for ct in cts]
        for (lm, ex), ct, ok, (lm0, ex0) in zip(tracks, cts, step_tracks(cts, prev, cur), before):
            ok1, lm1, ex1 = step_alone(lm, ex, prev, cur)
            if ok != ok1:
                bad += 1
            elif ok and not (np.array_equal(ct.lm, lm1) and np.array_equal(ct.extra, ex1)):
                bad += 1
            elif not ok and not (np.array_equal(ct.lm, lm0) and np.array_equal(ct.extra, ex0)):
                bad += 1
    return n, bad


def test_texture_points(rng, per_image=300):
    imgs = []
    for n in (S, 616):                                      # 616: a size that is not a multiple of 16
        imgs += [textured(rng, n),
                 ((np.indices((n, n)).sum(0) // 6) % 2 * 255).astype(np.uint8),     # equal corner scores
                 np.tile((np.arange(n) // 5 % 2 * 200 + 20).astype(np.uint8), (n, 1)),
                 np.full((n, n), 77, np.uint8)]
    trials = found = bad = 0
    for img in imgs:
        for _ in range(per_image):
            px = rng.uniform(-80, img.shape[0] + 80, 2) + rng.normal(0, rng.choice([3, 20, 60, 140]), (21, 2))
            if rng.random() < 0.05:
                px[rng.integers(21)] = rng.choice([-1e4, 1e4], 2)      # one wild landmark
            a, b = corners_whole(img, px), texture_points(img, px)
            trials += 1
            found += len(a) > 0
            bad += not (a.shape == b.shape and np.array_equal(a, b))
    return trials, found, bad


def ms(fn, n=100):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e3


def main():
    cv2.setNumThreads(1)
    rng = np.random.default_rng(0)
    print("OpenCV %s, numpy %s" % (cv2.__version__, np.__version__))
    n, bad_lk = test_step_tracks(rng)
    print("step_tracks: %d multi-hand steps, %d differ from stepping each hand alone" % (n, bad_lk))
    n, found, bad_tp = test_texture_points(rng)
    print("texture_points: %d hand outlines (%d with corners), %d differ from the whole-image search" % (n, found, bad_tp))

    prev, cur = frame_pair(rng)
    hands = [CamTrack(c + rng.normal(0, 30, (6, 2)), c + rng.normal(0, 40, (N_EXTRA, 2)))
             for c in (np.array([200.0, 300.0]), np.array([440.0, 320.0]))]
    sep = ms(lambda: [step_tracks([CamTrack(h.lm, h.extra)], prev, cur) for h in hands])
    tog = ms(lambda: step_tracks([CamTrack(h.lm, h.extra) for h in hands], prev, cur))
    px = np.array([300.0, 320.0]) + rng.normal(0, 45, (21, 2))
    whole, box = ms(lambda: corners_whole(prev, px)), ms(lambda: texture_points(prev, px))
    print("cost here: optical flow for 2 hands in one camera %.2f ms separately, %.2f ms together; "
          "texture points %.2f ms whole image, %.2f ms bounding box" % (sep, tog, whole, box))
    ok = not bad_lk and not bad_tp
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
