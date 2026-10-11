# Steam Frame optical hand tracking — current system (2026-10-11)

Goal: replace the Steam Frame controllers with optically tracked hands in any PC SteamVR app
(streamed with Valve's native Frame streaming), Quest-3-like. Set a controller down → that hand
switches to hand tracking; pick it up → back to the controller. No buttons from hands; pose +
finger skeleton only.

Current user verdict: works end to end (both hands, correct sides/orientation), but **too jittery**
compared with Quest 3. Position is smooth-ish only when the fast tracker holds; orientation and
fingers update slowly.

---------------------------------------------------------------------------------------------------
## 1. Hardware / platform

- Steam Frame: SteamOS (Arch-based), Snapdragon 8 Gen 3 (SM8650), Adreno 750, 8 cores
  (cpu0-1 2.27 GHz, cpu2-4 3.15 GHz, cpu5-6 2.96 GHz, cpu7 prime 3.05 GHz), ~15.6 GB RAM.
- Thermals are the hard limit: during a VR streaming session the CPU already idles ~67-81 °C.
  The headset **rebooted from overheating** (core 7 at 95-97 °C, fan at max) when hand tracking ran
  ~2 MediaPipe threads flat out. No usable NPU path found (no QNN/Hexagon TFLite delegate on system).
- Cameras (from XRService's factory calibration `sensorRigCalibration.json`):
  | pair | sensor | image | FOV | use |
  |---|---|---|---|---|
  | passthrough L/R | "Arcturus Camera V3" | 2464x2464 colour NV12, 60 fps, sensors mounted ±90° rotated | ~95-100° | **what we use** |
  | SLAM L/R | Quanta DVX | 1056x1024 mono, 60 Hz alternating scene / controller-LED exposure | ~180° fisheye each, ~55° apart, low/outward | not reachable live |
  | upper L/R | Quanta SMX | 640x480 mono | ~150° | not reachable live |
- Passthrough pair: 64.0 mm baseline; in head frame at (∓0.032, -0.005, -0.057) m, optical axes
  ~11° outward, ~12° down. Calibration validated live: ORB matches between the two cameras meet
  within 1.5 mm median (slot 1 = left camera, intrinsics apply to raw buffer pixels, no crop offset).
- PC: Windows 11, PC SteamVR, VS 2022 Build Tools + Windows 11 SDK. Frame streams via Valve vrlink.

---------------------------------------------------------------------------------------------------
## 2. Data flow

```
XRService (Valve, untouched) ──dma-buf NV12 (UBWC compressed)──▶ vrcompositor
                                                                     │  VK_LAYER_MRHAND (ours, passive):
                                                                     │  dup()s the imported camera fds
                                                                     ▼
                       unix socket $XDG_RUNTIME_DIR/mrhand/cam.sock (same-uid only)
                                                                     ▼
ptgrab (ours, C++/Vulkan, own low-priority GPU queue) ── /dev/shm/mrhand-pt (640x640 luma x2) ──▶
handtrack.live (Python) ── fast optical-flow tracker 60 Hz + MediaPipe ≤10 Hz ── palm pose + curls
     ── UDP FhHandPacket v2 (104→144 B, PC subscribes to headset:47811) ──▶
PC SteamVR driver "framehands" (C++ DLL) ── two virtual Frame controllers + 31-bone skeletons ──▶ apps
```

---------------------------------------------------------------------------------------------------
## 3. Headset components (`~/handtrack/` on the Frame, source mirrored in this repository)

### 3.1 VK_LAYER_MRHAND (`capture/handcam_layer.cpp`)
- Implicit user-scope Vulkan layer, loads **only into vrcompositor**, only while
  `~/.config/mrhand/enable-capture` exists; crash guard (3 compositor starts in 120 s → disables).
- Hooks only CreateImage / AllocateMemory (dma-buf import) / Bind / Free. No commands, no queues.
- Records format, extent, DRM modifier, plane layout; serves {metadata + dup'd fd} on the socket.
- Findings: compositor imports 8 passthrough buffers (2464², NV12, modifier 0x0500000000000001 =
  Qualcomm UBWC, CPU-unreadable) and 8 SLAM-sized buffers (1056x1024 NV12 linear) that stay **empty**
  in every room-view mode tried. Coexists with the separate chroma-key layer (VK_LAYER_MRCHROMA).

### 3.2 ptgrab (`capture/ptgrab.cpp`)
- Imports the passthrough dma-bufs into its own Vulkan device (UBWC modifier, foreign-queue acquire).
- Change detection: copies a 16-row luma strip of each buffer every ~4 ms, publishes a buffer once
  a later probe shows it stable (avoids tearing). Each camera: 2-buffer ring, 60 fps, cameras in sync.
- GPU downscale: plane0 → R8 2464² → blit 1280² → blit 640² (linear) → 400 KB host copy (~1.2 ms).
- Output: `/dev/shm/mrhand-pt` (header + 2 × 640² luma, per-camera seqlock, CLOCK_MONOTONIC
  "detected" timestamp). Runs `nice 10`, `taskset 0-6`.

### 3.3 handtrack.live (`handtrack/live.py`) — main loop
- Process: `nice 10`, affinity cores 0-6, OpenCV/BLAS single-threaded.
- **Fast path, every camera frame (60 Hz)** — `handtrack/fasttrack.py`:
  - Images rotated upright (cam0 CW, cam1 CCW), 640×640 gray.
  - Per hand, per camera: pyramidal Lucas-Kanade (21×21 window, 3 levels) on seed points =
    6 palm landmarks (0,1,5,9,13,17) + up to 30 `goodFeaturesToTrack` corners inside the convex hull
    of all 21 landmarks. Forward-backward check < 2.5 px, needs ≥4 good points. Palm-centre pixel
    moves by the **median** point shift.
  - 3D: triangulate the two cameras' palm-centre rays (accept gap < 3 cm); one camera only → keep
    the last distance along the new ray.
  - Plausibility gate: 10 cm–1 m from head, in front, ≤ 4 m/s apparent speed; failure → drop tracks,
    request immediate MediaPipe re-detect, hold last good position ≤ 0.3 s.
  - Output filtered with One Euro (min_cutoff 3.0, beta 0.8). Cost ~6-9 ms/frame with hands.
- **Slow path (MediaPipe), background thread**:
  - MediaPipe Tasks HandLandmarker (`hand_landmarker.task`, float16 full), VIDEO mode, num_hands=2,
    CPU/XNNPACK, one detector instance per camera. Handedness labels used as-is (inverting them
    swapped the hands).
  - Schedule: ≤10 runs/s while a hand was seen in the last 1 s, 5/s idle; alternate cameras; every
    6th run uses both cameras (stereo); a lost fast track triggers an immediate run on that camera.
  - Thermal guard (max CPU zone): ≥86 °C → 15 Hz cap, ≥89 °C → 8 Hz, ≥92 °C pause until <86 °C.
  - Measured cost per camera frame: ~22 ms (1 hand locked on), ~25 ms palm search alone,
    **~42-50 ms** when num_hands=2 and only one hand is visible (palm search runs every frame).
- Pose from a MediaPipe result (`handtrack/palm.py`):
  - Stereo: triangulate all 21 landmarks; pair hands across cameras by palm ray gap (<2.5 cm);
    gates: depth 0.10-1.30 m, wrist→middle-MCP 5-14 cm.
  - Single camera: SQPnP fit of MediaPipe's metric world landmarks to the 2D points (rays from the
    KB fisheye model) × per-hand scale learned on stereo frames (EMA; observed 1.03-1.16).
    Mono-vs-stereo palm error observed p50 3.5-5.6 cm.
  - Palm frame: centre = mean(wrist, idx MCP, mid MCP, pinky MCP); -z = wrist→mean(4 MCPs);
    x from index→pinky (sign per hand); y = z×x (back of hand). One Euro on centre (1.5, 0.6) and axes.
  - Finger curls from MediaPipe world landmarks: finger = (MCP+PIP+DIP flexion)/250°, thumb =
    (MCP+IP)/140°, One Euro (2.0, 0.5). Splay not computed (sent as 0).
- Composition sent at 60 Hz: **position = fast tracker; orientation + curls = latest MediaPipe
  result (≤10 Hz, held between runs)**. MediaPipe position used only if the fast tracker has
  nothing and it agrees with the last sent position within 10 cm.
- Debug view: MJPEG on 127.0.0.1:8765 (SSH tunnel to the PC), both cameras, MediaPipe skeleton,
  red circle = fast-tracked palm centre, palm xyz + curls. Logging: `~/handtrack/data/live_hands.jsonl`.
- Measured now: fast path 60 fps, ~50 positions/s per hand, capture→send p50 ~15-23 ms;
  MediaPipe ~5-10 runs/s; tracker CPU ~30 % idle (was 218 % before thread fixes).

### 3.4 Bridge protocol (`pcdriver/src/framehands_proto.h`, `handtrack/palm.py: Bridge`)
- PC sends `FhSubscribe` {magic "FHND", version 2, kind 1} to headset UDP 47811 every second;
  headset streams `FhHandPacket` to the sender (private LAN sources only; PC-initiated → no firewall rule).
- Packet: magic, version, seq, mode, 2 × FhHand {valid, confidence, pos[3] head frame m,
  quat[4] wxyz head-from-palm, ageMs (capture→send), curl[5], splay[5]}.

---------------------------------------------------------------------------------------------------
## 4. PC driver (`pcdriver/`, registered with `vrpathreg adddriver ...\pcdriver\framehands`)

- Two devices, presenting as **Frame controllers**: Prop_ControllerType "frame_controller",
  input profile `{frame_controller}/input/frame_controller_profile.json`, render model
  `{frame_controller}frame_controller_left/right`. So every app's Frame-controller bindings apply.
- Components: skeleton (/input/skeleton/left|right, VRSkeletalTracking_Full, both motion ranges)
  + haptic. **No buttons** are created/pressed.
- Skeleton: Valve's OpenVR SDK sample `handskeletonsimulation` (BSD-3, vendored in
  `third_party/openvr_handsim`): 5 curls + 5 splays → 31 bone transforms.
- Pose placement: palm frame of the open-hand reference skeleton computed in /pose/raw space
  (palm at (∓0.032, -0.018, 0.118) m); controller pose = measured palm pose × inverse(that), so the
  app's hand lands on the real hand.
- World pose: HMD raw pose at the sample's capture time (`GetRawTrackedDevicePoses(-age)`) × hand-in-head.
- Prediction: world-space linear/angular velocity by finite differences of successive samples,
  EMA smoothing 0.5; `poseTimeOffset = -age` (max 100 ms) so SteamVR extrapolates to display time.
  Safety: velocity ≤ 3 m/s, angular ≤ 15 rad/s, any sample jump > 15 cm → velocity zeroed.
  Stale > 250 ms → hold last pose, OutOfRange.
- Mode switching per hand (driver reads real controller poses):
  hand mode when the real controller is still (≤1 cm / 3°) for 2 s **and** the tracked palm is seen
  > 15 cm away from it; back to controller mode when it moves > 2 cm / 8°.
  In controller mode the virtual controller reports **disconnected** (so apps can't pick it over the real controller);
  in hand mode it connects and sets Prop_ControllerHandSelectionPriority high.
- Settings (`framehands/resources/settings/default.vrsettings`, section driver_framehands):
  headsetHost, port, stillSeconds, forceHandMode, predict, predictRotation, velocitySmoothing,
  maxPredictMs, lostHoldMs, handAwayCm, maxVelocity, maxAngularVelocity, maxJumpCm.

---------------------------------------------------------------------------------------------------
## 5. Where the jitter / roughness comes from (current understanding)

1. **Orientation and fingers update at MediaPipe rate (≤10 Hz, often 5-8 under heat)** and are held
   between runs → rotation steps; driver angular-velocity prediction from these sparse, noisy samples
   adds overshoot.
2. **Fast-tracker position noise**: median LK shift on a deforming, rotating hand; palm-centre pixel
   in each camera drifts independently between MediaPipe re-seeds, then snaps on re-seed
   (the re-seed is a hard reset, not a blend) → small jumps every MediaPipe run.
3. **Single-camera depth** (PnP with MediaPipe's metric model) is 3-6 cm off; it is used to
   initialise and whenever only one camera sees the hand.
4. **Track losses on palm rotation** (appearance change) → hold/re-detect → visible hitches.
5. **Two smoothing stages + prediction not designed together**: One Euro on the headset,
   EMA velocity + SteamVR extrapolation on the PC, using the packet age (assumes headset and PC
   clocks agree via age only; network jitter not modelled).
6. **Heat ceiling**: MediaPipe is ~20-50 ms per frame on CPU; more runs → thermal throttling.

## 6. Constraints / non-goals

- Must not disturb XRService/headset tracking, compositor frame timing, or the separate chroma-key
  layer. No bypassing OS protections (Yama ptrace_scope=1 blocks pidfd_getfd; not circumvented).
- Must stay within thermal budget (headset already rebooted once from heat).
- Hands must never take over a controller the user is holding.
- Future: finger/thumb articulation (already wired via skeleton), more cameras if a safe source exists.

## 7. Operations

- Headset: `~/handtrack/start.sh [live.py args]` / `~/handtrack/stop.sh`; logs /tmp/live.log,
  /tmp/ptgrab.log, /tmp/mrhandcam-<pid>.log. Layer: `~/handtrack/src/capture/install.sh|uninstall.sh`.
- PC: build `pcdriver\build.bat`; DLL can only be replaced while PC SteamVR is closed.
  Remove: `vrpathreg removedriver <repo>\pcdriver\framehands`.
- Change log: `~/handtrack/CHANGES.log`; investigation notes: `FINDINGS.md`.
