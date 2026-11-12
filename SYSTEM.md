# Steam Frame optical hand tracking — v4 single-model tracker (branch v4-single-tracker, 2026-11-12)

Goal: replace the Steam Frame controllers with optically tracked hands in any PC SteamVR app
(streamed with Valve's native Frame streaming), Quest-3-like. Set a controller down → that hand
switches to hand tracking; pick it up → back to the controller. No buttons from hands; pose +
finger skeleton only.

v3 user verdict (main branch): accurate in the end, but laggy, jittery and losing the hand then
snapping back. v4 replaces the three stitched trackers (optical flow + Mercury + MediaPipe, each
feeding a different part of the pose through its own filter) with **one keypoint model and one
filter per hand**. Everything still runs on the headset; the PC only receives poses.

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
handtrack.main (Python) ── Mercury keypoints ≤30/s per hand → palm fit → Kalman filter ── pose + curls
     (Mercury detector + MediaPipe only while a hand is missing)
     ── UDP FhHandPacket v3 (PC subscribes to headset:47811), every camera frame ──▶
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

### 3.3 The v4 tracker (`handtrack/main.py`, `tracker.py`, `handfit.py`, `mercury.py`)

One observation source, one state per hand:

```
every camera frame (60 Hz, both cameras paired by capture time, snapped to the 60 Hz grid)
  1. collect finished keypoint crops → palm fit → Kalman update            (as soon as they finish,
                                                                             also between frames)
  2. each tracked hand that is due (≤ hz=30/s, shared budget=70 crops/s × thermal scale):
     predict its pose to this frame → one 128×128 Mercury crop around the predicted joints in its
     primary camera; both cameras of the same frame at track start and every 0.5 s / 3 s
  3. an enabled hand missing → hand finders propose candidates: Mercury's 160×160 detector
     (in-process, ≤30/s) and MediaPipe (own process, ≤12/s); idle rates (10/s, 4/s) once the
     missing hands have been gone 2 s.  Each candidate is checked by the keypoint model in both
     mirror images; the better heatmap decides left vs right; a checked candidate starts a track
  4. send every enabled hand's pose, predicted to this frame's capture time
```

- **Keypoint model** (`mercury.py`, Monado Mercury `grayscale_keypoint_jan18.onnx`, port of Monado's
  crop/decode): stereographic crop around the predicted joints (right hands mirrored: the model sees
  left hands), temporal hint = predicted keypoints. Output per crop: 21 keypoint rays, relative depths,
  is-hand probability, mean heatmap peak (`hm`), 5 finger curls. ~11 ms per crop on the Frame (one
  ORT thread); two worker threads.
- **Palm fit** (`handfit.fit_palm`): the rigid palm (wrist + 4 knuckles) of a canonical adult palm
  scaled to the user, fitted to the keypoint rays of one or two views plus Mercury's relative depths
  (Levenberg-Marquardt, analytic Jacobian, Huber weights, best 4 of 5 joints for the residual).
  Starting points: the prediction, a closed form from the relative depths, and both planar-pose
  (IPPE) solutions; the prediction picks between near-equal solutions. Gate: residual ≤ 4°,
  plausible position. Covariance from JᵀJ (consistency checked in simulation: NEES ≈ 1–2.5).
- **Palm size** (`PalmTemplate`): only the scale is learned — from stereo frames by fitting position,
  rotation and scale together (`fit_scale`); a running median of 25 samples, moved ≤ 2 % per sample so
  depth glides. (One frame's triangulation at 64 mm baseline is ±3 cm per knuckle: far too noisy to
  learn a palm shape — doing that corrupted the template in testing.)
- **Filter** (`PosKF`, `RotKF`): constant velocity for position and rotation (error-state on SO(3)).
  Process noise follows the innovations, not the estimated speed (speed estimated from noisy
  observations fed back into the gain and made a still hand shake): an observation more surprising
  than its covariance allows is down-weighted the first time; a second surprise in the same direction,
  a big position surprise, or the position opening, raises the process noise at once; otherwise it
  decays to the floor (still hand: heavy smoothing; moving hand: open within 1–2 updates).
  Outlier gate + restart where the hand really is (two agreeing outliers), output blended over 0.12 s.
  A hand with no good observation glides to a stop (velocity decays after 50 ms), is held 0.25 s
  (0.6 s when the other hand is in front of it), then reported missing.
- **Two hands**: an observation is refused if the crop saw the other tracked hand (compared by palm
  direction in the image and by position) or if its heatmap confidence dropped below 0.85 × the
  hand's usual and it disagrees with the prediction (a mirrored hand = the wrong one). Two tracks
  within 2 cm for 0.25 s are one hand (the newer is dropped; a clap's contact is shorter).
- **Fingers**: Mercury's 5 curls from the same observation (provisional mapping thumb /2, fingers /4),
  One Euro (1.5, 1.5).
- **Output**: palm pose (same palm frame and quaternion as v3 — centre = mean(wrist, index, middle,
  pinky knuckles); −z along the fingers; y = back of the hand), confidence by observation age, curls.
- **Scheduling**: SCHED_IDLE on cores 0-4 by default (`start.sh`: `HANDTRACK_SCHED=other`,
  `HANDTRACK_CPUS` to try others); thermal guard scales the crop budget and search rates (≥88 °C 0.7,
  ≥90 °C 0.5, ≥92 °C 0.3 + no MediaPipe); the independent watchdog stops everything at ≥93 °C for 3 s.
- **Parameters**: `handtrack/tracker.py` DEFAULTS; overridable with `--set key=value` or live in
  `~/.config/mrhand/v4.conf` (`key = value` lines, re-read every second).
- **Telemetry**: `~/handtrack/data/telemetry_v4.csv` (poses + tracker events, rotated at 20 MB; summary:
  `tools/v4_report.py`);
  status line every 5 s in `/tmp/live.log` (`[T C] fast: …`, read by the dashboard panel).
- **Simulation**: `tools/sim_tracker.py` runs the real tracker in real time against synthetic hands
  with known truth (oracle keypoint model decoded by the real decode path; occlusion; finders with
  latency): still / reach / fast / slap / sudden stop / out-of-view / two hands crossing / clap / fingers.

### 3.4 Bridge protocol (`pcdriver/src/framehands_proto.h`, `handtrack/bridge.py`)
- PC sends `FhSubscribe` {magic "FHND", version 2 or 3, kind 1} to headset UDP 47811 every second
  (v3 adds a clock ping, answered with `FhPong`); headset streams `FhHandPacket` to the sender
  (private LAN sources only; PC-initiated → no firewall rule).
- Packet: magic, version, seq, (v3: tCapNs = camera capture time, headset CLOCK_MONOTONIC), mode,
  flags (v3: enabled hands), 2 × FhHand {valid, confidence, pos[3] head frame m, quat[4] wxyz
  head-from-palm, ageMs (capture→send), curl[5], splay[5]}. v4 sends one packet per camera frame with
  the pose predicted to that frame's capture time.

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
## 5. Why v3 felt laggy and jumpy, and what v4 changes

| v3 (main) | v4 |
|---|---|
| position from optical flow, orientation + fingers from Mercury/MediaPipe, each with its own filter and timing | one keypoint observation updates position, orientation and fingers together |
| re-seeds snapped the flow tracker to the slow path (jumps), flow lost on rotation (dropouts) | crops follow the filter's prediction; misses glide to a stop; re-acquisition blends |
| stacked One Euro filters + glide (130–175 ms orientation lag for usable smoothness) | one Kalman filter per hand, adaptive to its own surprise; prediction to the capture time |
| stereo triangulation of views from different times | stereo only from the same capture; single view uses the learned palm size |
| MediaPipe on a schedule even when the hand is tracked | finders run only while a hand is missing |

Remaining limits: a single view resolves the palm's tilt worst when it is edge-on (orientation error
rises to 15–40° in the simulator's fastest wrist turns); motion blur during very fast moves can drop
crops (the hand then glides and snaps back); Mercury's detector is unproven on Frame images, so
MediaPipe remains the reliable finder.

## 6. Constraints / non-goals

- Must not disturb XRService/headset tracking, compositor frame timing, or the separate chroma-key
  layer. No bypassing OS protections (Yama ptrace_scope=1 blocks pidfd_getfd; not circumvented).
- Must stay within thermal budget (headset already rebooted once from heat).
- Hands must never take over a controller the user is holding.
- Future: finger/thumb articulation (already wired via skeleton), more cameras if a safe source exists.

## 7. Operations

- Headset: `~/handtrack/start.sh [handtrack.main args]` / `~/handtrack/stop.sh` (the dashboard panel
  runs them); default args in `~/handtrack/start_args_v4`; logs /tmp/live.log, /tmp/ptgrab.log,
  /tmp/watchdog.log. Layer: `~/handtrack/src/capture/install.sh|uninstall.sh`.
- Timing on the headset: `tools/bench_models.py`. Tracker in simulation: `tools/sim_tracker.py`.
- Debug view: `HANDTRACK_HTTP=1 ~/handtrack/start.sh` (tracker not already running), then on the PC
  `ssh -N -L 8765:127.0.0.1:8765 frame` → http://localhost:8765 (stream) or `/snap.jpg` (one frame).
- First test of v4 on the headset: [docs/V4_TEST_PLAN.md](docs/V4_TEST_PLAN.md).
- PC: build `pcdriver\build.bat`; DLL can only be replaced while PC SteamVR is closed.
  Remove: `vrpathreg removedriver <repo>\pcdriver\framehands`.
- Change log: `~/handtrack/CHANGES.log`; investigation notes: `FINDINGS.md`.
