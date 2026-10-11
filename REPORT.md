# Steam Frame optical hand tracking — findings, measurements and limitations

Status snapshot: 2026-10-11, ~01:45 local. Written as a brief for an independent research pass whose
main goal is **lower latency**. The user's own assessment: *"All the tracking is amazing. It just is
delayed, clearly."* Secondary issues: occasional loss of a hand, occasional wrong depth, (fixed but
watch) hands collapsing onto one hand / left-right confusion, finger twitch when still.

Everything below is labelled **measured**, **estimated** or **unverified**. Numbers come from the
telemetry and benchmark tools in this repository unless stated otherwise.

---------------------------------------------------------------------------------------------------
## 1. Goal and hard constraints

- Optical hand tracking on the **Steam Frame** (SteamOS, Snapdragon 8 Gen 3 / Adreno 750, 8 cores,
  ~15.6 GB RAM) for **PC SteamVR apps streamed with Valve's native Frame streaming (vrlink)**.
- Per hand: put the physical controller down → that hand becomes optically tracked; pick it up →
  physical controller again. Optical hands produce **pose + 31-bone skeleton only, never buttons**.
- Do not modify XRService, headset tracking, vrcompositor behaviour or the separate chroma-key
  passthrough project. No OS-protection bypass, no camera ownership, no global Vulkan/driver changes.
- Native ARM64 Linux on the headset; no streaming camera images to the PC.
- Thermal: the headset once **rebooted under high load** (cpu7_middle 95–97 °C, fan at max). The
  cause is **unverified** (kernel CPU trip points are 110 °C critical; the fan controller's "max 95"
  is a fan set-point, not a shutdown). We run a separate watchdog anyway (see §7).

---------------------------------------------------------------------------------------------------
## 2. Cameras and access (what is actually reachable)

| Pair | Image | FOV | Live access for us |
|---|---|---|---|
| Passthrough L/R ("Arcturus Camera V3") | 2464×2464 NV12, UBWC-compressed, 60 fps, sensors mounted ±90° | ~95–100° | **yes** (only this pair) |
| SLAM L/R (Quanta DVX) | 1056×1024 mono, 60 Hz alternating scene / controller-LED exposure | ~180° fisheye each, ~55° apart, low/outward | **no** |
| Upper L/R (Quanta SMX) | 640×480 mono | ~150° | **no** |

- Passthrough pair: 64.0 mm baseline, at (∓0.032, −0.005, −0.057) m in head frame, axes ~11° out,
  ~12° down. Factory calibration (Kannala-Brandt 4) from XRService's dataset recorder;
  **validated live**: ORB stereo matches meet within 1.5 mm median.
- Access paths investigated (details: FINDINGS.md):
  - Public OpenVR `IVRTrackedCamera`: `HasCamera=1` but no frames/intrinsics on the Frame driver,
    even with `camera/enableCamera` at startup. **Dead end.**
  - `/dev/video99`: composited mirror only. `/dev/video0..16`: owned by XRService (not touched).
  - XRService IPC (`libArcturusPerception.so`, has StereoPair/Calibration returns): single-session,
    owned by driver_cv; second client could disturb tracking — **not attempted**.
  - Dataset recorder (`vrcmd --cmdhmd "start_tracking_recording 1 1 1"`): full calibration + one SLAM
    camera as H.264 (offline only).
  - **Used:** passive Vulkan layer in vrcompositor (VK_LAYER_MRHAND) dup()s the camera dma-bufs the
    compositor imports; our own Vulkan process (ptgrab) reads/decompresses them on its own queue.
  - The compositor also imports 8 SLAM-sized (1056×1024 NV12 linear) buffers; they stayed **empty**
    in every room-view mode tried. Handle borrowing from XRService (pidfd_getfd) is blocked by Yama
    ptrace_scope=1 — deliberately not bypassed.
- XRService itself ships MediaPipe palm/landmark models (used for controller "palm fallback");
  their output is not exposed.

---------------------------------------------------------------------------------------------------
## 3. Current architecture (what runs now)

```
XRService ─(UBWC NV12 dma-bufs)─▶ vrcompositor
                                     └─ VK_LAYER_MRHAND (passive: dup fds, no commands)
ptgrab (C++/Vulkan, own low-prio queue): strip-probe every ~4 ms, publish when stable,
        GPU downscale 2464²→640² luma → /dev/shm/mrhand-pt  (60 fps per camera, cameras in sync)
handtrack.live (Python, SCHED_IDLE, cores 0–4):
   fast path every frame : Lucas-Kanade on ~18 pts/hand/camera → stereo palm centre (60 Hz)
   discovery (MediaPipe HandLandmarker, separate process): alternating cameras; 2 Hz when both hands
       are tracked by Mercury, up to 7 Hz otherwise
   tracking (Monado Mercury keypoint model, separate process): one 128×128 stereographic crop per
       tracked hand per cycle, cameras alternating, ROI + temporal hint from previous joints;
       single view → 21 rays + per-joint relative depth → 3D hand (absolute depth anchored to the
       optical-flow stereo palm position)
   identity: 3D-position association, collapse guard (7 cm), geometric left/right sanity
   output 60 Hz: position (fast path) + orientation/curls (latest Mercury/MediaPipe), timestamps
       snapped to the 60 Hz camera grid
   ── UDP v3 (hand packets with capture time; ping/pong clock sync) ──▶
PC driver framehands v3 (SteamVR, C++): two virtual Frame controllers (pose + skeleton only),
   HMD pose history (pose at capture time), live-reloadable ini, presentation: interpolation with
   17 ms delay, no extrapolation; handoff by controller set-down + palm-away; disconnected when idle
SteamVR app → vrlink encode/stream → Frame decode → vrcompositor → display
```

---------------------------------------------------------------------------------------------------
## 4. Latency — where the time goes

### 4.1 Measured / estimated per stage

| Stage | Number | Kind |
|---|---|---|
| Photon → exposure end → XRService writes buffer | unknown (not observable to us) | **unmeasured** |
| ptgrab sees new buffer (strip poll every ~4 ms + one stable poll) | ~4–8 ms after write completes | estimated |
| GPU decompress + downscale + copy | 1.2–1.7 ms | measured |
| Fast path (optical flow, 2 hands × 2 cams) | 2–9 ms/frame (5.7 ms median benchmark) | measured |
| **Mercury keypoint crop** | 9.5–16 ms normal prio; **19–30 ms under SCHED_IDLE** | measured |
| Mercury cycle throughput | ~27 crops/s total → **~13 per hand per second**, one camera at a time | measured |
| Mercury stereo (both views within 40 ms) | **<5 % of updates** (almost always single-view) | measured |
| MediaPipe discovery run | 31–58 ms normal; **56–104 ms under SCHED_IDLE** | measured |
| capture → UDP send, position (fresh optical flow) | p50 **31–35 ms**, p95 66–100 ms, p99 82–250 ms | measured (900 s window) |
| capture → UDP send, orientation & finger curls | p50 **150–185 ms**, p95 300–470 ms | measured (window mixes MediaPipe-only and Mercury periods; see 4.3) |
| UDP to PC | ~1 ms (min RTT 1.3 ms) | measured |
| PC driver presentation | +17 ms interpolation delay (chosen), 0 extrapolation | configured / ablated |
| SteamVR app render + vrlink encode/stream/decode + Frame compositor | unmeasured; typically tens of ms for streaming | **unverified** |

### 4.2 Interpretation

- **Position** is the fast part (~50 ms to the PC incl. interpolation, before the streaming loop).
- **Orientation and fingers are the slow part**: a new articulated observation per hand arrives only
  ~13×/s with Mercury (and only 2–8×/s with the MediaPipe backend), each ~20–30 ms old when computed,
  then held, smoothed (rotation glide ~25–100 ms adaptive; curl One Euro 1.0/1.0) and interpolated
  (17 ms). Felt latency for wrist rotation / 3D articulation is therefore roughly 100–250 ms + the
  streaming round trip. This matches the user's report: straight-line motion feels good; angling the
  hand lags.
- The single biggest levers: **articulated update rate per hand** (crops/s) and **per-crop latency**,
  both CPU-bound on a CPU that is already hot from Valve's own workload (XRService ~73 % of a core).

### 4.3 Telemetry caveats

- Orientation source labels in telemetry (`mp_newest`, `mp_avg2`) do not distinguish Mercury from
  MediaPipe results (both write the same slot) — add a `merc` label before trusting per-backend ages.
- Capture timestamps are the time ptgrab saw a stable buffer, snapped to the 60 Hz grid — an estimate,
  not exposure time. XRService exposes no frame sequence/exposure timestamp to us.
- Motion-to-photon (through the PC app and back) has not been measured (needs a photodiode / external
  high-speed camera).

---------------------------------------------------------------------------------------------------
## 5. PC presentation ablation (synthetic replay, headset still on head; measured)

| Setting | Reversal sweep (true 150 mm) | Roughness while moving p95 | Rotation step p95 | Still p99 |
|---|---|---|---|---|
| Linear extrapolation 50 ms | 186 mm (+24 % overshoot) | 4–16 mm | 3.4° | 3.9 mm |
| No prediction | 151 mm | ~10 mm (stair-steps) | 3.3° | 0.6 mm |
| **Interpolation, 17 ms delay (live)** | **150 mm** | **3.4–4.8 mm** | **1.3°** | **0.5 mm** |
| Old: linear + angular 100 ms | 187 mm, up to 693 mm earlier | up to 33 mm | 2.4° | 1.6 mm |

Conclusion: extrapolation on these sparse/noisy samples overshoots (the 12 cm palm→controller offset
amplifies angular errors); interpolation costs one frame of delay. Any future predictor must be
measured against this harness (`tools/driver_ablation.py`, `tools/replay_sender.py`,
`pcdriver/build/pose_recorder.exe`).

---------------------------------------------------------------------------------------------------
## 6. Inference backends benchmarked (headset CPU, 1 thread, measured hot)

| Model | Input | Median |
|---|---|---|
| Mercury keypoint (`grayscale_keypoint_jan18.onnx`, ONNX Runtime 1.31 CPU) | 128×128 gray | 11.3 ms (9.5–16 ms live) |
| Mercury detector (`grayscale_detection_160x160.onnx`) | 160×160 | 5.9 ms |
| MediaPipe hand_landmarks_detector.tflite (LiteRT) | 224×224 | 23.3 ms |
| MediaPipe hand_detector.tflite (palm search) | 192×192 | 27.6 ms |
| MediaPipe Tasks full pipeline, 1 hand locked | 640² | ~22 ms |
| MediaPipe Tasks, num_hands=2 with one hand visible (palm search every frame) | 640² | ~42 ms |
| MediaPipe Tasks GPU delegate (works via non-EGL path) | 640² | ~20 ms, but compositor p95 GPU time 2.81→3.22 ms and dropped frames appeared |

- Mercury vs MediaPipe on identical live frames (shadow mode): is_hand ≥ 0.97 in 100 % of views,
  2D agreement 3–6 px median (640² image), 3D 1.4–3.4 cm median.
- Mercury curl output is consistent across views (e.g. index −3.30 vs −3.23); mapping to 0..1 is a
  **provisional guess** (thumb /2.0, fingers /4.0), not calibrated.
- **Mercury model license is unverified** (model repo has no license file): local evaluation only,
  not redistributed (fetched by `tools/fetch_models.sh`). Mercury code is BSL-1.0.
- No QNN/Hexagon NPU path available on the system; not pursued.

---------------------------------------------------------------------------------------------------
## 7. Thermal, CPU contention and passthrough banding

- During native VR streaming the hottest CPU zone idles at **71–86 °C before our workload**.
  XRService ~73 % of a core, eye tracking ~18 %, plus vrcompositor/vrlink.
- **Passthrough vertical banding** appeared with the Mercury backend. A/B: pipeline off → gone;
  ptgrab alone → no banding; tracker on → banding. XRService logged "Late WM_DONE ISR by > 1 ms" while
  the tracker ran. Mitigation now: tracker + workers under **SCHED_IDLE** on cores 0–4. Watch for
  regressions whenever compute is added; it is the clearest sign of starving Valve's real-time work.
- Thermal policy (user-set): MediaPipe half rate ≥ 88 °C, 2 Hz ≥ 90 °C, paused ≥ 92 °C (resume 88);
  fast path every other frame ≥ 90 °C; **separate watchdog stops the whole workload at ≥ 93 °C for
  3 s** (it fired once at 92.3 °C under the older 89 °C limit).
- SCHED_IDLE roughly doubles Mercury per-crop time (16 → 19–30 ms): the latency/contention trade-off
  is central.

---------------------------------------------------------------------------------------------------
## 8. Known limitations (everywhere)

**Capture**
- Only the front passthrough pair is usable live; hands low/at the sides leave its ~100° FOV.
- Frame readiness is inferred by pixel stability (no producer sequence number / fence available to us);
  tearing rate unquantified. Buffer identity is not a frame counter. Timestamps are estimates.
- Passthrough focus range is nominally 30 cm–∞; near hands may be soft (not quantified).
- ~10–20 % of 60 fps frames are not caught (gaps of 33 ms seen).

**Tracking**
- Articulated state ~13 updates/s per hand (Mercury), single-view almost always; true stereo joints <5 %.
- Single-view depth relies on Mercury's relative depths + an absolute anchor from optical-flow stereo;
  when optical flow loses one camera, depth is held (in/out motion stalls) — the "depth sometimes off"
  symptom. MediaPipe single-camera PnP depth was 3–9 cm off vs stereo.
- Optical flow loses the palm on fast rotation / palm↔back flips; hold ≤ 0.3 s then re-detect.
- No full articulated fitting (Mercury's Levenberg-Marquardt kinematic fit is not ported); joints are
  per-frame estimates, curls are 5 scalars → Valve's hand-simulation skeleton (splay = 0).
- Identity: fixed by 3D association + collapse guard + geometric L/R; crossing hands not well tested.
- Python orchestration: OpenCV's Python binding cannot share image pyramids across LK calls; GIL forced
  inference into separate processes (IPC copies of 640² images).

**PC / presentation**
- 17 ms interpolation delay by design; extrapolation overshoots on current sample quality.
- Clock sync by ping/pong (min-RTT quarter); network asymmetry unknown.
- HMD pose history uses `GetRawTrackedDevicePoses(0)` per RunFrame — whether that pose is itself
  predicted has not been audited.
- Skeleton fed by curls only (no per-joint retargeting); apps that ignore skeletal input show nothing.
- Virtual controllers present as `frame_controller` (bindings apply), no buttons.

**Operations**
- PC driver DLL can only be replaced with PC SteamVR closed.
- Earlier reboot cause unproven; watchdog is a safety net, not a diagnosis.

---------------------------------------------------------------------------------------------------
## 9. Where latency could come from next (hypotheses for the research pass)

Ordered by expected impact on felt latency of rotation/articulation. None verified yet.

1. **More articulated updates per hand at lower per-update latency** — e.g. both cameras every cycle
   for each hand (true stereo every update) at 30–60 Hz. Needs ~4–8× today's Mercury throughput
   without starving XRService. Candidates: GPU inference with a compositor-safe priority (ncnn
   Vulkan / TFLite GPU on Adreno — must prove no compositor regression), int8/fp16 quantised
   keypoint model, smaller crops, batching both views/hands in one inference.
2. **Move the hot loop out of Python** (C++ tracker: shared pyramids, zero-copy shm, no IPC image
   copies, one process); reuse Monado Mercury natively incl. its kinematic LM fit (consistent
   articulated hand, joint limits, temporal priors) — requires building Monado static libs on SteamOS.
3. **Motion model / prediction designed together with the presentation layer** (e.g. constant-
   velocity Kalman on palm pose + per-joint, with measurement-time-correct updates), then re-test
   extrapolation in the replay harness; only worthwhile once samples are frequent and clean.
4. **Cut fixed delays**: ptgrab poll/settle (event-driven readiness if any legitimate signal exists),
   PC interpolation 17 ms → adaptive jitter buffer, measure SteamVR/vrlink motion-to-photon.
5. **Coverage** (secondary): SLAM cameras would fix out-of-FOV losses but have no legitimate live
   path today.
6. **Valve's own hand pipeline**: XRService already runs MediaPipe palm/landmark models; if Valve ever
   exposes hand tracking, it would avoid all duplicate compute.

---------------------------------------------------------------------------------------------------
## 10. Reproduce / tools

- Headset: `~/handtrack/start.sh` / `stop.sh` (args default from `~/handtrack/start_args`,
  currently `--backend mercury`); dashboard panel "Hand Tracking" toggles the same.
- Telemetry: `tools/telemetry_report.py --last N` (per-component fresh-observation rates and ages).
- Benchmarks: `tools/validate_mercury.py`, shadow mode `--mercury-shadow`, optical-flow and model
  benchmarks (see git history), `tools/test_palm.py`, `tools/test_mercury_geom.py`, `tools/check_stereo.py`.
- PC: `tools/replay_sender.py` + `pcdriver/build/pose_recorder.exe` + `tools/driver_ablation.py`;
  live settings in `pcdriver/framehands/framehands_live.ini` (re-read every second).
- Rollback: headset `~/handtrack/revert_v1.sh` (tracker before blending changes),
  `stop.sh`, `capture/uninstall.sh` (layer), `overlay/install_overlay.sh --remove`;
  PC `vrpathreg removedriver <repo>\pcdriver\framehands`, working v2 DLL kept locally
  (SHA-256 E9C96E1C…), v3 DLL 7E84FD49….
