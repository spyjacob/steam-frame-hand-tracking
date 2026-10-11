# Research brief for pass 3: better algorithms for Steam Frame hand tracking

Purpose: a self-contained brief for a research session (papers, open-source systems, algorithms) to find
the next step forward. Read with [LATENCY_REPORT_V2.md](LATENCY_REPORT_V2.md) (latest measurements),
[REPORT.md](REPORT.md) (system and limitations), [SYSTEM.md](SYSTEM.md) (architecture).

## 1. The system in one paragraph

Optical hand tracking on the **Steam Frame** headset, delivered to PC SteamVR games over Valve's native
Frame streaming. The headset's two front passthrough cameras (2464² colour, 60 fps, 64 mm baseline,
~100° FOV) are read passively and downscaled to 640² grey per camera. A Python tracker on the headset
CPU combines: **60 Hz pyramidal Lucas-Kanade optical flow** on ~18 points per hand per camera with
stereo triangulation for palm position; **Monado Mercury's keypoint CNN** (128×128 crop per hand, one
camera per crop, alternating; 21 joints as rays + relative depth, finger curl scalars, is-hand
probability) for orientation and fingers at ~20–25 Hz per hand; **MediaPipe Hands** for discovery and
re-acquisition. A palm frame and 5 curl scalars go over UDP to a PC SteamVR driver, which shows two
virtual controllers (pose + Valve's simulated 31-bone skeleton driven by the curls). The user sets a
controller down to switch that hand to optical tracking.

## 2. Hard constraints (do not propose violating these)

- Compute: Snapdragon 8 Gen 3 / Adreno 750 / 8 CPU cores, **shared with Valve's own streaming,
  SLAM/tracking (XRService) and compositor**. The CPU already runs 70–85 °C before our work; our load
  pushes to 88–94 °C; a watchdog stops us at 93 °C for 3 s. **Thermal headroom is the binding limit.**
- No NPU path found (no QNN/Hexagon delegate). The GPU belongs to the compositor: a previous MediaPipe
  GPU test caused dropped compositor frames. Any GPU proposal must explain how it avoids that.
- Only the two front passthrough cameras are reachable (SLAM/side cameras are not). Hands low or at the
  sides leave the FOV. Exposure is not ours to control (passthrough-tuned, so motion blur on fast moves).
- Camera images and crops must stay on the headset. The PC (strong desktop GPU) may only receive
  extracted joints/rays/confidences/timestamps, so heavy *fitting* could run there; heavy *image*
  models cannot.
- No button/trigger/grip input synthesis: pose and skeleton only.
- No changes to Valve components, OS protections or system drivers.

## 3. Measured state (latest; details in LATENCY_REPORT_V2.md)

| Quantity | Value |
|---|---|
| Mercury result latency, camera → available (p50 / p95) | ~50 / 68 ms (of which ~26 ms is frame age before the crop starts, ~19 ms inference, ~2 ms collection) |
| Mercury inference per 128² crop (CPU, 1 thread, SCHED_IDLE) | 19 ms wall, 15.6 ms CPU |
| Articulated observations | ~24/s with one hand, roughly half per hand with two; true synchronized stereo rare |
| Wrist orientation lag behind raw observations (current default smoothing) | ~130 ms (95 ms with no smoothing, but then 17.7°/frame jitter when still) |
| Position (optical flow) capture → send | ~23–35 ms p50 |
| PC side | +17 ms interpolation, no extrapolation (earlier 50–100 ms predictors overshot badly) |
| Streaming round trip (PC render → encode → Frame display) | unmeasured; typically tens of ms |

## 4. What the user wants and what fails

The user likes smooth, stable hands for slow reaching and squeezing, but still finds the delay too
long. Specific failures, in priority order:

1. **Fast "slap" motion (forth and back, repeated)**: tracking is lost nearly every time. Optical flow
   loses the hand (blur, large inter-frame motion), the Mercury crop is then placed where the hand was,
   misses, and the hand drops to slow re-detection. A tested fix (higher speed gate, 4-level pyramid,
   velocity-extrapolated and enlarged crop) held the hand more often but produced wrong depth (single
   camera, depth held) and wrong orientation during the slap.
2. **Overall latency** of orientation and fingers (~130 ms behind, plus streaming), and choppy quick
   wrist turns.
3. **Finger independence**: moving the middle finger often raises the index finger instead.
   Curls are 5 scalars from Mercury's curl head (provisional mapping) feeding a simulated skeleton.
4. Occasional wrong depth when only one camera sees the hand.

## 5. Research questions

Please find and assess **concrete, implementable** options (papers with code, open-source trackers,
algorithms), each judged against the constraints in §2:

1. **Fast-motion robustness on egocentric stereo**: how do production trackers (Quest/UmeTrack-style
   systems, Ultraleap, Apple, academic egocentric work) keep fast or blurred hands? Motion models for
   ROI prediction, blur-robust keypoint models, detection-in-the-loop strategies, re-acquisition in one or
   two frames instead of hundreds of ms.
2. **Lower-latency articulated tracking at 30–60 Hz per hand within a few watts of CPU**: lighter
   keypoint models (with published accuracy and CPU latency on ARM), crop-size trade-offs, INT8 on ARM
   CPUs, model families that predict 3D joints from one view reliably.
3. **Temporal fusion instead of smoothing**: Kalman/IMM/UKF or learned filters for hand pose that cut lag
   without jitter, handling 20–25 Hz noisy articulated observations plus a 60 Hz palm position track,
   with out-of-order and differently aged measurements. What do production systems use?
4. **Kinematic hand-model fitting** (e.g. Mercury's Levenberg-Marquardt optimizer, MANO-based fitting,
   inverse kinematics) using rays/keypoints from one or two views, and whether running it on the PC from
   transmitted rays is sensible. Effect on finger independence and on depth from a single view.
5. **Stereo usage**: better ways to use the second camera than alternating single-view crops (time-aware
   stereo, epipolar-constrained keypoints, cheap second-view depth for the palm only).
6. **Prediction to display time** for hands: published approaches and safe horizons for streamed VR
   (our observations reach a PC game and come back as video).
7. **Finger-level output**: better curl/splay estimation or per-joint retargeting from 21 keypoints, to
   fix index/middle coupling.
8. **Compute placement**: realistic use of the Adreno GPU alongside a latency-critical compositor
   (Vulkan compute with low priority, ncnn, LiteRT GPU) and evidence on whether it can be done without
   dropping compositor frames.

## 6. What to deliver

- A ranked shortlist (top 5–8) of approaches: what it is, source (paper/repo/licence), expected effect
  on the four failures above, compute cost on an ARM CPU (measured numbers where published),
  integration effort in this architecture, and risks.
- For the top 2–3: a concrete implementation plan in this codebase (which files, what to measure,
  success criteria, rollback), ordered so the first step delivers something without a full rewrite.
- Clearly separate published evidence from estimates. Licences must allow local use; note anything with
  an unclear licence (the Mercury model's licence is itself unverified).

## 7. What has already been tried (do not re-propose without a new angle)

- MediaPipe-only tracking (too slow: 22–50 ms per frame on CPU, single-view depth 3–9 cm off).
- MediaPipe GPU delegate (compositor frame drops).
- 50–100 ms linear/angular extrapolation on the PC (overshoot up to 24 %).
- Lighter One Euro / no smoothing (lag down to 95 ms, jitter unacceptable when still).
- Speed gate 8 m/s + 4-level LK + velocity-extrapolated, enlarged Mercury crop for slaps (less loss,
  wrong depth and orientation).
- SLAM/side cameras (not reachable), XRService's own hand models (not exposed).
- Pass 3 (see LATENCY_REPORT_V2.md): no cross-time stereo, newest-only curls and hint reset on recovery
  (better metrics, but hint-less recovery crops coincided with explosive wrist motion: reverted);
  fast-path depth repair `--flow-v2` (kept: single-camera depth frames down ~10x); cross-camera agreement
  check for rebuilt views (better metrics, felt no better: reverted).
