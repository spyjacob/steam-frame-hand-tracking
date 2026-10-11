# Arcturus camera access — investigation log

Headset: `ssh frame` (steamos@192.168.8.180). Headset-side tree: `~/handtrack/` (src/, backup/, CHANGES.log).
PC-side source: this repository (`<repo>`).

## Verified (2026-10-10)

| Path | Result | What it provides |
|---|---|---|
| A. Public `IVRTrackedCamera_006` (probe/camprobe) | `HasCamera=1`, `AcquireVideoStreamingService` OK, but `GetCameraFrameSize` -> OperationFailed for all frame types; `Prop_NumCameras`=0, all camera props UnknownProperty. Tested at runtime and after a SteamVR restart with enableCamera=true | **nothing** — the Frame driver publishes no tracked-camera provider. Closed. |
| B1. `/dev/video99` / `v4l2cam` | `v4l2cam` uses IVRHeadsetView + IVROverlayView | final composited view only — unusable for stereo |
| B2. `driver_cv` debug requests (**tested**, see below) | `driver_cv.so` handles `start_tracking_recording %d %d %d` / `stop_tracking_recording` (via public `IVRSystem::DriverDebugRequest`), wiring to `XRIPCSession::setDatasetRecordingEnabled(bool x4)`; logs "Starting to record at path %s" | offline dataset: full rig calibration + 1 SLAM cam H.264 @30 fps + snapshots. Not live |
| B3. XR IPC (`libArcturusPerception.so`) | client has `handleReturn_StereoPair(MultiCameraColorFrame)`, `handleReturn_Calibration`, `getLastSegmentationResult`; protocol is single-session (`isThereASessionRunningInAnotherProcess`), owned by driver_cv | would give raw stereo, but a 2nd client risks disrupting tracking — **not attempted** |
| C. mr-chroma layer | shader override inside vrcompositor; exports no frames. (Note: its implicit layer .so is also mapped into XRService) | nothing reusable without modifying it |
| D/E. vrcompositor Vulkan tap (capture/handcam_layer.cpp, unfinished) | not built/deployed | NV12 passthrough images as delivered to the compositor (pre-reprojection) — needs validation whether distorted/raw |

Other facts:
- Camera ISP nodes: `/dev/video0..16` (msm_vfe*), XRService holds `/dev/video0`, `/dev/video3`, `/dev/v4l-subdev29`, ~290 dmabufs. Opening those ourselves would contend with XRService — ruled out.
- XRService ships `libmediapipe_linux_arm64.so`, `libonnxruntime`; no hand-pose output found in its OpenVR input manifest (Hand_Left/Right are controller poses).
- Calibration file `colorPassthroughCalibrationHistory.json` holds **extrinsics only**: passthroughCam0FromPassthroughCam1 = t(0.0024, 0.0628, -0.0117) m + quat(x,y,z,w)=(-0.00066, 0.1863, 0.9818, 0.0357) (~64 mm baseline, cams ~176° rolled relative to each other), plus frontCam0FromPassthroughCam0. **No intrinsics/distortion** — must come from another source.
- Cameras pause entirely while the headset is in standby (proximity sensor).

## Dataset recorder (B2) — tested 21:51, headset worn
`vrcmd --cmdhmd "start_tracking_recording 1 1 1"` / `"stop_tracking_recording"` (requests queue while in standby).
Output: `~/.config/openvr/config/cv/xrservice/datasets/Deckard_<date>.dataset/` (~370 MB per 55 s). Contains:
- `sensorRigCalibration.json` — **full factory intrinsics + extrinsics for all cameras**: slam_left/right (Quanta DVX 1056x1024,
  kb_spline fisheye, fx~328), upper_left/right (Quanta SMX 640x480), passthrough L/R ("Arcturus Camera V3", 2464x2464,
  kb_polar_spline, fx~1475). Positions in mm, axes given as plus_x/plus_z.
- `.occ` (Occipital chunk format; parser: tools/occ_extract.py): IMU (types 30/31, ~1 kHz), and H.264 video of **one SLAM camera
  at 30 fps** (scene exposure, stream 2) plus its interleaved **controller-LED exposure** frames (stream 4); counters alternate,
  so the sensor runs 60 Hz. Timestamps are steady 33.33 ms. **No passthrough (Arcturus color) video, no second camera in the stream.**
- `snapshots/{Beginning,End}Snapshot/` — single full-res PNGs of every camera incl. passthrough_left/right 2464x2464 RGB
  (raw fisheye, one camera ~180° rotated) and cameraMetadata.json (exposure/gain/timestamp).
Verdict: excellent calibration source + real offline test data; NOT a live or stereo frame source.

## Milestone 2 groundwork (RECORDED data, not live)
`handtrack/detect.py` (MediaPipe Tasks HandLandmarker, model `~/handtrack/models/hand_landmarker.task`,
sha256 fbc2a300...cde1) + `tools/detect_recording.py`. On rec1 stream2 (raw fisheye SLAM cam): 280/1637 frames with a hand,
27 ms/frame CPU. Visual check: correct on mid/far hands; misses a large close-up blurred hand; one doubtful small detection.
Handedness is flipped vs MediaPipe (outward-facing camera). Next: undistorted/rectified crops.

## Camera choice
Target = **SLAM pair** (Quanta DVX, 1056x1024 mono, ~180° fisheye each, ~55° apart): widest coverage. Scene frames at 30 fps
(alternate frames are short controller-LED exposures). Passthrough pair (~95-100° FOV) is the fallback.
Valve already ships MediaPipe palm_detection + hand_landmark (full/lite) in `drivers/cv/bin/linuxarm64/libmediapipe_linux_arm64.so`,
used by XRService's controller "Palm fallback" (EKF update with palm joints). Which cameras feed it is unverified, and its
output is not exposed.

## Where camera frames travel (read-only metadata, 2026-10-10 ~22:00)
- Named shm: `u1000-Shm_e975559a` (24 MB) / `_f13f643a` (12 MB) shared by vrserver+XRService = property tables, static.
  `XR_ServerResponse_High_Data` = calibration JSON text. **No images in POSIX shm.**
- dmabufs (fdinfo size/exp_name/ino, compared across processes). XRService shares exactly two image sets with **vrcompositor only**:
  - 8 x 9,564,160 B (udmabuf): passthrough-sized (2464x2464 NV12 with padding)
  - 8 x 1,769,472 B (udmabuf) = 1152x1024x1.5: **SLAM-sized NV12**. Content unverified (hypothesis: SLAM frames, maybe for a
    monochrome passthrough mode; `camera/monochromeTintHue` exists in default settings).
  XRService also keeps 32 x 1,769,472 B, 32 x 12,353,536 B, etc. to itself. vrlink receives none of them.
- Blocked / not pursued: `pidfd_getfd` (Yama ptrace_scope=1, so only the parent may do it; not bypassing that), and re-opening
  via `/proc/PID/fd` (denied by the Claude Code auto-mode classifier).
- Next: an identify-phase vrcompositor layer that lists the camera images the compositor imports and dumps one frame of each.

## Incident
Rollback first wrote `enableCamera=false` explicitly; the Frame's driver default is **true**, so the explicit false hid the
camera/passthrough options in SteamVR. Fixed by removing the key; vrsettings now identical to the original backup.

## Modifications made (rollback: `~/handtrack/rollback.sh`)
- `camera/enableCamera` set then **removed** (file identical to `~/handtrack/backup/steamvr.vrsettings.orig`).
- One SteamVR restart (SIGTERM vrserver; auto-relaunched in ~15 s).
- Dataset `Deckard_2026-10-10_21-51-47.dataset` created by the recorder (delete to reclaim 370 MB).
- `~/handtrack/models/hand_landmarker.task` downloaded.

## Live capture + palm-controller work (2026-10-10 ~23:00) — status at pause
- LIVE: VK_LAYER_MRHAND (capture/handcam_layer.cpp) in vrcompositor shares XRService's camera dma-bufs; capture/ptgrab
  imports the passthrough buffers (UBWC, mod 0x0500000000000001) on its own low-priority GPU queue -> /dev/shm/mrhand-pt.
  Both cams 60 fps, synced; 2-buffer ring per cam; cam1 = passthrough_left. SLAM-sized buffers exist but stay empty.
- handtrack/live.py: MediaPipe per cam (threads), ~25 fps/cam, ~50 ms latency, MJPEG debug on 127.0.0.1:8765 (ssh -L tunnel).
  User confirmed live tracking "looks great".
- handtrack/calib.py: KB model + rig->head alignment (0.3 mm residual); passthrough L/R at (-/+0.032, -0.005, -0.057) m head frame.
- pcdriver/: framehands SteamVR driver source (virtual L/R controllers, UDP subscribe to headset:47811, per-hand
  set-down-for-2s -> hand mode, pick-up -> controller mode via ControllerHandSelectionPriority). NOT BUILT yet: PC lacked
  Windows SDK; VS Installer adding Windows11SDK.26100 was started (UAC).
- TODO: stereo palm pose (triangulate wrist+MCPs, orientation, smoothing) + UDP sender on headset; build/register driver
  (vrpathreg adddriver), restart PC SteamVR, test role takeover in a PC VR app; tune palm->controller pose offset.
