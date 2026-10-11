# Latency pass 2 — report (2026-10-11)

Goal: reduce visible wrist/finger delay without losing the smoothness the user likes. Scope and rules
followed the pass-2 brief (no changes to XRService, compositor, Valve tracking, chroma key, OS
protections, thermal limits or watchdog; no new input synthesis; no images leave the headset).

Everything below is **measured** on the headset unless marked otherwise. Motion-to-photon (physical
hand → display) is still **unmeasured**; all lags here end at the UDP send on the headset.

## Environment

| Item | Value |
|---|---|
| Headset | Steam Frame, SteamOS, streaming from PC SteamVR (vrlink) |
| Python libs | onnxruntime 1.31.0 (providers: CPU, Azure; **no XNNPACK build**), opencv 5.0.0, numpy 2.5.4, mediapipe 1.1.0 |
| Mercury keypoint model | `grayscale_keypoint_jan18.onnx`, sha256 `40c0daa598cedb99…` |
| Mercury detector (unused) | `grayscale_detection_160x160.onnx`, sha256 `1f1a039a266e13dc…` |
| MediaPipe model | `hand_landmarker.task`, sha256 `fbc2a30080c3c557…` |
| Tracker process | `chrt --idle` (SCHED_IDLE), cores 0–4; Mercury in one worker process, ORT 1 intra-op thread |

## Three metrics, kept separate

1. **New-result latency**: camera-capture estimate → the tracker has the Mercury result
   (`R` telemetry rows: capture, submit, worker start/end, collect, CPU vs runqueue wait).
2. **Age at send**: age of the latest observation at each 60 Hz send, including holding
   (`F` rows; this is what REPORT.md's "150–185 ms" was).
3. **Motion-response lag**: the time shift that best aligns the sent orientation with the raw,
   unfiltered Mercury orientations (interpolated), while the wrist turns. Includes metric 1 plus filtering.

`tools/telemetry_report.py` prints all three; validated on synthetic data (recovers an injected 120 ms
lag exactly; nearest-sample comparison biased it to 80 ms, so interpolation is used).

## Results

### Baseline (both hands, original settings, batch scheduling)
Run 12:34–12:39 (this was the run first started while the user was in the headset; a later "baseline"
run accidentally wrote to the default telemetry file — see *Process issues*). Same settings either way.

| Metric | Value |
|---|---|
| New-result latency p50 / p95 | **70.8 / 113 ms** |
| — frame age when the crop is submitted | 30.8 ms (ptgrab poll + shm + fast path before submit) |
| — submit → worker start | 1.8 ms p50, 24.8 p95 (second hand of a 2-hand job waits for the first) |
| — inference wall / thread CPU / runqueue wait | 19.2 / 15.6 / 2.6 ms p50 |
| — worker end → tracker collects | **14.2 ms p50, 36 p95** (collected only once per camera frame) |
| Orientation motion lag (metric 3) | **170–185 ms** → filtering/glide ≈ 100–115 ms of it |
| Age at send (metric 2), orientation | 110–126 ms p50, 470–560 p95 |
| Heat | mean 84 °C, max 92; tracker paused its own MediaPipe at 94 °C twice |

### Filter ablation (right hand only, live profile switching, 40 s each)

| Profile | Orientation lag | Still-hand jitter p95 (deg/frame) |
|---|---|---|
| original (axis 1.0/0.3, glide 40 ms) | 175 / 170 ms | 4.1 / 2.4 |
| glide off | 140 | 5.3 |
| axis 3.0/0.5, glide off | 105 | 8.2 |
| axis filter off, glide off | **95** (floor) | **17.7** (unusable) |
| axis 3.0/0.5, glide 15, curl 2.5/1.5 | 130 | 13.0 |

User verdict: the original is the smoothest and preferred for slow reaching/grabbing; lighter ones
felt snappier but jittery.

### Adaptive profiles + early collect (right hand only)

| Profile (all with `--merc-early`) | Orientation lag | Still jitter p95 |
|---|---|---|
| original | 155 / 170 ms | 5.0 / 3.2 |
| **A: axis 0.4/1.2, glide 25** | **130** | **4.3** |
| B: axis 0.3/2.0, glide 25 (user was slapping) | 115 | 4.6 |

`--merc-early`: worker end → collect **14.2 → 2.0 ms p50** (p95 36 → 8–9); new-result latency
**60 → 49–50 ms p50** (p95 82 → 68). One hand: 19 → 24 accepted observations/s.

### Slap test (right hand, profile A, 60 s each)

| | old | `--max-speed 8 --lk-levels 4 --merc-predict-roi` |
|---|---|---|
| Mercury rejected (not a hand) | 123 | 37 |
| position frozen ("hold") frames | 318 | 202 |
| rejected as implausible jump | 32 | **126** |
| single-camera position frames | 411 | **734** |

User: the fix was less choppy but lost depth (hand far away for a while) and orientation during
repeated slaps. **Not adopted** (left behind switches).

### Health (all runs)
Late-ISR events: +2 to +6 per run (38 already logged that day with tracking off); dropped compositor
frames 0–2 per run; compositor GPU p95 2.4–3.6 ms; no passthrough banding reported. Peak 92 °C.

## What changed (final state)

**On by default**
- `--merc-early` (in `start_args`): collect a finished Mercury job and submit the next between camera
  frames, not once per frame. No visual change.
- Smoothing profile **A** is the new default (`PROFILE_DEFAULTS` in `live.py`).
- One Euro ignores same-time / older samples (were producing negative-dt velocity spikes when a slow
  MediaPipe result arrived after a newer Mercury one, and double-feeding on re-labelling).
- Telemetry: `R` rows per Mercury observation, `P` rows on profile change, `merc`/`mp` backend labels
  (were all labelled `mp_*`). Report: per-stage latency, motion lag, still jitter, `--seg` by profile.
- `tools/health.sh`: temperature, XRService late-ISR count, compositor dropped frames / GPU p95.
- `start.sh` now warns when a tracker is already running and the given args are NOT applied.

**Available, off by default**
- `~/.config/mrhand/profile` (live): `axis_filter`, `centre_filter`, `curl_filter`, `pos_filter`
  (`<min_cutoff> <beta>` or `off`), `rot_glide_ms`. `tools/profile_cycle.sh` cycles test sets.
- `--merc-sched perhand` (one hand per Mercury job, least recently served first) — **untested live**.
- `--max-speed`, `--lk-levels`, `--merc-predict-roi` — tested, not adopted.

## Rollback
- Previous smoothing: `cp ~/handtrack/src/tools/profile_smooth_v1.conf ~/.config/mrhand/profile`.
- Previous tracker code: `~/handtrack/backup/pre-latency2-20261011-1233/` (copy back into `~/handtrack/src/`).
- Previous start args: `echo "--backend mercury" > ~/handtrack/start_args`.
- PC driver unchanged in this pass.

## Process issues (honest record)
- One run's `--telemetry` path was ignored because a tracker started from the panel was already
  running; `start.sh` silently kept it. Fixed (warning). That run's data was recovered from the default
  file using the profile markers.
- A test start that showed as cancelled had actually run while the user was away.

## Commands
```
~/handtrack/start.sh --backend mercury --merc-early --telemetry ~/handtrack/data/lat2/X.csv
~/handtrack/src/tools/health.sh 300 ~/handtrack/data/lat2/X_health.csv &
~/handtrack/src/tools/profile_cycle.sh 45 adaptive &
cd ~/handtrack/src && ../venv/bin/python tools/telemetry_report.py ~/handtrack/data/lat2/X.csv --seg -2
cd ~/handtrack/src && ../venv/bin/python tools/telemetry_report.py ~/handtrack/data/lat2/X.csv --seg N
```

## Not done from the brief (untested proposals)
Per-hand scheduling live test; MediaPipe discovery suppression / Mercury detector for discovery;
in-process ORT worker (GIL release) and shared-memory frame slots; native C++ optical flow with
shared pyramids; SCHED_IDLE vs bounded SCHED_OTHER A/B; ORT profiling, XNNPACK build, INT8
quantization, ncnn Vulkan; PC interpolation 17 → 8 → 0 ms and short-horizon prediction; per-component
source timestamps in the UDP packet; kinematic fit; external motion-to-photon measurement.
See RESEARCH_BRIEF_V3.md.
