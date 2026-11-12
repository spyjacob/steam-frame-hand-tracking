# v4 tracker: deploy and first human test

This is for Claude (or a person) on the PC that reaches the headset with `ssh frame`, working with the
user, who wears the Frame. It takes about 45 minutes. The code is on branch `v4-single-tracker`, and the
previous tracker (v3) is on `main`. Rollback is in section 8.

## 1. What changes

| | v3 (`main`) | v4 (this branch) |
|---|---|---|
| Tracking | MediaPipe on both cameras + optical-flow fast path + Mercury refinement + stacked smoothing | Mercury keypoint model on each hand, at most 30×/s per hand → rigid palm fit → one Kalman filter per hand (position + rotation) |
| Finding hands | MediaPipe always running | only while a hand is missing (MediaPipe in its own process + Mercury detector); slower after 2 s |
| Process | `python -m handtrack.live` | `python -m handtrack.main` |
| Arguments | `~/handtrack/start_args` | `~/handtrack/start_args_v4` (empty = defaults); live tuning in `~/.config/mrhand/v4.conf` |
| Telemetry | `~/handtrack/data/telemetry.csv` | `~/handtrack/data/telemetry_v4.csv`; summary: `tools/v4_report.py` |

These are **unchanged**:

- ptgrab and the Vulkan layer
- the watchdog (≥93 °C for 3 s)
- the dashboard panel
- the `/tmp/live.log` and `live.pid` names
- UDP protocol v3 on port 47811, with the same palm frame and quaternion

The PC driver needs no rebuild or reinstall.

**Checked without the headset (dev Mac):**

- A closed-loop simulation of the real tracker (numbers in 3.4).
- The real Mercury models on life-size hand photos. Acquired at frame 4, left/right correct, position within about 1.4 cm, still-hand step p50 0.3 mm / 0.2°.
- `handtrack.main` end to end with a fake camera and a fake PC driver: 60 frames/s, 60 poses/s, valid v3 packets.

**Not checked:** anything on the Frame. That means real camera noise, inference time under SCHED_IDLE, heat and banding.

## 2. Rules for this test

- Change only files under `~/handtrack/`. Do not touch XRService, headset tracking, the compositor, the
  mr-chroma project or OS protections.
- Log every headset change in `~/handtrack/CHANGES.log` (`echo "$(date -Is) what" >> ~/handtrack/CHANGES.log`).
- Heat:
  - Check the hottest zone before each step and wait if it is above 85 °C:
    `cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1` (m°C).
  - The tracker slows itself at 88/90/92 °C, and the watchdog stops it at ≥93 °C for 3 s.
  - If the watchdog trips, let the headset cool and continue with one hand.
- Camera images (debug view, snapshots) show the user and their room. Keep them on the PC, and do not
  commit or upload them.
- If passthrough shows banding or SteamVR misbehaves, run `~/handtrack/stop.sh` first, then investigate.

## 3. Deploy

### 3.1 PC (in this repo)

```
git fetch origin
git checkout v4-single-tracker
git log -1 --format=%h                  # commit id, for the change log
git archive -o handtrack-v4.tar HEAD    # tracked files only: no images, models or build outputs
scp handtrack-v4.tar frame:/tmp/
```

### 3.2 Headset (`ssh frame`): stop, back up, install

Run this block in one shell session, because `STAMP` is a shell variable.

```
~/handtrack/stop.sh
cd ~/handtrack && STAMP=$(date +%Y%m%d-%H%M) && mkdir -p backup && echo $STAMP
tar -czf backup/pre-v4-$STAMP.tgz --exclude=__pycache__ --exclude=src/data --exclude=src/capture/build \
    src $(ls start.sh stop.sh start_args cpus sched 2>/dev/null)
ls -lh backup/pre-v4-$STAMP.tgz
tar -xf /tmp/handtrack-v4.tar -C ~/handtrack/src
cp src/headset/start.sh src/headset/stop.sh src/headset/start_args_v4 ~/handtrack/ && chmod +x start.sh stop.sh
echo "$(date -Is) v4 tracker installed (v4-single-tracker @<commit>): start.sh runs handtrack.main; backup backup/pre-v4-$STAMP.tgz; rollback src/docs/V4_TEST_PLAN.md section 8" >> CHANGES.log
```

What extracting over `src` does and doesn't touch:

- It overwrites tracked files and adds the new ones.
- The v3 files (`handtrack/live.py`, …) stay on disk, unused.
- `capture/build` (the built layer and ptgrab) is not touched, and this branch does not change `capture/`.
- The v3 `~/handtrack/start_args` stays where it is for the rollback.

Write the `STAMP` down; the rollback needs it.

### 3.3 Check the environment

```
cd ~/handtrack/src
../venv/bin/python -c "import sys,numpy,cv2,onnxruntime,mediapipe; print(sys.version.split()[0], numpy.__version__, cv2.__version__, onnxruntime.__version__, mediapipe.__version__, hasattr(cv2,'solvePnPGeneric'))"
../venv/bin/python -c "import handtrack.main, handtrack.mpworker; print('imports ok')"
sha256sum ~/handtrack/models/mercury/*.onnx
```

What to expect:

- Python 3.9 or newer, with `True` at the end of the first line.
- `imports ok`.
- Model checksums: keypoint `40c0daa5…`, detector `1f1a039a…` (full values in `tools/fetch_models.sh`; if the models are missing, run it).

### 3.4 Quick checks (tracker stopped, about a minute)

```
../venv/bin/python tools/sim_tracker.py still two
HANDTRACK_CPUS=0-4 chrt --idle 0 ../venv/bin/python tools/bench_models.py --seconds 3
```

`sim_tracker.py` runs the real tracker code in real time against synthetic hands. On the dev Mac it gave:

```
scenario hand  valid%  first  pos50  pos95  rot50  rot95  jit_mm  jit_deg    lag jump_mm   curl  swap
still    left    98.6   0.08   0.77   2.18    6.5   13.2    6.76     5.45    nan    34.6   0.02     0
two      left    99.0   0.08   2.24   5.74    8.4   15.4     nan      nan     33    93.8   0.06     0
two      right   97.5   0.20   1.27   2.98    7.1   12.3     nan      nan      0    44.0   0.02     0
```

On the headset, `swap` must be 0 and valid% ≥ 97; other values up to about 1.5× these are fine. The sim's `jit_*`
includes the first second while the filter settles. An exception here means a Python, numpy or OpenCV
difference: stop and report it.

`bench_models.py` times each piece of the tracker with the tracker's own CPU placement and SCHED_IDLE.
v3 measurements on the Frame (REPORT.md) put a Mercury keypoint crop at 9.5–16 ms at normal priority
and 19–30 ms under SCHED_IDLE, and the detector at about 6 ms. Two hands at 30/s need about 60 crops/s,
so `two crops at once … crops/s max` should be above 70. If it is lower, plan on T11 (normal scheduling)
or `--set hz=20`.

## 4. First start

**Starting it.** Either:

- **Normal:** in the headset, dashboard → **Hand Tracking** → **RIGHT** on.
- **With the debug view** (recommended for the first run): `HANDTRACK_HTTP=1 ~/handtrack/start.sh` over ssh,
  and the user switches hands in the panel. The tracker is already running, so the panel only changes which
  hands are on. Turning both hands off still stops everything.

**Startup lines.** Watch with `tail -f /tmp/live.log`. Within a few seconds these lines appear:

```
v4 tracker: pid …, cpus [0, 1, 2, 3, 4], policy IDLE
params: defaults
models: keypoint + detector + MediaPipe; 2 workers; shm 640x640
hands enabled: right
```

**Status line.** One appears every 5 s. It begins `[T C] fast:` (the panel reads the temperature from it).
Fields are per 5 s:

| Field | Expect | If not |
|---|---|---|
| `fast: N frames/s` | ~60 | 0 or missing: ptgrab is not delivering (`/tmp/ptgrab.log`; is SteamVR running, with `~/.config/mrhand/enable-capture` present?) |
| `ms/frame` | < 3 | the main loop is starved (CPU contention) |
| `poses/s L / R` | ~60 for each hand that is ON and in view | see `missed` and `tracked` |
| `latency p50/p95` | p50 ≲ 15 ms | time from frame arrival (ptgrab's timestamp) to UDP send |
| `crops N/s M ms` | ~30/s per tracked hand, plus a few while searching; M as measured by the bench | M much above the bench: CPU contention |
| `accepted (stereo)` | most crops accepted; a few stereo while the palm size is learned, then one every 3 s | — |
| `missed`, `outliers` | small compared with `accepted` | see section 6 |
| `obs age` | ≈ crop time + 5–15 ms | age of an observation when the filter uses it |
| `tracked: right(30/s,scale 1.0x)` | 25–30/s; `scale` 0.85–1.25 for an adult hand once learned | a scale stuck far from 1 means the depth is off: the hand appears too near or too far |
| `step mm p50/p95` | still hand: p50 < 1, p95 < 4 | jitter: see T1 and section 7 |
| `search: det mp verify acquired lost` | finders run only while a hand is missing | — |
| `scale` (at the end) | 1.0 (0.7/0.5/0.3 when hot) | thermal guard |

An `errors (n): …` line or a Python traceback means something broke. Copy it into the report. A
traceback at start means stop and roll back (section 8).

**The debug view.** It needs `HANDTRACK_HTTP=1`.

- On the PC, run `ssh -N -L 8765:127.0.0.1:8765 frame`.
- Open http://localhost:8765 for the stream, or save a single frame: `curl -s -o snap.jpg http://localhost:8765/snap.jpg`.
  Look at the frame, then delete it.
- What it shows: both cameras upright, left camera on the left.
  - Dots are the latest keypoints (left hand orange, right hand blue).
  - The white cross with `side x y z cm` is the pose sent to the PC.
  - Green circles are hand-finder candidates being checked.
- Frames are drawn only while someone is watching, so close it for latency judgements.

## 5. Human test

Set up as follows:

- Run a PC VR app that shows the Frame controllers or hands, such as SteamVR Home.
- Put the controllers down so that the hands take over.
- Use the right hand only unless a test says otherwise.

After each test, run `cd ~/handtrack/src && ../venv/bin/python tools/v4_report.py --last <seconds of the test>` on the
headset, and write down the report together with what the user says.

| # | Do | Looks right when | Numbers (`v4_report.py`) |
|---|---|---|---|
| T1 still | Hand still at ~40 cm, palm away from you, 15 s; then palm toward you, 15 s | the virtual hand is steady, with no shimmer and no slow wobble | step position p50 ≤ 1 mm, p95 ≤ 4 mm; rotation p95 well under v3's 4.3°/frame, ideally ≤ 2° |
| T2 slow | Slow reaches forward/back, left/right and up/down, plus slow wrist turns, 20 s | it follows with no visible delay and no stepping | sent ≥ 98 %, no gap > 0.1 s |
| T3 fast | Fast punches, waves, flicks and wrist rotations, 20 s | it keeps up; at worst a brief freeze on the fastest flicks, and it never flies off or jumps somewhere else | longest gap < 0.5 s; `restarted` 0–2 |
| T4 out / back | Hand out of view to your side, wait 2 s, bring it back. 5 times, plus at the edges of the view | it disappears about 0.25 s after leaving and reappears within about 0.3–0.5 s of being back in view | acquired ≈ lost ≈ 5 |
| T5 two hands | Both hands ON. Hold both up 10 s; cross one slowly in front of the other; clap; hold palms together 2 s | left stays left and right stays right. A hidden hand may freeze for up to 0.6 s, but never jumps onto the other hand | `duplicate dropped` ≈ number of palm-together holds; no swaps |
| T6 orientation | Palm down, palm up, palm toward you, thumb up (edge-on) and fist, each held 3 s; then turn between them | the virtual hand matches, with no 180° flips | rotation p95 during holds ≤ 2–3° |
| T7 fingers | Open and close the fist slowly, then fast; move single fingers | the curls follow (the mapping is provisional; the thumb is weakest) | — |
| T8 controller | Pick up the right controller and use it, set it down; 3 times | the controller pose while it is held; the hand takes over when it is set down; no fight between them | — (PC driver logic, unchanged) |
| T9 heat | Both hands, normal use for 10 min; run `~/handtrack/src/tools/health.sh 600` in a second ssh | the panel temperature stays below 90 °C and the watchdog never stops tracking | status `scale` stays 1.0 or 0.7 |
| T10 banding | Look at passthrough for 1 min with tracking on, then with it off | no vertical banding in either | — |
| T11 (optional) normal scheduling | `~/handtrack/stop.sh; HANDTRACK_SCHED=other ~/handtrack/start.sh`, then repeat T1, T3 and T10 | it feels snappier, with no banding | `crops … ms` and `obs age` lower than under SCHED_IDLE |

For T11: if banding appears, stop and restart without `HANDTRACK_SCHED`, which is the default SCHED_IDLE. If it is
clean and better, make it permanent with `echo other > ~/handtrack/sched`, and log the change.

For a direct comparison with v3, roll back (section 8), repeat T1–T3, then reinstall v4. To reinstall, run the
`tar -xf` and `cp` lines of 3.2 again; the archive is still in `/tmp`. Log both changes.

## 6. Reading the report

`tools/v4_report.py [--last S | --from T --to T] [--run N]` summarises the newest tracker run, meaning the time since
the tracker last started (`--run -2` is the run before). It shows, per hand:

- **sent %**, plus **gaps**: frames with no pose while the hand was ON (a hand out of view counts too).
- **frame-to-frame step**: position (mm) and rotation (°). For a still hand this is the jitter. For a
  moving hand it is speed/60.
- **capture → sent**: ms from frame to UDP send.
- **observations/s**: the stereo count and **age**, the palm-fit **residual** (°) and the **heatmap**
  confidence (wrong or garbage crops score low).
- **rejected crops**: why a keypoint result was not used.
  - `notahand`: Mercury's is-hand below `is_hand`.
  - `fit`: the palm fit's residual is above `gate_deg` (blur, part of the hand out of view).
  - `other`: the crop saw the other hand (expected in T5).
  - `lowhm`: heatmap confidence fell and the result disagreed with the prediction (mirror image or other hand).
  - `outlier`: too far from the prediction.
- **acquired / lost / duplicate dropped / restarted**:
  - `restarted`: two agreeing outliers moved the track to where the hand really is. It should be rare.
  - `duplicate dropped`: two tracks within 2 cm for 0.25 s.
- **candidates rejected**: hand-finder candidates that the keypoint model refused, by finder and reason.

## 7. Tuning (live, no restart)

Put `key = value` lines in `~/.config/mrhand/v4.conf`. The tracker re-reads the file every second, and
`/tmp/live.log` shows `params: …` when it changes. Delete a line to return to the default. All keys and
defaults are listed in `handtrack/tracker.py` (`DEFAULTS`). Log each change in `CHANGES.log`.

| Symptom | Change |
|---|---|
| A still hand shimmers (T1 step p95 high) | `sa_lo = 0.15` and `ra_lo = 1.5` (more smoothing at rest); or `attack = 12` (a bigger surprise is needed before the filter opens) |
| The hand is late to start moving, or trails | `attack = 6`; `sa_lo = 0.6` |
| It overshoots after sudden stops | `release = 0.6` |
| It drops during fast moves (T3: `fit` / `notahand` rejections) | `gate_deg = 5`; `is_hand = 0.9`; `hold_s = 0.35` |
| It drops at the edge of view or when partly hidden | `hold_s = 0.35`; `drop_s = 0.6` |
| It is slow to reappear (T4) | `mp_hz = 20` (more heat); `show_after = 1` (more risk of a ghost hand) |
| Hands swap or the wrong side is picked (T5) | `hm_drop = 0.9`; `chir_margin = 0.08`; `same_hand_deg = 12` |
| One hand vanishes while the palms touch | `collapse_m = 0.015` |
| Too hot | `hz = 20`; `budget = 50`; or track one hand |
| Fingers jitter or lag | `curl_filter = 1.0, 1.5` (steadier) or `3.0, 3.0` (faster); this applies to the next new track |

`start_args_v4` takes command-line options for the next start: `--set key=value`, `--no-mediapipe` (only after
the status line shows the Mercury detector acquiring hands: `search: det … acquired …`) and `--workers 3`.

## 8. Rollback to v3

```
~/handtrack/stop.sh
cd ~/handtrack && tar -xzf backup/pre-v4-<STAMP>.tgz    # restores src/ plus the v3 start.sh, stop.sh, start_args
echo "$(date -Is) rolled back to v3 (backup/pre-v4-<STAMP>.tgz)" >> CHANGES.log
~/handtrack/start.sh                                    # or use the panel
```

The v4 files (`handtrack/main.py` …) stay in `src`, unused.

If the backup is missing, rebuild the v3 tree from `main`:

1. On the PC, check out `main` and run `git archive -o handtrack-main.tar HEAD`.
2. Copy the archive to the headset and extract it into `~/handtrack/src` the same way as in 3.2.
3. Run `cp src/headset/start.sh src/headset/stop.sh ~/handtrack/`. The v3 `start_args` was never touched.

## 9. Known risks

- **First run on real images.** If the real keypoint noise is much higher than the simulated 0.7°, the hand
  will shimmer. The simulation at 1.5° plus slow inference was unusable (p95 17 mm / 48° per frame). If T1
  looks like that, the problem is the crops, not the filter: get a snapshot and the T1 report.
- **SCHED_IDLE roughly doubles crop time.** Observations then arrive older, and fast moves suffer. T11 tests normal
  scheduling, which is what caused banding in v3, so watch for it.
- **The Mercury detector is unproven on Frame images.** MediaPipe remains the main finder.
- **Fast edge-on wrist turns:** the palm keypoints line up, and the hand may glide or freeze briefly.
- **Motion blur on very fast moves:** the hand glides, then is re-acquired.
- **Palm size** is learned from stereo in the first seconds of a track. Check `scale` in the status line.
- **Finger curl mapping** is provisional.

## 10. What to report back

Report the following:

- the commit id and backup name
- the 3.3, 3.4 and first status-line output
- for each test, the `v4_report.py` block and the user's words
- the maximum temperature and whether there was banding
- any tuning changes
- the user's verdict compared with v3

Images are not part of the report.
