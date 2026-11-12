#!/bin/bash
# Start the hand-tracking pipeline: ptgrab (camera frames), the v4 tracker (handtrack.main: tracking +
# PC bridge) and the independent thermal watchdog.  Args are passed to handtrack.main; with no args
# the lines of ~/handtrack/start_args_v4 are used (# comments allowed).  The v3 file
# ~/handtrack/start_args is NOT read (its flags belong to the old tracker on the main branch).
#
# CPU placement / scheduling (env var, else file, else default):
#   HANDTRACK_CPUS   or ~/handtrack/cpus    cores for the tracker, e.g. 0-4 (default) or 2-6 (no little cores)
#   HANDTRACK_SCHED  or ~/handtrack/sched   idle (default: SCHED_IDLE, XRService/compositor always win the
#                                           CPU; passthrough banding was traced to tracker CPU contention)
#                                           or other (normal scheduling at nice 10: faster inference, watch
#                                           passthrough for banding)
#   HANDTRACK_HTTP=1                        debug camera view on the headset's 127.0.0.1:8765 only
#                                           (from the PC: ssh -L 8765:127.0.0.1:8765 frame, open localhost:8765)
cd ~/handtrack/src
R=$XDG_RUNTIME_DIR/mrhand; mkdir -p $R
ARGS="$*"
[ -z "$ARGS" ] && [ -f ~/handtrack/start_args_v4 ] && ARGS=$(grep -v '^[[:space:]]*#' ~/handtrack/start_args_v4 | tr '\n' ' ')
[ -z "$HANDTRACK_CPUS" ] && [ -f ~/handtrack/cpus ] && HANDTRACK_CPUS=$(head -1 ~/handtrack/cpus)
[ -z "$HANDTRACK_SCHED" ] && [ -f ~/handtrack/sched ] && HANDTRACK_SCHED=$(head -1 ~/handtrack/sched)
export HANDTRACK_CPUS=${HANDTRACK_CPUS:-0-4}
HANDTRACK_SCHED=${HANDTRACK_SCHED:-idle}
HTTP=--no-http; VIEW=""
[ "$HANDTRACK_HTTP" = 1 ] && HTTP="" && VIEW=", debug view on 127.0.0.1:8765"
case "$HANDTRACK_SCHED" in
  other) SCHED="" ;;
  *)     SCHED="chrt --idle 0"; HANDTRACK_SCHED=idle ;;
esac
if ! kill -0 "$(cat $R/ptgrab.pid 2>/dev/null)" 2>/dev/null; then
  DISABLE_MRCHROMA=1 DISABLE_MRHAND=1 nohup nice -n 10 taskset -c 0-6 capture/build/ptgrab stream > /tmp/ptgrab.log 2>&1 &
  echo $! > $R/ptgrab.pid; sleep 1
fi
if ! kill -0 "$(cat $R/live.pid 2>/dev/null)" 2>/dev/null; then
  # live.pid / /tmp/live.log keep their names: the dashboard panel reads them
  nohup $SCHED ../venv/bin/python -m handtrack.main $HTTP $ARGS > /tmp/live.log 2>&1 &
  echo $! > $R/live.pid
else
  # a tracker is already running (e.g. started from the dashboard panel): new settings are NOT applied
  if [ -n "$*" ]; then
    echo "WARNING: tracker already running (pid $(cat $R/live.pid)); args NOT applied: $* -- run stop.sh first" >&2
    ARGS="(already running, args not applied)"
  fi
  [ -n "$VIEW" ] && VIEW=", debug view NOT applied (tracker already running; run stop.sh first)"
fi
if ! kill -0 "$(cat $R/watchdog.pid 2>/dev/null)" 2>/dev/null; then
  nohup ../venv/bin/python -m handtrack.watchdog > /tmp/watchdog.log 2>&1 &
  echo $! > $R/watchdog.pid
fi
echo "running: ptgrab $(cat $R/ptgrab.pid)  tracker v4 $(cat $R/live.pid)  watchdog $(cat $R/watchdog.pid)  (args: ${ARGS:-none}, cpus $HANDTRACK_CPUS, sched $HANDTRACK_SCHED$VIEW)"
