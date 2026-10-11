#!/bin/bash
# Start the hand-tracking pipeline: ptgrab (camera frames), live (tracking + PC bridge),
# and the independent thermal watchdog.  Args are passed to handtrack.live; with no args the
# defaults in ~/handtrack/start_args are used.
# The tracker runs under SCHED_IDLE on HANDTRACK_CPUS (default 0-4) so XRService/compositor
# always win the CPU (passthrough banding was traced to tracker CPU contention).
cd ~/handtrack/src
R=$XDG_RUNTIME_DIR/mrhand; mkdir -p $R
ARGS="$*"
[ -z "$ARGS" ] && [ -f ~/handtrack/start_args ] && ARGS=$(cat ~/handtrack/start_args)
export HANDTRACK_CPUS=${HANDTRACK_CPUS:-0-4}
if ! kill -0 "$(cat $R/ptgrab.pid 2>/dev/null)" 2>/dev/null; then
  DISABLE_MRCHROMA=1 DISABLE_MRHAND=1 nohup nice -n 10 taskset -c 0-6 capture/build/ptgrab stream > /tmp/ptgrab.log 2>&1 &
  echo $! > $R/ptgrab.pid; sleep 1
fi
if ! kill -0 "$(cat $R/live.pid 2>/dev/null)" 2>/dev/null; then
  nohup chrt --idle 0 ../venv/bin/python -m handtrack.live --no-http $ARGS > /tmp/live.log 2>&1 &
  echo $! > $R/live.pid
elif [ -n "$*" ]; then
  # a tracker is already running (e.g. started from the dashboard panel): these args are NOT applied
  echo "WARNING: tracker already running (pid $(cat $R/live.pid)); args NOT applied: $* -- run stop.sh first" >&2
  ARGS="(already running, args not applied)"
fi
if ! kill -0 "$(cat $R/watchdog.pid 2>/dev/null)" 2>/dev/null; then
  nohup ../venv/bin/python -m handtrack.watchdog > /tmp/watchdog.log 2>&1 &
  echo $! > $R/watchdog.pid
fi
echo "running: ptgrab $(cat $R/ptgrab.pid)  live $(cat $R/live.pid)  watchdog $(cat $R/watchdog.pid)  (tracker args: $ARGS, cpus $HANDTRACK_CPUS, SCHED_IDLE)"
