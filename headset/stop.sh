#!/bin/bash
# Stop the hand-tracking pipeline (tracker, its MediaPipe worker, ptgrab, watchdog).
R=$XDG_RUNTIME_DIR/mrhand
names="live ptgrab"
[ "$1" = "--from-watchdog" ] || names="$names watchdog"
for n in $names; do f=$R/$n.pid; [ -f $f ] && kill "$(cat $f)" 2>/dev/null; rm -f $f; done
pkill -x ptgrab 2>/dev/null
# MediaPipe worker processes of the tracker (spawned children); match on our venv path only
for p in $(pgrep -f "handtrack/src/../venv/bin/python -c from multiprocessing"); do kill $p 2>/dev/null; done
echo stopped
