#!/bin/bash
# Headset health log for latency experiments (read-only observation, no system changes).
#   tools/health.sh <seconds> [out.csv]
# Every ~10 s: hottest thermal zone, cumulative XRService "Late WM_DONE ISR" count in today's logs,
# compositor dropped / mispresented frames over 4 s (mrctl timing, if the mr-chroma helper exists).
# Stop/rollback conditions for an experiment: banding (look), late-ISR count rising, dropped frames.
secs=${1:-60}
out=${2:-$HOME/handtrack/data/health.csv}
MRCTL=$HOME/.local/share/mrchroma/mrctl
export LD_LIBRARY_PATH=/opt/steamvr/bin/linuxarm64
mkdir -p "$(dirname "$out")"
[[ -s "$out" ]] || echo "t_unix,max_temp_c,late_isr_total,frames,dropped,mispresented,compositor_gpu_p95_ms" > "$out"
end=$(( $(date +%s) + secs ))
while [[ $(date +%s) -lt $end ]]; do
  t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
  isr=$(cat "$HOME"/.local/share/Steam/logs/XRService-$(date +%Y.%m.%d)*/*.log \
            "$HOME"/.local/share/Steam/logs/XRService-$(date +%Y.%m.%d)*.log 2>/dev/null | grep -c "Late WM_DONE")
  frames=; dropped=; mis=; gpu=
  if [[ -x "$MRCTL" ]]; then
    tm=$("$MRCTL" timing 4 2>/dev/null)
    frames=$(sed -n 's/^frames=\([0-9]*\).*/\1/p' <<<"$tm")
    dropped=$(sed -n 's/.*dropped=\([0-9]*\).*/\1/p' <<<"$tm")
    mis=$(sed -n 's/.*mispresented=\([0-9]*\).*/\1/p' <<<"$tm")
    gpu=$(sed -n 's/.*compositor GPU ms *: p50 [0-9.]* *p95 \([0-9.]*\).*/\1/p' <<<"$tm")
  else
    sleep 4
  fi
  echo "$(date +%s),$((t / 1000)),$isr,$frames,$dropped,$mis,$gpu" >> "$out"
  sleep 6
done
