#!/bin/bash
# Cycle smoothing profiles through ~/.config/mrhand/profile (read live by handtrack.live, which writes a
# P row to telemetry on every change).  Ends by deleting the file = the default smooth profile.
#   tools/profile_cycle.sh [seconds_per_profile] [set]
#   set "filters"  (default): current, glide off, lighter axis filter, axis filter off, light mix, current
#   set "adaptive": current / A / current / B, A and B = smoother when still, faster when moving
per=${1:-40}
set=${2:-filters}
F=$HOME/.config/mrhand/profile
CUR="axis_filter = 1.0 0.3
curl_filter = 1.0 1.0
rot_glide_ms = 40"
if [[ $set == adaptive ]]; then
  profiles=(
    "$CUR"
    "axis_filter = 0.4 1.2
curl_filter = 1.0 1.0
rot_glide_ms = 25"
    "$CUR
# repeat"
    "axis_filter = 0.3 2.0
curl_filter = 1.0 1.0
rot_glide_ms = 25"
  )
else
  profiles=(
    "$CUR"
    "axis_filter = 1.0 0.3
curl_filter = 1.0 1.0
rot_glide_ms = 0"
    "axis_filter = 3.0 0.5
curl_filter = 1.0 1.0
rot_glide_ms = 0"
    "axis_filter = off
curl_filter = 1.0 1.0
rot_glide_ms = 0"
    "axis_filter = 3.0 0.5
curl_filter = 2.5 1.5
rot_glide_ms = 15"
    "$CUR
# repeat of profile 0"
  )
fi
for i in "${!profiles[@]}"; do
  printf '%s\n' "${profiles[$i]}" > "$F.tmp" && mv "$F.tmp" "$F"
  echo "$(date +%T) profile $i"
  sleep "$per"
done
rm -f "$F"
echo "$(date +%T) done (default profile)"
