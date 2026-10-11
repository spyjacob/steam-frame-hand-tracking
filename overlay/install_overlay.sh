#!/bin/bash
# Build + register the "Hand Tracking" dashboard panel (user scope, removable).
#   install_overlay.sh            build, install, register manifest, enable autolaunch, start it
#   install_overlay.sh --remove   unregister, stop, delete installed files
set -euo pipefail
cd "$(dirname "$0")"
DEST="$HOME/.local/share/handtrack/overlay"
MANIFEST="$DEST/handtrack_overlay.vrmanifest"
SDK=/opt/steamvr/tools/hellovr_vulkan_linux/src/openvr/headers
LIB=/opt/steamvr/bin/linuxarm64
export LD_LIBRARY_PATH=$LIB

if [[ "${1:-}" == "--remove" ]]; then
  [[ -x "$DEST/handtrack_reg" ]] && "$DEST/handtrack_reg" remove "$MANIFEST" handtrack.toggle || true
  pkill -x handtrack_overl || true
  rm -rf "$DEST"
  echo "hand tracking panel removed"
  exit 0
fi

mkdir -p "$DEST"
g++ -O2 -std=c++17 -o "$DEST/handtrack_overlay" handtrack_overlay.cpp -I$SDK $LIB/libopenvr_api.so -Wl,-rpath,$LIB \
    -pthread $(pkg-config --cflags --libs freetype2)
g++ -O2 -std=c++17 -o "$DEST/handtrack_reg" handtrack_reg.cpp -I$SDK $LIB/libopenvr_api.so -Wl,-rpath,$LIB
cp icon.png "$DEST/"
cat > "$MANIFEST" <<EOF
{
  "source": "user",
  "applications": [{
    "app_key": "handtrack.toggle",
    "launch_type": "binary",
    "binary_path_linux": "$DEST/handtrack_overlay",
    "binary_path_linux_arm": "$DEST/handtrack_overlay",
    "arguments": "",
    "is_dashboard_overlay": true,
    "image_path": "$DEST/icon.png",
    "strings": { "en_us": { "name": "Hand Tracking", "description": "Turn optical hand tracking on/off" } }
  }]
}
EOF
"$DEST/handtrack_reg" add "$MANIFEST" handtrack.toggle
pkill -x handtrack_overl && sleep 1 || true
(nohup "$DEST/handtrack_overlay" >/tmp/handtrack_overlay.log 2>&1 &)
echo "installed; Hand Tracking panel running (log /tmp/handtrack_overlay.log)"
