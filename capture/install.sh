#!/bin/bash
# Install VK_LAYER_MRHAND (user scope). Takes effect at the next vrcompositor start,
# and only while ~/.config/mrhand/enable-capture exists.
set -euo pipefail
cd "$(dirname "$0")"
DEST="$HOME/.local/share/mrhand"
MAN="$HOME/.local/share/vulkan/implicit_layer.d/mrhand.json"
mkdir -p "$DEST" "$(dirname "$MAN")" "$HOME/.config/mrhand"
install -m 755 build/libVkLayer_mrhand.so "$DEST/"
cat > "$MAN" <<JSON
{
  "file_format_version": "1.2.0",
  "layer": {
    "name": "VK_LAYER_MRHAND",
    "type": "GLOBAL",
    "library_path": "$DEST/libVkLayer_mrhand.so",
    "api_version": "1.4.309",
    "implementation_version": "1",
    "description": "Passive camera-buffer registry for handtrack (loads only into vrcompositor)",
    "functions": { "vkNegotiateLoaderLayerInterfaceVersion": "mrhand_NegotiateLoaderLayerInterfaceVersion" },
    "disable_environment": { "DISABLE_MRHAND": "1" }
  }
}
JSON
echo "$(date -Is) install: VK_LAYER_MRHAND manifest $MAN" >> "$HOME/handtrack/CHANGES.log"
echo "installed. enable with: touch ~/.config/mrhand/enable-capture ; then restart SteamVR"
