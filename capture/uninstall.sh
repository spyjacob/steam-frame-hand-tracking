#!/bin/bash
# Remove VK_LAYER_MRHAND. Running compositor keeps it until SteamVR restarts.
rm -f "$HOME/.local/share/vulkan/implicit_layer.d/mrhand.json" "$HOME/.config/mrhand/enable-capture"
rm -f "$HOME/.local/share/mrhand/libVkLayer_mrhand.so"
echo "$(date -Is) uninstall: VK_LAYER_MRHAND removed" >> "$HOME/handtrack/CHANGES.log"
echo "removed; restart SteamVR to unload it from vrcompositor"
