#!/bin/bash
# Build VK_LAYER_MRHAND (vrcompositor camera-buffer registry) and ptgrab (camera frame grabber).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
g++ -std=c++17 -O2 -g -fPIC -shared -fvisibility=hidden -Wall -Wno-missing-field-initializers \
    handcam_layer.cpp -o build/libVkLayer_mrhand.so -lpthread
echo "built build/libVkLayer_mrhand.so"
g++ -std=c++17 -O2 -Wall ptgrab.cpp -o build/ptgrab -lvulkan -lpthread
echo "built build/ptgrab"
