#!/bin/bash
# Fetch the inference models into ~/handtrack/models (run on the headset).
#  - MediaPipe hand_landmarker.task (Apache-2.0, Google)
#  - Monado Mercury models from gitlab.freedesktop.org/monado/utilities/hand-tracking-models @37a8a81b
#    LICENSE UNVERIFIED (that repo has no license file): local evaluation only.
set -euo pipefail
M=~/handtrack/models; mkdir -p $M/mercury
curl -sSLo $M/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
for f in grayscale_keypoint_jan18.onnx grayscale_detection_160x160.onnx; do
  curl -sSLo $M/mercury/$f "https://gitlab.freedesktop.org/monado/utilities/hand-tracking-models/-/raw/37a8a81b/$f"
done
sha256sum $M/hand_landmarker.task $M/mercury/*.onnx
echo "expected: keypoint 40c0daa598cedb993b54fff17685231b7465d6db342656c401b01c2029efd1d5"
echo "          detector 1f1a039a266e13dc186bb884430ebd9c8216bdda680ab08a533d4c671f27ed36"
