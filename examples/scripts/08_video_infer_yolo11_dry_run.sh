#!/usr/bin/env bash
set -euo pipefail

# 사용 전 예시:
#   conda activate ltb_ultra
#   export INPUT_PATH="/path/to/video_or_video_folder"
#   export OUT_DIR="/path/to/video_inference_outputs"
#   export YOLO_WEIGHTS="/path/to/yolo11_best.pt"
#   export CLASS_YAML="/path/to/data.yaml"
#   ./examples/scripts/08_video_infer_yolo11_dry_run.sh

INPUT_PATH="${INPUT_PATH:?set INPUT_PATH}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:?set YOLO_WEIGHTS}"
CLASS_YAML="${CLASS_YAML:?set CLASS_YAML}"

lsbbox-video-infer-yolo11 \
  --input-path "$INPUT_PATH" \
  --out-dir "$OUT_DIR" \
  --weights "$YOLO_WEIGHTS" \
  --class-yaml "$CLASS_YAML" \
  --max-videos "${MAX_VIDEOS:-1}" \
  --max-frames "${MAX_FRAMES:-300}"
