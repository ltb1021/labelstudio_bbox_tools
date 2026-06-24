#!/usr/bin/env bash
set -euo pipefail

# 사용 전 예시:
#   conda activate ltb_rfdetr
#   export INPUT_PATH="/path/to/video_or_video_folder"
#   export OUT_DIR="/path/to/video_inference_outputs"
#   export RFDETR_WEIGHTS="/path/to/checkpoint_best_total.pth"
#   export CLASS_YAML="/path/to/data.yaml"
#   ./examples/scripts/09_video_infer_rfdetr_dry_run.sh

INPUT_PATH="${INPUT_PATH:?set INPUT_PATH}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
RFDETR_WEIGHTS="${RFDETR_WEIGHTS:?set RFDETR_WEIGHTS}"
CLASS_YAML="${CLASS_YAML:?set CLASS_YAML}"

lsbbox-video-infer-rfdetr \
  --input-path "$INPUT_PATH" \
  --out-dir "$OUT_DIR" \
  --weights "$RFDETR_WEIGHTS" \
  --class-yaml "$CLASS_YAML" \
  --model-variant "${MODEL_VARIANT:-medium}" \
  --max-videos "${MAX_VIDEOS:-1}" \
  --max-frames "${MAX_FRAMES:-300}"
