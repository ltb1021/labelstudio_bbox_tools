#!/usr/bin/env bash
set -euo pipefail

# 사용 전 예시:
#   export LEFT_VIDEO="/path/to/yolo11_visualized.mp4"
#   export RIGHT_VIDEO="/path/to/rfdetr_visualized.mp4"
#   export OUT_VIDEO="/path/to/compare_yolo11_rfdetr.mp4"
#   ./examples/scripts/10_video_compare_dry_run.sh

LEFT_VIDEO="${LEFT_VIDEO:?set LEFT_VIDEO}"
RIGHT_VIDEO="${RIGHT_VIDEO:?set RIGHT_VIDEO}"
OUT_VIDEO="${OUT_VIDEO:?set OUT_VIDEO}"

lsbbox-video-compare \
  --left-video "$LEFT_VIDEO" \
  --right-video "$RIGHT_VIDEO" \
  --out-video "$OUT_VIDEO" \
  --left-title "${LEFT_TITLE:-YOLO11}" \
  --right-title "${RIGHT_TITLE:-RF-DETR}" \
  --max-frames "${MAX_FRAMES:-300}"
