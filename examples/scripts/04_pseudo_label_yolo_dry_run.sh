#!/usr/bin/env bash
set -euo pipefail

# 실제 업로드 없이 project/task/image 경로와 class 설정만 확인하는 예시입니다.
lsbbox-pseudo-label-yolo \
  --project-id 123 \
  --weights "/path/to/yolo11_best.pt" \
  --class-yaml "/path/to/data.yaml" \
  --imgsz 640 \
  --conf 0.30 \
  --iou 0.60 \
  --batch-size 1 \
  --meta-tag "my_pseudo_label_run" \
  --import-id "my_import_id" \
  --pred-model "my_pseudo_label_run" \
  --max-tasks 20 \
  --dry-run
