#!/usr/bin/env bash
set -euo pipefail

# MMYOLO JSON을 기존 Label Studio task에 matching만 해보는 예시입니다.
lsbbox-import-annotations \
  --project-id 123 \
  --ann-source "/path/to/annotations_mmyolo.json" \
  --mirror-root "/path/to/dataset_root" \
  --image-match-mode fullpath \
  --upload-mode prediction \
  --meta-tag "my_annotation_import" \
  --import-id "my_annotation_import" \
  --pred-model "my_annotation_import" \
  --batch-size 200 \
  --dry-run
