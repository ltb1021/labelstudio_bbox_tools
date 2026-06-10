#!/usr/bin/env bash
set -euo pipefail

# 사용 전 예시:
#   export MMYOLO_JSON="/path/to/annotations_mmyolo.json"
#   ./examples/scripts/02_apply_bbox_ui_dry_run.sh

MMYOLO_JSON="${MMYOLO_JSON:?set MMYOLO_JSON to a COCO/MMYOLO json file}"
PROJECT_ID="${PROJECT_ID:-0}"

lsbbox-apply-ui \
  --project-id "$PROJECT_ID" \
  --mmyolo-json "$MMYOLO_JSON" \
  --shape bbox \
  --dry-run
