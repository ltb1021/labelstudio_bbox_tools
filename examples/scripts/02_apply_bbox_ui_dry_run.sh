#!/usr/bin/env bash
set -euo pipefail

lsbbox-apply-ui \
  --project-id 0 \
  --mmyolo-json "/share_ssd/ltb/Users/ltb/label_studio/박스_데이터셋_250507/박스_mmyolo/260609_형주책임님_small데이터셋_1차테스트용_export/annotations_mmyolo.json" \
  --shape bbox \
  --dry-run

