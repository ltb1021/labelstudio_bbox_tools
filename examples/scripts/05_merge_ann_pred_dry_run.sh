#!/usr/bin/env bash
set -euo pipefail

# sources-json 예시:
# [
#   {"kind": "pred", "model_ver": "model_version_a", "score_thr": 0.5},
#   {"kind": "pred", "model_ver": "model_version_b", "score_thr": 0.5}
# ]
lsbbox-merge-ann-pred \
  --project-id 123 \
  --sources-json "/path/to/sources.json" \
  --mode prediction \
  --new-model-ver "merged_prediction_name" \
  --meta-tag "merged_prediction_name" \
  --resolve keep_earlier \
  --iou-thr-base 0.5 \
  --max-tasks 20 \
  --dry-run
