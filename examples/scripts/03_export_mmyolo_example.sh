#!/usr/bin/env bash
set -euo pipefail

# 사용 전 예시:
#   export PROJECT_ID=123
#   export ANN_USER_ID=1
#   export OUT_DIR="/path/to/export/output"
#   ./examples/scripts/03_export_mmyolo_example.sh

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
ANN_USER_ID="${ANN_USER_ID:?set ANN_USER_ID}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"

lsbbox-export \
  --project-id "$PROJECT_ID" \
  --out-dir "$OUT_DIR" \
  --export-type ann \
  --ann-format mmyolo \
  --source-type ann \
  --ann-user-id "$ANN_USER_ID" \
  --ann-min-lead 0
