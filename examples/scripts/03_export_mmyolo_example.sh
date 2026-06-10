#!/usr/bin/env bash
set -euo pipefail

# Fill PROJECT_ID and ANN_USER_ID before running.
PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
ANN_USER_ID="${ANN_USER_ID:?set ANN_USER_ID}"

lsbbox-export \
  --project-id "$PROJECT_ID" \
  --out-dir "/share_ssd/ltb/Users/ltb/label_studio/260610_labelstudio_export_refactoring_test" \
  --export-type ann \
  --ann-format mmyolo \
  --source-type ann \
  --ann-user-id "$ANN_USER_ID" \
  --ann-min-lead 0

