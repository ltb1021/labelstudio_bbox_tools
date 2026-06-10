#!/usr/bin/env bash
set -euo pipefail

# 사용 전 예시:
#   export SRC_ROOT="/path/to/images"
#   export LABEL_STUDIO_DOC_ROOT="/path/to/label_studio_data"
#   ./examples/scripts/01_import_images_dry_run.sh

SRC_ROOT="${SRC_ROOT:?set SRC_ROOT to an image folder}"
DOC_ROOT="${LABEL_STUDIO_DOC_ROOT:?set LABEL_STUDIO_DOC_ROOT}"
PROJECT_TITLE="${PROJECT_TITLE:-refactor_smoke_import}"

lsbbox-import-images \
  --src-root "$SRC_ROOT" \
  --doc-root "$DOC_ROOT" \
  --project-title "$PROJECT_TITLE" \
  --dry-run
