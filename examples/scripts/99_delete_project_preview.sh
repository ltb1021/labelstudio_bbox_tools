#!/usr/bin/env bash
set -euo pipefail

# 위험 작업 preview 예시입니다. 실제 삭제하지 않습니다.
lsbbox-delete-project-safe \
  --project-id 123 \
  --page-size 1000 \
  --batch-size 1000 \
  --sleep-sec 0.15 \
  --dry-run
