#!/usr/bin/env bash
set -euo pipefail

lsbbox-import-images \
  --src-root "/share_ssd/ltb/Users/ltb/label_studio/운영서버_도메인맞춤_추가학습용/260608/오탐/쓰러짐/images" \
  --doc-root "/share_ssd/ltb/Users/ltb/label_studio" \
  --project-title "refactor_smoke_import" \
  --dry-run

