#!/usr/bin/env bash
set -euo pipefail

# 영상 파일 또는 영상 폴더를 실제 저장 없이 preview하는 예시입니다.
# --recursive를 붙이지 않으면 input 폴더 바로 아래의 영상 파일만 찾습니다.
lsbbox-extract-video-frames \
  --input-path "/path/to/video_or_video_folder" \
  --out-dir "/share_ssd/ltb/Users/ltb/label_studio/frames_export" \
  --interval-seconds 2.0 \
  --image-format jpg \
  --jpg-quality 95 \
  --max-videos 3 \
  --max-frames-per-video 20 \
  --dry-run
