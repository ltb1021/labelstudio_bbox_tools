#!/usr/bin/env bash
set -euo pipefail

# YOLO11 official pose pretrained weight로 영상 pose inference 입력/출력 설정을 미리 확인합니다.
# 실제 모델 로드와 영상 저장을 하려면 마지막 줄에 --run을 추가하세요.

lsbbox-pose-infer-yolo11 \
  --input-path "/path/to/video_or_video_folder" \
  --out-dir "/path/to/pose_inference_outputs" \
  --weights "yolo11x-pose.pt" \
  --device "cuda:0" \
  --conf 0.25 \
  --keypoint-conf 0.20 \
  --max-videos 1 \
  --max-frames 300
