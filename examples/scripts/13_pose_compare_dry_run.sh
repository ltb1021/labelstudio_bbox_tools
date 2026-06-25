#!/usr/bin/env bash
set -euo pipefail

# 이미 생성된 YOLO11 pose/RF-DETR pose 시각화 영상을 좌우로 합쳐 비교할 설정을 미리 확인합니다.
# 실제 비교 영상을 저장하려면 마지막 줄에 --run을 추가하세요.

lsbbox-video-compare \
  --left-video "/path/to/yolo11_pose_visualized.mp4" \
  --right-video "/path/to/rfdetr_keypoint_visualized.mp4" \
  --out-video "/path/to/compare_yolo11_rfdetr_pose.mp4" \
  --left-title "YOLO11 Pose" \
  --right-title "RF-DETR Keypoint" \
  --max-frames 300
