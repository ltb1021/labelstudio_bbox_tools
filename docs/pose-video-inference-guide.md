# Pose Video Inference Guide

이 문서는 YOLO11 Pose와 RF-DETR Keypoint Preview official pretrained weight를 영상에 적용하고, 사람 bbox와 skeleton을 시각화해서 비교하는 workflow를 설명한다.

## 목적

- 쓰러짐 감지, 앉음 오탐 분리처럼 사람 자세가 중요한 상황을 영상으로 빠르게 확인한다.
- 영상 파일 하나 또는 영상 폴더 안의 여러 영상을 pose inference한다.
- bbox, skeleton line, keypoint dot, `class score` label을 영상 위에 그려 MP4로 저장한다.
- YOLO11 Pose 결과와 RF-DETR Keypoint 결과를 좌우로 합친 비교 영상을 만든다.
- Label Studio project나 task는 수정하지 않는다.

## 중요한 전제

YOLO11 Pose와 RF-DETR Keypoint Preview는 official pretrained 기준으로 사람이 있는 위치와 keypoints를 모델이 함께 예측한다. 외부 custom detector bbox를 full-frame 안에서 그대로 주입해서 그 위치만 pose head가 계산하도록 하는 구조는 기본 API의 주된 사용 방식이 아니다.

운영 시스템에서 custom detector를 먼저 쓰고 싶다면 보통 두 가지 방법이 있다.

```text
post-filter 방식:
custom detection full-frame + pose model full-frame -> IoU가 맞는 pose만 유지
```

이 방식은 구현은 쉽지만 pose model도 전체 frame을 보기 때문에 속도 이득이 거의 없고, pose model이 작은 사람을 못 찾는 문제도 그대로 남을 수 있다.

```text
top-down crop 방식:
custom detection full-frame -> 사람 bbox crop/padding/resize -> crop별 pose inference -> 원본 좌표로 복원
```

이 방식은 작은 사람을 crop으로 키워 볼 수 있어 recall 개선 가능성이 있지만, 사람 수만큼 pose inference가 늘어 batch/tracking 최적화가 필요하다. 이번 1차 구현은 full-frame official pose model 비교에 집중하고, detection-gated crop 방식은 추후 별도 기능으로 분리하는 것을 권장한다.

## Notebook 위치

```text
examples/notebooks/pose_inference/
```

권장 순서:

1. `01_yolo11_pose_video_visualize.ipynb`
2. `02_rfdetr_pose_video_visualize.ipynb`
3. `03_compare_pose_videos.ipynb`

## Conda Env

YOLO11 pose notebook은 보통 `ltb_ultra` env에서 실행한다.

RF-DETR keypoint notebook은 RF-DETR package가 editable 설치된 `ltb_rfdetr` env에서 실행한다.

좌우 비교 notebook은 model을 load하지 않으므로 OpenCV와 Pillow가 있는 env면 충분하다.

## Official Pretrained 기본값

YOLO11은 RF-DETR Keypoint Preview와 체급을 최대한 맞춘 비교를 위해 기본값을 큰 모델로 둔다.

```python
MODEL_WEIGHTS = "yolo11x-pose.pt"
```

RF-DETR은 `RFDETRKeypointPreview`의 official default pretrained weight를 사용한다.

```python
MODEL_WEIGHTS = None
```

처음 실제 inference를 실행할 때 weight가 local cache에 없으면 각 package가 weight 다운로드를 시도할 수 있다. 네트워크가 막힌 서버에서는 미리 weight를 받아두거나 cache 경로를 확인해야 한다.

## Input 탐색 규칙

`INPUT_PATH`에는 영상 파일 하나 또는 영상 폴더를 넣을 수 있다.

- 파일이면 그 파일 하나만 처리한다.
- 폴더이면 기본값 `RECURSIVE=False`라서 대상 폴더 바로 아래 영상만 처리한다.
- 하위 폴더까지 모두 탐색하려면 `RECURSIVE=True`로 바꾼다.

지원 확장자는 기존 video inference와 같다.

```text
.mp4, .avi, .mov, .mkv, .webm, .m4v, .mpg, .mpeg
```

## 안전 기본값

처음 notebook을 열면 실제 inference가 실행되지 않는다.

```python
RUN_PREVIEW = False
RUN_INFERENCE = False
RUN_COMPARE = False
DRY_RUN = True
MAX_VIDEOS = 1
MAX_FRAMES = 300
```

먼저 `RUN_PREVIEW=True`로 영상 목록과 metadata를 확인한다. 실제 model inference와 영상 저장은 `RUN_INFERENCE=True`, `DRY_RUN=False`로 바꾼 뒤 실행한다.

## 출력 구조

실제 inference를 실행하면 `OUT_DIR` 아래에 시간 기반 run folder가 생긴다.

```text
OUT_DIR/
  20260625_153000__yolo11_pose/
    run_config.json
    predictions.jsonl
    videos_summary.json
    videos_summary.csv
    videos/
      input_video__yolo11_pose.mp4
```

주요 파일:

- `videos/*.mp4`: bbox, skeleton, keypoint, label이 그려진 시각화 영상
- `predictions.jsonl`: frame별 pose instance와 keypoint 좌표/score
- `videos_summary.json`: 영상별 처리 frame 수, instance 수, keypoint 수, 출력 경로
- `run_config.json`: 실행 설정 기록

영상, weight, inference output은 Git에 넣지 않는다.

## 시각화 규칙

- official pretrained는 `person` 1-class를 기본으로 사용한다.
- bbox는 class color로 표시한다.
- skeleton은 COCO person 17 keypoints 연결 규칙을 사용한다.
- keypoint dot은 `KEYPOINT_CONF` 이상일 때만 그린다.
- skeleton line은 연결된 두 keypoint가 모두 `KEYPOINT_CONF` 이상일 때만 그린다.
- label은 bbox 좌상단 근처에 두되, 화면 밖으로 나가지 않게 clamp하고 겹침 후보 위치를 순서대로 시도한다.
- 한글 title/label이 필요한 경우 Nanum/Noto CJK font를 자동 탐색한다.

## CLI 예시

YOLO11 pose dry-run은 model을 load하지 않고 영상 metadata만 확인한다.

```bash
lsbbox-pose-infer-yolo11 \
  --input-path "/path/to/video_or_video_folder" \
  --out-dir "/path/to/pose_inference_outputs" \
  --weights "yolo11x-pose.pt" \
  --device "cuda:0" \
  --max-videos 1 \
  --max-frames 300
```

실제 실행은 `--run`을 추가한다.

```bash
lsbbox-pose-infer-yolo11 \
  --input-path "/path/to/video_or_video_folder" \
  --out-dir "/path/to/pose_inference_outputs" \
  --weights "yolo11x-pose.pt" \
  --device "cuda:0" \
  --max-videos 1 \
  --max-frames 300 \
  --run
```

RF-DETR Keypoint Preview도 같은 방식이다. `--weights`를 생략하면 official default pretrained weight를 사용한다.

```bash
lsbbox-pose-infer-rfdetr \
  --input-path "/path/to/video_or_video_folder" \
  --out-dir "/path/to/pose_inference_outputs" \
  --device "cuda" \
  --max-videos 1 \
  --max-frames 300
```

좌우 비교 영상은 이미 생성된 두 결과 영상을 입력으로 받는다.

```bash
lsbbox-video-compare \
  --left-video "/path/to/yolo11_pose_visualized.mp4" \
  --right-video "/path/to/rfdetr_keypoint_visualized.mp4" \
  --out-video "/path/to/compare_yolo11_rfdetr_pose.mp4" \
  --left-title "YOLO11 Pose" \
  --right-title "RF-DETR Keypoint" \
  --max-frames 300
```

비교 영상도 실제 저장하려면 `--run`을 추가한다.
