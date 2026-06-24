# Video Inference Visualization Guide

이 문서는 YOLO11과 RF-DETR weight를 같은 영상에 적용하고, bbox/label이 그려진 결과 영상을 비교하는 workflow를 설명한다.

## 목적

- 영상 파일 하나 또는 영상 폴더 안의 여러 영상을 inference한다.
- bbox와 `class score` 라벨을 영상 위에 그려서 MP4로 저장한다.
- YOLO11 결과와 RF-DETR 결과를 좌우로 합친 비교 영상을 만든다.
- Label Studio project나 task는 수정하지 않는다.

## Notebook 위치

```text
examples/notebooks/video_inference/
```

권장 순서:

1. `01_yolo11_video_inference_visualize.ipynb`
2. `02_rfdetr_video_inference_visualize.ipynb`
3. `03_compare_visualized_videos.ipynb`

## Conda Env

YOLO11 notebook은 보통 `ltb_ultra` env에서 실행한다.

RF-DETR notebook은 RF-DETR package가 editable 설치된 `ltb_rfdetr` env에서 실행한다.

좌우 비교 notebook은 model을 load하지 않으므로 OpenCV와 Pillow가 있는 env면 충분하다.

## Class YAML

두 model 모두 같은 28-class YAML을 사용하는 것을 권장한다.

```python
CLASS_YAML = Path("/path/to/data.yaml")
EXPECTED_CLASS_COUNT = 28
STRICT_CLASS_COUNT = True
```

class id와 class name 순서가 학습 때 사용한 순서와 다르면 영상에 표시되는 class 이름이 틀릴 수 있다. 따라서 실행 전에 class check cell에서 0번부터 27번까지 순서를 확인한다.

## Input 탐색 규칙

`INPUT_PATH`에는 영상 파일 하나 또는 영상 폴더를 넣을 수 있다.

- 파일이면 그 파일 하나만 처리한다.
- 폴더이면 기본값 `RECURSIVE=False`라서 대상 폴더 바로 아래 영상만 처리한다.
- 하위 폴더까지 모두 탐색하려면 `RECURSIVE=True`로 바꾼다.

지원 확장자:

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
  20260624_153000__yolo11/
    run_config.json
    predictions.jsonl
    videos_summary.json
    videos_summary.csv
    videos/
      input_video__yolo11.mp4
```

주요 파일:

- `videos/*.mp4`: bbox와 label이 그려진 시각화 영상
- `predictions.jsonl`: frame별 detection 정보
- `videos_summary.json`: 영상별 처리 frame 수, bbox 수, 출력 경로
- `run_config.json`: 실행 설정 기록

dataset, weight, export 결과, inference output 영상은 Git에 넣지 않는다.

## 시각화 규칙

- class YAML 순서를 기준으로 class별 고정 color를 만든다.
- bbox는 class color로 표시한다.
- label은 기본적으로 bbox 좌상단 근처에 둔다.
- label이 화면 밖으로 나가면 안쪽으로 clamp한다.
- label끼리 겹치면 몇 가지 후보 위치를 순서대로 시도한다.
- label 텍스트는 `class_name score` 형식이다.
- 한글 class 이름이 깨지지 않도록 Nanum/Noto CJK font를 자동 탐색한다.

## CLI 예시

dry-run은 model을 load하지 않고 영상 metadata만 확인한다.

```bash
lsbbox-video-infer-yolo11       --input-path "/path/to/video_or_video_folder"       --out-dir "/path/to/video_inference_outputs"       --weights "/path/to/yolo11_best.pt"       --class-yaml "/path/to/data.yaml"       --max-videos 1       --max-frames 300
```

실제 실행은 `--run`을 추가한다.

```bash
lsbbox-video-infer-yolo11       --input-path "/path/to/video_or_video_folder"       --out-dir "/path/to/video_inference_outputs"       --weights "/path/to/yolo11_best.pt"       --class-yaml "/path/to/data.yaml"       --max-videos 1       --max-frames 300       --run
```

RF-DETR도 같은 방식이다.

```bash
lsbbox-video-infer-rfdetr       --input-path "/path/to/video_or_video_folder"       --out-dir "/path/to/video_inference_outputs"       --weights "/path/to/checkpoint_best_total.pth"       --class-yaml "/path/to/data.yaml"       --model-variant medium       --max-videos 1       --max-frames 300
```

좌우 비교 영상은 이미 생성된 두 결과 영상을 입력으로 받는다.

```bash
lsbbox-video-compare       --left-video "/path/to/yolo11_visualized.mp4"       --right-video "/path/to/rfdetr_visualized.mp4"       --out-video "/path/to/compare_yolo11_rfdetr.mp4"       --left-title "YOLO11"       --right-title "RF-DETR"       --max-frames 300
```

비교 영상도 실제 저장하려면 `--run`을 추가한다.
