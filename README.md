# labelstudio_bbox_tools

Label Studio object detection bbox workflow를 정리한 Python package다. 영상 frame 추출, 이미지 import, MMYOLO/COCO JSON 기반 bbox UI 설정, annotation export, YOLO/RF-DETR pseudo labeling, annotation/prediction merge, 기존 annotation import, project delete 보조 기능을 포함한다.

이 repo는 **notebook-first** 사용을 기본으로 한다. 기존처럼 notebook cell을 순서대로 실행하면서 경로, project id, class 목록, export 결과를 눈으로 확인할 수 있게 구성한다. shell script는 반복 작업이나 자동화가 필요할 때 보조로 사용한다.

## Current Scope

- 영상 파일 또는 영상 폴더에서 라벨링용 frame 이미지 추출
- YOLO11/RF-DETR weight로 영상 inference 결과를 bbox/label overlay 영상으로 시각화하고 좌우 비교
- 이미지 폴더를 Label Studio project에 import
- MMYOLO/COCO `categories`에서 class list 추출
- Label Studio bbox 또는 polygon UI 적용
- manual annotation 또는 model prediction을 `mmyolo`, `yolo`, `yolo_obb`로 export
- YOLO weight로 Label Studio prediction 생성
- RF-DETR custom weight로 Label Studio prediction 생성
- 여러 annotation/prediction source를 merge해서 새 prediction 또는 annotation 생성
- 기존 MMYOLO annotation JSON을 Label Studio task에 annotation 또는 prediction으로 import
- 대량 task project 삭제를 위한 safe preview/delete helper

아직 별도 범위로 남겨둔 기능:

- MMYOLO to YOLO 변환 고도화
- Docker container 생성 자동화
- dataset preprocessing/editing notebook 추가 정리

## Install

권장 환경은 현재 RTX4090 서버 기준 기본 Label Studio/YOLO 작업은 `ltb_ultra` conda env다. RF-DETR pseudo labeling과 RF-DETR 영상 inference는 RF-DETR package가 editable 설치된 `ltb_rfdetr` env에서 실행한다.

```bash
cd /path/to/labelstudio_bbox_tools
conda activate ltb_ultra
python -m pip install -e .
cp .env.example .env
```

`.env`에는 본인 서버의 Label Studio 접속 정보를 채운다. 실제 API key는 Git에 올리지 않는다.

새 CLI 명령이 추가된 뒤 shell에서 `lsbbox-pseudo-label-yolo` 같은 명령을 찾지 못하면, 같은 conda env에서 `python -m pip install -e .`를 한 번 더 실행한다. editable install은 코드 변경은 바로 따라가지만, console script entry point가 새로 생긴 경우 재설치가 필요할 수 있다.

`python -m pip install -e .`가 무슨 뜻인지 잘 모르겠다면 [docs/editable-install-guide.md](docs/editable-install-guide.md)를 먼저 읽는다. `.env`를 어떻게 채우는지 헷갈리면 [docs/env-guide.md](docs/env-guide.md)를 확인한다.

## `.env` 설정

`.env.example`을 복사해서 `.env`를 만든 뒤 값을 채운다.

```text
LABEL_STUDIO_URL=http://your-server-ip:9225
LABEL_STUDIO_API_KEY=your-local-token
LABEL_STUDIO_DOC_ROOT=/path/to/label_studio_data
```

- `LABEL_STUDIO_URL`: 브라우저에서 접속하는 Label Studio 주소
- `LABEL_STUDIO_API_KEY`: Label Studio 사용자 계정의 API token
- `LABEL_STUDIO_DOC_ROOT`: Label Studio container가 local files로 접근할 수 있는 host 경로

`LABEL_STUDIO_API_KEY`는 README, notebook, commit에 적지 않는다.

## Notebook 사용법

Label Studio notebook은 아래 폴더에 있다.

```text
examples/notebooks/labelstudio/
```

권장 순서:

1. `00_ls_check_environment.ipynb`: 환경과 `.env` 설정 확인
2. 영상이 입력이면 `07_ls_extract_video_frames.ipynb`: 영상에서 라벨링용 frame 이미지 추출
3. `01_ls_import_images.ipynb`: 이미지 폴더 import
4. `02_ls_apply_bbox_ui_from_mmyolo.ipynb`: MMYOLO/COCO JSON에서 class를 읽어 bbox UI 적용
5. Label Studio 브라우저에서 bbox labeling
6. `03_ls_export_mmyolo.ipynb`: annotation을 MMYOLO 형식으로 export
7. `04_ls_pseudo_label_yolo.ipynb`: YOLO weight로 prediction 생성
8. `04_2_ls_pseudo_label_rfdetr.ipynb`: RF-DETR weight로 prediction 생성
9. `05_ls_merge_ann_pred.ipynb`: ann/pred source merge
10. `06_ls_import_annotations_mmyolo.ipynb`: 기존 MMYOLO annotation import
11. `99_ls_delete_project_safe.ipynb`: 위험 작업인 project 삭제 preview/delete

각 notebook은 처음에는 실제 Label Studio 데이터를 바꾸지 않는 안전한 기본값으로 시작한다.

- `DRY_RUN=True`
- `PREVIEW_ONLY=True`
- `RUN_EXPORT=False`
- `RUN_*=False` 또는 `CONFIRM` 값 미입력

출력을 확인한 뒤 필요한 경우에만 값을 바꿔서 실행한다.

Label Studio와 직접 관련 없는 영상 inference 비교 notebook은 아래 폴더에 있다.

```text
examples/notebooks/video_inference/
```

권장 순서:

1. `01_yolo11_video_inference_visualize.ipynb`: `ltb_ultra` env에서 YOLO11 weight 영상 inference/시각화
2. `02_rfdetr_video_inference_visualize.ipynb`: `ltb_rfdetr` env에서 RF-DETR checkpoint 영상 inference/시각화
3. `03_compare_visualized_videos.ipynb`: 이미 생성된 두 시각화 영상을 좌우로 합쳐 비교

video inference notebook도 기본값은 `RUN_PREVIEW=False`, `RUN_INFERENCE=False`, `RUN_COMPARE=False`, `DRY_RUN=True`로 시작한다.

자세한 notebook 운영 정책은 [docs/notebook-workflow-guide.md](docs/notebook-workflow-guide.md)를 본다. 2차 고급 workflow는 [docs/workflow-labelstudio-advanced.md](docs/workflow-labelstudio-advanced.md)를 본다. 영상 frame 추출은 [docs/video-frame-extraction-guide.md](docs/video-frame-extraction-guide.md)를 본다. YOLO11/RF-DETR 영상 inference 비교는 [docs/video-inference-visualization-guide.md](docs/video-inference-visualization-guide.md)를 본다.

## Shell Command 사용법

notebook으로 검증한 뒤 같은 기능을 command로 반복 실행할 수 있다.

영상 frame 추출 dry-run 예시:

```bash
lsbbox-extract-video-frames \
  --input-path "/path/to/video_or_video_folder" \
  --out-dir "/path/to/frames_export" \
  --interval-seconds 2.0 \
  --max-videos 3 \
  --max-frames-per-video 20 \
  --dry-run
```

YOLO11 영상 inference dry-run 예시:

```bash
lsbbox-video-infer-yolo11 \
  --input-path "/path/to/video_or_video_folder" \
  --out-dir "/path/to/video_inference_outputs" \
  --weights "/path/to/yolo11_best.pt" \
  --class-yaml "/path/to/data.yaml" \
  --max-videos 1 \
  --max-frames 300
```

RF-DETR 영상 inference dry-run 예시:

```bash
lsbbox-video-infer-rfdetr \
  --input-path "/path/to/video_or_video_folder" \
  --out-dir "/path/to/video_inference_outputs" \
  --weights "/path/to/checkpoint_best_total.pth" \
  --class-yaml "/path/to/data.yaml" \
  --model-variant medium \
  --max-videos 1 \
  --max-frames 300
```

실제 영상을 쓰려면 각 command에 `--run`을 추가한다.

좌우 비교 영상 dry-run 예시:

```bash
lsbbox-video-compare \
  --left-video "/path/to/yolo11_visualized.mp4" \
  --right-video "/path/to/rfdetr_visualized.mp4" \
  --out-video "/path/to/compare_yolo11_rfdetr.mp4" \
  --left-title "YOLO11" \
  --right-title "RF-DETR" \
  --max-frames 300
```

이미지 import dry-run 예시:

```bash
lsbbox-import-images \
  --src-root "/path/to/images" \
  --doc-root "/path/to/label_studio_data" \
  --project-title "my_bbox_project" \
  --dry-run
```

UI 설정 preview 예시:

```bash
lsbbox-apply-ui \
  --project-id 123 \
  --mmyolo-json "/path/to/annotations_mmyolo.json" \
  --shape bbox \
  --dry-run
```

MMYOLO export 예시:

```bash
lsbbox-export \
  --project-id 123 \
  --out-dir "/path/to/export/output" \
  --export-type ann \
  --ann-format mmyolo \
  --source-type ann \
  --ann-user-id 1 \
  --ann-min-lead 0
```

기본적으로 선택한 `ANN_USER_ID`의 bbox가 없는 task는 MMYOLO `images[]`에서도 제외된다.
검수용 negative image처럼 bbox가 없는 image까지 의도적으로 남겨야 하면 `--include-empty-images`를 추가한다.

```bash
lsbbox-export \
  --project-id 123 \
  --out-dir "/path/to/export/output" \
  --export-type ann \
  --ann-format mmyolo \
  --source-type ann \
  --ann-user-id 1 \
  --ann-min-lead 0 \
  --include-empty-images
```

YOLO pseudo labeling dry-run 예시:

```bash
lsbbox-pseudo-label-yolo \
  --project-id 123 \
  --weights "/path/to/yolo11_best.pt" \
  --class-yaml "/path/to/data.yaml" \
  --imgsz 640 \
  --max-tasks 20 \
  --dry-run
```

RF-DETR pseudo labeling dry-run 예시:

```bash
lsbbox-pseudo-label-rfdetr \
  --project-id 123 \
  --weights "/path/to/checkpoint_best_total.pth" \
  --class-yaml "/path/to/data.yaml" \
  --model-variant medium \
  --conf 0.30 \
  --iou 0.60 \
  --max-tasks 20 \
  --dry-run
```

## Path Model

Label Studio local files는 경로 개념이 중요하다. 자세한 설명은 [docs/path-and-mount-guide.md](docs/path-and-mount-guide.md)를 본다.

핵심 용어:

- `doc_root`: Label Studio container가 local-files로 서빙하는 root
- `src_root`: import할 이미지 폴더
- `mirror_root`: `src_root`가 `doc_root` 밖일 때 symlink 구조를 만들 기준 root

## Safety

- 실제 token은 repo에 저장하지 않는다.
- dataset, export 결과, model weights는 Git에 넣지 않는다.
- 기존 실무 폴더는 source of truth로 보존하고, 이 repo에는 검증된 core 기능만 가져온다.
- 실제 Label Studio project/task를 생성하거나 수정하기 전에는 notebook의 dry-run/preview를 먼저 확인한다.
- project delete 기능은 실제 삭제 전 `DRY_RUN=True` preview와 `CONFIRM="delete"` 이중 안전장치를 반드시 거친다.
