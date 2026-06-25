# Notebook Workflow Guide

이 repo는 notebook을 수동 확인과 반복 작업의 기본 진입점으로 둔다. shell script는 같은 작업을 빠르게 반복하거나 자동화할 때 사용한다.

## Notebook 위치

Label Studio 관련 notebook은 아래 폴더에 둔다.

```text
examples/notebooks/labelstudio/
```

현재 Label Studio notebook은 다음과 같다.

```text
00_ls_check_environment.ipynb
01_ls_import_images.ipynb
02_ls_apply_bbox_ui_from_mmyolo.ipynb
03_ls_export_mmyolo.ipynb
04_ls_pseudo_label_yolo.ipynb
04_2_ls_pseudo_label_rfdetr.ipynb
05_ls_merge_ann_pred.ipynb
06_ls_import_annotations_mmyolo.ipynb
07_ls_extract_video_frames.ipynb
99_ls_delete_project_safe.ipynb
```

파일명에 `ls_`를 붙인 이유는 나중에 Label Studio와 직접 관련 없는 dataset 전처리, YOLO 변환, pseudo labeling notebook이 추가되어도 쉽게 구분하기 위해서다.

Label Studio와 직접 관련 없는 영상 inference 비교 notebook은 아래 폴더에 둔다.

```text
examples/notebooks/video_inference/
```

현재 video inference notebook은 다음과 같다.

```text
01_yolo11_video_inference_visualize.ipynb
02_rfdetr_video_inference_visualize.ipynb
03_compare_visualized_videos.ipynb
```

이 notebook들은 Label Studio API를 호출하지 않는다. YOLO11과 RF-DETR weight를 같은 영상에 적용한 뒤 bbox/label overlay 영상을 만들고, 필요하면 두 결과를 좌우로 합쳐 비교한다.

사람 pose inference 비교 notebook은 아래 폴더에 둔다.

```text
examples/notebooks/pose_inference/
```

현재 pose inference notebook은 다음과 같다.

```text
01_yolo11_pose_video_visualize.ipynb
02_rfdetr_pose_video_visualize.ipynb
03_compare_pose_videos.ipynb
04_yolo11_pose_fallback_crop_test.ipynb
05_rfdetr_pose_fallback_crop_test.ipynb
```

이 notebook들은 official pretrained pose/keypoint weight로 사람 skeleton을 영상에 그린다. `04`/`05`는 custom detector가 찾았지만 full-frame pose가 놓친 b 케이스를 crop batch로 재추론한다. Label Studio API를 호출하지 않는다.

## 권장 실행 순서

1. `00_ls_check_environment.ipynb`
2. 영상이 입력이면 `07_ls_extract_video_frames.ipynb`로 라벨링용 image frame 생성
3. `01_ls_import_images.ipynb`
4. `02_ls_apply_bbox_ui_from_mmyolo.ipynb`
5. Label Studio 브라우저에서 bbox labeling
6. `03_ls_export_mmyolo.ipynb`
7. 필요하면 `04_ls_pseudo_label_yolo.ipynb`
8. 필요하면 `04_2_ls_pseudo_label_rfdetr.ipynb`
9. 필요하면 `05_ls_merge_ann_pred.ipynb`
10. 필요하면 `06_ls_import_annotations_mmyolo.ipynb`
11. 위험 작업이 필요할 때만 `99_ls_delete_project_safe.ipynb`

## 안전 기본값

각 notebook은 처음 열었을 때 실제 서버 데이터를 바로 바꾸지 않도록 구성한다.

- import notebook: `DRY_RUN=True`
- UI 설정 notebook: `PREVIEW_ONLY=True`
- export notebook: `RUN_EXPORT=False`
- video/pseudo/merge/import/delete notebook: `DRY_RUN=True` 또는 별도 실행 flag가 꺼진 상태
- video inference notebook: `RUN_PREVIEW=False`, `RUN_INFERENCE=False`, `RUN_COMPARE=False`, `DRY_RUN=True`
- pose inference notebook: `RUN_PREVIEW=False`, `RUN_INFERENCE=False`, `RUN_COMPARE=False`, `DRY_RUN=True`

이 값을 바꾸기 전에는 출력 메시지를 보고 경로, project id, class 목록이 맞는지 확인한다.

## `.env` 사용 방식

notebook에는 API key를 직접 적지 않는다. 대신 repo root의 `.env` 파일에서 읽는다.

```text
LABEL_STUDIO_URL=http://your-server-ip:9225
LABEL_STUDIO_API_KEY=your-local-token
LABEL_STUDIO_DOC_ROOT=/path/to/label_studio_data
```

`.env`는 `.gitignore`에 포함되어 있으므로 Git에 올라가지 않는다. `.env.example`은 어떤 값을 채워야 하는지 보여주는 템플릿이다.

## notebook과 shell script의 역할 차이

notebook은 사람이 눈으로 확인하면서 단계별로 실행하기 좋다. class 목록, 생성될 local-files URL, export 통계 같은 중간 결과를 확인하기 쉽다.

shell script는 이미 검증된 설정을 반복 실행하기 좋다. 예를 들어 매일 같은 폴더 구조로 export하는 작업은 script가 더 편할 수 있다.

이 repo에서는 notebook을 먼저 검증하고, 반복되는 작업만 script로 옮기는 방식을 기본 정책으로 둔다.
