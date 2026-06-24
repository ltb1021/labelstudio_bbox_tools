# Label Studio Advanced Workflows

이 문서는 2차 리팩토링 범위인 pseudo labeling, annotation/prediction merge, 기존 annotation import, project delete 기능의 사용 기준을 정리한다.

## 공통 안전 원칙

- notebook 기본값은 실제 서버를 바꾸지 않는 값으로 둔다.
- 실제 POST/DELETE를 하기 전에는 project id, 대상 task 수, class 목록, model_version/tag 이름을 눈으로 확인한다.
- `LABEL_STUDIO_API_KEY`는 `.env`에만 저장한다.
- weight, dataset, export 결과는 Git에 넣지 않는다.

## YOLO Pseudo Labeling

notebook: `examples/notebooks/labelstudio/04_ls_pseudo_label_yolo.ipynb`

이 기능은 기존 image task에 YOLO 추론 결과를 `prediction`으로 추가한다.

주요 설정:

- `WEIGHTS_YOLO`: 사용할 `.pt` weight 경로.
- `CLASS_YAML`: YOLO dataset yaml 경로. `names` 필드에서 class 순서를 읽는다.
- `MANUAL_CLASSES`: yaml이 없을 때 쓰는 fallback class list.
- `CONF`: 모델 추론의 기본 confidence threshold.
- `IOU`: Ultralytics 내부 NMS의 기본 IoU threshold.
- `CLASS_THRESH`: class별 confidence threshold.
- `CLASS_IOU`: class별 2차 NMS IoU threshold.

`CLASS_IOU`는 모델이 한 번 NMS를 한 뒤에도 같은 class 안에서 겹치는 bbox가 많이 남을 때 추가로 정리하기 위한 값이다. 예를 들어 `small_worker`와 `worker`처럼 비슷한 class가 혼동되는 문제는 `CLASS_THRESH`와 class 설계, merge 단계의 `CLASS_GROUPS`까지 함께 조정해야 한다.

## RF-DETR Pseudo Labeling

notebook: `examples/notebooks/labelstudio/04_2_ls_pseudo_label_rfdetr.ipynb`

이 기능은 기존 image task에 RF-DETR custom checkpoint 추론 결과를 `prediction`으로 추가한다. 기본 UX는 YOLO pseudo labeling notebook과 최대한 맞춘다.

주요 설정:

- `MODEL_WEIGHTS`: RF-DETR checkpoint 경로. 보통 `checkpoint_best_total.pth`를 직접 지정한다.
- `MODEL_VARIANT`: `auto`, `nano`, `small`, `medium`, `large` 중 선택한다. checkpoint에 model metadata가 있으면 `auto`가 편할 수 있다.
- `CLASS_YAML`: 학습 때 사용한 class order와 같은 `names`를 가진 yaml이어야 한다.
- `CONF`: RF-DETR `predict(..., threshold=...)`에 전달되는 기본 confidence threshold다.
- `CLASS_THRESH`: class별 threshold. 낮은 class별 threshold를 쓰려면 `CONF`도 충분히 낮게 잡아야 한다.
- `IOU`, `CLASS_IOU`: RF-DETR 결과에 한 번 더 적용하는 classwise NMS 설정이다. RF-DETR 자체 후처리와 중복될 수 있으므로 중복 bbox가 많을 때만 조정한다.

기본값은 `DRY_RUN=True`이므로 model load, inference, upload를 하지 않는다. 실제 업로드는 notebook에서 `DRY_RUN=False`로 바꾼 뒤 실행한다.

## Annotation/Prediction Merge

notebook: `examples/notebooks/labelstudio/05_ls_merge_ann_pred.ipynb`

이 기능은 여러 source에서 bbox를 읽고 겹치는 bbox를 규칙에 따라 하나로 정리해서 새 prediction 또는 annotation을 만든다.

source 종류:

- `SrcAnn(user_id, min_lead)`: 특정 사용자의 annotation.
- `SrcPred(model_ver, score_thr)`: 특정 model_version의 prediction.

충돌 해결 규칙:

- `resolve="keep_earlier"`: `sources` list에서 앞에 있는 source가 우선이다. annotation은 score가 없거나 신뢰할 주력 weight가 정해진 경우가 많아서 이 방식이 실무적으로 안정적이다.
- `resolve="higher_score"`: 겹치는 bbox 중 score가 높은 것을 남긴다.

`CLASS_GROUPS`는 서로 다른 class라도 같은 그룹 안이면 겹치는 bbox를 충돌로 본다. worker/signalman 계열처럼 모델이 비슷하게 보는 class들을 한 그룹으로 묶어 중복 bbox를 줄일 때 사용한다.

## MMYOLO Annotation Import

notebook: `examples/notebooks/labelstudio/06_ls_import_annotations_mmyolo.ipynb`

이 기능은 기존 annotation JSON을 이미 import된 Label Studio image task에 붙인다.

권장 설정:

- `image_match_mode="fullpath"`: 같은 basename 이미지가 여러 폴더에 있을 수 있으므로 full path 기준 matching을 우선한다.
- `upload_mode="prediction"`: 원본은 annotation이어도 Label Studio에는 prediction으로 넣어두면 merge/검수 workflow에서 분리해 다루기 쉽다.
- `BATCH_SIZE`: 처음에는 200 이하로 시작하고 서버 상태를 보면서 조정한다.

## Project Delete

notebook: `examples/notebooks/labelstudio/99_ls_delete_project_safe.ipynb`

이 기능은 위험도가 높다. 대량 task project를 브라우저에서 삭제하기 어렵거나 실패할 때만 사용한다.

안전 실행 순서:

1. `DRY_RUN=True`로 project title, task count, sample task ids를 확인한다.
2. 실제 삭제 전 `PROJECT_ID`와 브라우저의 project title을 다시 확인한다.
3. `DRY_RUN=False`, `CONFIRM="delete"`, `MAX_BATCHES=1`로 한 batch만 테스트한다.
4. 문제가 없을 때 `MAX_BATCHES=None`으로 전체 task를 삭제한다.
5. project meta 삭제는 `DELETE_PROJECT_META=True`일 때만 수행한다.
