# Basic Bbox Workflow

이 문서는 Label Studio에서 bbox labeling을 할 때의 기본 흐름을 설명한다. 처음 사용하는 사람은 shell command보다 notebook 흐름을 먼저 따라가는 것을 권장한다.

## 전체 흐름

1. 환경 확인
2. 이미지 import
3. bbox UI와 class 목록 적용
4. 브라우저에서 labeling
5. MMYOLO export
6. export 결과 간단 확인

## 1. 환경 확인

`examples/notebooks/labelstudio/00_ls_check_environment.ipynb`를 실행한다.

확인할 것:

- 현재 Python kernel이 원하는 conda env인지
- `labelstudio_bbox_tools` package가 import되는지
- `.env`에 필요한 값이 있는지
- 필요하면 Label Studio 연결이 되는지

## 2. 이미지 import

`01_ls_import_images.ipynb`를 사용한다.

중요 변수:

- `SRC_ROOT`: import할 이미지 폴더
- `PROJECT_ID`: `None`이면 새 project 생성, 숫자면 기존 project에 추가
- `PROJECT_TITLE`: 새 project 생성 시 사용할 제목
- `SLICE_SPEC`: 처음 테스트할 이미지 일부만 고르는 범위
- `DRY_RUN`: 실제 import 여부

처음에는 `SLICE_SPEC=":5"`, `DRY_RUN=True`로 확인한다. 생성될 `/data/local-files/?d=...` URL이 자연스럽게 보이면 실제 import로 넘어간다.

## 3. bbox UI 적용

`02_ls_apply_bbox_ui_from_mmyolo.ipynb`를 사용한다.

중요 변수:

- `PROJECT_ID`: UI를 적용할 Label Studio project id
- `MMYOLO_JSON`: class 정보를 읽을 JSON
- `SHAPE_TYPE`: 보통 `bbox`
- `PREVIEW_ONLY`: 실제 project 수정 여부

처음에는 `PREVIEW_ONLY=True`로 class 목록과 XML을 확인한다.

## 4. 브라우저 labeling

Label Studio 브라우저에서 project를 열고 bbox 작업을 진행한다.

project id는 브라우저 URL이나 project 목록에서 확인한다. 이 id는 export notebook에서도 사용한다.

## 5. MMYOLO export

`03_ls_export_mmyolo.ipynb`를 사용한다.

중요 변수:

- `PROJECT_ID`: export할 project id
- `OUT_DIR`: export 결과를 저장할 폴더
- `SOURCE_TYPE`: `ann` 또는 `pred`
- `ANN_USER_ID`: 수동 annotation 작업자 id
- `ANN_MIN_LEAD`: lead_time 필터
- `RUN_EXPORT`: 실제 export 여부

처음에는 `RUN_EXPORT=False`로 설정값만 확인한다.

## 6. 결과 확인

export가 끝나면 보통 아래 파일을 확인한다.

```text
annotations_mmyolo.json
classes.txt
images_all.txt
```

`03_ls_export_mmyolo.ipynb` 마지막 cell은 `annotations_mmyolo.json`의 image, annotation, category 수를 간단히 출력한다.
