# COCO Dataset Pack Guide

이 문서는 COCO bbox JSON에 들어있는 이미지 경로와 annotation을 이용해, 다른 사람에게 전달하기 쉬운 dataset 폴더를 만드는 방법을 설명한다. Label Studio API를 호출하지 않고 local 파일만 읽고 복사한다.

## 목적

- COCO JSON 안의 `images[].file_name` 절대경로에서 원본 이미지를 찾는다.
- 이미지를 `images/{split}/` 아래로 복사한다.
- COCO bbox를 YOLO txt label로 변환해서 `labels/{split}/` 아래에 저장한다.
- 원본 추적을 위해 `export_manifest.csv`와 `export_summary.json`을 남긴다.
- 필요하면 portable한 `annotations/{split}_coco.json`도 함께 저장한다.

## Notebook 위치

```text
examples/notebooks/dataset_tools/01_coco_pack_images_labels.ipynb
```

처음 기본값은 안전하다.

```python
DRY_RUN = True
RUN_PACK = False
```

먼저 dry-run으로 class 수, 이미지 수, annotation 수, missing image 수를 확인한 뒤 실제 실행한다.

## 입력 방식

Split을 직접 지정하는 방식을 가장 권장한다.

```python
COCO_INPUTS = {
    "train": Path("/path/to/train_coco.json"),
    "val": Path("/path/to/val_coco.json"),
}
```

파일명에 `train`, `val`, `test`가 명확히 들어 있으면 list로 둬도 자동 판정한다.

```python
COCO_INPUTS = [
    Path("/path/to/my_train.json"),
    Path("/path/to/my_val.json"),
]
```

## 출력 구조

실제 실행하면 `OUT_DIR` 아래에 다음 구조가 생긴다.

```text
OUT_DIR/
  images/
    train/
    val/
  labels/
    train/
    val/
  annotations/
    train_coco.json
    val_coco.json
  data.yaml
  export_manifest.csv
  export_summary.json
```

- `images/{split}`: 전달용 이미지 복사본
- `labels/{split}`: YOLO txt label
- `annotations/{split}_coco.json`: `file_name`을 새 `images/{split}` 경로로 바꾼 COCO JSON
- `data.yaml`: Ultralytics YOLO 학습용 class list
- `export_manifest.csv`: 원본 이미지 경로와 새 파일명을 매핑한 표
- `export_summary.json`: class 수, 이미지 수, annotation 수, 누락 이미지 수 요약

## 파일명 충돌 방지

서로 다른 폴더에 같은 파일명 이미지가 있을 수 있다. 단순히 `000001.jpg` 같은 basename만 쓰면 덮어쓰기 위험이 있다.

이 tool은 새 파일명에 원본 이미지 부모 경로 일부와 짧은 hash를 넣는다.

```text
source_folder_tag__a1b2c3d4e5__original_name.jpg
```

그래도 충돌하면 `__dup001` suffix를 추가한다. 원본 경로는 `export_manifest.csv`에 남기므로 나중에 추적할 수 있다.

## Class 순서

COCO `categories`를 `category_id` 기준으로 정렬한 뒤 YOLO의 0-based class index로 변환한다.

예를 들어 COCO category id가 `1`, `3`이면 YOLO index는 다음처럼 다시 매핑된다.

```text
COCO category_id 1 -> YOLO class 0
COCO category_id 3 -> YOLO class 1
```

`data.yaml`의 `names` 순서와 label txt의 class index가 이 순서와 일치한다. 여러 JSON을 같이 넣을 때 category id/name 순서가 다르면 기본값에서는 중단한다.

## 주요 옵션

```python
LABEL_FORMAT = "yolo"
SAVE_REWRITTEN_COCO = True
COPY_MODE = "copy"
INCLUDE_EMPTY_IMAGES = True
FAIL_ON_MISSING = False
CLIP_BBOX = True
OVERWRITE = False
```

- `LABEL_FORMAT="yolo"`: YOLO txt label을 생성한다. COCO만 필요하면 `"none"`으로 둔다.
- `SAVE_REWRITTEN_COCO=True`: 새 image 경로를 담은 COCO JSON도 저장한다.
- `COPY_MODE="copy"`: 다른 사람에게 전달할 dataset이면 copy가 가장 안전하다.
- `INCLUDE_EMPTY_IMAGES=True`: annotation 없는 이미지도 포함하고 빈 txt를 만든다.
- `FAIL_ON_MISSING=False`: 이미지 파일이 없으면 summary에 기록하고 계속한다. 엄격하게 중단하려면 True.
- `CLIP_BBOX=True`: bbox가 이미지 경계를 살짝 벗어나면 경계 안으로 자른 뒤 YOLO label을 만든다.
- `OVERWRITE=False`: 기존 output 파일이 있으면 중단한다.

## CLI 예시

Dry-run:

```bash
lsbbox-pack-coco-dataset \
  --input train=/path/to/train_coco.json \
  --input val=/path/to/val_coco.json \
  --out-dir /path/to/output_dataset
```

실제 복사/저장:

```bash
lsbbox-pack-coco-dataset \
  --input train=/path/to/train_coco.json \
  --input val=/path/to/val_coco.json \
  --out-dir /path/to/output_dataset \
  --run
```

새 CLI 명령이 shell에서 보이지 않으면 해당 conda env에서 editable install을 다시 실행한다.

```bash
python -m pip install -e .
```

## Git 주의

생성된 images, labels, annotations, manifest, summary는 전달용 산출물이다. 대량 dataset과 이미지 파일은 Git에 넣지 않는다.
