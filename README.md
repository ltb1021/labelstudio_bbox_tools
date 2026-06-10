# labelstudio_bbox_tools

Label Studio object detection bbox workflow를 정리한 Python package다. 1차 범위는 이미지 import, MMYOLO/COCO JSON 기반 bbox UI 설정, annotation export다.

이 repo는 **notebook-first** 사용을 기본으로 한다. 기존처럼 notebook cell을 순서대로 실행하면서 경로, project id, class 목록, export 결과를 눈으로 확인할 수 있게 구성한다. shell script는 반복 작업이나 자동화가 필요할 때 보조로 사용한다.

## Current Scope

- 이미지 폴더를 Label Studio project에 import
- MMYOLO/COCO `categories`에서 class list 추출
- Label Studio bbox 또는 polygon UI 적용
- manual annotation 또는 model prediction을 `mmyolo`, `yolo`, `yolo_obb`로 export

아직 1차 범위에 포함하지 않은 기능:

- pseudo labeling
- MMYOLO to YOLO 변환
- ann/pred merge 고도화
- Docker container 생성 자동화

## Install

권장 환경은 현재 RTX4090 서버 기준 `ltb_ultra` conda env다.

```bash
cd /path/to/labelstudio_bbox_tools
conda activate ltb_ultra
python -m pip install -e .
cp .env.example .env
```

`.env`에는 본인 서버의 Label Studio 접속 정보를 채운다. 실제 API key는 Git에 올리지 않는다.

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
2. `01_ls_import_images.ipynb`: 이미지 폴더 import
3. `02_ls_apply_bbox_ui_from_mmyolo.ipynb`: MMYOLO/COCO JSON에서 class를 읽어 bbox UI 적용
4. Label Studio 브라우저에서 bbox labeling
5. `03_ls_export_mmyolo.ipynb`: annotation을 MMYOLO 형식으로 export

각 notebook은 처음에는 실제 Label Studio 데이터를 바꾸지 않는 안전한 기본값으로 시작한다.

- `DRY_RUN=True`
- `PREVIEW_ONLY=True`
- `RUN_EXPORT=False`

출력을 확인한 뒤 필요한 경우에만 값을 바꿔서 실행한다.

자세한 notebook 운영 정책은 [docs/notebook-workflow-guide.md](docs/notebook-workflow-guide.md)를 본다.

## Shell Command 사용법

notebook으로 검증한 뒤 같은 기능을 command로 반복 실행할 수 있다.

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
