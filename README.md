# labelstudio_bbox_tools

Label Studio object detection bbox workflow를 정리하기 위한 작은 Python package다. 1차 범위는 이미지 import, MMYOLO/COCO JSON 기반 bbox UI 설정, annotation export다.

## Current Scope

- 이미지 폴더를 Label Studio project에 import
- MMYOLO/COCO `categories`에서 class list 추출
- Label Studio bbox 또는 polygon UI 적용
- manual annotation 또는 model prediction을 `mmyolo`, `yolo`, `yolo_obb`로 export

## RTX4090 Quick Start

```bash
cd /share_ssd/ltb/Users/ltb/git_repos/labelstudio_bbox_tools
conda activate ltb_ultra
python -m pip install -e .
cp .env.example .env
```

`.env`에는 실제 `LABEL_STUDIO_API_KEY`를 로컬에서만 채운다. 이 파일은 Git에서 제외된다.

## Dry Run Checks

```bash
lsbbox-import-images \
  --src-root "/share_ssd/ltb/Users/ltb/label_studio/운영서버_도메인맞춤_추가학습용/260608/오탐/쓰러짐/images" \
  --project-title "refactor_smoke_import" \
  --dry-run

lsbbox-apply-ui \
  --project-id 0 \
  --mmyolo-json "/share_ssd/ltb/Users/ltb/label_studio/박스_데이터셋_250507/박스_mmyolo/260609_형주책임님_small데이터셋_1차테스트용_export/annotations_mmyolo.json" \
  --shape bbox \
  --dry-run
```

## Basic Workflow

1. `lsbbox-import-images`로 이미지 import
2. `lsbbox-apply-ui`로 bbox UI와 class list 적용
3. Label Studio 브라우저에서 labeling
4. `lsbbox-export`로 MMYOLO export

예시:

```bash
lsbbox-export \
  --project-id 123 \
  --out-dir "/share_ssd/ltb/Users/ltb/label_studio/260610_labelstudio_export_refactoring_test" \
  --export-type ann \
  --ann-format mmyolo \
  --source-type ann \
  --ann-user-id 1 \
  --ann-min-lead 0
```

## Path Model

자세한 설명은 [docs/path-and-mount-guide.md](docs/path-and-mount-guide.md)를 본다.

- `LABEL_STUDIO_DOC_ROOT`: Label Studio container가 local-files로 서빙하는 root
- `src_root`: import할 이미지 폴더
- `mirror_root`: `src_root`가 `doc_root` 밖일 때 symlink 구조를 만들 기준 root

## Safety

- 실제 token은 repo에 저장하지 않는다.
- dataset, export 결과, model weights는 Git에 넣지 않는다.
- 기존 실무 폴더는 source of truth로 보존하고, 이 repo에는 검증된 core 기능만 가져온다.

