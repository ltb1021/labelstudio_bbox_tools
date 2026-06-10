# Basic Bbox Workflow

1. 이미지 폴더를 Label Studio project로 import한다.
2. MMYOLO/COCO JSON의 `categories`에서 class list를 추출한다.
3. 해당 project에 bbox UI를 적용한다.
4. 브라우저에서 수동 labeling을 진행한다.
5. 원하는 `source_type`, `ann_user_id`, `ann_min_lead` 조건으로 MMYOLO JSON을 export한다.

## Dry Run

```bash
cp .env.example .env
python -m pip install -e .
lsbbox-import-images --src-root "<image-dir>" --project-title "test" --dry-run
lsbbox-apply-ui --project-id 0 --mmyolo-json "<annotations_mmyolo.json>" --dry-run
```

`--dry-run`은 Label Studio project/task를 만들지 않는다.

## Mutating Commands

아래 명령은 Label Studio project 또는 task를 실제로 수정한다.

```bash
lsbbox-import-images --src-root "<image-dir>" --project-title "my_dataset"
lsbbox-apply-ui --project-id 123 --mmyolo-json "<annotations_mmyolo.json>"
lsbbox-export --project-id 123 --out-dir "<export-dir>" --ann-user-id 1
```

