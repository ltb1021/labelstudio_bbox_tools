# Local Instructions for labelstudio_bbox_tools

## Role
- 이 repo는 Label Studio object detection bbox workflow를 재사용 가능한 Python package, notebook, shell script, 문서로 정리하는 프로젝트다.
- 1차 핵심 흐름은 image import, bbox UI/class setup, MMYOLO export다.

## Working Rules
- 사용자 수동 작업 가이드는 notebook-first로 작성한다.
- shell script는 반복 실행과 자동화용으로 유지하되, README에서는 notebook 사용법을 먼저 안내한다.
- notebook 파일은 `examples/notebooks/labelstudio/` 아래에 두고 파일명에 `ls_` prefix를 붙인다.
- notebook은 기본적으로 dry-run, preview, RUN_EXPORT=False처럼 안전한 값에서 시작해야 한다.
- token, password, private key, 실제 `.env` 값은 notebook, README, commit에 넣지 않는다.
- dataset, export 결과, model weights는 Git에 넣지 않는다.
- 4090 서버 특화 절대경로는 문서의 기본 예시로 고정하지 말고, 사용자가 직접 채울 placeholder와 개념 설명을 우선한다.

## Documentation Style
- 한국어 중심으로 설명한다.
- 초보자도 따라갈 수 있도록 용어를 먼저 풀고, 그 다음 명령과 예시를 제시한다.
- `doc_root`, `mirror_root`, editable install, conda env, Label Studio local-files 같은 개념은 단순한 비유와 실무 의미를 함께 설명한다.

## Validation
- 실제 Label Studio project/task를 생성하거나 수정하는 검증은 실행 전 사용자에게 한 번 더 확인한다.
- 기본 검증은 `compileall`, notebook JSON parse, dry-run, import smoke check 순서로 한다.
