# Path And Mount Guide

Label Studio의 `/data/local-files/?d=...` URL은 container가 접근 가능한 `doc_root`를 기준으로 동작한다.

## Core Terms

- `doc_root`: Label Studio container가 local files로 서빙하는 root. 현재 RTX4090에서는 `/share_ssd/ltb/Users/ltb/label_studio`.
- `src_root`: import하려는 실제 이미지 폴더.
- `mirror_root`: `src_root`가 `doc_root` 밖에 있을 때, 기존 host 경로 구조를 `doc_root/img_symlinks` 아래에 보존하기 위한 기준 root.

## Recommended Rule

가능하면 dataset을 Label Studio container에 bind mount된 `doc_root` 또는 같은 경로로 접근 가능한 mount root 아래에 둔다.

`src_root`가 `doc_root` 내부이면 symlink가 필요 없다. `src_root`가 외부이면 `mirror_root`를 명시해서 symlink 경로를 안정적으로 만든다.

## Docker Notes

현재 RTX4090의 Label Studio container는 다음 mount를 사용한다.

- `/share_ssd/ltb/Users/ltb/label_studio`
- `/share_T7_50`
- `/share_T5_48`
- `/share_T7_49`

다른 Ubuntu 서버에서는 같은 절대경로를 강제하지 말고, container 생성 시 host dataset root를 명시적으로 bind mount한 뒤 `.env`의 `LABEL_STUDIO_DOC_ROOT`와 맞춘다.

