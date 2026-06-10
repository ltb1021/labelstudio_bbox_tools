# GitHub Remote Setup Guide

로컬 commit 확인 후 GitHub에서 빈 repo를 만든다. 예시는 `labelstudio_bbox_tools` 기준이다.

```bash
cd /share_ssd/ltb/Users/ltb/git_repos/labelstudio_bbox_tools
git remote add origin git@github.com:ltb1021/labelstudio_bbox_tools.git
git branch -M main
git push -u origin main
```

이미 remote가 있으면 먼저 확인한다.

```bash
git remote -v
```

실제 `.env`, dataset, export 결과, model weights는 push하지 않는다.

