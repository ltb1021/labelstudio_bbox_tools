# Environment File Guide

이 문서는 `.env` 파일을 어떻게 만들고 수정해서 쓰는지 설명한다.

## 결론

4090 서버에서는 사용자가 직접 `.env` 파일을 만들고, 본인 서버 설정을 채워서 사용하면 된다.

`.env.example`은 양식만 보여주는 파일이다. 실제 작업에서는 아래처럼 복사해서 `.env`를 만든다.

```bash
cp .env.example .env
```

그 다음 `.env` 파일을 열어서 본인 서버 값으로 바꾼다.

```text
LABEL_STUDIO_URL=http://your-server-ip:9225
LABEL_STUDIO_API_KEY=your-local-token
LABEL_STUDIO_DOC_ROOT=/path/to/label_studio_data
```

## `.env`를 쓰는 이유

Label Studio 작업에는 자주 바뀌거나 공개하면 안 되는 값이 있다.

- 서버 주소
- API token
- 서버별 dataset mount root
- Label Studio local-files root

이 값을 notebook이나 Python 코드에 직접 적으면 문제가 생긴다.

- GitHub에 token이 올라갈 위험이 있다.
- 다른 서버에서 재사용할 때 경로를 모두 찾아 바꿔야 한다.
- 같은 코드인데 사람마다 다른 값을 써야 하는 상황을 관리하기 어렵다.

그래서 코드는 공통으로 두고, 서버마다 다른 값은 `.env`에 둔다.

## 각 항목 설명

### LABEL_STUDIO_URL

브라우저나 API에서 접속하는 Label Studio 주소다.

예시 형식:

```text
LABEL_STUDIO_URL=http://your-server-ip:9225
```

Docker container 내부 port가 아니라, 사용자가 접속하는 host IP와 port를 쓴다.

### LABEL_STUDIO_API_KEY

Label Studio 사용자 계정의 API token이다. 이 값으로 Python 코드가 project 생성, task import, label config 수정, export 요청을 할 수 있다.

주의:

- 채팅에 붙여넣지 않는다.
- README에 쓰지 않는다.
- notebook cell에 직접 쓰지 않는다.
- Git commit에 포함하지 않는다.

### LABEL_STUDIO_DOC_ROOT

Label Studio가 `/data/local-files/?d=...` URL로 파일을 찾을 때 기준이 되는 root 경로다.

예를 들어 `LABEL_STUDIO_DOC_ROOT=/data/label_studio`이고 task image URL이 아래와 같다면,

```text
/data/local-files/?d=my_dataset/images/a.jpg
```

Label Studio는 실제 파일을 아래처럼 찾는다고 생각하면 된다.

```text
/data/label_studio/my_dataset/images/a.jpg
```

## `.env`는 Git에 올라가나?

올라가지 않는다. `.gitignore`에 `.env`가 들어 있다.

대신 `.env.example`은 Git에 올린다. 이 파일은 실제 값 없이 어떤 항목이 필요한지만 보여준다.

## 여러 서버에서 쓰는 방법

서버마다 `.env`를 따로 만든다.

- 4090 서버의 `.env`
- 다른 Ubuntu 서버의 `.env`
- 테스트 PC의 `.env`

코드는 그대로 두고 `.env`만 각 서버에 맞게 바꾸는 방식이 가장 안전하다.

## 값이 잘 읽히는지 확인

`examples/notebooks/labelstudio/00_ls_check_environment.ipynb`를 실행한다.

또는 terminal에서 다음처럼 확인할 수 있다.

```bash
python -c "from labelstudio_bbox_tools.config import settings_from_env; s=settings_from_env('.env'); print(s.url); print(s.doc_root); print('<api key hidden>' if s.api_key else '<missing>')"
```
