# Path And Mount Guide

Label Studio image import/export에서 가장 자주 헷갈리는 부분은 경로다. 특히 Docker container로 Label Studio를 실행하면 "host terminal에서 보이는 경로"와 "Label Studio container가 접근할 수 있는 경로"를 구분해야 한다.

## 핵심 개념

### host path

Ubuntu terminal에서 보이는 실제 파일 경로다.

예시 형식:

```text
/path/on/host/datasets/images/a.jpg
```

Python script는 기본적으로 host path를 보고 파일을 찾는다.

### container path

Docker container 안에서 보이는 경로다. Docker를 실행할 때 `-v host_path:container_path` 형태로 mount하면 container가 host 파일을 볼 수 있다.

### local-files URL

Label Studio는 browser task에 직접 host absolute path를 넣는 대신 아래 같은 URL을 자주 사용한다.

```text
/data/local-files/?d=relative/path/to/image.jpg
```

여기서 `d=` 뒤의 값은 `doc_root` 기준 상대경로라고 생각하면 된다.

## 이 repo의 용어

### doc_root

Label Studio가 local files로 서빙하는 root다.

`.env`에서는 아래 값으로 설정한다.

```text
LABEL_STUDIO_DOC_ROOT=/path/to/label_studio_data
```

`/data/local-files/?d=a/b.jpg`가 있으면 실제 파일은 보통 아래처럼 해석된다.

```text
{doc_root}/a/b.jpg
```

### src_root

이번에 import하려는 이미지 폴더다.

```text
/path/to/my/images
```

### mirror_root

`src_root`가 `doc_root` 밖에 있을 때 사용하는 기준 경로다.

이 repo는 `doc_root/img_symlinks/...` 아래에 symlink를 만들 수 있다. 이때 `mirror_root`를 기준으로 원래 폴더 구조를 최대한 보존한다.

## 가장 단순한 권장 구조

처음 세팅하는 서버라면 이미지를 `doc_root` 아래에 두는 것이 가장 쉽다.

```text
label_studio_data/
  datasets/
    my_project/
      images/
        a.jpg
        b.jpg
```

이 경우 `src_root`는 아래처럼 된다.

```text
/path/to/label_studio_data/datasets/my_project/images
```

이미 `doc_root` 안에 있으므로 `mirror_root`가 필요 없다.

## doc_root 밖의 dataset을 import하는 경우

이미지가 다른 디스크나 다른 mount root에 있을 수 있다.

```text
/path/to/large_disk/datasets/my_project/images
```

이 경우 Label Studio가 그 경로를 직접 볼 수 없다면 import가 실패하거나 browser에서 이미지가 안 보일 수 있다.

해결 방법은 두 가지다.

1. Docker container 실행 시 해당 dataset root를 mount한다.
2. `doc_root/img_symlinks` 아래에 symlink를 만들어 local-files URL이 `doc_root` 기준으로 동작하게 한다.

이 repo의 `lsbbox-import-images`는 두 번째 방식을 도와준다.

## mirror_root 예시

다음 구조를 가정한다.

```text
/path/to/large_disk/datasets/project_a/images/a.jpg
```

`mirror_root`를 아래처럼 잡는다.

```text
/path/to/large_disk/datasets
```

그러면 symlink는 대략 아래처럼 만들어진다.

```text
{doc_root}/img_symlinks/project_a/images/a.jpg -> /path/to/large_disk/datasets/project_a/images/a.jpg
```

Label Studio task에는 아래 같은 URL이 들어간다.

```text
/data/local-files/?d=img_symlinks/project_a/images/a.jpg
```

## 다른 Ubuntu 서버로 옮길 때

다른 서버에서 같은 절대경로를 맞추려고 하지 말고, 다음 순서로 생각한다.

1. Label Studio data root를 정한다.
2. Docker container가 그 root를 볼 수 있게 mount한다.
3. `.env`의 `LABEL_STUDIO_DOC_ROOT`를 그 root로 설정한다.
4. dataset을 가능하면 그 root 아래에 둔다.
5. 외부 dataset을 써야 하면 `mirror_root`를 명시한다.

## 문제 해결

### 브라우저에서 이미지가 깨진다

가능성이 높은 원인:

- task의 `/data/local-files/?d=...` 경로가 잘못됨
- `LABEL_STUDIO_DOC_ROOT`가 실제 container local-files root와 다름
- Docker container가 해당 host path를 mount하지 않음
- symlink가 container에서 따라갈 수 없는 경로를 가리킴

먼저 import notebook에서 dry-run URL을 확인하고, 그 URL의 `d=` 값이 `doc_root` 아래 실제 파일로 이어지는지 확인한다.

### host에서는 파일이 있는데 Label Studio에서는 안 보인다

host Python은 파일을 볼 수 있지만 Docker container는 못 볼 수 있다. Docker run 또는 compose에서 bind mount가 되어 있는지 확인해야 한다.
