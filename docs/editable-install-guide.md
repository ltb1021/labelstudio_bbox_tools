# Editable Install Guide

이 문서는 `python -m pip install -e .`가 무엇을 하는지 설명한다. Label Studio 도구를 notebook에서 편하게 쓰려면 이 명령을 한 번 실행하는 것이 좋다.

## 한 줄 요약

`python -m pip install -e .`는 현재 repo를 Python 환경에 "개발 모드 패키지"로 등록한다.

그 결과 notebook이나 script에서 아래처럼 바로 import할 수 있다.

```python
from labelstudio_bbox_tools.importers.image_import import import_images
```

## 왜 필요한가

Python은 `import labelstudio_bbox_tools`를 실행할 때 정해진 검색 위치를 뒤진다. 이 검색 위치를 보통 `sys.path`라고 부른다.

아무 설정이 없으면 Python은 현재 폴더, 설치된 package 폴더, 표준 라이브러리 폴더 정도만 찾는다. repo 안의 `src/labelstudio_bbox_tools`는 자동으로 찾지 못할 수 있다.

예전 notebook에서는 이런 문제를 해결하려고 `sys.path.append(...)`를 자주 썼다. 하지만 이 방식은 notebook 위치가 바뀌면 깨지기 쉽고, 다른 서버에서 재사용할 때도 헷갈린다.

`pip install -e .`를 해두면 Python 환경이 이 repo를 package로 기억하므로 notebook 위치와 상관없이 import가 안정적으로 동작한다.

## 명령을 나눠서 이해하기

```bash
python -m pip install -e .
```

- `python`: 현재 conda 환경의 Python을 사용한다.
- `-m pip`: 그 Python에 연결된 pip를 실행한다. `pip` 명령만 쓰는 것보다 어떤 환경에 설치되는지 덜 헷갈린다.
- `install`: package를 설치한다.
- `-e`: editable mode, 즉 개발 모드로 설치한다.
- `.`: 현재 폴더를 설치 대상으로 사용한다. 현재 폴더에는 `pyproject.toml`이 있어야 한다.

## 일반 설치와 editable 설치 차이

일반 설치는 package 파일을 Python 환경의 site-packages 폴더로 복사한다. 이후 원본 코드를 고쳐도 다시 설치하기 전까지 반영되지 않을 수 있다.

editable 설치는 site-packages에 원본 repo를 가리키는 연결 정보를 만든다. 그래서 repo 안의 `.py` 파일을 수정하면 다음 Python 실행부터 바로 반영된다.

실무적으로는 다음 의미다.

- 개발 중인 내 repo를 notebook에서 바로 import할 수 있다.
- 코드를 고칠 때마다 재설치하지 않아도 된다.
- conda 환경마다 따로 실행해야 한다.

## conda 환경과의 관계

`pip install -e .`는 모든 conda 환경에 한 번에 적용되지 않는다. 현재 활성화된 conda 환경 하나에만 적용된다.

예를 들어 아래 순서로 실행하면 `ltb_ultra` 환경에만 등록된다.

```bash
conda activate ltb_ultra
cd /path/to/labelstudio_bbox_tools
python -m pip install -e .
```

나중에 다른 환경에서 쓰려면 그 환경에서도 다시 실행한다.

```bash
conda activate another_env
cd /path/to/labelstudio_bbox_tools
python -m pip install -e .
```

## 설치가 되었는지 확인

```bash
python -c "import labelstudio_bbox_tools; print(labelstudio_bbox_tools.__version__)"
```

오류 없이 version이 출력되면 import 준비가 된 것이다.

## 제거 방법

현재 conda 환경에서 등록을 해제하려면 다음을 실행한다.

```bash
python -m pip uninstall labelstudio-bbox-tools
```

이 명령은 package 등록만 해제한다. repo 폴더 자체를 삭제하지는 않는다.

## 자주 헷갈리는 점

### Q. 이 명령이 내 코드를 GitHub에 올리나요?

아니다. 이 명령은 현재 PC의 Python 환경에 package를 등록할 뿐이다. Git commit, push와는 관련이 없다.

### Q. `.env`나 dataset도 설치되나요?

아니다. Python package 코드만 import 가능하게 만든다. `.env`, dataset, export 결과, model weights는 별도로 관리한다.

### Q. notebook에서 `sys.path.append`를 계속 써야 하나요?

권장하지 않는다. 다만 처음 설치 전 확인용 notebook에서는 repo 내부 실행을 돕기 위해 임시 fallback을 둘 수 있다. 실제 사용은 editable install을 기준으로 하는 것이 좋다.
