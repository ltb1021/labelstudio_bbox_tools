# Video Frame Extraction Guide

이 문서는 영상 파일을 bbox 라벨링용 이미지 frame으로 나누는 workflow를 설명한다. 결과 이미지는 `01_ls_import_images.ipynb`로 Label Studio에 import할 수 있는 일반 이미지 폴더가 된다.

## 언제 쓰나

- CCTV, 현장 촬영 영상, 녹화 파일에서 일정 간격으로 라벨링 이미지를 만들 때
- 한 영상 안에서 너무 비슷한 frame을 모두 라벨링하지 않고 대표 frame만 뽑고 싶을 때
- 나중에 이미지가 원본 영상의 몇 초 지점에서 나왔는지 추적하고 싶을 때

## Notebook

```text
examples/notebooks/labelstudio/07_ls_extract_video_frames.ipynb
```

기본 흐름은 다음과 같다.

1. `INPUT_PATH`에 영상 파일 또는 영상 폴더를 넣는다.
2. `OUT_DIR`에 frame을 저장할 별도 output 경로를 넣는다.
3. `RECURSIVE=False`이면 대상 폴더 바로 아래 영상만 찾는다.
4. `RECURSIVE=True`이면 하위 폴더까지 모두 찾는다.
5. `INTERVAL_SECONDS=2.0`처럼 몇 초마다 한 장을 뽑을지 정한다.
6. 먼저 preview cell에서 예상 영상 수, 예상 frame 수, output 경로를 확인한다.
7. 실제 저장할 때만 `RUN_EXTRACT=True`로 바꾼다.

## Sampling Mode

세 가지 방식 중 하나만 사용한다. 가장 추천하는 기본값은 `INTERVAL_SECONDS`다.

- `INTERVAL_SECONDS=2.0`: 영상 FPS를 읽어서 약 2초마다 한 장을 저장한다.
- `EVERY_N_FRAMES=60`: 원본 frame 기준 60장마다 한 장을 저장한다. FPS가 이상하게 읽히는 영상에 유용하다.
- `TARGET_FPS=0.5`: 결과 이미지를 초당 0.5장 정도로 맞춘다. 2초마다 한 장과 비슷하다.

`INTERVAL_SECONDS`를 쓰면 OpenCV가 읽은 FPS를 기준으로 frame 간격을 계산한다. 예를 들어 30 FPS 영상에서 `INTERVAL_SECONDS=2.0`이면 약 60 frame마다 저장한다.

## Output Structure

기본 구조는 다음과 같다.

```text
OUT_DIR/
  frames/
    video_stem/
      video_stem__idx000000__frame00000000__t000000.000s.jpg
      video_stem__idx000001__frame00000060__t000002.000s.jpg
  manifests/
    frames_manifest.csv
    videos_summary.json
```

파일명에는 원본 영상 이름, 저장 순서, 원본 frame index, timestamp가 들어간다. 나중에 문제가 있는 이미지가 어느 영상의 어느 시점에서 나왔는지 추적하기 위한 정보다.

## Manifest

실제 저장을 실행하면 `manifests/frames_manifest.csv`와 `manifests/videos_summary.json`이 만들어진다.

주요 필드:

- `video_path`: 원본 영상 경로
- `video_rel_path`: 입력 폴더 기준 상대 영상 경로
- `frame_path`: 저장된 이미지 경로
- `frame_index`: 원본 영상에서의 frame index
- `timestamp_seconds`: 원본 영상 기준 시간
- `fps`, `width`, `height`: OpenCV가 읽은 영상 정보
- `labelstudio_import_hint`: 추후 Label Studio import 연계에 활용할 수 있는 frame 경로

현재는 Label Studio project import까지 자동으로 연결하지 않는다. 먼저 frame 추출 결과를 눈으로 확인하고, 이후 `01_ls_import_images.ipynb`로 image import를 실행하는 방식을 권장한다.

## CLI Preview

```bash
lsbbox-extract-video-frames \
  --input-path "/path/to/video_or_video_folder" \
  --out-dir "/path/to/frames_export" \
  --interval-seconds 2.0 \
  --max-videos 3 \
  --max-frames-per-video 20 \
  --dry-run
```

하위 폴더까지 찾으려면 `--recursive`를 추가한다.

## Safety

- 기본 notebook은 실제 저장 cell이 꺼져 있다.
- 기존 frame이 있으면 기본적으로 덮어쓰지 않고 건너뛴다.
- `SKIP_EXISTING=False` 또는 CLI `--overwrite`는 기존 이미지를 다시 쓸 수 있으므로 output 경로를 먼저 확인한다.
- 긴 영상 폴더는 처음에 `MAX_VIDEOS`와 `MAX_FRAMES_PER_VIDEO`를 작은 값으로 두고 preview한다.
