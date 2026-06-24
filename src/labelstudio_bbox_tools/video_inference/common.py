from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    class_id: int
    class_name: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "xyxy": [float(value) for value in self.xyxy],
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "score": float(self.score),
        }


@dataclass
class VideoSummary:
    video_path: str
    video_rel_path: str
    output_video_path: str | None
    fps: float
    frame_count: int
    duration_seconds: float | None
    width: int
    height: int
    processed_frames: int = 0
    boxes_written: int = 0
    dry_run: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class VideoInferenceResult:
    model_name: str
    input_path: Path
    out_dir: Path
    run_dir: Path | None
    videos: list[VideoSummary] = field(default_factory=list)
    predictions_path: Path | None = None
    summary_path: Path | None = None
    run_config_path: Path | None = None
    dry_run: bool = True

    @property
    def processed_frames(self) -> int:
        return sum(item.processed_frames for item in self.videos)

    @property
    def boxes_written(self) -> int:
        return sum(item.boxes_written for item in self.videos)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_path": str(self.input_path),
            "out_dir": str(self.out_dir),
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "video_count": len(self.videos),
            "processed_frames": self.processed_frames,
            "boxes_written": self.boxes_written,
            "predictions_path": str(self.predictions_path) if self.predictions_path else None,
            "summary_path": str(self.summary_path) if self.summary_path else None,
            "run_config_path": str(self.run_config_path) if self.run_config_path else None,
            "dry_run": self.dry_run,
            "videos": [item.as_dict() for item in self.videos],
        }


def import_cv2():
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for video inference. Install opencv-python in the active conda env."
        ) from exc
    return cv2


def safe_name(value: str) -> str:
    value = re.sub(r'[\\/:*?"<>|\s#]+', "_", str(value).strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "item"


def iter_video_files(
    input_path: str | Path,
    *,
    recursive: bool = False,
    extensions: Iterable[str] = VIDEO_EXTENSIONS,
) -> list[Path]:
    root = Path(input_path).expanduser().resolve()
    normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    if root.is_file():
        if root.suffix.lower() not in normalized_exts:
            raise ValueError(f"Input file is not a supported video extension: {root}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in normalized_exts)


def video_rel_path(video_path: Path, input_path: Path) -> Path:
    video_path = video_path.expanduser().resolve()
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        return Path(video_path.name)
    try:
        return video_path.relative_to(input_path)
    except ValueError:
        return Path(video_path.name)


def output_video_path(run_dir: Path, model_name: str, rel_path: Path) -> Path:
    parent_parts = [safe_name(part) for part in rel_path.parent.parts if part not in {"", "."}]
    parent = Path(*parent_parts) if parent_parts else Path()
    filename = f"{safe_name(rel_path.stem)}__{safe_name(model_name)}.mp4"
    return run_dir / "videos" / parent / filename


def make_run_dir(out_dir: str | Path, model_name: str, run_name: str | None = None) -> Path:
    out_root = Path(out_dir).expanduser().resolve()
    if run_name is None:
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__{safe_name(model_name)}"
    run_dir = out_root / safe_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def video_metadata(video_path: Path) -> tuple[float, int, int, int, float | None]:
    cv2 = import_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    duration = frame_count / fps if fps > 0 and frame_count > 0 else None
    return fps, frame_count, width, height, duration


def preview_video_inputs(
    *,
    input_path: str | Path,
    out_dir: str | Path,
    model_name: str,
    recursive: bool = False,
    max_videos: int | None = None,
    run_name: str | None = None,
) -> VideoInferenceResult:
    input_root = Path(input_path).expanduser().resolve()
    videos = iter_video_files(input_root, recursive=recursive)
    if max_videos is not None:
        videos = videos[: int(max_videos)]
    result = VideoInferenceResult(
        model_name=model_name,
        input_path=input_root,
        out_dir=Path(out_dir).expanduser().resolve(),
        run_dir=None,
        dry_run=True,
    )
    print(f"[dry-run] videos_found={len(videos):,}, recursive={recursive}, run_name={run_name or '<auto>'}")
    for video_path in videos:
        rel_path = video_rel_path(video_path, input_root)
        try:
            fps, frame_count, width, height, duration = video_metadata(video_path)
            summary = VideoSummary(
                video_path=str(video_path),
                video_rel_path=str(rel_path),
                output_video_path=str(output_video_path(Path(out_dir), model_name, rel_path)),
                fps=fps,
                frame_count=frame_count,
                duration_seconds=duration,
                width=width,
                height=height,
                dry_run=True,
            )
            print(f"[dry-run] {rel_path} fps={fps:.3f} frames={frame_count:,} size={width}x{height}")
        except Exception as exc:
            summary = VideoSummary(
                video_path=str(video_path),
                video_rel_path=str(rel_path),
                output_video_path=None,
                fps=0.0,
                frame_count=0,
                duration_seconds=None,
                width=0,
                height=0,
                dry_run=True,
                error=str(exc),
            )
            print(f"[dry-run][error] {rel_path}: {exc}")
        result.videos.append(summary)
    return result


def _frame_in_range(frame_index: int, fps: float, start_seconds: float | None, end_seconds: float | None) -> bool:
    if fps <= 0:
        return True
    timestamp = frame_index / fps
    if start_seconds is not None and timestamp < float(start_seconds):
        return False
    if end_seconds is not None and timestamp > float(end_seconds):
        return False
    return True


def write_run_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_videos_summary(path: Path, result: VideoInferenceResult) -> None:
    path.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_csv(path: Path, result: VideoInferenceResult) -> None:
    fieldnames = [
        "video_path",
        "video_rel_path",
        "output_video_path",
        "fps",
        "frame_count",
        "duration_seconds",
        "width",
        "height",
        "processed_frames",
        "boxes_written",
        "dry_run",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in result.videos:
            writer.writerow(item.as_dict())


def run_video_inference(
    *,
    input_path: str | Path,
    out_dir: str | Path,
    model_name: str,
    class_names: Sequence[str],
    color_map: dict[str, tuple[int, int, int]],
    predict_frame: Callable[[Any, int, float, Path | None], Sequence[Detection]],
    recursive: bool = False,
    max_videos: int | None = None,
    frame_stride: int = 1,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    max_frames: int | None = None,
    run_name: str | None = None,
    font_path: str | Path | None = None,
    font_size: int = 20,
    line_width: int = 3,
    score_digits: int = 2,
    codec: str = "mp4v",
    overwrite: bool = False,
    run_config: dict[str, Any] | None = None,
) -> VideoInferenceResult:
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    cv2 = import_cv2()
    from labelstudio_bbox_tools.video_inference.draw import draw_detections_on_bgr

    input_root = Path(input_path).expanduser().resolve()
    videos = iter_video_files(input_root, recursive=recursive)
    if max_videos is not None:
        videos = videos[: int(max_videos)]

    run_dir = make_run_dir(out_dir, model_name, run_name)
    predictions_path = run_dir / "predictions.jsonl"
    summary_path = run_dir / "videos_summary.json"
    summary_csv_path = run_dir / "videos_summary.csv"
    run_config_path = run_dir / "run_config.json"

    result = VideoInferenceResult(
        model_name=model_name,
        input_path=input_root,
        out_dir=Path(out_dir).expanduser().resolve(),
        run_dir=run_dir,
        predictions_path=predictions_path,
        summary_path=summary_path,
        run_config_path=run_config_path,
        dry_run=False,
    )

    config = dict(run_config or {})
    config.update(
        {
            "model_name": model_name,
            "input_path": str(input_root),
            "out_dir": str(Path(out_dir).expanduser().resolve()),
            "run_dir": str(run_dir),
            "recursive": recursive,
            "max_videos": max_videos,
            "frame_stride": frame_stride,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "max_frames": max_frames,
            "class_count": len(class_names),
            "font_path": str(font_path) if font_path else None,
            "codec": codec,
        }
    )
    write_run_config(run_config_path, config)

    with predictions_path.open("w", encoding="utf-8") as pred_file:
        for video_path in tqdm(videos, desc=f"{model_name} videos"):
            rel_path = video_rel_path(video_path, input_root)
            out_video = output_video_path(run_dir, model_name, rel_path)
            out_video.parent.mkdir(parents=True, exist_ok=True)
            if out_video.exists() and not overwrite:
                raise FileExistsError(f"Output video already exists: {out_video}")

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                result.videos.append(
                    VideoSummary(
                        video_path=str(video_path),
                        video_rel_path=str(rel_path),
                        output_video_path=str(out_video),
                        fps=0.0,
                        frame_count=0,
                        duration_seconds=None,
                        width=0,
                        height=0,
                        dry_run=False,
                        error="Could not open video",
                    )
                )
                continue

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = frame_count / fps if fps > 0 and frame_count > 0 else None
            writer = cv2.VideoWriter(
                str(out_video),
                cv2.VideoWriter_fourcc(*codec),
                fps if fps > 0 else 30.0,
                (width, height),
            )
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Could not open output video writer: {out_video}")

            summary = VideoSummary(
                video_path=str(video_path),
                video_rel_path=str(rel_path),
                output_video_path=str(out_video),
                fps=fps,
                frame_count=frame_count,
                duration_seconds=duration,
                width=width,
                height=height,
                dry_run=False,
            )
            frame_index = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if not _frame_in_range(frame_index, fps, start_seconds, end_seconds):
                        frame_index += 1
                        continue
                    if frame_index % frame_stride != 0:
                        frame_index += 1
                        continue
                    if max_frames is not None and summary.processed_frames >= max_frames:
                        break

                    timestamp = frame_index / fps if fps > 0 else 0.0
                    detections = list(predict_frame(frame, frame_index, timestamp, video_path))
                    drawn = draw_detections_on_bgr(
                        frame,
                        detections,
                        color_map=color_map,
                        font_path=font_path,
                        font_size=font_size,
                        line_width=line_width,
                        score_digits=score_digits,
                    )
                    writer.write(drawn)
                    summary.processed_frames += 1
                    summary.boxes_written += len(detections)

                    for det in detections:
                        row = {
                            "model_name": model_name,
                            "video_path": str(video_path),
                            "video_rel_path": str(rel_path),
                            "output_video_path": str(out_video),
                            "frame_index": frame_index,
                            "timestamp_seconds": timestamp,
                            **det.as_dict(),
                        }
                        pred_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    frame_index += 1
            finally:
                cap.release()
                writer.release()

            result.videos.append(summary)

    write_videos_summary(summary_path, result)
    write_summary_csv(summary_csv_path, result)
    print(f"[ok] run_dir={run_dir}")
    print(f"[ok] videos={len(result.videos):,}, frames={result.processed_frames:,}, boxes={result.boxes_written:,}")
    return result
