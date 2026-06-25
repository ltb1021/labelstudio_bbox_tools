from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from tqdm import tqdm

from labelstudio_bbox_tools.video_inference.classes import load_class_names, make_class_color_map
from labelstudio_bbox_tools.video_inference.common import (
    import_cv2,
    iter_video_files,
    make_run_dir,
    output_video_path,
    preview_video_inputs,
    video_rel_path,
)

COCO_PERSON_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


@dataclass(frozen=True)
class PoseKeypoint:
    name: str
    x: float
    y: float
    confidence: float | None = None

    def is_drawable(self, threshold: float = 0.2) -> bool:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            return False
        if self.confidence is None:
            return True
        return math.isfinite(self.confidence) and self.confidence >= threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x": float(self.x),
            "y": float(self.y),
            "confidence": None if self.confidence is None else float(self.confidence),
        }


@dataclass(frozen=True)
class PoseInstance:
    xyxy: tuple[float, float, float, float]
    class_id: int
    class_name: str
    score: float
    keypoints: tuple[PoseKeypoint, ...]
    raw_score: float | None = None
    source: str | None = None
    detection_class_name: str | None = None
    detection_score: float | None = None
    detection_xyxy: tuple[float, float, float, float] | None = None
    crop_xyxy: tuple[float, float, float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "xyxy": [float(value) for value in self.xyxy],
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "score": float(self.score),
            "raw_score": None if self.raw_score is None else float(self.raw_score),
            "source": self.source,
            "detection_class_name": self.detection_class_name,
            "detection_score": None if self.detection_score is None else float(self.detection_score),
            "detection_xyxy": None if self.detection_xyxy is None else [float(value) for value in self.detection_xyxy],
            "crop_xyxy": None if self.crop_xyxy is None else [float(value) for value in self.crop_xyxy],
            "keypoints": [keypoint.as_dict() for keypoint in self.keypoints],
        }


@dataclass
class PoseVideoSummary:
    video_path: str
    video_rel_path: str
    output_video_path: str | None
    fps: float
    frame_count: int
    duration_seconds: float | None
    width: int
    height: int
    processed_frames: int = 0
    instances_written: int = 0
    keypoints_written: int = 0
    dry_run: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PoseInferenceResult:
    model_name: str
    input_path: Path
    out_dir: Path
    run_dir: Path | None
    videos: list[PoseVideoSummary] = field(default_factory=list)
    predictions_path: Path | None = None
    summary_path: Path | None = None
    run_config_path: Path | None = None
    dry_run: bool = True

    @property
    def processed_frames(self) -> int:
        return sum(item.processed_frames for item in self.videos)

    @property
    def instances_written(self) -> int:
        return sum(item.instances_written for item in self.videos)

    @property
    def keypoints_written(self) -> int:
        return sum(item.keypoints_written for item in self.videos)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_path": str(self.input_path),
            "out_dir": str(self.out_dir),
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "video_count": len(self.videos),
            "processed_frames": self.processed_frames,
            "instances_written": self.instances_written,
            "keypoints_written": self.keypoints_written,
            "predictions_path": str(self.predictions_path) if self.predictions_path else None,
            "summary_path": str(self.summary_path) if self.summary_path else None,
            "run_config_path": str(self.run_config_path) if self.run_config_path else None,
            "dry_run": self.dry_run,
            "videos": [item.as_dict() for item in self.videos],
        }


def load_pose_class_names(
    *,
    class_yaml: str | Path | None = None,
    manual_classes: Sequence[str] | None = None,
    expected_count: int | None = None,
    strict_count: bool = False,
) -> list[str]:
    if class_yaml or manual_classes:
        return load_class_names(
            class_yaml=class_yaml,
            manual_classes=manual_classes,
            expected_count=expected_count,
            strict_count=strict_count,
        )
    return ["person"]


def keypoints_from_xy_conf(
    xy: Sequence[Sequence[float]],
    confidence: Sequence[float] | None = None,
    *,
    names: Sequence[str] = COCO_PERSON_KEYPOINT_NAMES,
) -> tuple[PoseKeypoint, ...]:
    keypoints: list[PoseKeypoint] = []
    for idx, point in enumerate(xy):
        if idx >= len(names) or len(point) < 2:
            continue
        conf = None
        if confidence is not None and idx < len(confidence):
            conf = float(confidence[idx])
        keypoints.append(PoseKeypoint(name=names[idx], x=float(point[0]), y=float(point[1]), confidence=conf))
    return tuple(keypoints)


def _frame_in_range(frame_index: int, fps: float, start_seconds: float | None, end_seconds: float | None) -> bool:
    if fps <= 0:
        return True
    timestamp = frame_index / fps
    if start_seconds is not None and timestamp < float(start_seconds):
        return False
    if end_seconds is not None and timestamp > float(end_seconds):
        return False
    return True


def _write_run_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_videos_summary(path: Path, result: PoseInferenceResult) -> None:
    path.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary_csv(path: Path, result: PoseInferenceResult) -> None:
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
        "instances_written",
        "keypoints_written",
        "dry_run",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in result.videos:
            writer.writerow(item.as_dict())


def preview_pose_video_inputs(
    *,
    input_path: str | Path,
    out_dir: str | Path,
    model_name: str,
    recursive: bool = False,
    max_videos: int | None = None,
    run_name: str | None = None,
):
    return preview_video_inputs(
        input_path=input_path,
        out_dir=out_dir,
        model_name=model_name,
        recursive=recursive,
        max_videos=max_videos,
        run_name=run_name,
    )


def run_pose_video_inference(
    *,
    input_path: str | Path,
    out_dir: str | Path,
    model_name: str,
    class_names: Sequence[str],
    color_map: dict[str, tuple[int, int, int]],
    predict_frame: Callable[[Any, int, float, Path | None], Sequence[PoseInstance]],
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
    keypoint_radius: int = 4,
    keypoint_conf: float = 0.2,
    score_digits: int = 2,
    draw_bbox: bool = True,
    draw_skeleton: bool = True,
    draw_keypoints: bool = True,
    codec: str = "mp4v",
    overwrite: bool = False,
    run_config: dict[str, Any] | None = None,
) -> PoseInferenceResult:
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    cv2 = import_cv2()
    from labelstudio_bbox_tools.pose_inference.draw import draw_pose_instances_on_bgr

    input_root = Path(input_path).expanduser().resolve()
    videos = iter_video_files(input_root, recursive=recursive)
    if max_videos is not None:
        videos = videos[: int(max_videos)]

    run_dir = make_run_dir(out_dir, model_name, run_name)
    predictions_path = run_dir / "predictions.jsonl"
    summary_path = run_dir / "videos_summary.json"
    summary_csv_path = run_dir / "videos_summary.csv"
    run_config_path = run_dir / "run_config.json"

    result = PoseInferenceResult(
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
            "font_size": font_size,
            "line_width": line_width,
            "keypoint_radius": keypoint_radius,
            "keypoint_conf": keypoint_conf,
            "draw_bbox": draw_bbox,
            "draw_skeleton": draw_skeleton,
            "draw_keypoints": draw_keypoints,
            "codec": codec,
        }
    )
    _write_run_config(run_config_path, config)

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
                    PoseVideoSummary(
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

            summary = PoseVideoSummary(
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
                    instances = list(predict_frame(frame, frame_index, timestamp, video_path))
                    drawn = draw_pose_instances_on_bgr(
                        frame,
                        instances,
                        color_map=color_map,
                        font_path=font_path,
                        font_size=font_size,
                        line_width=line_width,
                        keypoint_radius=keypoint_radius,
                        keypoint_conf=keypoint_conf,
                        score_digits=score_digits,
                        draw_bbox=draw_bbox,
                        draw_skeleton=draw_skeleton,
                        draw_keypoints=draw_keypoints,
                    )
                    writer.write(drawn)
                    summary.processed_frames += 1
                    summary.instances_written += len(instances)
                    summary.keypoints_written += sum(
                        1 for instance in instances for keypoint in instance.keypoints if keypoint.is_drawable(keypoint_conf)
                    )

                    for instance in instances:
                        row = {
                            "model_name": model_name,
                            "video_path": str(video_path),
                            "video_rel_path": str(rel_path),
                            "output_video_path": str(out_video),
                            "frame_index": frame_index,
                            "timestamp_seconds": timestamp,
                            **instance.as_dict(),
                        }
                        pred_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    frame_index += 1
            finally:
                cap.release()
                writer.release()

            result.videos.append(summary)

    _write_videos_summary(summary_path, result)
    _write_summary_csv(summary_csv_path, result)
    print(f"[ok] run_dir={run_dir}")
    print(
        f"[ok] videos={len(result.videos):,}, frames={result.processed_frames:,}, "
        f"instances={result.instances_written:,}, keypoints={result.keypoints_written:,}"
    )
    return result


def default_pose_color_map(class_names: Sequence[str] | None = None) -> dict[str, tuple[int, int, int]]:
    return make_class_color_map(class_names or ["person"])

