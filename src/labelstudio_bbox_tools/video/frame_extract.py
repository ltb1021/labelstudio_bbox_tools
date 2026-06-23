"""Extract representative frames from video files for bbox labeling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


@dataclass
class VideoSummary:
    video_path: str
    video_rel_path: str
    output_dir: str
    fps: float
    frame_count: int
    duration_seconds: float | None
    width: int
    height: int
    planned_frames: int
    saved_frames: int = 0
    skipped_existing: int = 0
    failed_frames: int = 0
    dry_run: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "video_rel_path": self.video_rel_path,
            "output_dir": self.output_dir,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
            "width": self.width,
            "height": self.height,
            "planned_frames": self.planned_frames,
            "saved_frames": self.saved_frames,
            "skipped_existing": self.skipped_existing,
            "failed_frames": self.failed_frames,
            "dry_run": self.dry_run,
            "error": self.error,
        }


@dataclass
class ExtractResult:
    input_path: Path
    out_dir: Path
    videos: list[VideoSummary] = field(default_factory=list)
    manifest_path: Path | None = None
    summary_path: Path | None = None
    manifest_rows: int = 0
    dry_run: bool = True

    @property
    def planned_frames(self) -> int:
        return sum(item.planned_frames for item in self.videos)

    @property
    def saved_frames(self) -> int:
        return sum(item.saved_frames for item in self.videos)

    @property
    def skipped_existing(self) -> int:
        return sum(item.skipped_existing for item in self.videos)

    @property
    def failed_frames(self) -> int:
        return sum(item.failed_frames for item in self.videos)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "out_dir": str(self.out_dir),
            "video_count": len(self.videos),
            "planned_frames": self.planned_frames,
            "saved_frames": self.saved_frames,
            "skipped_existing": self.skipped_existing,
            "failed_frames": self.failed_frames,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "summary_path": str(self.summary_path) if self.summary_path else None,
            "manifest_rows": self.manifest_rows,
            "dry_run": self.dry_run,
            "videos": [item.as_dict() for item in self.videos],
        }


def _import_cv2():
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for video frame extraction. "
            "Install it in the active environment, for example: python -m pip install 'labelstudio-bbox-tools[video]'"
        ) from exc
    return cv2


def _safe_name(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|\s#]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "video"


def _iter_video_files(input_path: Path, recursive: bool, extensions: Iterable[str] = VIDEO_EXTENSIONS) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    if input_path.is_file():
        if input_path.suffix.lower() not in normalized_exts:
            raise ValueError(f"Input file is not a supported video extension: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*" if recursive else "*"
    videos = [path for path in input_path.glob(pattern) if path.is_file() and path.suffix.lower() in normalized_exts]
    return sorted(videos)


def _video_rel_path(video_path: Path, input_path: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    video_path = video_path.expanduser().resolve()
    if input_path.is_file():
        return Path(video_path.name)
    try:
        return video_path.relative_to(input_path)
    except ValueError:
        return Path(video_path.name)


def _video_output_dir(out_dir: Path, video_rel_path: Path) -> Path:
    parent = Path(*[_safe_name(part) for part in video_rel_path.parent.parts if part not in {"", "."}])
    stem = _safe_name(video_rel_path.stem)
    return out_dir / "frames" / parent / stem if str(parent) != "." else out_dir / "frames" / stem


def _select_step_frames(
    *,
    fps: float,
    interval_seconds: float | None,
    every_n_frames: int | None,
    target_fps: float | None,
) -> tuple[int, str, float]:
    selected = [interval_seconds is not None, every_n_frames is not None, target_fps is not None]
    if sum(selected) != 1:
        raise ValueError("Set exactly one of interval_seconds, every_n_frames, or target_fps.")
    if every_n_frames is not None:
        if every_n_frames < 1:
            raise ValueError("every_n_frames must be >= 1")
        return int(every_n_frames), "every_n_frames", float(every_n_frames)
    if fps <= 0:
        raise ValueError("Video FPS is unavailable; use every_n_frames for this video.")
    if interval_seconds is not None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        return max(1, int(round(fps * interval_seconds))), "interval_seconds", float(interval_seconds)
    assert target_fps is not None
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")
    return max(1, int(round(fps / target_fps))), "target_fps", float(target_fps)


def _frame_indices(
    *,
    fps: float,
    frame_count: int,
    step_frames: int,
    start_seconds: float | None,
    end_seconds: float | None,
    max_frames_per_video: int | None,
) -> list[int]:
    start_frame = 0
    end_frame = frame_count
    if fps > 0 and start_seconds is not None:
        if start_seconds < 0:
            raise ValueError("start_seconds must be >= 0")
        start_frame = max(0, int(math.ceil(start_seconds * fps)))
    if fps > 0 and end_seconds is not None:
        if end_seconds < 0:
            raise ValueError("end_seconds must be >= 0")
        end_frame = min(frame_count, int(math.floor(end_seconds * fps)))
    if end_frame < start_frame:
        raise ValueError("end_seconds must be greater than start_seconds")
    indices = list(range(start_frame, end_frame, step_frames))
    if max_frames_per_video is not None:
        if max_frames_per_video < 1:
            raise ValueError("max_frames_per_video must be >= 1")
        indices = indices[:max_frames_per_video]
    return indices


def _imwrite_params(cv2: Any, image_format: str, jpg_quality: int) -> list[int]:
    fmt = image_format.lower().lstrip(".")
    if fmt in {"jpg", "jpeg"}:
        return [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    if fmt == "png":
        return [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    raise ValueError("image_format must be one of: jpg, jpeg, png")


def _frame_file_name(video_stem: str, saved_index: int, frame_index: int, timestamp_seconds: float, image_format: str) -> str:
    ext = image_format.lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    return f"{_safe_name(video_stem)}__idx{saved_index:06d}__frame{frame_index:08d}__t{timestamp_seconds:010.3f}s.{ext}"


def _open_video_metadata(video_path: Path) -> tuple[Any, float, int, int, int]:
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return cap, fps, frame_count, width, height


def extract_video_frames(
    *,
    input_path: str | Path,
    out_dir: str | Path,
    recursive: bool = False,
    interval_seconds: float | None = 2.0,
    every_n_frames: int | None = None,
    target_fps: float | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    max_frames_per_video: int | None = None,
    max_videos: int | None = None,
    image_format: str = "jpg",
    jpg_quality: int = 95,
    skip_existing: bool = True,
    dry_run: bool = True,
    write_manifest: bool = True,
) -> ExtractResult:
    """Extract frames from a video file or a directory of videos.

    Dry-run mode reads video metadata and reports planned frame paths without writing images.
    """

    cv2 = _import_cv2()
    input_root = Path(input_path).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    videos = _iter_video_files(input_root, recursive=recursive)
    if max_videos is not None:
        if max_videos < 1:
            raise ValueError("max_videos must be >= 1")
        videos = videos[:max_videos]

    result = ExtractResult(input_path=input_root, out_dir=out_root, dry_run=dry_run)
    manifest_rows: list[dict[str, Any]] = []
    write_params = _imwrite_params(cv2, image_format, jpg_quality)

    for video_path in tqdm(videos, desc="Videos"):
        rel_path = _video_rel_path(video_path, input_root)
        video_out_dir = _video_output_dir(out_root, rel_path)
        summary = VideoSummary(
            video_path=str(video_path),
            video_rel_path=str(rel_path),
            output_dir=str(video_out_dir),
            fps=0.0,
            frame_count=0,
            duration_seconds=None,
            width=0,
            height=0,
            planned_frames=0,
            dry_run=dry_run,
        )
        result.videos.append(summary)

        cap = None
        try:
            cap, fps, frame_count, width, height = _open_video_metadata(video_path)
            duration = frame_count / fps if fps > 0 and frame_count > 0 else None
            step_frames, mode_name, mode_value = _select_step_frames(
                fps=fps,
                interval_seconds=interval_seconds,
                every_n_frames=every_n_frames,
                target_fps=target_fps,
            )
            indices = _frame_indices(
                fps=fps,
                frame_count=frame_count,
                step_frames=step_frames,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                max_frames_per_video=max_frames_per_video,
            )

            summary.fps = fps
            summary.frame_count = frame_count
            summary.duration_seconds = duration
            summary.width = width
            summary.height = height
            summary.planned_frames = len(indices)

            if not dry_run:
                video_out_dir.mkdir(parents=True, exist_ok=True)

            for saved_index, frame_index in enumerate(indices):
                timestamp = frame_index / fps if fps > 0 else float(frame_index)
                frame_name = _frame_file_name(rel_path.stem, saved_index, frame_index, timestamp, image_format)
                frame_path = video_out_dir / frame_name
                row = {
                    "video_path": str(video_path),
                    "video_rel_path": str(rel_path),
                    "video_name": video_path.name,
                    "frame_path": str(frame_path),
                    "frame_rel_path": str(frame_path.relative_to(out_root)) if frame_path.is_absolute() else str(frame_path),
                    "frame_index": frame_index,
                    "saved_index": saved_index,
                    "timestamp_seconds": round(timestamp, 6),
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "sample_mode": mode_name,
                    "sample_value": mode_value,
                    "image_format": image_format,
                    "status": "planned" if dry_run else "pending",
                    "labelstudio_import_hint": str(frame_path),
                }

                if dry_run:
                    manifest_rows.append(row)
                    continue

                if frame_path.exists() and skip_existing:
                    summary.skipped_existing += 1
                    row["status"] = "skipped_existing"
                    manifest_rows.append(row)
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    summary.failed_frames += 1
                    row["status"] = "failed_read"
                    manifest_rows.append(row)
                    continue

                ok = cv2.imwrite(str(frame_path), frame, write_params)
                if not ok:
                    summary.failed_frames += 1
                    row["status"] = "failed_write"
                    manifest_rows.append(row)
                    continue

                summary.saved_frames += 1
                row["status"] = "saved"
                manifest_rows.append(row)
        except Exception as exc:  # pragma: no cover - intentionally reported in result.
            summary.error = str(exc)
        finally:
            if cap is not None:
                cap.release()

    result.manifest_rows = len(manifest_rows)
    if write_manifest and not dry_run:
        manifest_dir = out_root / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "frames_manifest.csv"
        summary_path = manifest_dir / "videos_summary.json"
        result.manifest_path = manifest_path
        result.summary_path = summary_path
        if manifest_rows:
            with manifest_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0].keys()))
                writer.writeheader()
                writer.writerows(manifest_rows)
        else:
            manifest_path.write_text("", encoding="utf-8")
        summary_path.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract video frames for Label Studio bbox labeling.")
    parser.add_argument("--input-path", required=True, help="Video file path or directory containing video files.")
    parser.add_argument("--out-dir", required=True, help="Output directory for frames and manifests.")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories when input-path is a directory.")
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument("--interval-seconds", type=float, default=2.0, help="Extract one frame every N seconds. Default: 2.0")
    sample_group.add_argument("--every-n-frames", type=int, help="Extract one frame every N original frames.")
    sample_group.add_argument("--target-fps", type=float, help="Extract frames at an approximate target FPS.")
    parser.add_argument("--start-seconds", type=float, help="Start time in seconds.")
    parser.add_argument("--end-seconds", type=float, help="End time in seconds.")
    parser.add_argument("--max-frames-per-video", type=int, help="Limit extracted frames per video.")
    parser.add_argument("--max-videos", type=int, help="Limit number of videos processed.")
    parser.add_argument("--image-format", default="jpg", choices=["jpg", "jpeg", "png"])
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frame images.")
    parser.add_argument("--dry-run", action="store_true", help="Preview planned work without writing images.")
    parser.add_argument("--no-manifest", action="store_true", help="Do not write manifest files.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    interval_seconds = args.interval_seconds
    if args.every_n_frames is not None or args.target_fps is not None:
        interval_seconds = None

    result = extract_video_frames(
        input_path=args.input_path,
        out_dir=args.out_dir,
        recursive=args.recursive,
        interval_seconds=interval_seconds,
        every_n_frames=args.every_n_frames,
        target_fps=args.target_fps,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        max_frames_per_video=args.max_frames_per_video,
        max_videos=args.max_videos,
        image_format=args.image_format,
        jpg_quality=args.jpg_quality,
        skip_existing=not args.overwrite,
        dry_run=args.dry_run,
        write_manifest=not args.no_manifest,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
