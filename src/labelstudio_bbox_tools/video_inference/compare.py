from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from labelstudio_bbox_tools.video_inference.common import import_cv2
from labelstudio_bbox_tools.video_inference.draw import draw_title_bar_bgr


@dataclass
class VideoCompareSummary:
    left_video: str
    right_video: str
    out_video: str
    left_title: str
    right_title: str
    fps: float
    output_width: int
    output_height: int
    written_frames: int
    dry_run: bool

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _metadata(video_path: Path) -> tuple[float, int, int, int]:
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
    return fps, frame_count, width, height


def combine_videos_side_by_side(
    *,
    left_video: str | Path,
    right_video: str | Path,
    out_video: str | Path,
    left_title: str = "YOLO11",
    right_title: str = "RF-DETR",
    font_path: str | Path | None = None,
    font_size: int = 28,
    title_bar_height: int = 48,
    target_height: int | None = None,
    fps: float | None = None,
    codec: str = "mp4v",
    max_frames: int | None = None,
    overwrite: bool = False,
    dry_run: bool = True,
) -> VideoCompareSummary:
    cv2 = import_cv2()
    left_path = Path(left_video).expanduser().resolve()
    right_path = Path(right_video).expanduser().resolve()
    out_path = Path(out_video).expanduser().resolve()

    left_fps, left_frames, left_w, left_h = _metadata(left_path)
    right_fps, right_frames, right_w, right_h = _metadata(right_path)
    base_height = int(target_height or min(left_h, right_h))
    left_out_w = int(round(left_w * base_height / left_h))
    right_out_w = int(round(right_w * base_height / right_h))
    output_width = left_out_w + right_out_w
    output_height = base_height + title_bar_height
    output_fps = float(fps or left_fps or right_fps or 30.0)
    planned_frames = min(left_frames, right_frames)
    if max_frames is not None:
        planned_frames = min(planned_frames, int(max_frames))

    print(
        f"[info] left={left_w}x{left_h}@{left_fps:.3f} frames={left_frames:,}, "
        f"right={right_w}x{right_h}@{right_fps:.3f} frames={right_frames:,}"
    )
    print(f"[info] output={output_width}x{output_height}@{output_fps:.3f}, planned_frames={planned_frames:,}")

    if dry_run:
        return VideoCompareSummary(
            left_video=str(left_path),
            right_video=str(right_path),
            out_video=str(out_path),
            left_title=left_title,
            right_title=right_title,
            fps=output_fps,
            output_width=output_width,
            output_height=output_height,
            written_frames=0,
            dry_run=True,
        )

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output video already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    left_cap = cv2.VideoCapture(str(left_path))
    right_cap = cv2.VideoCapture(str(right_path))
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*codec),
        output_fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        left_cap.release()
        right_cap.release()
        raise RuntimeError(f"Could not open output video writer: {out_path}")

    written = 0
    try:
        for _ in tqdm(range(planned_frames), desc="Combine videos"):
            ok_left, left_frame = left_cap.read()
            ok_right, right_frame = right_cap.read()
            if not ok_left or not ok_right:
                break
            left_frame = cv2.resize(left_frame, (left_out_w, base_height), interpolation=cv2.INTER_AREA)
            right_frame = cv2.resize(right_frame, (right_out_w, base_height), interpolation=cv2.INTER_AREA)
            left_frame = draw_title_bar_bgr(
                left_frame,
                title=left_title,
                font_path=font_path,
                font_size=font_size,
                bar_height=title_bar_height,
            )
            right_frame = draw_title_bar_bgr(
                right_frame,
                title=right_title,
                font_path=font_path,
                font_size=font_size,
                bar_height=title_bar_height,
            )
            combined = cv2.hconcat([left_frame, right_frame])
            writer.write(combined)
            written += 1
    finally:
        left_cap.release()
        right_cap.release()
        writer.release()

    summary = VideoCompareSummary(
        left_video=str(left_path),
        right_video=str(right_path),
        out_video=str(out_path),
        left_title=left_title,
        right_title=right_title,
        fps=output_fps,
        output_width=output_width,
        output_height=output_height,
        written_frames=written,
        dry_run=False,
    )
    (out_path.with_suffix(".summary.json")).write_text(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_path} frames={written:,}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine two visualized videos side-by-side for comparison.")
    parser.add_argument("--left-video", required=True)
    parser.add_argument("--right-video", required=True)
    parser.add_argument("--out-video", required=True)
    parser.add_argument("--left-title", default="YOLO11")
    parser.add_argument("--right-title", default="RF-DETR")
    parser.add_argument("--font-path")
    parser.add_argument("--font-size", type=int, default=28)
    parser.add_argument("--title-bar-height", type=int, default=48)
    parser.add_argument("--target-height", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run", action="store_true", help="Actually write the output video. Omit for dry-run preview.")
    args = parser.parse_args()

    summary = combine_videos_side_by_side(
        left_video=args.left_video,
        right_video=args.right_video,
        out_video=args.out_video,
        left_title=args.left_title,
        right_title=args.right_title,
        font_path=args.font_path,
        font_size=args.font_size,
        title_bar_height=args.title_bar_height,
        target_height=args.target_height,
        fps=args.fps,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
        dry_run=not args.run,
    )
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
