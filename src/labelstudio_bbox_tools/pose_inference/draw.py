from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from labelstudio_bbox_tools.pose_inference.common import PoseInstance
from labelstudio_bbox_tools.video_inference.classes import color_for_class
from labelstudio_bbox_tools.video_inference.common import Detection
from labelstudio_bbox_tools.video_inference.draw import (
    _intersects,
    _label_box_candidates,
    _text_fill_for_background,
    _text_size,
    draw_detections_on_bgr,
    load_font,
)

COCO_PERSON_SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]

DEFAULT_SOURCE_COLOR_MAP = {
    "pose_only": (80, 180, 255),
    "matched_full_frame_pose": (80, 255, 120),
    "fallback_crop_pose": (255, 210, 80),
    "fallback_failed_detection": (255, 80, 80),
}

DEFAULT_SOURCE_LABEL_PREFIXES = {
    "fallback_crop_pose": "fallback_pose",
    "fallback_failed_detection": "custom_detect",
}

DEFAULT_DASHED_BBOX_SOURCES = {"fallback_crop_pose"}


def _rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def _in_frame(x: float, y: float, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def _source_color(
    instance: PoseInstance,
    *,
    color_map: Mapping[str, tuple[int, int, int]],
    color_by_source: bool,
    source_color_map: Mapping[str, tuple[int, int, int]],
) -> tuple[int, int, int]:
    source = instance.source or ""
    if color_by_source and source in source_color_map:
        return source_color_map[source]
    return color_for_class(instance.class_name, dict(color_map))


def _source_label(
    instance: PoseInstance,
    *,
    source_label_prefixes: Mapping[str, str],
    score_digits: int,
) -> str:
    source = instance.source or ""
    prefix = source_label_prefixes.get(source)
    if prefix:
        return f"{prefix} {instance.class_name} {instance.score:.{score_digits}f}"
    return f"{instance.class_name} {instance.score:.{score_digits}f}"


def _draw_solid_rectangle(draw, xyxy: tuple[int, int, int, int], color: tuple[int, int, int], line_width: int) -> None:
    x1, y1, x2, y2 = xyxy
    for offset in range(max(1, int(line_width))):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)


def _draw_dashed_line(draw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int], width: int) -> None:
    x1, y1 = start
    x2, y2 = end
    dash = max(6, int(width) * 4)
    gap = max(4, int(width) * 3)
    if y1 == y2:
        x = x1
        while x <= x2:
            draw.line((x, y1, min(x + dash, x2), y2), fill=color, width=width)
            x += dash + gap
    elif x1 == x2:
        y = y1
        while y <= y2:
            draw.line((x1, y, x2, min(y + dash, y2)), fill=color, width=width)
            y += dash + gap


def _draw_dashed_rectangle(draw, xyxy: tuple[int, int, int, int], color: tuple[int, int, int], line_width: int) -> None:
    x1, y1, x2, y2 = xyxy
    width = max(1, int(line_width))
    _draw_dashed_line(draw, (x1, y1), (x2, y1), color, width)
    _draw_dashed_line(draw, (x1, y2), (x2, y2), color, width)
    _draw_dashed_line(draw, (x1, y1), (x1, y2), color, width)
    _draw_dashed_line(draw, (x2, y1), (x2, y2), color, width)


def _draw_source_aware_bboxes_on_bgr(
    frame_bgr,
    instances: Sequence[PoseInstance],
    *,
    color_map: Mapping[str, tuple[int, int, int]],
    font_path: str | Path | None,
    font_size: int,
    line_width: int,
    score_digits: int,
    color_by_source: bool,
    source_color_map: Mapping[str, tuple[int, int, int]],
    source_label_prefixes: Mapping[str, str],
    dashed_bbox_sources: set[str],
):
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("OpenCV, numpy, and Pillow are required for drawing pose boxes") from exc

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image)
    font = load_font(font_path, font_size)
    image_width, image_height = image.size
    used_label_boxes: list[tuple[int, int, int, int]] = []

    for instance in instances:
        x1, y1, x2, y2 = [int(round(value)) for value in instance.xyxy]
        x1 = max(0, min(x1, image_width - 1))
        x2 = max(0, min(x2, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        y2 = max(0, min(y2, image_height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        color = _source_color(
            instance,
            color_map=color_map,
            color_by_source=color_by_source,
            source_color_map=source_color_map,
        )
        if instance.source in dashed_bbox_sources:
            _draw_dashed_rectangle(draw, (x1, y1, x2, y2), color, line_width)
        else:
            _draw_solid_rectangle(draw, (x1, y1, x2, y2), color, line_width)

        label = _source_label(instance, source_label_prefixes=source_label_prefixes, score_digits=score_digits)
        text_w, text_h = _text_size(draw, label, font)
        pad_x = max(4, font_size // 4)
        pad_y = max(3, font_size // 6)
        label_w = text_w + pad_x * 2
        label_h = text_h + pad_y * 2
        candidates = _label_box_candidates(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            label_width=label_w,
            label_height=label_h,
            image_width=image_width,
            image_height=image_height,
        )
        chosen = candidates[0]
        for candidate in candidates:
            if not any(_intersects(candidate, used) for used in used_label_boxes):
                chosen = candidate
                break
        used_label_boxes.append(chosen)
        draw.rectangle(chosen, fill=color)
        draw.text((chosen[0] + pad_x, chosen[1] + pad_y), label, font=font, fill=_text_fill_for_background(color))

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_pose_instances_on_bgr(
    frame_bgr,
    instances: Sequence[PoseInstance],
    *,
    color_map: dict[str, tuple[int, int, int]],
    font_path: str | Path | None = None,
    font_size: int = 20,
    line_width: int = 3,
    keypoint_radius: int = 4,
    keypoint_conf: float = 0.2,
    score_digits: int = 2,
    draw_bbox: bool = True,
    draw_skeleton: bool = True,
    draw_keypoints: bool = True,
    color_by_source: bool = False,
    source_color_map: Mapping[str, tuple[int, int, int]] | None = None,
    source_label_prefixes: Mapping[str, str] | None = None,
    dashed_bbox_sources: Sequence[str] | None = None,
):
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for drawing pose video frames") from exc

    drawn = frame_bgr.copy()
    height, width = drawn.shape[:2]
    source_colors = dict(DEFAULT_SOURCE_COLOR_MAP)
    if source_color_map:
        source_colors.update(source_color_map)

    for instance in instances:
        rgb = _source_color(
            instance,
            color_map=color_map,
            color_by_source=color_by_source,
            source_color_map=source_colors,
        )
        bgr = _rgb_to_bgr(rgb)
        points = list(instance.keypoints)

        if draw_skeleton:
            for start_idx, end_idx in COCO_PERSON_SKELETON:
                if start_idx >= len(points) or end_idx >= len(points):
                    continue
                start = points[start_idx]
                end = points[end_idx]
                if not start.is_drawable(keypoint_conf) or not end.is_drawable(keypoint_conf):
                    continue
                if not _in_frame(start.x, start.y, width, height) or not _in_frame(end.x, end.y, width, height):
                    continue
                cv2.line(
                    drawn,
                    (int(round(start.x)), int(round(start.y))),
                    (int(round(end.x)), int(round(end.y))),
                    bgr,
                    max(1, int(line_width)),
                    lineType=cv2.LINE_AA,
                )

        if draw_keypoints:
            radius = max(1, int(keypoint_radius))
            for keypoint in points:
                if not keypoint.is_drawable(keypoint_conf):
                    continue
                if not _in_frame(keypoint.x, keypoint.y, width, height):
                    continue
                center = (int(round(keypoint.x)), int(round(keypoint.y)))
                cv2.circle(drawn, center, radius + 1, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(drawn, center, radius, bgr, thickness=-1, lineType=cv2.LINE_AA)

    if not draw_bbox:
        return drawn

    label_prefixes = dict(DEFAULT_SOURCE_LABEL_PREFIXES)
    if source_label_prefixes:
        label_prefixes.update(source_label_prefixes)
    dashed_sources = set(dashed_bbox_sources or [])
    use_source_aware_bbox = (
        color_by_source
        or bool(dashed_sources)
        or any((instance.source or "") in label_prefixes for instance in instances)
    )
    if use_source_aware_bbox:
        return _draw_source_aware_bboxes_on_bgr(
            drawn,
            instances,
            color_map=color_map,
            font_path=font_path,
            font_size=font_size,
            line_width=line_width,
            score_digits=score_digits,
            color_by_source=color_by_source,
            source_color_map=source_colors,
            source_label_prefixes=label_prefixes,
            dashed_bbox_sources=dashed_sources,
        )

    detections = [
        Detection(
            xyxy=instance.xyxy,
            class_id=instance.class_id,
            class_name=instance.class_name,
            score=instance.score,
        )
        for instance in instances
    ]
    return draw_detections_on_bgr(
        drawn,
        detections,
        color_map=color_map,
        font_path=font_path,
        font_size=font_size,
        line_width=line_width,
        score_digits=score_digits,
    )
