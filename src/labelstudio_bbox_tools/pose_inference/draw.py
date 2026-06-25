from __future__ import annotations

from pathlib import Path
from typing import Sequence

from labelstudio_bbox_tools.pose_inference.common import PoseInstance
from labelstudio_bbox_tools.video_inference.classes import color_for_class
from labelstudio_bbox_tools.video_inference.common import Detection
from labelstudio_bbox_tools.video_inference.draw import draw_detections_on_bgr

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


def _rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def _in_frame(x: float, y: float, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


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
):
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for drawing pose video frames") from exc

    drawn = frame_bgr.copy()
    height, width = drawn.shape[:2]

    for instance in instances:
        rgb = color_for_class(instance.class_name, color_map)
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

