from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from labelstudio_bbox_tools.pose_inference.common import PoseInstance, PoseKeypoint
from labelstudio_bbox_tools.pseudo_label.yolo import classwise_nms_indices
from labelstudio_bbox_tools.video_inference.common import Detection, import_cv2, safe_name


@dataclass(frozen=True)
class PoseDetectionMatch:
    detection_index: int
    pose_index: int
    iou: float


@dataclass(frozen=True)
class PoseDetectionCases:
    pose_only: list[PoseInstance]
    detection_only: list[Detection]
    matched: list[tuple[Detection, PoseInstance, float]]

    @property
    def pose_only_count(self) -> int:
        return len(self.pose_only)

    @property
    def detection_only_count(self) -> int:
        return len(self.detection_only)

    @property
    def matched_count(self) -> int:
        return len(self.matched)


FALLBACK_SUCCESS_SOURCE = "fallback_crop_pose"
FALLBACK_FAILED_SOURCE = "fallback_failed_detection"
FALLBACK_DEBUG_SUCCESS = "fallback_success"
FALLBACK_DEBUG_FAILED = "fallback_failed"


@dataclass
class CropRegion:
    detection: Detection
    crop_xyxy: tuple[int, int, int, int]
    crop_bgr: object


@dataclass
class FallbackDebugCropRecord:
    case: str
    video_path: str
    frame_index: int
    crop_index: int
    detection: Detection
    crop_xyxy: tuple[int, int, int, int]
    crop_bgr: object


class FallbackDebugSaver:
    def __init__(
        self,
        *,
        save_frames: bool = False,
        save_crops: bool = False,
        max_items_per_case_per_video: int | None = 200,
        jpeg_quality: int = 95,
    ) -> None:
        self.save_frames = bool(save_frames)
        self.save_crops = bool(save_crops)
        self.max_items_per_case_per_video = max_items_per_case_per_video
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self.pending_crops: dict[tuple[str, int], list[FallbackDebugCropRecord]] = {}
        self.frame_counts: Counter[tuple[str, str]] = Counter()
        self.crop_counts: Counter[tuple[str, str]] = Counter()
        self.records: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.save_frames or self.save_crops

    def add_crop(
        self,
        *,
        case: str,
        video_path: Path | None,
        frame_index: int,
        crop_index: int,
        crop: CropRegion,
    ) -> None:
        if not self.save_crops:
            return
        key = (str(video_path) if video_path else "<unknown>", int(frame_index))
        self.pending_crops.setdefault(key, []).append(
            FallbackDebugCropRecord(
                case=case,
                video_path=key[0],
                frame_index=int(frame_index),
                crop_index=int(crop_index),
                detection=crop.detection,
                crop_xyxy=crop.crop_xyxy,
                crop_bgr=crop.crop_bgr,
            )
        )

    def _can_save(self, counter: Counter[tuple[str, str]], video_key: str, case: str) -> bool:
        if self.max_items_per_case_per_video is None:
            return True
        return counter[(video_key, case)] < int(self.max_items_per_case_per_video)

    def _write_image(self, path: Path, image_bgr) -> bool:
        cv2 = import_cv2()
        path.parent.mkdir(parents=True, exist_ok=True)
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        return bool(cv2.imwrite(str(path), image_bgr, params))

    def after_frame(
        self,
        *,
        frame_bgr,
        drawn_bgr,
        instances: Sequence[PoseInstance],
        video_path: Path,
        video_rel_path: Path,
        output_video_path: Path,
        frame_index: int,
        timestamp_seconds: float,
        run_dir: Path,
    ) -> None:
        video_key = safe_name(str(Path(video_rel_path).with_suffix("")))
        sources = {instance.source for instance in instances}
        frame_cases: list[str] = []
        if FALLBACK_SUCCESS_SOURCE in sources:
            frame_cases.append(FALLBACK_DEBUG_SUCCESS)
        if FALLBACK_FAILED_SOURCE in sources:
            frame_cases.append(FALLBACK_DEBUG_FAILED)

        if self.save_frames:
            for case in frame_cases:
                if not self._can_save(self.frame_counts, video_key, case):
                    continue
                out_path = run_dir / "debug_frames" / case / f"{video_key}__f{int(frame_index):06d}.jpg"
                if self._write_image(out_path, drawn_bgr):
                    self.frame_counts[(video_key, case)] += 1
                    self.records.append(
                        {
                            "kind": "frame",
                            "case": case,
                            "video_path": str(video_path),
                            "video_rel_path": str(video_rel_path),
                            "output_video_path": str(output_video_path),
                            "frame_index": int(frame_index),
                            "timestamp_seconds": float(timestamp_seconds),
                            "path": str(out_path),
                        }
                    )

        pending_key = (str(video_path) if video_path else "<unknown>", int(frame_index))
        pending = self.pending_crops.pop(pending_key, [])
        if self.save_crops:
            for record in pending:
                case = record.case
                if not self._can_save(self.crop_counts, video_key, case):
                    continue
                det_name = safe_name(record.detection.class_name)
                out_path = (
                    run_dir
                    / "debug_crops"
                    / case
                    / f"{video_key}__f{int(frame_index):06d}__crop{record.crop_index:02d}__{det_name}.jpg"
                )
                if self._write_image(out_path, record.crop_bgr):
                    self.crop_counts[(video_key, case)] += 1
                    self.records.append(
                        {
                            "kind": "crop",
                            "case": case,
                            "video_path": str(video_path),
                            "video_rel_path": str(video_rel_path),
                            "frame_index": int(frame_index),
                            "timestamp_seconds": float(timestamp_seconds),
                            "crop_index": int(record.crop_index),
                            "crop_xyxy": [int(value) for value in record.crop_xyxy],
                            "detection": record.detection.as_dict(),
                            "path": str(out_path),
                        }
                    )

    def write_summary(self, run_dir: Path) -> Path | None:
        if not self.enabled:
            return None
        path = run_dir / "fallback_debug_summary.json"
        payload = {
            "save_frames": self.save_frames,
            "save_crops": self.save_crops,
            "max_items_per_case_per_video": self.max_items_per_case_per_video,
            "jpeg_quality": self.jpeg_quality,
            "frame_counts": {f"{video}::{case}": count for (video, case), count in self.frame_counts.items()},
            "crop_counts": {f"{video}::{case}": count for (video, case), count in self.crop_counts.items()},
            "records": self.records,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def filter_detections_by_class(
    detections: Sequence[Detection],
    target_classes: Sequence[str] | None = None,
) -> list[Detection]:
    if not target_classes:
        return list(detections)
    targets = {str(name) for name in target_classes}
    return [det for det in detections if det.class_name in targets]


def split_pose_detection_cases(
    *,
    pose_instances: Sequence[PoseInstance],
    detections: Sequence[Detection],
    match_iou: float = 0.3,
) -> PoseDetectionCases:
    candidates: list[PoseDetectionMatch] = []
    for det_idx, detection in enumerate(detections):
        for pose_idx, pose in enumerate(pose_instances):
            iou = box_iou(detection.xyxy, pose.xyxy)
            if iou >= match_iou:
                candidates.append(PoseDetectionMatch(det_idx, pose_idx, iou))

    candidates.sort(key=lambda item: item.iou, reverse=True)
    used_detections: set[int] = set()
    used_poses: set[int] = set()
    matched: list[tuple[Detection, PoseInstance, float]] = []
    for candidate in candidates:
        if candidate.detection_index in used_detections or candidate.pose_index in used_poses:
            continue
        used_detections.add(candidate.detection_index)
        used_poses.add(candidate.pose_index)
        matched.append((detections[candidate.detection_index], pose_instances[candidate.pose_index], candidate.iou))

    pose_only = [pose for idx, pose in enumerate(pose_instances) if idx not in used_poses]
    detection_only = [det for idx, det in enumerate(detections) if idx not in used_detections]
    return PoseDetectionCases(pose_only=pose_only, detection_only=detection_only, matched=matched)


def _clip_crop_box(
    xyxy: Sequence[float],
    *,
    width: int,
    height: int,
    padding_ratio: float = 0.15,
    min_crop_size: int = 32,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 1 or bh <= 1:
        return None
    pad_x = bw * float(padding_ratio)
    pad_y = bh * float(padding_ratio)
    cx1 = max(0, int(round(x1 - pad_x)))
    cy1 = max(0, int(round(y1 - pad_y)))
    cx2 = min(width, int(round(x2 + pad_x)))
    cy2 = min(height, int(round(y2 + pad_y)))
    if cx2 - cx1 < int(min_crop_size) or cy2 - cy1 < int(min_crop_size):
        return None
    return cx1, cy1, cx2, cy2


def make_crop_regions(
    frame_bgr,
    detections: Sequence[Detection],
    *,
    padding_ratio: float = 0.15,
    min_crop_size: int = 32,
    max_crops: int | None = None,
) -> list[CropRegion]:
    height, width = frame_bgr.shape[:2]
    sorted_detections = sorted(detections, key=lambda item: item.score, reverse=True)
    if max_crops is not None:
        sorted_detections = sorted_detections[: int(max_crops)]

    crops: list[CropRegion] = []
    for detection in sorted_detections:
        crop_xyxy = _clip_crop_box(
            detection.xyxy,
            width=width,
            height=height,
            padding_ratio=padding_ratio,
            min_crop_size=min_crop_size,
        )
        if crop_xyxy is None:
            continue
        x1, y1, x2, y2 = crop_xyxy
        crops.append(CropRegion(detection=detection, crop_xyxy=crop_xyxy, crop_bgr=frame_bgr[y1:y2, x1:x2].copy()))
    return crops


def detection_to_pose_instance(
    detection: Detection,
    *,
    source: str = FALLBACK_FAILED_SOURCE,
    class_name: str | None = None,
) -> PoseInstance:
    return PoseInstance(
        xyxy=tuple(float(value) for value in detection.xyxy),
        class_id=int(detection.class_id),
        class_name=class_name or detection.class_name,
        score=float(detection.score),
        raw_score=float(detection.score),
        keypoints=tuple(),
        source=source,
        detection_class_name=detection.class_name,
        detection_score=float(detection.score),
        detection_xyxy=tuple(float(value) for value in detection.xyxy),
    )


def attach_detection_context(
    pose: PoseInstance,
    detection: Detection | None,
    *,
    source: str,
    crop_xyxy: tuple[int, int, int, int] | None = None,
) -> PoseInstance:
    if detection is None:
        return replace(pose, source=source)
    return replace(
        pose,
        source=source,
        detection_class_name=detection.class_name,
        detection_score=detection.score,
        detection_xyxy=tuple(float(value) for value in detection.xyxy),
        crop_xyxy=None if crop_xyxy is None else tuple(float(value) for value in crop_xyxy),
    )


def offset_pose_from_crop(
    pose: PoseInstance,
    crop: CropRegion,
    *,
    source: str = "fallback_crop_pose",
) -> PoseInstance:
    offset_x, offset_y, _, _ = crop.crop_xyxy
    x1, y1, x2, y2 = pose.xyxy
    keypoints = tuple(
        PoseKeypoint(
            name=keypoint.name,
            x=keypoint.x + offset_x,
            y=keypoint.y + offset_y,
            confidence=keypoint.confidence,
        )
        for keypoint in pose.keypoints
    )
    shifted = replace(
        pose,
        xyxy=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
        keypoints=keypoints,
    )
    return attach_detection_context(shifted, crop.detection, source=source, crop_xyxy=crop.crop_xyxy)


def apply_pose_nms(
    instances: Sequence[PoseInstance],
    *,
    iou: float = 0.5,
) -> list[PoseInstance]:
    instances = list(instances)
    if not instances:
        return []
    class_names: list[str] = []
    remapped_class_ids: list[int] = []
    for instance in instances:
        if instance.class_name not in class_names:
            class_names.append(instance.class_name)
        remapped_class_ids.append(class_names.index(instance.class_name))
    boxes = [instance.xyxy for instance in instances]
    scores = [instance.raw_score if instance.raw_score is not None else instance.score for instance in instances]
    keep = classwise_nms_indices(boxes, remapped_class_ids, scores, class_names, None, default_iou=iou)
    return [instances[idx] for idx in keep]
