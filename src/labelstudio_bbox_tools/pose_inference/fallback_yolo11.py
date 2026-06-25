from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from labelstudio_bbox_tools.pose_inference.common import PoseInstance, load_pose_class_names, preview_pose_video_inputs, run_pose_video_inference
from labelstudio_bbox_tools.pose_inference.fallback_crop import (
    apply_pose_nms,
    attach_detection_context,
    filter_detections_by_class,
    make_crop_regions,
    offset_pose_from_crop,
    split_pose_detection_cases,
)
from labelstudio_bbox_tools.pose_inference.yolo11 import _result_to_pose_instances
from labelstudio_bbox_tools.video_inference.classes import load_class_names, make_class_color_map
from labelstudio_bbox_tools.video_inference.yolo11 import _result_to_detections


def _chunked(items: Sequence, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def run_yolo11_pose_fallback_crop_test(
    *,
    input_path: str | Path,
    out_dir: str | Path,
    detector_weights: str | Path,
    detector_class_yaml: str | Path | None = None,
    detector_manual_classes: Sequence[str] | None = None,
    pose_weights: str | Path = "yolo11x-pose.pt",
    pose_class_yaml: str | Path | None = None,
    pose_manual_classes: Sequence[str] | None = None,
    target_detection_classes: Sequence[str] | None = None,
    device: str | None = None,
    detector_imgsz: int | tuple[int, int] | None = 640,
    pose_imgsz: int | tuple[int, int] | None = 640,
    detector_conf: float = 0.25,
    detector_iou: float = 0.5,
    pose_conf: float = 0.25,
    pose_iou: float = 0.45,
    keypoint_conf: float = 0.2,
    match_iou: float = 0.3,
    crop_padding_ratio: float = 0.15,
    min_crop_size: int = 32,
    fallback_batch_size: int = 8,
    max_fallback_crops_per_frame: int | None = 4,
    max_pose_per_crop: int = 1,
    final_nms_iou: float = 0.5,
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
    score_digits: int = 2,
    draw_bbox: bool = True,
    draw_skeleton: bool = True,
    draw_keypoints: bool = True,
    codec: str = "mp4v",
    overwrite: bool = False,
    dry_run: bool = True,
):
    if fallback_batch_size < 1:
        raise ValueError("fallback_batch_size must be >= 1")
    if max_pose_per_crop < 1:
        raise ValueError("max_pose_per_crop must be >= 1")

    detector_class_names = load_class_names(
        class_yaml=detector_class_yaml,
        manual_classes=detector_manual_classes,
        expected_count=None,
        strict_count=False,
    )
    pose_class_names = load_pose_class_names(
        class_yaml=pose_class_yaml,
        manual_classes=pose_manual_classes,
        expected_count=None,
        strict_count=False,
    )
    color_map = make_class_color_map(pose_class_names)
    model_name = "yolo11_pose_fallback_crop"

    if dry_run:
        print(
            f"[dry-run] model={model_name}, detector_weights={detector_weights}, pose_weights={pose_weights}, "
            f"target_detection_classes={list(target_detection_classes or []) or '<all>'}"
        )
        return preview_pose_video_inputs(
            input_path=input_path,
            out_dir=out_dir,
            model_name=model_name,
            recursive=recursive,
            max_videos=max_videos,
            run_name=run_name,
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required. Activate the YOLO conda env, for example ltb_ultra.") from exc

    detector_model = YOLO(str(detector_weights))
    pose_model = YOLO(str(pose_weights))
    if device:
        detector_model.to(device)
        pose_model.to(device)

    stats: Counter[str] = Counter()

    def detector_predict(frame_bgr) -> list:
        kwargs: dict[str, Any] = {"source": frame_bgr, "conf": detector_conf, "iou": detector_iou, "verbose": False}
        if detector_imgsz is not None:
            kwargs["imgsz"] = detector_imgsz
        if device:
            kwargs["device"] = device
        results = detector_model.predict(**kwargs)
        if not results:
            return []
        detections = _result_to_detections(results[0], detector_class_names)
        return filter_detections_by_class(detections, target_detection_classes)

    def pose_predict_full(frame_bgr) -> list[PoseInstance]:
        kwargs: dict[str, Any] = {"source": frame_bgr, "conf": pose_conf, "iou": pose_iou, "verbose": False}
        if pose_imgsz is not None:
            kwargs["imgsz"] = pose_imgsz
        if device:
            kwargs["device"] = device
        results = pose_model.predict(**kwargs)
        if not results:
            return []
        return _result_to_pose_instances(results[0], pose_class_names)

    def pose_predict_crops(crops) -> list[PoseInstance]:
        if not crops:
            return []
        restored: list[PoseInstance] = []
        for crop_batch in _chunked(crops, fallback_batch_size):
            kwargs: dict[str, Any] = {
                "source": [crop.crop_bgr for crop in crop_batch],
                "conf": pose_conf,
                "iou": pose_iou,
                "verbose": False,
                "batch": len(crop_batch),
            }
            if pose_imgsz is not None:
                kwargs["imgsz"] = pose_imgsz
            if device:
                kwargs["device"] = device
            results = pose_model.predict(**kwargs)
            for crop, result in zip(crop_batch, results, strict=False):
                crop_poses = _result_to_pose_instances(result, pose_class_names)
                crop_poses.sort(key=lambda item: item.raw_score if item.raw_score is not None else item.score, reverse=True)
                for pose in crop_poses[:max_pose_per_crop]:
                    restored.append(offset_pose_from_crop(pose, crop))
        return restored

    def predict_frame(frame_bgr, frame_index: int, timestamp: float, video_path: Path | None) -> list[PoseInstance]:
        detections = detector_predict(frame_bgr)
        full_poses = pose_predict_full(frame_bgr)
        cases = split_pose_detection_cases(pose_instances=full_poses, detections=detections, match_iou=match_iou)

        stats["frames"] += 1
        stats["pose_only"] += cases.pose_only_count
        stats["detection_only"] += cases.detection_only_count
        stats["matched"] += cases.matched_count

        outputs: list[PoseInstance] = []
        outputs.extend(attach_detection_context(pose, None, source="pose_only") for pose in cases.pose_only)
        outputs.extend(
            attach_detection_context(pose, detection, source="matched_full_frame_pose")
            for detection, pose, _iou in cases.matched
        )

        crops = make_crop_regions(
            frame_bgr,
            cases.detection_only,
            padding_ratio=crop_padding_ratio,
            min_crop_size=min_crop_size,
            max_crops=max_fallback_crops_per_frame,
        )
        stats["fallback_crops"] += len(crops)
        fallback_poses = pose_predict_crops(crops)
        stats["fallback_pose_recovered"] += len(fallback_poses)
        outputs.extend(fallback_poses)
        return apply_pose_nms(outputs, iou=final_nms_iou)

    result = run_pose_video_inference(
        input_path=input_path,
        out_dir=out_dir,
        model_name=model_name,
        class_names=pose_class_names,
        color_map=color_map,
        predict_frame=predict_frame,
        recursive=recursive,
        max_videos=max_videos,
        frame_stride=frame_stride,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        max_frames=max_frames,
        run_name=run_name,
        font_path=font_path,
        font_size=font_size,
        line_width=line_width,
        keypoint_radius=keypoint_radius,
        keypoint_conf=keypoint_conf,
        score_digits=score_digits,
        draw_bbox=draw_bbox,
        draw_skeleton=draw_skeleton,
        draw_keypoints=draw_keypoints,
        codec=codec,
        overwrite=overwrite,
        run_config={
            "detector_weights": str(detector_weights),
            "detector_class_yaml": str(detector_class_yaml) if detector_class_yaml else None,
            "pose_weights": str(pose_weights),
            "pose_class_yaml": str(pose_class_yaml) if pose_class_yaml else None,
            "target_detection_classes": list(target_detection_classes or []),
            "device": device,
            "detector_imgsz": detector_imgsz,
            "pose_imgsz": pose_imgsz,
            "detector_conf": detector_conf,
            "detector_iou": detector_iou,
            "pose_conf": pose_conf,
            "pose_iou": pose_iou,
            "match_iou": match_iou,
            "crop_padding_ratio": crop_padding_ratio,
            "fallback_batch_size": fallback_batch_size,
            "max_fallback_crops_per_frame": max_fallback_crops_per_frame,
            "max_pose_per_crop": max_pose_per_crop,
            "final_nms_iou": final_nms_iou,
        },
    )
    if result.run_dir:
        path = result.run_dir / "fallback_cases_summary.json"
        path.write_text(json.dumps(dict(stats), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] fallback_cases_summary={path}")
    return result
