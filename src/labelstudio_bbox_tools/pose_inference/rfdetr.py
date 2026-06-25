from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from labelstudio_bbox_tools.pose_inference.common import (
    COCO_PERSON_KEYPOINT_NAMES,
    PoseInstance,
    keypoints_from_xy_conf,
    load_pose_class_names,
    preview_pose_video_inputs,
    run_pose_video_inference,
)
from labelstudio_bbox_tools.pseudo_label.yolo import classwise_nms_indices
from labelstudio_bbox_tools.video_inference.classes import make_class_color_map


@dataclass(frozen=True)
class RfDetrKeypointLoadInfo:
    load_mode: str
    model_class: str
    model_weights: str | None


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalise_shape(value: str | int | None) -> tuple[int, int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return int(value), int(value)
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        return size, size
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise ValueError("shape must be an int or 'height,width'")


def _load_rfdetr_keypoint_model(
    *,
    model_weights: str | Path | None = None,
    device: str | None = None,
) -> tuple[Any, RfDetrKeypointLoadInfo]:
    try:
        import rfdetr
    except ImportError as exc:
        raise RuntimeError("RF-DETR is required. Activate the RF-DETR conda env, for example ltb_rfdetr.") from exc

    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device

    model_cls = getattr(rfdetr, "RFDETRKeypointPreview", None)
    if model_cls is None:
        raise RuntimeError("The installed RF-DETR package does not expose RFDETRKeypointPreview.")

    if model_weights in (None, ""):
        model = model_cls(**kwargs)
        return model, RfDetrKeypointLoadInfo("official_default", type(model).__name__, None)

    weights_path = str(model_weights)
    try:
        model = model_cls.from_checkpoint(weights_path, **kwargs)
        return model, RfDetrKeypointLoadInfo("from_checkpoint", type(model).__name__, weights_path)
    except Exception as checkpoint_exc:
        try:
            model = model_cls(pretrain_weights=weights_path, **kwargs)
        except Exception as constructor_exc:
            raise RuntimeError("Failed to load RF-DETR keypoint weights with RFDETRKeypointPreview.") from constructor_exc
        print(f"[warn] from_checkpoint failed for RFDETRKeypointPreview: {checkpoint_exc}")
        return model, RfDetrKeypointLoadInfo("pretrain_weights", type(model).__name__, weights_path)


def _keypoints_to_pose_instances(
    key_points: Any,
    class_names: Sequence[str],
    *,
    keypoint_names: Sequence[str] = COCO_PERSON_KEYPOINT_NAMES,
) -> list[PoseInstance]:
    xy_all = _to_list(getattr(key_points, "xy", None))
    conf_all = _to_list(getattr(key_points, "keypoint_confidence", None))
    det_scores = _to_list(getattr(key_points, "detection_confidence", None))
    raw_class_ids = _to_list(getattr(key_points, "class_id", None))
    data = getattr(key_points, "data", {}) or {}
    boxes = _to_list(data.get("xyxy"))
    raw_class_names = _to_list(data.get("class_name"))

    instances: list[PoseInstance] = []
    for idx, xy in enumerate(xy_all):
        if idx >= len(boxes):
            continue
        raw_class_id = int(raw_class_ids[idx]) if idx < len(raw_class_ids) and raw_class_ids[idx] is not None else 0
        class_name = ""
        if idx < len(raw_class_names) and raw_class_names[idx]:
            class_name = str(raw_class_names[idx])
        elif 0 <= raw_class_id < len(class_names):
            class_name = class_names[raw_class_id]
        else:
            class_name = class_names[0] if class_names else "person"
        if class_name == "__background__":
            continue
        class_id = class_names.index(class_name) if class_name in class_names else raw_class_id
        raw_score = float(det_scores[idx]) if idx < len(det_scores) and det_scores[idx] is not None else 1.0
        conf = conf_all[idx] if idx < len(conf_all) else None
        box = boxes[idx]
        instances.append(
            PoseInstance(
                xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                class_id=class_id,
                class_name=class_name,
                score=_clamp_score(raw_score),
                raw_score=raw_score,
                keypoints=keypoints_from_xy_conf(xy, conf, names=keypoint_names),
            )
        )
    return instances


def _filter_pose_instances(
    instances: Sequence[PoseInstance],
    class_names: Sequence[str],
    *,
    enable_nms: bool = True,
    iou: float = 0.5,
    max_instances_per_frame: int | None = None,
    min_visible_keypoints: int | None = None,
    keypoint_conf: float = 0.2,
) -> list[PoseInstance]:
    filtered = list(instances)

    if min_visible_keypoints is not None:
        min_count = int(min_visible_keypoints)
        filtered = [
            instance
            for instance in filtered
            if sum(1 for keypoint in instance.keypoints if keypoint.is_drawable(keypoint_conf)) >= min_count
        ]

    if enable_nms and filtered:
        boxes = [instance.xyxy for instance in filtered]
        class_ids = [instance.class_id for instance in filtered]
        # RF-DETR keypoint detection_confidence can exceed 1.0 after uncertainty fusion.
        # Use raw_score for ranking, but keep the clamped display score for labels.
        scores = [instance.raw_score if instance.raw_score is not None else instance.score for instance in filtered]
        keep = classwise_nms_indices(boxes, class_ids, scores, class_names, None, default_iou=iou)
        filtered = [filtered[idx] for idx in keep]

    if max_instances_per_frame is not None:
        max_count = int(max_instances_per_frame)
        filtered.sort(key=lambda item: item.raw_score if item.raw_score is not None else item.score, reverse=True)
        filtered = filtered[:max_count]

    return filtered


def run_rfdetr_pose_video_inference(
    *,
    input_path: str | Path,
    out_dir: str | Path,
    model_weights: str | Path | None = None,
    class_yaml: str | Path | None = None,
    manual_classes: Sequence[str] | None = None,
    device: str | None = None,
    conf: float = 0.25,
    iou: float = 0.5,
    keypoint_conf: float = 0.2,
    enable_nms: bool = True,
    max_instances_per_frame: int | None = None,
    min_visible_keypoints: int | None = None,
    shape: tuple[int, int] | None = None,
    recursive: bool = False,
    max_videos: int | None = None,
    frame_stride: int = 1,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    max_frames: int | None = None,
    expected_class_count: int | None = None,
    strict_class_count: bool = False,
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
    class_names = load_pose_class_names(
        class_yaml=class_yaml,
        manual_classes=manual_classes,
        expected_count=expected_class_count,
        strict_count=strict_class_count,
    )
    color_map = make_class_color_map(class_names)
    model_name = "rfdetr_keypoint"
    if dry_run:
        print(f"[dry-run] model={model_name}, weights={model_weights or '<official-default>'}, classes={class_names}")
        return preview_pose_video_inputs(
            input_path=input_path,
            out_dir=out_dir,
            model_name=model_name,
            recursive=recursive,
            max_videos=max_videos,
            run_name=run_name,
        )

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("opencv-python is required in the RF-DETR env for video frame handling.") from exc

    model, load_info = _load_rfdetr_keypoint_model(model_weights=model_weights, device=device)
    print(f"[info] loaded RF-DETR keypoint class={load_info.model_class}, mode={load_info.load_mode}")

    def predict_frame(frame_bgr, frame_index: int, timestamp: float, video_path: Path | None) -> list[PoseInstance]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        kwargs: dict[str, Any] = {"threshold": conf, "include_source_image": False}
        if shape is not None:
            kwargs["shape"] = shape
        predictions = model.predict(frame_rgb, **kwargs)
        if isinstance(predictions, list):
            predictions = predictions[0] if predictions else None
        if predictions is None:
            return []
        instances = _keypoints_to_pose_instances(predictions, class_names)
        return _filter_pose_instances(
            instances,
            class_names,
            enable_nms=enable_nms,
            iou=iou,
            max_instances_per_frame=max_instances_per_frame,
            min_visible_keypoints=min_visible_keypoints,
            keypoint_conf=keypoint_conf,
        )

    return run_pose_video_inference(
        input_path=input_path,
        out_dir=out_dir,
        model_name=model_name,
        class_names=class_names,
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
            "model_weights": str(model_weights) if model_weights else None,
            "class_yaml": str(class_yaml) if class_yaml else None,
            "device": device,
            "conf": conf,
            "iou": iou,
            "keypoint_conf": keypoint_conf,
            "enable_nms": enable_nms,
            "max_instances_per_frame": max_instances_per_frame,
            "min_visible_keypoints": min_visible_keypoints,
            "shape": shape,
            "load_info": load_info.__dict__,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RF-DETR keypoint video inference and save skeleton visualization videos.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--class-yaml")
    parser.add_argument("--device")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--keypoint-conf", type=float, default=0.2)
    parser.add_argument("--disable-nms", action="store_true")
    parser.add_argument("--max-instances-per-frame", type=int)
    parser.add_argument("--min-visible-keypoints", type=int)
    parser.add_argument("--shape")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-seconds", type=float)
    parser.add_argument("--end-seconds", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--font-path")
    parser.add_argument("--font-size", type=int, default=20)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--keypoint-radius", type=int, default=4)
    parser.add_argument("--hide-bbox", action="store_true")
    parser.add_argument("--hide-skeleton", action="store_true")
    parser.add_argument("--hide-keypoints", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run", action="store_true", help="Actually load the model and write videos. Omit for dry-run preview.")
    args = parser.parse_args()

    result = run_rfdetr_pose_video_inference(
        input_path=args.input_path,
        out_dir=args.out_dir,
        model_weights=args.weights,
        class_yaml=args.class_yaml,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        keypoint_conf=args.keypoint_conf,
        enable_nms=not args.disable_nms,
        max_instances_per_frame=args.max_instances_per_frame,
        min_visible_keypoints=args.min_visible_keypoints,
        shape=_normalise_shape(args.shape),
        recursive=args.recursive,
        max_videos=args.max_videos,
        frame_stride=args.frame_stride,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        max_frames=args.max_frames,
        run_name=args.run_name,
        font_path=args.font_path,
        font_size=args.font_size,
        line_width=args.line_width,
        keypoint_radius=args.keypoint_radius,
        draw_bbox=not args.hide_bbox,
        draw_skeleton=not args.hide_skeleton,
        draw_keypoints=not args.hide_keypoints,
        overwrite=args.overwrite,
        dry_run=not args.run,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

