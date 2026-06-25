from __future__ import annotations

import argparse
import json
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
from labelstudio_bbox_tools.video_inference.classes import make_class_color_map
from labelstudio_bbox_tools.video_inference.yolo11 import _normalise_imgsz


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _result_to_pose_instances(
    result: Any,
    class_names: Sequence[str],
    *,
    keypoint_names: Sequence[str] = COCO_PERSON_KEYPOINT_NAMES,
) -> list[PoseInstance]:
    boxes_obj = getattr(result, "boxes", None)
    keypoints_obj = getattr(result, "keypoints", None)
    if boxes_obj is None or keypoints_obj is None or len(boxes_obj) == 0:
        return []

    xy_all = _to_list(getattr(keypoints_obj, "xy", None))
    conf_all = _to_list(getattr(keypoints_obj, "conf", None))

    instances: list[PoseInstance] = []
    for idx, box in enumerate(boxes_obj):
        class_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else 0
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
        score = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 1.0
        xy = xy_all[idx] if idx < len(xy_all) else []
        conf = conf_all[idx] if idx < len(conf_all) else None
        instances.append(
            PoseInstance(
                xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                class_id=class_id,
                class_name=class_name,
                score=score,
                keypoints=keypoints_from_xy_conf(xy, conf, names=keypoint_names),
            )
        )
    return instances


def run_yolo11_pose_video_inference(
    *,
    input_path: str | Path,
    out_dir: str | Path,
    model_weights: str | Path = "yolo11x-pose.pt",
    class_yaml: str | Path | None = None,
    manual_classes: Sequence[str] | None = None,
    device: str | None = None,
    imgsz: int | tuple[int, int] | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    keypoint_conf: float = 0.2,
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
    model_name = "yolo11_pose"
    if dry_run:
        print(f"[dry-run] model={model_name}, weights={model_weights}, classes={class_names}")
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

    model = YOLO(str(model_weights))
    if device:
        model.to(device)

    def predict_frame(frame_bgr, frame_index: int, timestamp: float, video_path: Path | None) -> list[PoseInstance]:
        kwargs: dict[str, Any] = {
            "source": frame_bgr,
            "conf": conf,
            "iou": iou,
            "verbose": False,
        }
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
        if device:
            kwargs["device"] = device
        results = model.predict(**kwargs)
        if not results:
            return []
        return _result_to_pose_instances(results[0], class_names)

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
            "model_weights": str(model_weights),
            "class_yaml": str(class_yaml) if class_yaml else None,
            "device": device,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "keypoint_conf": keypoint_conf,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO11 pose video inference and save skeleton visualization videos.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--weights", default="yolo11x-pose.pt")
    parser.add_argument("--class-yaml")
    parser.add_argument("--device")
    parser.add_argument("--imgsz")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--keypoint-conf", type=float, default=0.2)
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

    result = run_yolo11_pose_video_inference(
        input_path=args.input_path,
        out_dir=args.out_dir,
        model_weights=args.weights,
        class_yaml=args.class_yaml,
        device=args.device,
        imgsz=_normalise_imgsz(args.imgsz),
        conf=args.conf,
        iou=args.iou,
        keypoint_conf=args.keypoint_conf,
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

