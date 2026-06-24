from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Sequence

from labelstudio_bbox_tools.pseudo_label.rfdetr import _detections_to_candidates, _load_rfdetr_model
from labelstudio_bbox_tools.pseudo_label.yolo import classwise_nms_indices
from labelstudio_bbox_tools.video_inference.classes import load_class_names, make_class_color_map
from labelstudio_bbox_tools.video_inference.common import Detection, preview_video_inputs, run_video_inference


def _candidate_to_detections(
    *,
    boxes: Sequence[Sequence[float]],
    class_ids: Sequence[int],
    scores: Sequence[float],
    class_names: Sequence[str],
    keep: Sequence[int],
    conf: float,
) -> list[Detection]:
    detections: list[Detection] = []
    for idx in keep:
        class_id = int(class_ids[idx])
        if not 0 <= class_id < len(class_names):
            continue
        score = float(scores[idx])
        if score < conf:
            continue
        box = boxes[idx]
        detections.append(
            Detection(
                xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                class_id=class_id,
                class_name=class_names[class_id],
                score=score,
            )
        )
    return detections


def run_rfdetr_video_inference(
    *,
    input_path: str | Path,
    out_dir: str | Path,
    model_weights: str | Path,
    class_yaml: str | Path | None = None,
    manual_classes: Sequence[str] | None = None,
    model_variant: str = "medium",
    device: str | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    recursive: bool = False,
    max_videos: int | None = None,
    frame_stride: int = 1,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    max_frames: int | None = None,
    expected_class_count: int | None = 28,
    strict_class_count: bool = True,
    run_name: str | None = None,
    font_path: str | Path | None = None,
    font_size: int = 20,
    line_width: int = 3,
    score_digits: int = 2,
    codec: str = "mp4v",
    overwrite: bool = False,
    temp_frame_dir: str | Path | None = None,
    dry_run: bool = True,
):
    class_names = load_class_names(
        class_yaml=class_yaml,
        manual_classes=manual_classes,
        expected_count=expected_class_count,
        strict_count=strict_class_count,
    )
    color_map = make_class_color_map(class_names)
    model_name = "rfdetr"
    if dry_run:
        print(
            f"[dry-run] model={model_name}, variant={model_variant}, weights={model_weights}, "
            f"classes={len(class_names)}"
        )
        return preview_video_inputs(
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

    model, load_info = _load_rfdetr_model(model_weights=model_weights, model_variant=model_variant, device=device)
    print(f"[info] loaded RF-DETR class={load_info.model_class}, mode={load_info.load_mode}, variant={load_info.model_variant}")

    temp_context = None
    if temp_frame_dir is None:
        temp_context = tempfile.TemporaryDirectory(prefix="lsbbox_rfdetr_frames_")
        temp_root = Path(temp_context.name)
    else:
        temp_root = Path(temp_frame_dir).expanduser().resolve()
        temp_root.mkdir(parents=True, exist_ok=True)

    def predict_frame(frame_bgr, frame_index: int, timestamp: float, video_path: Path | None) -> list[Detection]:
        stem = video_path.stem if video_path else "frame"
        temp_path = temp_root / f"{stem}__frame{frame_index:08d}.jpg"
        cv2.imwrite(str(temp_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        detections_raw = model.predict(str(temp_path), threshold=conf, include_source_image=False)
        if isinstance(detections_raw, list):
            detections_raw = detections_raw[0] if detections_raw else None
        boxes, class_ids, scores = _detections_to_candidates(detections_raw)
        if not boxes:
            return []
        keep = classwise_nms_indices(boxes, class_ids, scores, class_names, None, default_iou=iou)
        return _candidate_to_detections(
            boxes=boxes,
            class_ids=class_ids,
            scores=scores,
            class_names=class_names,
            keep=keep,
            conf=conf,
        )

    try:
        return run_video_inference(
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
            score_digits=score_digits,
            codec=codec,
            overwrite=overwrite,
            run_config={
                "model_weights": str(model_weights),
                "class_yaml": str(class_yaml) if class_yaml else None,
                "model_variant": model_variant,
                "device": device,
                "conf": conf,
                "iou": iou,
                "temp_frame_dir": str(temp_frame_dir) if temp_frame_dir else None,
                "load_info": load_info.__dict__,
            },
        )
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RF-DETR video inference and save bbox visualization videos.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--class-yaml", required=True)
    parser.add_argument("--model-variant", default="medium")
    parser.add_argument("--device")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--temp-frame-dir")
    parser.add_argument("--run", action="store_true", help="Actually load the model and write videos. Omit for dry-run preview.")
    args = parser.parse_args()

    result = run_rfdetr_video_inference(
        input_path=args.input_path,
        out_dir=args.out_dir,
        model_weights=args.weights,
        class_yaml=args.class_yaml,
        model_variant=args.model_variant,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
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
        overwrite=args.overwrite,
        temp_frame_dir=args.temp_frame_dir,
        dry_run=not args.run,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
