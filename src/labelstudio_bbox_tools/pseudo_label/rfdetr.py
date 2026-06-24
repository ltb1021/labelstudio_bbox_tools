from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from labelstudio_bbox_tools.config import settings_from_env
from labelstudio_bbox_tools.ls_api import get_label_names, iter_project_tasks
from labelstudio_bbox_tools.ls_client import make_client
from labelstudio_bbox_tools.paths import resolve_local_file_url
from labelstudio_bbox_tools.pseudo_label.yolo import (
    PseudoLabelSummary,
    _make_ls_result,
    _parse_json_map,
    classwise_nms_indices,
    load_yolo_class_names,
)

RFDETR_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


@dataclass(frozen=True)
class RfDetrLoadInfo:
    model_variant: str
    load_mode: str
    model_class: str


def load_rfdetr_class_names(
    *,
    class_yaml: str | Path | None = None,
    manual_classes: Sequence[str] | None = None,
) -> list[str]:
    return load_yolo_class_names(class_yaml=class_yaml, manual_classes=manual_classes)


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _detections_to_candidates(detections: Any) -> tuple[list[list[float]], list[int], list[float]]:
    xyxy = getattr(detections, "xyxy", None)
    if xyxy is None:
        return [], [], []

    boxes_raw = _to_list(xyxy)
    class_raw = _to_list(getattr(detections, "class_id", None))
    confidence_raw = _to_list(getattr(detections, "confidence", None))

    boxes: list[list[float]] = []
    class_ids: list[int] = []
    scores: list[float] = []
    for idx, box in enumerate(boxes_raw):
        if idx >= len(class_raw):
            continue
        score = confidence_raw[idx] if idx < len(confidence_raw) and confidence_raw[idx] is not None else 1.0
        boxes.append([float(v) for v in box])
        class_ids.append(int(class_raw[idx]))
        scores.append(float(score))
    return boxes, class_ids, scores


def _predict_threshold(conf: float, class_conf_map: dict[str, float] | None) -> float:
    thresholds = [float(conf)]
    if class_conf_map:
        thresholds.extend(float(value) for value in class_conf_map.values())
    return max(0.0, min(thresholds))


def _load_rfdetr_model(
    *,
    model_weights: str | Path,
    model_variant: str = "medium",
    device: str | None = None,
) -> tuple[Any, RfDetrLoadInfo]:
    try:
        import rfdetr
    except ImportError as exc:
        raise RuntimeError("RF-DETR is required. Activate the RF-DETR conda env or install the local RF-DETR package.") from exc

    weights_path = str(model_weights)
    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device

    normalized_variant = model_variant.lower().strip()
    if normalized_variant in {"auto", "from_checkpoint"}:
        model = rfdetr.RFDETR.from_checkpoint(weights_path, **kwargs)
        return model, RfDetrLoadInfo(normalized_variant, "from_checkpoint", type(model).__name__)

    if normalized_variant not in RFDETR_VARIANTS:
        raise ValueError(f"model_variant must be one of: auto, {', '.join(RFDETR_VARIANTS)}")

    model_cls = getattr(rfdetr, RFDETR_VARIANTS[normalized_variant])
    try:
        model = model_cls.from_checkpoint(weights_path, **kwargs)
        return model, RfDetrLoadInfo(normalized_variant, "from_checkpoint", type(model).__name__)
    except Exception as checkpoint_exc:
        try:
            model = model_cls(pretrain_weights=weights_path, **kwargs)
        except Exception as constructor_exc:
            raise RuntimeError(
                f"Failed to load RF-DETR weights with {model_cls.__name__}. "
                "Try model_variant='auto' if the checkpoint contains model metadata."
            ) from constructor_exc
        print(f"[warn] from_checkpoint failed for {model_cls.__name__}: {checkpoint_exc}")
        return model, RfDetrLoadInfo(normalized_variant, "pretrain_weights", type(model).__name__)


def _write_local_results(
    *,
    out_path: Path,
    save_format: str,
    ls_results: list[dict[str, Any]],
    class_names: Sequence[str],
) -> None:
    if save_format == "json":
        out_path.write_text(json.dumps(ls_results, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    lines = []
    for item in ls_results:
        label = item["value"]["rectanglelabels"][0]
        class_id = class_names.index(label)
        x = item["value"]["x"] / 100
        y = item["value"]["y"] / 100
        w = item["value"]["width"] / 100
        h = item["value"]["height"] / 100
        lines.append(f"{class_id} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def auto_label_rfdetr(
    *,
    ls_url: str,
    api_key: str,
    project_id: int,
    model_weights: str | Path,
    doc_root: str | Path,
    class_yaml: str | Path | None = None,
    classes: Sequence[str] | None = None,
    model_variant: str = "medium",
    device: str | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    batch_size: int = 1,
    class_conf_map: dict[str, float] | None = None,
    class_iou_map: dict[str, float] | None = None,
    save_local: bool = False,
    local_out_dir: str | Path | None = None,
    save_format: str = "json",
    pred_model: str | None = None,
    import_id: str | None = None,
    meta_tag: str | None = None,
    dry_run: bool = True,
    max_tasks: int | None = None,
    page_size: int = 1000,
) -> PseudoLabelSummary:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if save_format not in {"json", "yolo"}:
        raise ValueError("save_format must be 'json' or 'yolo'")

    class_names = load_rfdetr_class_names(class_yaml=class_yaml, manual_classes=classes)
    client = make_client(ls_url, api_key)
    project = client.get_project(project_id)
    label_names = get_label_names(project.label_config, shape="bbox")
    tasks = list(iter_project_tasks(client, project_id, page_size=page_size, max_tasks=max_tasks))

    image_items: list[tuple[dict, Path]] = []
    skipped = 0
    for task in tasks:
        image_url = task.get("data", {}).get("image")
        if not image_url:
            skipped += 1
            continue
        try:
            image_path = resolve_local_file_url(image_url, doc_root)
        except Exception:
            skipped += 1
            continue
        if not image_path.exists():
            skipped += 1
            continue
        image_items.append((task, image_path))

    print(
        f"[info] classes={len(class_names)}, tasks={len(tasks):,}, "
        f"images_found={len(image_items):,}, skipped={skipped:,}, model_variant={model_variant}"
    )
    if dry_run:
        for task, image_path in image_items[:5]:
            print(f"[dry-run] task={task.get('id')} image={image_path}")
        return PseudoLabelSummary(project_id, len(tasks), len(image_items), 0, 0, skipped, True)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image-size inspection.") from exc

    model, load_info = _load_rfdetr_model(model_weights=model_weights, model_variant=model_variant, device=device)
    print(f"[info] loaded RF-DETR class={load_info.model_class}, mode={load_info.load_mode}, variant={load_info.model_variant}")

    local_out: Path | None = None
    if save_local:
        if not local_out_dir:
            raise ValueError("local_out_dir is required when save_local=True")
        local_out = Path(local_out_dir).expanduser()
        local_out.mkdir(parents=True, exist_ok=True)

    tasks_with_predictions = 0
    boxes_written = 0
    model_version = pred_model or Path(str(model_weights)).stem
    predict_threshold = _predict_threshold(conf, class_conf_map)

    for start in tqdm(range(0, len(image_items), batch_size), desc="Predict & upload"):
        batch = image_items[start : start + batch_size]
        for task, image_path in batch:
            detections = model.predict(str(image_path), threshold=predict_threshold, include_source_image=False)
            if isinstance(detections, list):
                detections = detections[0] if detections else None
            boxes, class_ids, scores = _detections_to_candidates(detections)
            if not boxes:
                skipped += 1
                continue
            keep = classwise_nms_indices(boxes, class_ids, scores, class_names, class_iou_map, default_iou=iou)
            ls_results: list[dict[str, Any]] = []
            with Image.open(image_path) as image:
                width, height = image.size

            for idx in keep:
                class_id = class_ids[idx]
                class_name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
                score = float(scores[idx])
                threshold = float((class_conf_map or {}).get(class_name, (class_conf_map or {}).get("default", conf)))
                if score < threshold:
                    continue
                ls_results.append(
                    _make_ls_result(
                        box=boxes[idx],
                        image_width=width,
                        image_height=height,
                        label=class_name,
                        score=score,
                        from_name=label_names.from_name,
                        to_name=label_names.to_name,
                    )
                )

            if not ls_results:
                skipped += 1
                continue

            payload: dict[str, Any] = {
                "task": task["id"],
                "result": ls_results,
                "score": 0,
                "model_version": model_version,
            }
            if import_id is not None:
                payload["import_id"] = import_id
            if meta_tag is not None:
                payload["meta"] = {"batch_tag": meta_tag}
            client.make_request("POST", "/api/predictions/", json=payload)

            tasks_with_predictions += 1
            boxes_written += len(ls_results)

            if local_out:
                out_path = local_out / f"{image_path.stem}.{save_format}"
                _write_local_results(out_path=out_path, save_format=save_format, ls_results=ls_results, class_names=class_names)

    print(f"[ok] uploaded_tasks={tasks_with_predictions:,}, boxes={boxes_written:,}, skipped={skipped:,}")
    return PseudoLabelSummary(project_id, len(tasks), len(image_items), tasks_with_predictions, boxes_written, skipped, False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Label Studio predictions with an RF-DETR model.")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--class-yaml")
    parser.add_argument("--classes", help="Comma-separated fallback classes. YAML wins when both are provided.")
    parser.add_argument("--model-variant", default="medium", choices=["auto", "nano", "small", "medium", "large"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--class-conf-json")
    parser.add_argument("--class-iou-json")
    parser.add_argument("--pred-model")
    parser.add_argument("--import-id")
    parser.add_argument("--meta-tag")
    parser.add_argument("--device")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--run", action="store_true", help="Actually upload predictions. Overrides --dry-run.")
    args = parser.parse_args()

    settings = settings_from_env()
    manual_classes = [item.strip() for item in args.classes.split(",")] if args.classes else None
    summary = auto_label_rfdetr(
        ls_url=settings.url,
        api_key=settings.api_key,
        project_id=args.project_id,
        model_weights=args.weights,
        doc_root=settings.doc_root,
        class_yaml=args.class_yaml,
        classes=manual_classes,
        model_variant=args.model_variant,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        batch_size=args.batch_size,
        class_conf_map=_parse_json_map(args.class_conf_json),
        class_iou_map=_parse_json_map(args.class_iou_json),
        pred_model=args.pred_model,
        import_id=args.import_id,
        meta_tag=args.meta_tag,
        dry_run=not args.run,
        max_tasks=args.max_tasks,
    )
    print(summary.as_dict())


if __name__ == "__main__":
    main()
