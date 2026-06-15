from __future__ import annotations

import argparse
import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from labelstudio_bbox_tools.config import settings_from_env
from labelstudio_bbox_tools.ls_api import get_label_names, iter_project_tasks
from labelstudio_bbox_tools.ls_client import make_client
from labelstudio_bbox_tools.paths import resolve_local_file_url


@dataclass(frozen=True)
class PseudoLabelSummary:
    project_id: int
    tasks_seen: int
    images_found: int
    tasks_with_predictions: int
    boxes_written: int
    skipped: int
    dry_run: bool

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def load_yolo_class_names(
    *,
    class_yaml: str | Path | None = None,
    manual_classes: Sequence[str] | None = None,
) -> list[str]:
    if class_yaml:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YOLO class yaml files") from exc
        data = yaml.safe_load(Path(class_yaml).read_text(encoding="utf-8")) or {}
        names = data.get("names")
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names, key=lambda x: int(x))]
        if isinstance(names, list):
            return [str(name) for name in names]
        raise ValueError("YOLO yaml must contain a names dict or list")

    if manual_classes:
        return [str(name) for name in manual_classes]
    raise ValueError("Provide class_yaml or manual_classes")


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
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


def classwise_nms_indices(
    boxes: Sequence[Sequence[float]],
    class_ids: Sequence[int],
    scores: Sequence[float],
    classes: Sequence[str],
    class_iou_map: dict[str, float] | None,
    *,
    default_iou: float = 0.5,
) -> list[int]:
    keep: list[int] = []
    for class_id in sorted(set(int(cid) for cid in class_ids)):
        indices = [idx for idx, cid in enumerate(class_ids) if int(cid) == class_id]
        indices.sort(key=lambda idx: float(scores[idx]), reverse=True)
        class_name = classes[class_id] if 0 <= class_id < len(classes) else str(class_id)
        iou_thr = float((class_iou_map or {}).get(class_name, (class_iou_map or {}).get("default", default_iou)))

        selected: list[int] = []
        for idx in indices:
            if all(_box_iou(boxes[idx], boxes[prev]) < iou_thr for prev in selected):
                selected.append(idx)
        keep.extend(selected)
    return sorted(keep)


def _parse_json_map(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    data = json.loads(raw)
    return {str(key): float(value) for key, value in data.items()}


def _normalise_imgsz(value: str | int | None) -> int | tuple[int, int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return value
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) == 1:
        return int(parts[0])
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise ValueError("imgsz must be an int or 'width,height'")


def _result_to_candidates(result: Any) -> tuple[list[list[float]], list[int], list[float]]:
    if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
        return [], [], []
    boxes: list[list[float]] = []
    class_ids: list[int] = []
    scores: list[float] = []
    for box in result.boxes:
        boxes.append([float(v) for v in box.xyxy[0].tolist()])
        class_ids.append(int(box.cls[0].item()))
        scores.append(float(box.conf[0].item()))
    return boxes, class_ids, scores


def _make_ls_result(
    *,
    box: Sequence[float],
    image_width: int,
    image_height: int,
    label: str,
    score: float,
    from_name: str,
    to_name: str,
) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return {
        "id": str(uuid.uuid4()),
        "type": "rectanglelabels",
        "to_name": to_name,
        "from_name": from_name,
        "value": {
            "x": x1 / image_width * 100,
            "y": y1 / image_height * 100,
            "width": (x2 - x1) / image_width * 100,
            "height": (y2 - y1) / image_height * 100,
            "rotation": 0,
            "rectanglelabels": [label],
        },
        "original_width": image_width,
        "original_height": image_height,
        "image_rotation": 0,
        "score": float(score),
    }


def auto_label_yolo(
    *,
    ls_url: str,
    api_key: str,
    project_id: int,
    model_weights: str | Path,
    doc_root: str | Path,
    class_yaml: str | Path | None = None,
    classes: Sequence[str] | None = None,
    device: str | None = None,
    imgsz: int | tuple[int, int] | None = None,
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

    class_names = load_yolo_class_names(class_yaml=class_yaml, manual_classes=classes)
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

    print(f"[info] classes={len(class_names)}, tasks={len(tasks):,}, images_found={len(image_items):,}, skipped={skipped:,}")
    if dry_run:
        for task, image_path in image_items[:5]:
            print(f"[dry-run] task={task.get('id')} image={image_path}")
        return PseudoLabelSummary(project_id, len(tasks), len(image_items), 0, 0, skipped, True)

    try:
        from PIL import Image
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Pseudo labeling requires Pillow and ultralytics. Install the pseudo extra or use an env that already has them.") from exc

    model = YOLO(model_weights)
    if device:
        model.to(device)

    local_out: Path | None = None
    if save_local:
        if not local_out_dir:
            raise ValueError("local_out_dir is required when save_local=True")
        local_out = Path(local_out_dir).expanduser()
        local_out.mkdir(parents=True, exist_ok=True)

    tasks_with_predictions = 0
    boxes_written = 0
    model_version = pred_model or Path(str(model_weights)).stem

    for start in tqdm(range(0, len(image_items), batch_size), desc="Predict & upload"):
        batch = image_items[start : start + batch_size]
        paths = [str(path) for _, path in batch]
        results = model.predict(source=paths, imgsz=imgsz, conf=conf, iou=iou, batch=len(paths), verbose=False)

        for result, (task, image_path) in zip(results, batch):
            boxes, class_ids, scores = _result_to_candidates(result)
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
                if save_format == "json":
                    out_path.write_text(json.dumps(ls_results, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
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

    print(f"[ok] uploaded_tasks={tasks_with_predictions:,}, boxes={boxes_written:,}, skipped={skipped:,}")
    return PseudoLabelSummary(project_id, len(tasks), len(image_items), tasks_with_predictions, boxes_written, skipped, False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Label Studio predictions with an Ultralytics YOLO model.")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--class-yaml")
    parser.add_argument("--classes", help="Comma-separated fallback classes. YAML wins when both are provided.")
    parser.add_argument("--imgsz")
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
    summary = auto_label_yolo(
        ls_url=settings.url,
        api_key=settings.api_key,
        project_id=args.project_id,
        model_weights=args.weights,
        doc_root=settings.doc_root,
        class_yaml=args.class_yaml,
        classes=manual_classes,
        device=args.device,
        imgsz=_normalise_imgsz(args.imgsz),
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
