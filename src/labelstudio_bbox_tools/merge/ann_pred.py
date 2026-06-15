from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from labelstudio_bbox_tools.config import settings_from_env
from labelstudio_bbox_tools.ls_api import completed_by_id, get_label_names, iter_project_task_ids, pick_latest, safe_json
from labelstudio_bbox_tools.ls_client import make_client
from labelstudio_bbox_tools.paths import resolve_local_file_url


@dataclass(frozen=True)
class SrcAnn:
    user_id: int
    min_lead: float = 0.0
    kind: str = "ann"


@dataclass(frozen=True)
class SrcPred:
    model_ver: str
    score_thr: float = 0.0
    kind: str = "pred"


@dataclass(frozen=True)
class MergeSummary:
    project_id: int
    tasks_seen: int
    tasks_written: int
    objects_written: int
    dry_run: bool

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    denom = area1 + area2 - inter
    return inter / denom if denom > 0 else 0.0


def list_project_model_versions(ls_url: str, api_key: str, project_id: int, *, with_counts: bool = False) -> Any:
    client = make_client(ls_url, api_key)
    versions: dict[str, int] = {}
    for task_id in iter_project_task_ids(client, project_id):
        try:
            response = client.make_request("GET", "/api/predictions/", params={"task": task_id})
            data = safe_json(response)
        except Exception:
            data = []
        predictions = data.get("results") if isinstance(data, dict) else data
        for pred in predictions or []:
            model_version = pred.get("model_version")
            if model_version:
                versions[model_version] = versions.get(model_version, 0) + 1
    ordered = sorted(versions)
    return (ordered, versions) if with_counts else ordered


def _select_ann_for_task(client: Any, task_id: int, user_id: int, min_lead: float) -> dict | None:
    try:
        anns = safe_json(client.make_request("GET", f"/api/tasks/{task_id}/annotations/"))
    except Exception:
        anns = []
    candidates = []
    for ann in anns or []:
        if completed_by_id(ann) != user_id:
            continue
        lead_time = ann.get("lead_time")
        if lead_time is None or float(lead_time) < min_lead:
            continue
        candidates.append(ann)
    return pick_latest(candidates)


def _select_pred_for_task(client: Any, task_id: int, model_ver: str) -> dict | None:
    try:
        data = safe_json(client.make_request("GET", "/api/predictions/", params={"task": task_id}))
    except Exception:
        data = []
    preds = data.get("results") if isinstance(data, dict) else data
    candidates = [pred for pred in preds or [] if pred.get("model_version") == model_ver]
    return pick_latest(candidates)


def _task_image_size(client: Any, task_id: int, doc_root: Path | None) -> tuple[int | None, int | None]:
    if doc_root is None:
        return None, None
    try:
        task = safe_json(client.make_request("GET", f"/api/tasks/{task_id}"))
    except Exception:
        return None, None
    image_url = task.get("data", {}).get("image")
    if not image_url:
        return None, None
    try:
        from PIL import Image
        image_path = resolve_local_file_url(image_url, doc_root)
        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return None, None


def _xyxy_from_result(result: dict, width: int | None, height: int | None):
    rw = result.get("original_width")
    rh = result.get("original_height")
    if rw and rh:
        width = width or int(float(rw))
        height = height or int(float(rh))
    if not width or not height:
        return (0.0, 0.0, 0.0, 0.0), None, None
    value = result["value"]
    x1 = float(value["x"]) / 100 * width
    y1 = float(value["y"]) / 100 * height
    w = float(value["width"]) / 100 * width
    h = float(value["height"]) / 100 * height
    return (x1, y1, x1 + w, y1 + h), width, height


def _build_group_index(class_groups: Sequence[Sequence[str]] | None) -> dict[str, int]:
    index: dict[str, int] = {}
    for group_id, group in enumerate(class_groups or []):
        for label in group:
            index[str(label)] = group_id
    return index


def _same_group(label_a: str, label_b: str, group_index: dict[str, int]) -> bool:
    return label_a in group_index and label_b in group_index and group_index[label_a] == group_index[label_b]


def _pair_iou_threshold(
    label_a: str,
    label_b: str,
    base_threshold: float,
    per_class_threshold: dict[str, float] | None,
    group_index: dict[str, int],
    grouped_threshold: float | None,
) -> float | None:
    if label_a == label_b:
        return float((per_class_threshold or {}).get(label_a, base_threshold))
    if _same_group(label_a, label_b, group_index):
        return float(grouped_threshold if grouped_threshold is not None else base_threshold)
    return None


def _try_add_box(
    boxes: list[tuple[str, tuple[float, float, float, float], float, int]],
    label: str,
    box: tuple[float, float, float, float],
    score: float,
    *,
    resolve: str,
    base_threshold: float,
    per_class_threshold: dict[str, float] | None,
    group_index: dict[str, int],
    grouped_threshold: float | None,
    source_order: int,
) -> None:
    for idx, (prev_label, prev_box, prev_score, _) in enumerate(boxes):
        threshold = _pair_iou_threshold(label, prev_label, base_threshold, per_class_threshold, group_index, grouped_threshold)
        if threshold is None:
            continue
        if _iou(box, prev_box) >= threshold:
            if resolve == "keep_earlier":
                return
            if resolve == "higher_score":
                if score <= prev_score:
                    return
                boxes[idx] = (label, box, score, source_order)
                return
            raise ValueError("resolve must be 'keep_earlier' or 'higher_score'")
    boxes.append((label, box, score, source_order))


def _collect_boxes_for_task(
    client: Any,
    task_id: int,
    sources: Sequence[SrcAnn | SrcPred],
    *,
    doc_root: Path | None,
    resolve: str,
    iou_thr_base: float,
    iou_thr_per_class: dict[str, float] | None,
    class_groups: Sequence[Sequence[str]] | None,
    grouped_iou_thr: float | None,
) -> tuple[list[tuple[str, tuple[float, float, float, float], float, int]], int | None, int | None]:
    group_index = _build_group_index(class_groups)
    boxes: list[tuple[str, tuple[float, float, float, float], float, int]] = []
    width = height = None
    loaded_image_size = False

    for source_order, source in enumerate(sources):
        item = None
        if source.kind == "ann":
            item = _select_ann_for_task(client, task_id, source.user_id, source.min_lead)  # type: ignore[arg-type]
        elif source.kind == "pred":
            item = _select_pred_for_task(client, task_id, source.model_ver)  # type: ignore[arg-type]
        if not item:
            continue

        for result in item.get("result", []) or []:
            if result.get("type") != "rectanglelabels":
                continue
            labels = result.get("value", {}).get("rectanglelabels") or []
            if not labels:
                continue
            score = 1.0 if source.kind == "ann" else float(result.get("score", item.get("score", 0.0)) or 0.0)
            if source.kind == "pred" and score < getattr(source, "score_thr", 0.0):
                continue
            box, width, height = _xyxy_from_result(result, width, height)
            if (not width or not height) and not loaded_image_size:
                width, height = _task_image_size(client, task_id, doc_root)
                loaded_image_size = True
                box, width, height = _xyxy_from_result(result, width, height)
            if not width or not height:
                continue
            _try_add_box(
                boxes,
                str(labels[0]),
                box,
                score,
                resolve=resolve,
                base_threshold=iou_thr_base,
                per_class_threshold=iou_thr_per_class,
                group_index=group_index,
                grouped_threshold=grouped_iou_thr,
                source_order=source_order,
            )
    return boxes, width, height


def _make_ls_result(label: str, box: Sequence[float], score: float, width: int, height: int, *, from_name: str, to_name: str) -> dict:
    x1, y1, x2, y2 = [float(v) for v in box]
    return {
        "id": str(uuid.uuid4()),
        "type": "rectanglelabels",
        "from_name": from_name,
        "to_name": to_name,
        "origin": "manual",
        "image_rotation": 0,
        "original_width": width,
        "original_height": height,
        "value": {
            "x": x1 / width * 100,
            "y": y1 / height * 100,
            "width": (x2 - x1) / width * 100,
            "height": (y2 - y1) / height * 100,
            "rotation": 0,
            "rectanglelabels": [label],
        },
        "score": score,
    }


def _merge(
    *,
    mode: str,
    ls_url: str,
    api_key: str,
    project_id: int,
    sources: Sequence[SrcAnn | SrcPred],
    iou_thr_base: float = 0.5,
    iou_thr_per_class: dict[str, float] | None = None,
    class_groups: Sequence[Sequence[str]] | None = None,
    grouped_iou_thr: float | None = None,
    resolve: str = "keep_earlier",
    new_model_ver: str | None = None,
    new_import_id: str | None = None,
    meta_tag: str | None = None,
    completed_by: int = 1,
    doc_root: str | Path | None = None,
    dry_run: bool = True,
    max_tasks: int | None = None,
) -> MergeSummary:
    if mode not in {"prediction", "annotation"}:
        raise ValueError("mode must be prediction or annotation")
    if resolve not in {"keep_earlier", "higher_score"}:
        raise ValueError("resolve must be keep_earlier or higher_score")
    if mode == "prediction" and not new_model_ver:
        raise ValueError("new_model_ver is required for prediction merge")
    if mode == "annotation" and not new_import_id:
        new_import_id = "merged_" + uuid.uuid4().hex[:8]

    client = make_client(ls_url, api_key)
    project = client.get_project(project_id)
    label_names = get_label_names(project.label_config, shape="bbox")
    root = Path(doc_root).expanduser() if doc_root else None

    task_count = 0
    tasks_written = 0
    objects_written = 0
    for task_id in iter_project_task_ids(client, project_id, max_tasks=max_tasks):
        task_count += 1
        boxes, width, height = _collect_boxes_for_task(
            client,
            task_id,
            sources,
            doc_root=root,
            resolve=resolve,
            iou_thr_base=iou_thr_base,
            iou_thr_per_class=iou_thr_per_class,
            class_groups=class_groups,
            grouped_iou_thr=grouped_iou_thr,
        )
        if not boxes or not width or not height:
            continue
        results = [
            _make_ls_result(label, box, score, width, height, from_name=label_names.from_name, to_name=label_names.to_name)
            for label, box, score, _source_order in boxes
        ]
        tasks_written += 1
        objects_written += len(results)
        if dry_run:
            continue

        if mode == "prediction":
            payload: dict[str, Any] = {"task": task_id, "result": results, "model_version": new_model_ver, "score": 0}
            if meta_tag:
                payload["meta"] = {"merged_tag": meta_tag}
            client.make_request("POST", "/api/predictions/", json=payload)
        else:
            payload = {"task": task_id, "result": results, "completed_by": completed_by, "import_id": new_import_id}
            if meta_tag:
                payload["meta"] = {"merged_tag": meta_tag}
            client.make_request("POST", f"/api/tasks/{task_id}/annotations/", json=payload)

    print(f"[{'dry-run' if dry_run else 'ok'}] tasks_seen={task_count:,}, tasks_written={tasks_written:,}, objects={objects_written:,}")
    return MergeSummary(project_id, task_count, tasks_written, objects_written, dry_run)


def merge_to_new_prediction(**kwargs: Any) -> MergeSummary:
    return _merge(mode="prediction", **kwargs)


def merge_to_new_annotation(**kwargs: Any) -> MergeSummary:
    return _merge(mode="annotation", **kwargs)


def _load_sources_json(path: str | Path) -> list[SrcAnn | SrcPred]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    sources: list[SrcAnn | SrcPred] = []
    for item in items:
        if item["kind"] == "ann":
            sources.append(SrcAnn(user_id=int(item["user_id"]), min_lead=float(item.get("min_lead", 0.0))))
        elif item["kind"] == "pred":
            sources.append(SrcPred(model_ver=str(item["model_ver"]), score_thr=float(item.get("score_thr", 0.0))))
        else:
            raise ValueError(f"Unknown source kind: {item['kind']}")
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Label Studio annotations/predictions into a new prediction or annotation.")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--sources-json", required=True, help="JSON list of source specs.")
    parser.add_argument("--mode", choices=["prediction", "annotation"], default="prediction")
    parser.add_argument("--new-model-ver")
    parser.add_argument("--new-import-id")
    parser.add_argument("--meta-tag")
    parser.add_argument("--resolve", choices=["keep_earlier", "higher_score"], default="keep_earlier")
    parser.add_argument("--iou-thr-base", type=float, default=0.5)
    parser.add_argument("--class-groups-json")
    parser.add_argument("--grouped-iou-thr", type=float)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    settings = settings_from_env()
    class_groups = json.loads(Path(args.class_groups_json).read_text(encoding="utf-8")) if args.class_groups_json else None
    func = merge_to_new_prediction if args.mode == "prediction" else merge_to_new_annotation
    summary = func(
        ls_url=settings.url,
        api_key=settings.api_key,
        project_id=args.project_id,
        sources=_load_sources_json(args.sources_json),
        iou_thr_base=args.iou_thr_base,
        class_groups=class_groups,
        grouped_iou_thr=args.grouped_iou_thr,
        resolve=args.resolve,
        new_model_ver=args.new_model_ver,
        new_import_id=args.new_import_id,
        meta_tag=args.meta_tag,
        doc_root=settings.doc_root,
        dry_run=not args.run,
        max_tasks=args.max_tasks,
    )
    print(summary.as_dict())


if __name__ == "__main__":
    main()
