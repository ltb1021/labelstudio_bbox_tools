from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labelstudio_bbox_tools.config import settings_from_env
from labelstudio_bbox_tools.ls_api import get_label_names, iter_project_tasks, iter_project_task_ids, safe_json
from labelstudio_bbox_tools.ls_client import make_client


@dataclass(frozen=True)
class AnnotationImportSummary:
    project_id: int
    total_images: int
    matched_tasks: int
    objects: int
    uploaded_tasks: int
    dry_run: bool

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _decode_rel(url: str) -> str:
    from urllib.parse import unquote
    return unquote(url.split("?d=", 1)[1]) if "?d=" in url else unquote(url)


def fetch_task_map(client: Any, project_id: int, *, page_size: int = 1000, verbose: bool = False) -> dict[str, dict]:
    by_basename: dict[str, list[dict]] = {}
    by_rel: dict[str, dict] = {}
    count = 0
    for task in iter_project_tasks(client, project_id, page_size=page_size):
        image_url = task.get("data", {}).get("image", "")
        rel = _decode_rel(image_url)
        info = {"id": task["id"], "url": image_url, "rel": rel, "base": Path(rel).name.lower()}
        by_basename.setdefault(info["base"], []).append(info)
        by_rel[rel.lower()] = info
        count += 1
    if verbose:
        print(f"[fetch] collected={count:,}")
    return {"by_basename": by_basename, "by_rel": by_rel}


def _load_mmyolo(path: Path) -> tuple[dict[int, tuple[str, int, int]], dict[int, list[tuple[int, tuple[float, float, float, float]]]], dict[int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    images = {int(item["id"]): (item["file_name"], int(item["width"]), int(item["height"])) for item in data.get("images", [])}
    anns: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for ann in data.get("annotations", []):
        bbox = tuple(float(v) for v in ann["bbox"])
        anns.setdefault(int(ann["image_id"]), []).append((int(ann["category_id"]), bbox))
    id_to_name = {int(item["id"]): str(item["name"]) for item in data.get("categories", [])}
    return images, anns, id_to_name


def _rect_result(x: float, y: float, w: float, h: float, image_width: int, image_height: int, label: str, from_name: str, to_name: str) -> dict:
    def pct(value: float, size: int) -> float:
        return max(0.0, min(100.0, value / size * 100)) if size else 0.0

    return {
        "id": str(uuid.uuid4()),
        "type": "rectanglelabels",
        "to_name": to_name,
        "from_name": from_name,
        "origin": "manual",
        "original_width": image_width,
        "original_height": image_height,
        "image_rotation": 0,
        "value": {
            "x": pct(x, image_width),
            "y": pct(y, image_height),
            "width": pct(w, image_width),
            "height": pct(h, image_height),
            "rotation": 0,
            "rectanglelabels": [label],
        },
    }


def iter_mmyolo_matches(
    ann_source: str | Path,
    task_map: dict[str, dict],
    valid_labels: set[str],
    *,
    from_name: str,
    to_name: str,
    image_match_mode: str,
    mirror_root: str | Path,
):
    images, anns, id_to_name = _load_mmyolo(Path(ann_source))
    mirror = Path(mirror_root).expanduser().resolve()
    for image_id, (file_name, width, height) in images.items():
        file_path = Path(file_name)
        if image_match_mode == "fullpath":
            try:
                rel = file_path.expanduser().resolve().relative_to(mirror).as_posix().lower()
            except Exception:
                continue
            info = task_map["by_rel"].get(rel)
            if info is None:
                info = next((value for key, value in task_map["by_rel"].items() if key.endswith(rel)), None)
        elif image_match_mode == "basename":
            info = (task_map["by_basename"].get(file_path.name.lower()) or [None])[0]
        else:
            raise ValueError("image_match_mode must be fullpath or basename")
        if not info:
            continue

        results = []
        for category_id, bbox in anns.get(image_id, []):
            label = id_to_name.get(category_id, str(category_id))
            if label not in valid_labels:
                continue
            x, y, w, h = bbox
            results.append(_rect_result(x, y, w, h, width, height, label, from_name, to_name))
        if results:
            yield int(info["id"]), results


def _upload_annotation(client: Any, task_id: int, results: list[dict], *, meta_tag: str | None, import_id: str | None, completed_by: int = 1) -> bool:
    payload: dict[str, Any] = {"result": results, "completed_by": completed_by}
    if meta_tag:
        payload["meta"] = {"batch_tag": meta_tag}
    if import_id:
        payload["import_id"] = import_id
    client.make_request("POST", f"/api/tasks/{task_id}/annotations/", json=payload)
    return True


def _upload_prediction(client: Any, task_id: int, results: list[dict], *, meta_tag: str | None, import_id: str | None, model_version: str) -> bool:
    payload: dict[str, Any] = {"task": task_id, "result": results, "score": 0, "model_version": model_version}
    if meta_tag:
        payload["meta"] = {"batch_tag": meta_tag}
    if import_id:
        payload["import_id"] = import_id
    client.make_request("POST", "/api/predictions/", json=payload)
    return True


def import_annotations_to_project(
    *,
    dataset_type: str,
    ann_source: str | Path,
    project_id: int,
    ls_url: str,
    api_key: str,
    meta_tag: str | None = None,
    import_id: str | None = None,
    pred_model: str | None = None,
    pred_tag: str | None = None,
    dummy_pred: bool = True,
    target_shape: str = "bbox",
    image_match_mode: str = "basename",
    upload_mode: str = "annotation",
    auto_task: bool = False,
    mirror_root: str | Path | None = None,
    batch_size: int = 300,
    dry_run: bool = True,
) -> AnnotationImportSummary:
    if dataset_type.lower() != "mmyolo":
        raise ValueError("Only mmyolo is supported in the refactored importer")
    if target_shape != "bbox":
        raise ValueError("Only bbox import is currently supported")
    if upload_mode not in {"annotation", "prediction"}:
        raise ValueError("upload_mode must be annotation or prediction")
    if auto_task:
        raise NotImplementedError("auto_task is intentionally not enabled in this refactor")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if mirror_root is None:
        raise ValueError("mirror_root is required for reliable path matching")

    pred_tag = pred_tag or meta_tag
    import_id = import_id or meta_tag
    pred_model = pred_model or "dummy_pred"

    client = make_client(ls_url, api_key)
    project = client.get_project(project_id)
    label_names = get_label_names(project.label_config, shape=target_shape)
    task_map = fetch_task_map(client, project_id, verbose=True)
    total_images = len(json.loads(Path(ann_source).read_text(encoding="utf-8")).get("images", []))

    batch: list[tuple[int, list[dict]]] = []
    matched = 0
    objects = 0
    uploaded = 0

    def flush(items: list[tuple[int, list[dict]]]) -> int:
        if dry_run:
            return len(items)
        count = 0
        for task_id, results in items:
            if upload_mode == "annotation":
                _upload_annotation(client, task_id, results, meta_tag=meta_tag, import_id=import_id)
                if dummy_pred:
                    _upload_prediction(client, task_id, [], meta_tag=pred_tag, import_id=import_id, model_version=pred_model)
            else:
                _upload_prediction(client, task_id, results, meta_tag=pred_tag, import_id=import_id, model_version=pred_model)
            count += 1
        return count

    for task_id, results in iter_mmyolo_matches(
        ann_source,
        task_map,
        set(label_names.labels),
        from_name=label_names.from_name,
        to_name=label_names.to_name,
        image_match_mode=image_match_mode,
        mirror_root=mirror_root,
    ):
        matched += 1
        objects += len(results)
        batch.append((task_id, results))
        if len(batch) >= batch_size:
            uploaded += flush(batch)
            batch.clear()

    if batch:
        uploaded += flush(batch)

    print(f"[{'dry-run' if dry_run else 'ok'}] project={project_id}, total_images={total_images:,}, matched={matched:,}, uploaded={uploaded:,}, objects={objects:,}")
    return AnnotationImportSummary(project_id, total_images, matched, objects, uploaded, dry_run)


def delete_by_import_id(
    *,
    project_id: int,
    ls_url: str,
    api_key: str,
    import_id: str | None = None,
    meta_tag: str | None = None,
    do_predictions: bool = True,
    dry_run: bool = True,
) -> dict[str, int | bool]:
    if not import_id and not meta_tag:
        raise ValueError("import_id or meta_tag is required")
    client = make_client(ls_url, api_key)
    ann_ids: list[int] = []
    pred_ids: list[int] = []
    for task_id in iter_project_task_ids(client, project_id):
        try:
            annotations = safe_json(client.make_request("GET", f"/api/tasks/{task_id}/annotations/"))
        except Exception:
            annotations = []
        for ann in annotations or []:
            if (import_id and str(ann.get("import_id")) == str(import_id)) or (meta_tag and ann.get("meta", {}).get("batch_tag") == meta_tag):
                ann_ids.append(int(ann["id"]))
        if do_predictions:
            try:
                data = safe_json(client.make_request("GET", "/api/predictions/", params={"task": task_id}))
            except Exception:
                data = []
            predictions = data.get("results") if isinstance(data, dict) else data
            for pred in predictions or []:
                if (import_id and str(pred.get("import_id")) == str(import_id)) or (meta_tag and pred.get("meta", {}).get("batch_tag") == meta_tag):
                    pred_ids.append(int(pred["id"]))
    if not dry_run:
        for ann_id in ann_ids:
            client.make_request("DELETE", f"/api/annotations/{ann_id}")
        for pred_id in pred_ids:
            client.make_request("DELETE", f"/api/predictions/{pred_id}")
    return {"matched_annotations": len(ann_ids), "matched_predictions": len(pred_ids), "dry_run": dry_run}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import MMYOLO annotations into existing Label Studio tasks.")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--ann-source", required=True)
    parser.add_argument("--mirror-root", required=True)
    parser.add_argument("--image-match-mode", choices=["basename", "fullpath"], default="fullpath")
    parser.add_argument("--upload-mode", choices=["annotation", "prediction"], default="prediction")
    parser.add_argument("--meta-tag")
    parser.add_argument("--import-id")
    parser.add_argument("--pred-model")
    parser.add_argument("--pred-tag")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    settings = settings_from_env()
    summary = import_annotations_to_project(
        dataset_type="mmyolo",
        ann_source=args.ann_source,
        project_id=args.project_id,
        ls_url=settings.url,
        api_key=settings.api_key,
        meta_tag=args.meta_tag,
        import_id=args.import_id,
        pred_model=args.pred_model,
        pred_tag=args.pred_tag,
        image_match_mode=args.image_match_mode,
        upload_mode=args.upload_mode,
        mirror_root=args.mirror_root,
        batch_size=args.batch_size,
        dry_run=not args.run,
    )
    print(summary.as_dict())


if __name__ == "__main__":
    main()
