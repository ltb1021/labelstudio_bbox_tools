from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from labelstudio_bbox_tools.config import settings_from_env
from labelstudio_bbox_tools.ls_client import make_client
from labelstudio_bbox_tools.paths import resolve_local_file_url

YOLO_HEADER = "# class cx cy w h (normalized)\n"
YOLO_OBB_HEADER = "# class x1 y1 x2 y2 x3 y3 x4 y4 (normalized)\n"


@dataclass(frozen=True)
class ExportResult:
    out_dir: Path
    task_count: int
    image_count: int
    annotation_count: int
    source_matched_count: int = 0
    skipped_no_source_count: int = 0
    skipped_empty_result_count: int = 0


def _round6(value: float) -> float:
    return round(float(value), 6)


def _make_path(base: Path, rel: Path) -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_txt(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def _get_labels(label_config_xml: str) -> list[str]:
    return re.findall(r'<Label[^>]+value="([^"]+)"', label_config_xml or "")


def _image_size(task: dict, src: Path, first_result: dict | None) -> tuple[int | None, int | None]:
    width = (first_result or {}).get("original_width") or task.get("meta", {}).get("width")
    height = (first_result or {}).get("original_height") or task.get("meta", {}).get("height")
    if not width or not height:
        try:
            with Image.open(src) as image:
                width, height = image.size
        except Exception:
            return None, None
    return int(width), int(height)


def _fetch_tasks(project, page_size: int = 500) -> list[dict]:
    try:
        project.get_tasks(page=1, page_size=1)
        paged = True
    except TypeError:
        paged = False

    if not paged:
        return list(project.get_tasks())

    tasks = []
    page = 1
    while True:
        try:
            batch = project.get_tasks(page=page, page_size=page_size)
        except Exception:
            time.sleep(1)
            continue
        if not batch:
            break
        tasks.extend(batch)
        page += 1
    return tasks


def _relative_for_export(src: Path, doc_root: Path) -> Path:
    try:
        return src.relative_to(doc_root)
    except ValueError:
        return src.relative_to("/")


def _select_annotation(task: dict, ann_user_id: int, ann_min_lead: float, accept_lead_time_none: bool) -> dict | None:
    def user_id(annotation: dict):
        completed_by = annotation.get("completed_by")
        return completed_by.get("id") if isinstance(completed_by, dict) else completed_by

    candidates = []
    for annotation in task.get("annotations", []):
        if user_id(annotation) != ann_user_id:
            continue
        lead_time = annotation.get("lead_time")
        if lead_time is None:
            if not (accept_lead_time_none and ann_min_lead <= 0):
                continue
        elif float(lead_time) < float(ann_min_lead):
            continue
        candidates.append(annotation)

    if not candidates:
        return None
    candidates.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)
    return candidates[0]


def _select_prediction(task: dict, pred_model_ver: str) -> dict | None:
    candidates = [p for p in task.get("predictions", []) if p.get("model_version") == pred_model_ver]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)
    return candidates[0]


def _mmyolo_file_name(src: Path, doc_root: Path, mode: str) -> str:
    if mode == "absolute":
        return str(src)
    if mode == "doc-root-relative":
        return str(src.relative_to(doc_root))
    raise ValueError("mmyolo_file_name must be 'absolute' or 'doc-root-relative'")


def _rectangle_results(results: list[dict], class_to_id: dict[str, int]) -> list[dict]:
    selected = []
    for result in results:
        if result.get("type") != "rectanglelabels":
            continue
        labels = result.get("value", {}).get("rectanglelabels") or []
        if not labels or labels[0] not in class_to_id:
            continue
        selected.append(result)
    return selected


def export_project(
    *,
    project_id: int,
    ls_url: str,
    api_key: str,
    out_dir: str | Path,
    doc_root: str | Path,
    export_type: str = "both",
    ann_format: str = "mmyolo",
    source_type: str = "ann",
    ann_user_id: int | None = None,
    ann_min_lead: float = 0.0,
    accept_lead_time_none: bool = True,
    pred_model_ver: str | None = None,
    copy_symlink_target: bool = True,
    only_finished: bool = True,
    write_images_list: bool = True,
    mmyolo_file_name: str = "absolute",
    include_empty_images: bool = False,
) -> ExportResult:
    if export_type not in {"img", "ann", "both"}:
        raise ValueError("export_type must be one of: img, ann, both")
    if ann_format not in {"yolo", "yolo_obb", "mmyolo"}:
        raise ValueError("ann_format must be one of: yolo, yolo_obb, mmyolo")
    if source_type not in {"ann", "pred"}:
        raise ValueError("source_type must be one of: ann, pred")
    if source_type == "ann" and ann_user_id is None:
        raise ValueError("ann_user_id is required when source_type='ann'")
    if source_type == "pred" and not pred_model_ver:
        raise ValueError("pred_model_ver is required when source_type='pred'")

    doc_root_path = Path(doc_root).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    image_root = out_path / "images"
    ann_root = out_path / ("labels" if ann_format != "mmyolo" else "")
    image_root.mkdir(parents=True, exist_ok=True)

    project = make_client(ls_url, api_key).get_project(project_id)
    tasks = _fetch_tasks(project)
    if only_finished:
        tasks = [task for task in tasks if task.get("is_labeled") or task.get("completed_at")]
    print(f"[info] tasks={len(tasks):,}")

    labels = _get_labels(project.label_config)
    (out_path / "classes.txt").write_text("\n".join(labels), encoding="utf-8")
    class_to_id = {name: idx for idx, name in enumerate(labels)}

    coco = None
    ann_id = 1
    if ann_format == "mmyolo":
        coco = {
            "info": {},
            "licenses": [],
            "categories": [{"id": idx + 1, "name": name, "supercategory": "root"} for idx, name in enumerate(labels)],
            "images": [],
            "annotations": [],
        }

    exported_images = []
    source_matched_count = 0
    skipped_no_source_count = 0
    skipped_empty_result_count = 0

    for task in tqdm(tasks, desc="Exporting"):
        src = resolve_local_file_url(task["data"]["image"], doc_root_path)
        rel_for_copy = _relative_for_export(src, doc_root_path)

        candidate = None
        if source_type == "ann":
            candidate = _select_annotation(task, int(ann_user_id), ann_min_lead, accept_lead_time_none)
        else:
            candidate = _select_prediction(task, str(pred_model_ver))

        results = (candidate.get("result") or []) if candidate else []
        rectangle_results = _rectangle_results(results, class_to_id)
        if candidate:
            source_matched_count += 1

        if not include_empty_images and not rectangle_results:
            if candidate is None:
                skipped_no_source_count += 1
            else:
                skipped_empty_result_count += 1
            continue

        if export_type in {"img", "both"}:
            dst = _make_path(image_root, rel_for_copy)
            if not dst.exists():
                copy_src = src.resolve() if src.is_symlink() and copy_symlink_target else src
                shutil.copy2(copy_src, dst)
            exported_images.append(dst)

        if export_type in {"ann", "both"}:
            if ann_format == "mmyolo":
                width, height = _image_size(task, src, rectangle_results[0] if rectangle_results else None)
                if not width or not height:
                    continue
                assert coco is not None
                coco["images"].append(
                    {
                        "id": task["id"],
                        "file_name": _mmyolo_file_name(src, doc_root_path, mmyolo_file_name),
                        "width": width,
                        "height": height,
                    }
                )
                for result in rectangle_results:
                    class_name = result["value"]["rectanglelabels"][0]
                    x_px = result["value"]["x"] * width / 100
                    y_px = result["value"]["y"] * height / 100
                    box_w = result["value"]["width"] * width / 100
                    box_h = result["value"]["height"] * height / 100
                    coco["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": task["id"],
                            "category_id": class_to_id[class_name] + 1,
                            "bbox": [_round6(x_px), _round6(y_px), _round6(box_w), _round6(box_h)],
                            "area": _round6(box_w * box_h),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1
            else:
                if not rectangle_results and not include_empty_images:
                    continue
                width, height = _image_size(task, src, rectangle_results[0] if rectangle_results else None)
                if not width or not height:
                    continue
                header = YOLO_HEADER if ann_format == "yolo" else YOLO_OBB_HEADER
                lines = [header]
                for result in rectangle_results:
                    class_id = class_to_id[result["value"]["rectanglelabels"][0]]
                    x = result["value"]["x"] / 100
                    y = result["value"]["y"] / 100
                    box_w = result["value"]["width"] / 100
                    box_h = result["value"]["height"] / 100
                    if ann_format == "yolo":
                        lines.append(f"{class_id} {_round6(x + box_w / 2)} {_round6(y + box_h / 2)} {_round6(box_w)} {_round6(box_h)}\n")
                    else:
                        x1, y1 = _round6(x), _round6(y)
                        x2, y2 = _round6(x + box_w), _round6(y)
                        x3, y3 = _round6(x + box_w), _round6(y + box_h)
                        x4, y4 = _round6(x), _round6(y + box_h)
                        lines.append(f"{class_id} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
                _write_txt(_make_path(ann_root, rel_for_copy.with_suffix(".txt")), lines)

        if export_type in {"img", "both"} or (write_images_list and ann_format == "mmyolo"):
            exported_images.append(_make_path(image_root, rel_for_copy))

    annotation_count = 0
    if export_type in {"ann", "both"} and ann_format == "mmyolo":
        assert coco is not None
        annotation_count = len(coco["annotations"])
        (out_path / "annotations_mmyolo.json").write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

    if write_images_list:
        existing = sorted({path.relative_to(out_path) for path in exported_images if path.exists()})
        (out_path / "images_all.txt").write_text("\n".join(map(str, existing)), encoding="utf-8")

    print(
        "[info] source_matched="
        f"{source_matched_count:,}, skipped_no_source={skipped_no_source_count:,}, "
        f"skipped_empty_result={skipped_empty_result_count:,}"
    )
    print(f"[ok] export complete: {out_path}")
    image_count = len(coco["images"]) if coco else len({path for path in exported_images if path.exists()})
    return ExportResult(
        out_path,
        len(tasks),
        image_count,
        annotation_count,
        source_matched_count,
        skipped_no_source_count,
        skipped_empty_result_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Label Studio annotations/predictions.")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--export-type", default="ann", choices=["img", "ann", "both"])
    parser.add_argument("--ann-format", default="mmyolo", choices=["yolo", "yolo_obb", "mmyolo"])
    parser.add_argument("--source-type", default="ann", choices=["ann", "pred"])
    parser.add_argument("--ann-user-id", type=int)
    parser.add_argument("--ann-min-lead", type=float, default=0.0)
    parser.add_argument("--reject-lead-time-none", action="store_true")
    parser.add_argument("--pred-model-ver")
    parser.add_argument("--include-unfinished", action="store_true")
    parser.add_argument("--include-empty-images", action="store_true")
    parser.add_argument("--mmyolo-file-name", default="absolute", choices=["absolute", "doc-root-relative"])
    parser.add_argument("--dotenv", default=".env")
    args = parser.parse_args()

    settings = settings_from_env(args.dotenv)
    export_project(
        project_id=args.project_id,
        ls_url=settings.url,
        api_key=settings.api_key,
        out_dir=args.out_dir,
        doc_root=settings.doc_root,
        export_type=args.export_type,
        ann_format=args.ann_format,
        source_type=args.source_type,
        ann_user_id=args.ann_user_id,
        ann_min_lead=args.ann_min_lead,
        accept_lead_time_none=not args.reject_lead_time_none,
        pred_model_ver=args.pred_model_ver,
        only_finished=not args.include_unfinished,
        mmyolo_file_name=args.mmyolo_file_name,
        include_empty_images=args.include_empty_images,
    )


if __name__ == "__main__":
    main()

