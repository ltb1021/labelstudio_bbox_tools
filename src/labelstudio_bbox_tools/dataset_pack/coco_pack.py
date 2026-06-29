"""Package COCO bbox datasets into portable images/labels folders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image
from tqdm import tqdm

VALID_SPLITS = {"train", "val", "test"}
IMAGE_PATH_FIELDS = ("file_name", "path", "local_path", "coco_url")


@dataclass(frozen=True)
class CocoInput:
    path: Path
    split: str

    def as_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "split": self.split}


@dataclass
class SplitSummary:
    split: str
    json_paths: list[str] = field(default_factory=list)
    images_total: int = 0
    images_included: int = 0
    images_copied: int = 0
    images_missing: int = 0
    labels_written: int = 0
    annotations_total: int = 0
    annotations_written: int = 0
    annotations_skipped_invalid: int = 0
    annotations_clipped: int = 0
    empty_images: int = 0
    filename_collisions: int = 0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class CocoPackResult:
    out_dir: Path
    inputs: list[CocoInput]
    dry_run: bool = True
    label_format: str = "yolo"
    copy_mode: str = "copy"
    save_rewritten_coco: bool = True
    data_yaml_path: Path | None = None
    manifest_path: Path | None = None
    summary_path: Path | None = None
    class_names: list[str] = field(default_factory=list)
    split_summaries: dict[str, SplitSummary] = field(default_factory=dict)
    missing_images: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def images_total(self) -> int:
        return sum(item.images_total for item in self.split_summaries.values())

    @property
    def images_included(self) -> int:
        return sum(item.images_included for item in self.split_summaries.values())

    @property
    def annotations_written(self) -> int:
        return sum(item.annotations_written for item in self.split_summaries.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "inputs": [item.as_dict() for item in self.inputs],
            "dry_run": self.dry_run,
            "label_format": self.label_format,
            "copy_mode": self.copy_mode,
            "save_rewritten_coco": self.save_rewritten_coco,
            "data_yaml_path": str(self.data_yaml_path) if self.data_yaml_path else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "summary_path": str(self.summary_path) if self.summary_path else None,
            "class_count": len(self.class_names),
            "class_names": self.class_names,
            "images_total": self.images_total,
            "images_included": self.images_included,
            "annotations_written": self.annotations_written,
            "missing_image_count": len(self.missing_images),
            "missing_images": self.missing_images[:100],
            "warnings": self.warnings,
            "splits": {split: summary.as_dict() for split, summary in self.split_summaries.items()},
        }


def safe_name(value: str, *, max_length: int = 96) -> str:
    value = re.sub(r'[\\/:*?"<>|\s#]+', "_", str(value).strip())
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = "item"
    if len(value) > max_length:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        value = f"{value[: max_length - 9]}_{digest}"
    return value


def short_hash(value: str, length: int = 10) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:length]


def infer_split_from_filename(path: str | Path) -> str | None:
    stem = Path(path).stem.lower()
    tokens = re.split(r"[^a-z0-9]+", stem)
    for split in ("train", "val", "test"):
        if split in tokens:
            return split
    for split in ("train", "val", "test"):
        if f"_{split}_" in f"_{stem}_":
            return split
    return None


def _normalise_inputs(
    coco_inputs: Mapping[str, str | Path | Sequence[str | Path]] | Sequence[str | Path],
    *,
    split_map: Mapping[str, str] | None = None,
    split_mode: str = "infer_from_filename",
    default_split: str | None = None,
) -> list[CocoInput]:
    inputs: list[CocoInput] = []
    if isinstance(coco_inputs, Mapping):
        for split, paths in coco_inputs.items():
            if split not in VALID_SPLITS:
                raise ValueError(f"Invalid split: {split}. Use one of {sorted(VALID_SPLITS)}")
            if isinstance(paths, (str, Path)):
                path_list: Sequence[str | Path] = [paths]
            else:
                path_list = list(paths)
            for path in path_list:
                inputs.append(CocoInput(path=Path(path).expanduser().resolve(), split=split))
    else:
        for path_value in coco_inputs:
            path = Path(path_value).expanduser().resolve()
            split = None
            if split_map:
                split = split_map.get(str(path)) or split_map.get(path.name) or split_map.get(path.stem)
            if split is None and split_mode == "infer_from_filename":
                split = infer_split_from_filename(path)
            if split is None:
                split = default_split
            if split not in VALID_SPLITS:
                raise ValueError(
                    f"Could not determine split for {path}. Use a mapping like {{'train': path}} or provide split_map."
                )
            inputs.append(CocoInput(path=path, split=split))
    if not inputs:
        raise ValueError("At least one COCO JSON path is required.")
    for item in inputs:
        if not item.path.is_file():
            raise FileNotFoundError(f"COCO JSON does not exist: {item.path}")
    return inputs


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric_sort_key(value: Any) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except Exception:
        return (1, str(value))


def _category_signature(categories: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    ordered = sorted(categories, key=lambda item: _numeric_sort_key(item.get("id")))
    return [(str(item.get("id")), str(item.get("name"))) for item in ordered]


def _load_category_order(inputs: Sequence[CocoInput], *, strict_class_names: bool) -> tuple[list[dict[str, Any]], list[str]]:
    reference: list[tuple[str, str]] | None = None
    reference_categories: list[dict[str, Any]] | None = None
    for item in inputs:
        data = _load_json(item.path)
        categories = data.get("categories") or []
        if not categories:
            raise ValueError(f"COCO JSON has no categories: {item.path}")
        signature = _category_signature(categories)
        if reference is None:
            reference = signature
            reference_categories = [dict(cat) for cat in sorted(categories, key=lambda x: _numeric_sort_key(x.get("id")))]
        elif strict_class_names and signature != reference:
            raise ValueError(
                f"Category id/name order differs in {item.path}. Disable strict_class_names only if you intentionally merge schemas."
            )
    assert reference_categories is not None
    return reference_categories, [name for _cat_id, name in reference or []]


def _annotation_map(annotations: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for ann in annotations:
        by_image[str(ann.get("image_id"))].append(ann)
    return by_image


def _resolve_image_path(image: Mapping[str, Any], json_path: Path, fields: Sequence[str]) -> Path | None:
    for field in fields:
        value = image.get(field)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path
        return (json_path.parent / path).resolve()
    return None


def _image_size(image: Mapping[str, Any], image_path: Path | None) -> tuple[int, int]:
    width = image.get("width")
    height = image.get("height")
    try:
        width_i = int(float(width))
        height_i = int(float(height))
        if width_i > 0 and height_i > 0:
            return width_i, height_i
    except Exception:
        pass
    if image_path and image_path.is_file():
        with Image.open(image_path) as img:
            return int(img.width), int(img.height)
    raise ValueError(f"Image width/height is missing and file cannot be opened: {image_path}")


def _source_tag(image_path: Path, parts: int = 3) -> str:
    parent_parts = [part for part in image_path.parent.parts if part not in {"", os.sep}]
    selected = parent_parts[-parts:] if parent_parts else ["image"]
    return safe_name("__".join(selected), max_length=72)


def _dest_image_name(image_path: Path, *, used_names: set[str], collision_counter: Counter[str]) -> str:
    suffix = image_path.suffix.lower() or ".jpg"
    stem = safe_name(image_path.stem, max_length=72)
    base = f"{_source_tag(image_path)}__{short_hash(str(image_path.resolve() if image_path.exists() else image_path))}__{stem}{suffix}"
    candidate = base
    idx = 1
    while candidate in used_names:
        collision_counter["filename_collisions"] += 1
        candidate = f"{Path(base).stem}__dup{idx:03d}{suffix}"
        idx += 1
    used_names.add(candidate)
    return candidate


def _coco_bbox_to_yolo(
    bbox: Sequence[Any],
    *,
    width: int,
    height: int,
    clip_bbox: bool,
) -> tuple[float, float, float, float, bool] | None:
    if len(bbox) < 4:
        return None
    x, y, w, h = [float(v) for v in bbox[:4]]
    if w <= 0 or h <= 0 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = x, y, x + w, y + h
    clipped = False
    if clip_bbox:
        nx1 = max(0.0, min(float(width), x1))
        ny1 = max(0.0, min(float(height), y1))
        nx2 = max(0.0, min(float(width), x2))
        ny2 = max(0.0, min(float(height), y2))
        clipped = (nx1, ny1, nx2, ny2) != (x1, y1, x2, y2)
        x1, y1, x2, y2 = nx1, ny1, nx2, ny2
    if x2 <= x1 or y2 <= y1:
        return None
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh, clipped


def _write_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy_image(src: Path, dst: Path, *, copy_mode: str, overwrite: bool) -> bool:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Output image already exists: {dst}")
        if dst.is_file() or dst.is_symlink():
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(src, dst)
    elif copy_mode == "symlink":
        dst.symlink_to(src)
    elif copy_mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError("copy_mode must be one of: copy, symlink, hardlink")
    return True


def _write_data_yaml(path: Path, *, class_names: Sequence[str], splits: Sequence[str], overwrite: bool) -> None:
    lines = ["# Generated by labelstudio_bbox_tools.dataset_pack.coco_pack", f"path: {path.parent.as_posix()}"]
    for split in splits:
        lines.append(f"{split}: images/{split}")
    lines.append("names:")
    for idx, name in enumerate(class_names):
        escaped = str(name).replace('"', '\\"')
        lines.append(f"  {idx}: \"{escaped}\"")
    _write_text(path, "\n".join(lines) + "\n", overwrite=overwrite)


def pack_coco_dataset(
    *,
    coco_inputs: Mapping[str, str | Path | Sequence[str | Path]] | Sequence[str | Path],
    out_dir: str | Path,
    split_map: Mapping[str, str] | None = None,
    split_mode: str = "infer_from_filename",
    default_split: str | None = None,
    label_format: str = "yolo",
    save_rewritten_coco: bool = True,
    copy_mode: str = "copy",
    include_empty_images: bool = True,
    collision_policy: str = "source_tag_hash",
    fail_on_missing: bool = False,
    overwrite: bool = False,
    dry_run: bool = True,
    strict_class_names: bool = True,
    clip_bbox: bool = True,
    image_path_fields: Sequence[str] = IMAGE_PATH_FIELDS,
    show_progress: bool = True,
) -> CocoPackResult:
    if label_format not in {"yolo", "none"}:
        raise ValueError("label_format must be 'yolo' or 'none'.")
    if collision_policy != "source_tag_hash":
        raise ValueError("Only collision_policy='source_tag_hash' is currently supported.")
    inputs = _normalise_inputs(coco_inputs, split_map=split_map, split_mode=split_mode, default_split=default_split)
    out_root = Path(out_dir).expanduser().resolve()
    categories, class_names = _load_category_order(inputs, strict_class_names=strict_class_names)
    cat_id_to_index = {str(cat["id"]): idx for idx, cat in enumerate(categories)}

    result = CocoPackResult(
        out_dir=out_root,
        inputs=inputs,
        dry_run=dry_run,
        label_format=label_format,
        copy_mode=copy_mode,
        save_rewritten_coco=save_rewritten_coco,
        class_names=class_names,
    )
    for split in sorted({item.split for item in inputs}):
        result.split_summaries[split] = SplitSummary(split=split)

    manifest_rows: list[dict[str, Any]] = []
    rewritten_by_split: dict[str, dict[str, Any]] = {}
    used_names_by_split: dict[str, set[str]] = defaultdict(set)

    for item in inputs:
        data = _load_json(item.path)
        split_summary = result.split_summaries[item.split]
        split_summary.json_paths.append(str(item.path))
        images = data.get("images") or []
        annotations = data.get("annotations") or []
        anns_by_image = _annotation_map(annotations)
        split_summary.images_total += len(images)
        split_summary.annotations_total += len(annotations)
        split_coco = rewritten_by_split.setdefault(
            item.split,
            {
                "info": data.get("info", {}),
                "licenses": data.get("licenses", []),
                "images": [],
                "annotations": [],
                "categories": categories,
            },
        )
        image_id_map: dict[tuple[str, str], int] = {}

        iterator = tqdm(images, desc=f"pack {item.split} {item.path.name}", disable=not show_progress)
        for image in iterator:
            original_image_id = str(image.get("id"))
            image_anns = list(anns_by_image.get(original_image_id, []))
            if not include_empty_images and not image_anns:
                continue
            image_path = _resolve_image_path(image, item.path, image_path_fields)
            missing = image_path is None or not image_path.is_file()
            if missing:
                split_summary.images_missing += 1
                result.missing_images.append(str(image_path or image.get("file_name") or original_image_id))
                if fail_on_missing:
                    raise FileNotFoundError(f"Missing image for COCO image id={original_image_id}: {image_path}")
                continue
            try:
                width, height = _image_size(image, image_path)
            except Exception as exc:
                split_summary.images_missing += 1
                warning = f"Could not determine image size for image id={original_image_id}: {exc}"
                result.warnings.append(warning)
                if fail_on_missing:
                    raise
                continue
            dest_name = _dest_image_name(
                image_path,
                used_names=used_names_by_split[item.split],
                collision_counter=CounterProxy(split_summary),
            )
            dest_image_rel = Path("images") / item.split / dest_name
            dest_label_rel = Path("labels") / item.split / f"{Path(dest_name).stem}.txt"
            dest_image_path = out_root / dest_image_rel
            dest_label_path = out_root / dest_label_rel

            yolo_lines: list[str] = []
            valid_ann_count = 0
            new_image_id = len(split_coco["images"]) + 1
            image_id_map[(str(item.path), original_image_id)] = new_image_id
            for ann in image_anns:
                cat_key = str(ann.get("category_id"))
                if cat_key not in cat_id_to_index:
                    split_summary.annotations_skipped_invalid += 1
                    continue
                bbox = ann.get("bbox") or []
                converted = _coco_bbox_to_yolo(bbox, width=width, height=height, clip_bbox=clip_bbox)
                if converted is None:
                    split_summary.annotations_skipped_invalid += 1
                    continue
                cx, cy, bw, bh, clipped = converted
                if clipped:
                    split_summary.annotations_clipped += 1
                class_index = cat_id_to_index[cat_key]
                yolo_lines.append(f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                valid_ann_count += 1
                if save_rewritten_coco:
                    new_ann = dict(ann)
                    new_ann["id"] = len(split_coco["annotations"]) + 1
                    new_ann["image_id"] = new_image_id
                    split_coco["annotations"].append(new_ann)

            if not include_empty_images and valid_ann_count == 0:
                continue

            split_summary.images_included += 1
            if valid_ann_count == 0:
                split_summary.empty_images += 1
            split_summary.annotations_written += valid_ann_count

            if save_rewritten_coco:
                new_image = dict(image)
                new_image["id"] = new_image_id
                new_image["file_name"] = dest_image_rel.as_posix()
                new_image["width"] = width
                new_image["height"] = height
                split_coco["images"].append(new_image)

            if not dry_run:
                if not missing:
                    _copy_image(image_path, dest_image_path, copy_mode=copy_mode, overwrite=overwrite)
                    split_summary.images_copied += 1
                if label_format == "yolo":
                    _write_text(dest_label_path, "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), overwrite=overwrite)
                    split_summary.labels_written += 1

            manifest_rows.append(
                {
                    "split": item.split,
                    "source_json": str(item.path),
                    "original_image_id": original_image_id,
                    "original_image_path": str(image_path),
                    "copied_image_path": str(dest_image_path),
                    "label_path": str(dest_label_path) if label_format == "yolo" else "",
                    "new_file_name": dest_name,
                    "width": width,
                    "height": height,
                    "annotation_count": valid_ann_count,
                    "missing_image": missing,
                }
            )

    if not dry_run:
        if label_format == "yolo":
            data_yaml_path = out_root / "data.yaml"
            _write_data_yaml(
                data_yaml_path,
                class_names=class_names,
                splits=sorted(result.split_summaries),
                overwrite=overwrite,
            )
            result.data_yaml_path = data_yaml_path
        if save_rewritten_coco:
            for split, coco in rewritten_by_split.items():
                path = out_root / "annotations" / f"{split}_coco.json"
                _write_text(path, json.dumps(coco, ensure_ascii=False, indent=2) + "\n", overwrite=overwrite)
        manifest_path = out_root / "export_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {manifest_path}")
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "split",
                "source_json",
                "original_image_id",
                "original_image_path",
                "copied_image_path",
                "label_path",
                "new_file_name",
                "width",
                "height",
                "annotation_count",
                "missing_image",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        result.manifest_path = manifest_path
        summary_path = out_root / "export_summary.json"
        result.summary_path = summary_path
        _write_text(summary_path, json.dumps(result.as_dict(), ensure_ascii=False, indent=2) + "\n", overwrite=True)

    print(
        f"[{'dry-run' if dry_run else 'ok'}] inputs={len(inputs)}, images={result.images_included:,}/{result.images_total:,}, "
        f"annotations={result.annotations_written:,}, missing_images={len(result.missing_images):,}, out_dir={out_root}"
    )
    for split, summary in sorted(result.split_summaries.items()):
        print(
            f"[summary] {split}: images={summary.images_included:,}, anns={summary.annotations_written:,}, "
            f"empty={summary.empty_images:,}, missing={summary.images_missing:,}, invalid_anns={summary.annotations_skipped_invalid:,}"
        )
    return result


class CounterProxy(Counter[str]):
    def __init__(self, summary: SplitSummary):
        super().__init__()
        self.summary = summary

    def __setitem__(self, key: str, value: int) -> None:
        if key == "filename_collisions":
            self.summary.filename_collisions += max(0, int(value) - int(self.get(key, 0)))
        super().__setitem__(key, value)


def _parse_input_args(values: Sequence[str]) -> dict[str, Path] | list[Path]:
    mapped: dict[str, Path] = {}
    plain: list[Path] = []
    for value in values:
        if "=" in value:
            split, path = value.split("=", 1)
            split = split.strip()
            if split not in VALID_SPLITS:
                raise ValueError(f"Invalid split in input {value!r}")
            mapped[split] = Path(path).expanduser()
        else:
            plain.append(Path(value).expanduser())
    if mapped and plain:
        raise ValueError("Do not mix split=path inputs with plain paths.")
    return mapped or plain


def main() -> None:
    parser = argparse.ArgumentParser(description="Package COCO bbox JSON images into portable images/labels folders.")
    parser.add_argument("--input", action="append", required=True, help="COCO JSON path or split=path. Repeatable.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label-format", default="yolo", choices=["yolo", "none"])
    parser.add_argument("--no-coco", action="store_true", help="Do not write rewritten annotations/<split>_coco.json")
    parser.add_argument("--copy-mode", default="copy", choices=["copy", "symlink", "hardlink"])
    parser.add_argument("--exclude-empty-images", action="store_true")
    parser.add_argument("--fail-on-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run", action="store_true", help="Actually copy/write files. Omit for dry-run preview.")
    args = parser.parse_args()

    result = pack_coco_dataset(
        coco_inputs=_parse_input_args(args.input),
        out_dir=args.out_dir,
        label_format=args.label_format,
        save_rewritten_coco=not args.no_coco,
        copy_mode=args.copy_mode,
        include_empty_images=not args.exclude_empty_images,
        fail_on_missing=args.fail_on_missing,
        overwrite=args.overwrite,
        dry_run=not args.run,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
