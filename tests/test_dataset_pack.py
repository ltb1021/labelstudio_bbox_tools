import json
from pathlib import Path

from PIL import Image

from labelstudio_bbox_tools.dataset_pack.coco_pack import infer_split_from_filename, pack_coco_dataset


def _write_image(path: Path, size=(100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(20, 30, 40)).save(path)


def _write_coco(path: Path, images, annotations) -> None:
    path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [
                    {"id": 1, "name": "person", "supercategory": ""},
                    {"id": 3, "name": "vehicle", "supercategory": ""},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_infer_split_from_filename():
    assert infer_split_from_filename("sample_train_annotations.json") == "train"
    assert infer_split_from_filename("sample-val.json") == "val"
    assert infer_split_from_filename("sample.json") is None


def test_pack_coco_dataset_writes_yolo_labels_and_manifest(tmp_path):
    img_a = tmp_path / "src_a" / "same.jpg"
    img_b = tmp_path / "src_b" / "same.jpg"
    _write_image(img_a)
    _write_image(img_b)
    coco = tmp_path / "dataset_train.json"
    _write_coco(
        coco,
        images=[
            {"id": "a", "file_name": str(img_a), "width": 100, "height": 80},
            {"id": "b", "file_name": str(img_b), "width": 100, "height": 80},
        ],
        annotations=[
            {"id": "ann-a", "image_id": "a", "category_id": 1, "bbox": [10, 20, 30, 40], "iscrowd": 0},
            {"id": "ann-b", "image_id": "b", "category_id": 3, "bbox": [0, 0, 50, 20], "iscrowd": 0},
        ],
    )

    out_dir = tmp_path / "packed"
    result = pack_coco_dataset(coco_inputs=[coco], out_dir=out_dir, dry_run=False, show_progress=False)

    image_files = sorted((out_dir / "images" / "train").glob("*.jpg"))
    label_files = sorted((out_dir / "labels" / "train").glob("*.txt"))
    assert len(image_files) == 2
    assert len({path.name for path in image_files}) == 2
    assert len(label_files) == 2
    assert (out_dir / "data.yaml").is_file()
    assert (out_dir / "annotations" / "train_coco.json").is_file()
    assert (out_dir / "export_manifest.csv").is_file()
    assert result.split_summaries["train"].annotations_written == 2
    assert any(line.startswith("0 ") for line in label_files[0].read_text(encoding="utf-8").splitlines()) or any(
        line.startswith("0 ") for line in label_files[1].read_text(encoding="utf-8").splitlines()
    )
    assert any(line.startswith("1 ") for line in label_files[0].read_text(encoding="utf-8").splitlines()) or any(
        line.startswith("1 ") for line in label_files[1].read_text(encoding="utf-8").splitlines()
    )


def test_pack_coco_dataset_dry_run_does_not_write_outputs(tmp_path):
    img = tmp_path / "src" / "img.jpg"
    _write_image(img)
    coco = tmp_path / "dataset_val.json"
    _write_coco(
        coco,
        images=[{"id": 1, "file_name": str(img), "width": 100, "height": 80}],
        annotations=[],
    )
    out_dir = tmp_path / "packed"
    result = pack_coco_dataset(coco_inputs={"val": coco}, out_dir=out_dir, dry_run=True, show_progress=False)
    assert result.dry_run
    assert result.split_summaries["val"].empty_images == 1
    assert not out_dir.exists()
