import json

from labelstudio_bbox_tools.ui.class_sources import collect_classes_mmyolo


def test_collect_classes_mmyolo_sorts_by_id(tmp_path):
    path = tmp_path / "annotations_mmyolo.json"
    path.write_text(
        json.dumps(
            {
                "categories": [
                    {"id": 2, "name": "helmet"},
                    {"id": 1, "name": "person"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert collect_classes_mmyolo(path) == ["person", "helmet"]

