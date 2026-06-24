import json

from PIL import Image

from labelstudio_bbox_tools.exporters import project_export


LABEL_CONFIG = '<View><RectangleLabels name="label" toName="image"><Label value="person"/></RectangleLabels></View>'


def _rect(x=10, y=20, width=30, height=40):
    return {
        "type": "rectanglelabels",
        "original_width": 100,
        "original_height": 80,
        "value": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "rectanglelabels": ["person"],
        },
    }


def _annotation(user_id, results, updated_at):
    return {
        "completed_by": {"id": user_id},
        "lead_time": 1.0,
        "updated_at": updated_at,
        "result": results,
    }


def _task(task_id, image_name, annotations):
    return {
        "id": task_id,
        "data": {"image": image_name},
        "annotations": annotations,
        "is_labeled": True,
    }


class _FakeProject:
    label_config = LABEL_CONFIG


class _FakeClient:
    def get_project(self, project_id):
        return _FakeProject()


def _prepare_export(monkeypatch, tmp_path):
    doc_root = tmp_path / "doc_root"
    doc_root.mkdir()
    for name in ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]:
        Image.new("RGB", (100, 80), color="white").save(doc_root / name)

    tasks = [
        _task(101, "a.jpg", [_annotation(1, [_rect()], "2026-01-01T00:00:00Z")]),
        _task(102, "b.jpg", [_annotation(9, [_rect()], "2026-01-02T00:00:00Z")]),
        _task(103, "c.jpg", [
            _annotation(9, [_rect(x=15)], "2026-01-03T00:00:00Z"),
            _annotation(1, [_rect(x=25)], "2026-01-04T00:00:00Z"),
        ]),
        _task(104, "d.jpg", [_annotation(1, [], "2026-01-05T00:00:00Z")]),
    ]

    monkeypatch.setattr(project_export, "make_client", lambda _url, _api_key: _FakeClient())
    monkeypatch.setattr(project_export, "_fetch_tasks", lambda _project: tasks)
    monkeypatch.setattr(project_export, "resolve_local_file_url", lambda value, doc_root_path: doc_root_path / value)
    return doc_root


def test_export_mmyolo_skips_images_without_selected_user_bbox_by_default(monkeypatch, tmp_path):
    doc_root = _prepare_export(monkeypatch, tmp_path)
    out_dir = tmp_path / "out_default"

    result = project_export.export_project(
        project_id=244,
        ls_url="http://labelstudio.local",
        api_key="dummy",
        out_dir=out_dir,
        doc_root=doc_root,
        export_type="ann",
        ann_format="mmyolo",
        source_type="ann",
        ann_user_id=1,
    )

    data = json.loads((out_dir / "annotations_mmyolo.json").read_text(encoding="utf-8"))

    assert [image["id"] for image in data["images"]] == [101, 103]
    assert [annotation["image_id"] for annotation in data["annotations"]] == [101, 103]
    assert result.source_matched_count == 3
    assert result.skipped_no_source_count == 1
    assert result.skipped_empty_result_count == 1


def test_export_mmyolo_can_keep_empty_images_when_requested(monkeypatch, tmp_path):
    doc_root = _prepare_export(monkeypatch, tmp_path)
    out_dir = tmp_path / "out_include_empty"

    project_export.export_project(
        project_id=244,
        ls_url="http://labelstudio.local",
        api_key="dummy",
        out_dir=out_dir,
        doc_root=doc_root,
        export_type="ann",
        ann_format="mmyolo",
        source_type="ann",
        ann_user_id=1,
        include_empty_images=True,
    )

    data = json.loads((out_dir / "annotations_mmyolo.json").read_text(encoding="utf-8"))

    assert [image["id"] for image in data["images"]] == [101, 102, 103, 104]
    assert [annotation["image_id"] for annotation in data["annotations"]] == [101, 103]
