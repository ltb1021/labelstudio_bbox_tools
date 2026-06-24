from labelstudio_bbox_tools.video_inference.classes import load_class_names, make_class_color_map
from labelstudio_bbox_tools.video_inference.common import Detection
from labelstudio_bbox_tools.video_inference.draw import draw_detections_on_bgr, resolve_font_path


def test_load_class_names_from_yaml_dict_keeps_numeric_order(tmp_path):
    path = tmp_path / "data.yaml"
    yaml_text = "\n".join(["names:", "  1: helmet", "  0: person", ""])
    path.write_text(yaml_text, encoding="utf-8")
    assert load_class_names(class_yaml=path, expected_count=2) == ["person", "helmet"]


def test_make_class_color_map_is_stable_and_rgb():
    classes = [f"class_{idx}" for idx in range(28)]
    color_map = make_class_color_map(classes)
    assert list(color_map) == classes
    assert len(set(color_map.values())) == 28
    assert all(len(rgb) == 3 and all(0 <= channel <= 255 for channel in rgb) for rgb in color_map.values())


def test_draw_detections_with_korean_label_keeps_frame_shape():
    import numpy as np

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    detections = [Detection(xyxy=(5, 5, 60, 50), class_id=0, class_name="작업자", score=0.87)]
    color_map = {"작업자": (240, 40, 40)}
    drawn = draw_detections_on_bgr(frame, detections, color_map=color_map, font_size=14, line_width=2)
    assert drawn.shape == frame.shape
    assert int(drawn.sum()) > 0


def test_resolve_font_path_finds_optional_font_or_none():
    font = resolve_font_path()
    assert font is None or font.is_file()
