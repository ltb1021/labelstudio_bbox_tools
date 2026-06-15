from labelstudio_bbox_tools.pseudo_label.yolo import classwise_nms_indices, load_yolo_class_names


def test_classwise_nms_uses_per_class_threshold():
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]]
    class_ids = [0, 0, 0]
    scores = [0.9, 0.8, 0.7]
    keep = classwise_nms_indices(boxes, class_ids, scores, ["worker"], {"worker": 0.5})
    assert keep == [0, 2]


def test_load_yolo_class_names_manual_fallback():
    assert load_yolo_class_names(manual_classes=["a", "b"]) == ["a", "b"]
