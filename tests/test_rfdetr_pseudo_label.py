from types import SimpleNamespace

from labelstudio_bbox_tools.pseudo_label.rfdetr import _detections_to_candidates, _predict_threshold


def test_detections_to_candidates_reads_supervision_like_fields():
    detections = SimpleNamespace(
        xyxy=[[0, 1, 10, 11], [20, 21, 30, 31]],
        class_id=[2, 3],
        confidence=[0.8, 0.7],
    )
    boxes, class_ids, scores = _detections_to_candidates(detections)
    assert boxes == [[0.0, 1.0, 10.0, 11.0], [20.0, 21.0, 30.0, 31.0]]
    assert class_ids == [2, 3]
    assert scores == [0.8, 0.7]


def test_predict_threshold_uses_lowest_class_threshold_for_prefilter():
    assert _predict_threshold(0.3, {"default": 0.25, "hard_class": 0.1}) == 0.1
    assert _predict_threshold(0.3, None) == 0.3
