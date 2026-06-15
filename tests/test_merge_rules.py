from labelstudio_bbox_tools.merge.ann_pred import _build_group_index, _pair_iou_threshold


def test_grouped_iou_threshold_applies_to_group_members():
    groups = [["worker", "signalman"]]
    group_index = _build_group_index(groups)
    assert _pair_iou_threshold("worker", "signalman", 0.5, None, group_index, 0.4) == 0.4
    assert _pair_iou_threshold("worker", "helmet", 0.5, None, group_index, 0.4) is None
