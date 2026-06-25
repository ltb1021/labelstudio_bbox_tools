import numpy as np

from labelstudio_bbox_tools.pose_inference.common import PoseInstance, keypoints_from_xy_conf
from labelstudio_bbox_tools.pose_inference.draw import draw_pose_instances_on_bgr
from labelstudio_bbox_tools.pose_inference.rfdetr import _filter_pose_instances, _keypoints_to_pose_instances
from labelstudio_bbox_tools.video_inference.classes import make_class_color_map
from labelstudio_bbox_tools.video_inference.common import Detection
from labelstudio_bbox_tools.pose_inference.fallback_crop import make_crop_regions, offset_pose_from_crop, split_pose_detection_cases


def test_keypoints_from_xy_conf_uses_coco_names():
    keypoints = keypoints_from_xy_conf([[10, 20], [30, 40]], [0.9, 0.1])
    assert [item.name for item in keypoints] == ["nose", "left_eye"]
    assert keypoints[0].is_drawable(0.2)
    assert not keypoints[1].is_drawable(0.2)


def test_draw_pose_instances_keeps_frame_shape():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    keypoints = keypoints_from_xy_conf(
        [[40, 20], [35, 18], [45, 18], [32, 20], [48, 20], [30, 45], [60, 45], [25, 70], [65, 70], [20, 92], [70, 92], [35, 85], [55, 85], [35, 105], [55, 105], [35, 115], [55, 115]],
        [0.9] * 17,
    )
    instance = PoseInstance(xyxy=(20, 10, 80, 118), class_id=0, class_name="person", score=0.88, keypoints=keypoints)
    drawn = draw_pose_instances_on_bgr(
        frame,
        [instance],
        color_map=make_class_color_map(["person"]),
        font_size=14,
        line_width=2,
        keypoint_radius=3,
    )
    assert drawn.shape == frame.shape
    assert int(drawn.sum()) > 0


class _FakeKeyPoints:
    xy = np.array([[[10, 20], [30, 40]]], dtype=np.float32)
    keypoint_confidence = np.array([[0.8, 0.7]], dtype=np.float32)
    detection_confidence = np.array([1.23], dtype=np.float32)
    class_id = np.array([1], dtype=np.int64)
    data = {
        "xyxy": np.array([[5, 10, 50, 60]], dtype=np.float32),
        "class_name": np.array(["person"], dtype=object),
    }


def test_rfdetr_keypoints_to_pose_instances_uses_data_xyxy_and_class_name():
    instances = _keypoints_to_pose_instances(_FakeKeyPoints(), ["person"])
    assert len(instances) == 1
    assert instances[0].xyxy == (5.0, 10.0, 50.0, 60.0)
    assert instances[0].class_id == 0
    assert instances[0].class_name == "person"
    assert instances[0].score == 1.0
    assert abs(instances[0].raw_score - 1.23) < 1e-5
    assert [point.name for point in instances[0].keypoints] == ["nose", "left_eye"]


def test_rfdetr_pose_filter_applies_nms_and_visible_keypoint_filter():
    high = PoseInstance(
        xyxy=(0, 0, 100, 100),
        class_id=0,
        class_name="person",
        score=1.0,
        raw_score=1.4,
        keypoints=keypoints_from_xy_conf([[10, 10], [20, 20], [30, 30]], [0.9, 0.8, 0.7]),
    )
    duplicate = PoseInstance(
        xyxy=(2, 2, 102, 102),
        class_id=0,
        class_name="person",
        score=0.95,
        raw_score=0.95,
        keypoints=keypoints_from_xy_conf([[11, 11], [21, 21], [31, 31]], [0.9, 0.8, 0.7]),
    )
    weak = PoseInstance(
        xyxy=(200, 200, 260, 260),
        class_id=0,
        class_name="person",
        score=0.7,
        raw_score=0.7,
        keypoints=keypoints_from_xy_conf([[210, 210], [220, 220]], [0.9, 0.1]),
    )
    filtered = _filter_pose_instances(
        [duplicate, weak, high],
        ["person"],
        enable_nms=True,
        iou=0.5,
        min_visible_keypoints=2,
        keypoint_conf=0.2,
    )
    assert filtered == [high]



def test_fallback_crop_case_split_and_offset_restore():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    pose = PoseInstance(
        xyxy=(10, 10, 50, 80),
        class_id=0,
        class_name="person",
        score=0.9,
        keypoints=keypoints_from_xy_conf([[20, 20], [30, 30]], [0.9, 0.8]),
    )
    matched_det = Detection(xyxy=(12, 12, 52, 82), class_id=0, class_name="person", score=0.8)
    missed_det = Detection(xyxy=(90, 30, 130, 100), class_id=0, class_name="person", score=0.7)

    cases = split_pose_detection_cases(pose_instances=[pose], detections=[matched_det, missed_det], match_iou=0.3)
    assert cases.matched_count == 1
    assert cases.detection_only == [missed_det]
    assert cases.pose_only == []

    crops = make_crop_regions(frame, cases.detection_only, padding_ratio=0.0, min_crop_size=10)
    assert len(crops) == 1
    crop_pose = PoseInstance(
        xyxy=(0, 0, 20, 40),
        class_id=0,
        class_name="person",
        score=0.6,
        keypoints=keypoints_from_xy_conf([[5, 6]], [0.9]),
    )
    restored = offset_pose_from_crop(crop_pose, crops[0])
    assert restored.source == "fallback_crop_pose"
    assert restored.detection_class_name == "person"
    assert restored.xyxy == (90, 30, 110, 70)
    assert restored.keypoints[0].x == 95
    assert restored.keypoints[0].y == 36
