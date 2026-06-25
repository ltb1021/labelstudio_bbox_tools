import numpy as np

from labelstudio_bbox_tools.pose_inference.common import PoseInstance, keypoints_from_xy_conf
from labelstudio_bbox_tools.pose_inference.draw import draw_pose_instances_on_bgr
from labelstudio_bbox_tools.pose_inference.rfdetr import _keypoints_to_pose_instances
from labelstudio_bbox_tools.video_inference.classes import make_class_color_map


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
    detection_confidence = np.array([0.91], dtype=np.float32)
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
    assert abs(instances[0].score - 0.91) < 1e-5
    assert [point.name for point in instances[0].keypoints] == ["nose", "left_eye"]
