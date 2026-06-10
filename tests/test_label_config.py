from labelstudio_bbox_tools.ui.label_config import make_label_config_xml


def test_make_bbox_label_config():
    xml = make_label_config_xml(["person", "helmet"], shape="bbox")
    assert "<RectangleLabels" in xml
    assert '<Label value="person"/>' in xml
    assert '<Label value="helmet"/>' in xml


def test_make_polygon_label_config():
    xml = make_label_config_xml(["worker"], shape="polygon")
    assert "<PolygonLabels" in xml
    assert '<Label value="worker"/>' in xml

