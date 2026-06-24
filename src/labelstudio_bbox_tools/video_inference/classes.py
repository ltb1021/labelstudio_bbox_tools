from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Sequence


def load_class_names(
    *,
    class_yaml: str | Path | None = None,
    manual_classes: Sequence[str] | None = None,
    expected_count: int | None = 28,
    strict_count: bool = True,
) -> list[str]:
    """Load class names from a YOLO-style YAML file or a manual class list.

    The class order is important because model outputs usually contain class ids, not names.
    """

    if class_yaml:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read class YAML files") from exc
        data = yaml.safe_load(Path(class_yaml).expanduser().read_text(encoding="utf-8")) or {}
        names = data.get("names")
        if isinstance(names, dict):
            classes = [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
        elif isinstance(names, list):
            classes = [str(name) for name in names]
        else:
            raise ValueError("Class YAML must contain a names dict or list")
    elif manual_classes:
        classes = [str(name) for name in manual_classes]
    else:
        raise ValueError("Provide class_yaml or manual_classes")

    if expected_count is not None and len(classes) != int(expected_count):
        message = f"Expected {expected_count} classes, but found {len(classes)}"
        if strict_count:
            raise ValueError(message)
        print(f"[warn] {message}")
    return classes


def make_class_color_map(class_names: Sequence[str]) -> dict[str, tuple[int, int, int]]:
    """Create a deterministic, visually separated RGB color for each class name."""

    count = len(class_names)
    if count == 0:
        return {}
    # Golden-ratio hue stepping keeps neighbouring class ids from getting near-identical colours.
    golden_ratio = 0.618033988749895
    color_map: dict[str, tuple[int, int, int]] = {}
    for idx, name in enumerate(class_names):
        hue = (idx * golden_ratio) % 1.0
        saturation = 0.78 if idx % 2 == 0 else 0.92
        value = 0.95 if idx % 3 != 0 else 0.78
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        color_map[str(name)] = (int(red * 255), int(green * 255), int(blue * 255))
    return color_map


def color_for_class(class_name: str, color_map: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    return color_map.get(class_name, (255, 255, 255))
