from __future__ import annotations

import json
from pathlib import Path


def collect_classes_mmyolo(json_path: str | Path) -> list[str]:
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = data.get("categories") or []
    if not categories:
        raise RuntimeError(f"No categories found in {path}")

    classes = []
    for category in sorted(categories, key=lambda item: item.get("id", 0)):
        name = category.get("name")
        if name:
            classes.append(str(name))
    if not classes:
        raise RuntimeError(f"No category names found in {path}")
    return classes

