from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class LabelNames:
    from_name: str
    to_name: str
    labels: list[str]


def safe_json(response: Any) -> Any:
    if hasattr(response, "json"):
        try:
            return response.json()
        except Exception:
            pass
    if hasattr(response, "content"):
        try:
            return json.loads(response.content.decode("utf-8", "ignore"))
        except Exception:
            pass
    if isinstance(response, (bytes, bytearray)):
        try:
            return json.loads(response.decode("utf-8", "ignore"))
        except Exception:
            pass
    return response


def response_items(data: Any) -> list[dict]:
    data = safe_json(data)
    if isinstance(data, dict):
        items = data.get("results", [])
    else:
        items = data
    return list(items or [])


def iter_project_tasks(
    client: Any,
    project_id: int,
    *,
    page_size: int = 1000,
    max_tasks: int | None = None,
    fields: str | None = None,
) -> Iterable[dict]:
    yielded = 0
    page = 1
    while True:
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if fields:
            params["fields"] = fields
        try:
            response = client.make_request(
                "GET",
                f"/api/projects/{project_id}/tasks/",
                params=params,
            )
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                break
            raise

        data = safe_json(response)
        items = response_items(data)
        if not items:
            break
        for task in items:
            yield task
            yielded += 1
            if max_tasks is not None and yielded >= max_tasks:
                return
        page += 1
        if isinstance(data, dict) and not data.get("next"):
            break
        if not isinstance(data, dict):
            break


def iter_project_task_ids(
    client: Any,
    project_id: int,
    *,
    page_size: int = 1000,
    max_tasks: int | None = None,
) -> Iterable[int]:
    for task in iter_project_tasks(
        client,
        project_id,
        page_size=page_size,
        max_tasks=max_tasks,
        fields="id",
    ):
        yield int(task["id"])


def get_label_names(label_config_xml: str, *, shape: str = "bbox") -> LabelNames:
    root = ET.fromstring(label_config_xml or "<View />")
    candidates = []
    if shape == "polygon":
        candidates.append(".//PolygonLabels")
    else:
        candidates.append(".//RectangleLabels")
    candidates.extend([".//RectangleLabels", ".//PolygonLabels"])

    tag = None
    for query in candidates:
        tag = root.find(query)
        if tag is not None:
            break
    if tag is None:
        return LabelNames(from_name="tag", to_name="image", labels=[])

    return LabelNames(
        from_name=tag.attrib.get("name", "tag"),
        to_name=tag.attrib.get("toName", "image"),
        labels=[label.attrib["value"] for label in tag.findall("Label") if "value" in label.attrib],
    )


def completed_by_id(annotation: dict) -> int | None:
    completed_by = annotation.get("completed_by")
    if isinstance(completed_by, dict):
        value = completed_by.get("id")
        return int(value) if value is not None else None
    if completed_by is None:
        return None
    return int(completed_by)


def pick_latest(items: list[dict]) -> dict | None:
    if not items:
        return None

    def key(item: dict) -> str:
        return str(item.get("updated_at") or item.get("created_at") or "")

    return sorted(items, key=key, reverse=True)[0]
