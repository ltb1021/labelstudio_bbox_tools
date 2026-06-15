from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass
from typing import Any

from labelstudio_bbox_tools.config import settings_from_env
from labelstudio_bbox_tools.ls_api import iter_project_task_ids, safe_json
from labelstudio_bbox_tools.ls_client import make_client


@dataclass(frozen=True)
class ProjectDeleteSummary:
    project_id: int
    title: str | None
    initial_task_count: int | None
    deleted_tasks: int
    deleted_project: bool
    dry_run: bool

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def inspect_project(ls_url: str, api_key: str, project_id: int) -> dict[str, Any]:
    client = make_client(ls_url, api_key)
    project = client.get_project(project_id)
    params = getattr(project, "params", {}) or {}
    return {
        "id": getattr(project, "id", project_id),
        "title": params.get("title"),
        "created_at": params.get("created_at"),
        "task_count": params.get("task_count"),
    }


def _bulk_delete_tasks(client: Any, project_id: int, task_ids: list[int]) -> None:
    try:
        client.make_request("POST", f"/api/projects/{project_id}/tasks/delete", json={"ids": task_ids}, timeout=600)
        return
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status not in (404, 405):
            raise
    try:
        client.make_request("POST", "/api/tasks/bulk_delete", json={"ids": task_ids}, timeout=600)
        return
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status not in (404, 405):
            raise
    for task_id in task_ids:
        client.make_request("DELETE", f"/api/tasks/{task_id}/", timeout=120)


def delete_project_safely(
    *,
    ls_url: str,
    api_key: str,
    project_id: int,
    page_size: int = 1000,
    batch_size: int = 1000,
    sleep_sec: float = 0.15,
    delete_project_meta: bool = False,
    dry_run: bool = True,
    confirm: str = "",
    max_batches: int | None = None,
) -> ProjectDeleteSummary:
    if page_size < 1 or batch_size < 1:
        raise ValueError("page_size and batch_size must be >= 1")

    client = make_client(ls_url, api_key)
    project = client.get_project(project_id)
    params = getattr(project, "params", {}) or {}
    title = params.get("title")
    initial_task_count = params.get("task_count")

    if dry_run:
        preview_ids = list(itertools.islice(iter_project_task_ids(client, project_id, page_size=page_size), min(batch_size, 10)))
        print(f"[dry-run] project_id={project_id}, title={title!r}, task_count={initial_task_count}, sample_ids={preview_ids}")
        return ProjectDeleteSummary(project_id, title, initial_task_count, 0, False, True)

    if confirm.lower() != "delete":
        raise RuntimeError("Set confirm='delete' to actually delete tasks/project")

    deleted = 0
    batches = 0
    while True:
        task_ids = list(itertools.islice(iter_project_task_ids(client, project_id, page_size=page_size), batch_size))
        if not task_ids:
            break
        _bulk_delete_tasks(client, project_id, task_ids)
        deleted += len(task_ids)
        batches += 1
        print(f"[delete] cumulative_deleted={deleted:,}")
        if max_batches is not None and batches >= max_batches:
            break
        time.sleep(sleep_sec)

    deleted_project = False
    if delete_project_meta and max_batches is None:
        response = client.make_request("DELETE", f"/api/projects/{project_id}/")
        data = safe_json(response)
        status_code = getattr(response, "status_code", None)
        deleted_project = status_code in (200, 202, 204) or data in ({}, None)
        print(f"[delete] project_meta_deleted={deleted_project}")

    return ProjectDeleteSummary(project_id, title, initial_task_count, deleted, deleted_project, False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely preview or delete a large Label Studio project.")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--sleep-sec", type=float, default=0.15)
    parser.add_argument("--delete-project-meta", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--confirm", default="")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    settings = settings_from_env()
    summary = delete_project_safely(
        ls_url=settings.url,
        api_key=settings.api_key,
        project_id=args.project_id,
        page_size=args.page_size,
        batch_size=args.batch_size,
        sleep_sec=args.sleep_sec,
        delete_project_meta=args.delete_project_meta,
        dry_run=not args.run,
        confirm=args.confirm,
        max_batches=args.max_batches,
    )
    print(summary.as_dict())


if __name__ == "__main__":
    main()
