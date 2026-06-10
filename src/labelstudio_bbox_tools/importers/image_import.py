from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import requests

from labelstudio_bbox_tools.config import settings_from_env
from labelstudio_bbox_tools.ls_client import make_client
from labelstudio_bbox_tools.paths import collect_images, doc_relative_path, local_file_url, prefixed_project_title


@dataclass(frozen=True)
class ImageImportResult:
    project_id: int | None
    total_found: int
    total_selected: int
    task_count: int
    project_title: str | None


def _parse_slice(spec: str | None, total: int) -> slice:
    if not spec:
        return slice(0, total)
    start, stop = spec.split(":", 1)
    return slice(int(start) if start else 0, int(stop) if stop else total)


def _patch_project_title(ls_url: str, api_key: str, project_id: int, title: str) -> None:
    url = f"{ls_url.rstrip('/')}/api/projects/{project_id}"
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
    response = requests.patch(url, headers=headers, json={"title": title}, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Project title update failed: status={response.status_code}")


def build_image_tasks(
    files: list[Path],
    *,
    doc_root: str | Path,
    mirror_root: str | Path | None,
    create_links: bool,
) -> list[dict]:
    tasks = []
    for image_path in files:
        rel = doc_relative_path(
            image_path,
            doc_root=doc_root,
            mirror_root=mirror_root,
            create_link=create_links,
        )
        tasks.append({"data": {"image": local_file_url(rel)}})
    return tasks


def import_images(
    *,
    src_root: str | Path,
    ls_url: str,
    api_key: str,
    doc_root: str | Path,
    recursive: bool = False,
    slice_spec: str | None = None,
    project_id: int | None = None,
    project_title: str | None = None,
    mirror_root: str | Path | None = None,
    prefix_project_id_in_title: bool = True,
    update_existing_title: bool = False,
    dry_run: bool = False,
) -> ImageImportResult:
    files = collect_images(src_root, recursive=recursive)
    selected = files[_parse_slice(slice_spec, len(files))]
    title_base = project_title or Path(src_root).name

    tasks = build_image_tasks(
        selected,
        doc_root=doc_root,
        mirror_root=mirror_root,
        create_links=not dry_run,
    )

    if dry_run:
        print(f"[dry-run] found={len(files):,}, selected={len(selected):,}, tasks={len(tasks):,}")
        for task in tasks[:5]:
            print(f"[dry-run] {task['data']['image']}")
        return ImageImportResult(project_id, len(files), len(selected), 0, None)

    client = make_client(ls_url, api_key)
    if project_id is not None:
        project = client.get_project(project_id)
        print(f"[info] using existing project ID={project.id}")
        should_rename = update_existing_title
    else:
        label_config = "<View><Image name='image' value='$image'/></View>"
        project = client.start_project(title=title_base, label_config=label_config)
        print(f"[info] created project ID={project.id}")
        should_rename = True

    final_title = getattr(project, "title", None)
    if prefix_project_id_in_title and should_rename:
        final_title = prefixed_project_title(project.id, title_base)
        _patch_project_title(ls_url, api_key, project.id, final_title)
        print(f"[info] project title updated: {final_title}")

    response = project.import_tasks(tasks)
    task_count = response.get("task_count", len(response)) if isinstance(response, dict) else len(response)
    print(f"[ok] imported {task_count:,} tasks into project ID={project.id}")
    return ImageImportResult(project.id, len(files), len(selected), task_count, final_title)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import image files into a Label Studio project.")
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--project-id", type=int)
    parser.add_argument("--project-title")
    parser.add_argument("--mirror-root")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--slice")
    parser.add_argument("--update-existing-title", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--ls-url")
    parser.add_argument("--api-key")
    parser.add_argument("--doc-root")
    args = parser.parse_args()

    if args.dry_run:
        if args.dotenv:
            from labelstudio_bbox_tools.config import load_dotenv

            load_dotenv(args.dotenv)
        doc_root = args.doc_root or os.environ.get("LABEL_STUDIO_DOC_ROOT")
        if not doc_root:
            raise RuntimeError("--doc-root or LABEL_STUDIO_DOC_ROOT is required for dry-run")
        ls_url = args.ls_url or os.environ.get("LABEL_STUDIO_URL", "http://dry-run.local")
        api_key = args.api_key or os.environ.get("LABEL_STUDIO_API_KEY", "dry-run")
    else:
        settings = settings_from_env(args.dotenv)
        doc_root = args.doc_root or settings.doc_root
        ls_url = args.ls_url or settings.url
        api_key = args.api_key or settings.api_key

    import_images(
        src_root=args.src_root,
        ls_url=ls_url,
        api_key=api_key,
        doc_root=doc_root,
        recursive=args.recursive,
        slice_spec=args.slice,
        project_id=args.project_id,
        project_title=args.project_title,
        mirror_root=args.mirror_root,
        update_existing_title=args.update_existing_title,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

