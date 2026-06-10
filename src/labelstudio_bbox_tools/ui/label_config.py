from __future__ import annotations

import argparse
from collections.abc import Sequence

from labelstudio_bbox_tools.config import settings_from_env
from labelstudio_bbox_tools.ls_client import make_client
from labelstudio_bbox_tools.ui.class_sources import collect_classes_mmyolo


def make_label_config_xml(classes: Sequence[str], shape: str = "bbox") -> str:
    if shape not in {"bbox", "polygon"}:
        raise ValueError("shape must be 'bbox' or 'polygon'")
    tag = "RectangleLabels" if shape == "bbox" else "PolygonLabels"
    labels = "\n".join(f'    <Label value="{label}"/>' for label in classes)
    return f"""
<View>
  <Image name="image" value="$image"/>
  <{tag} name="tag" toName="image" strokeWidth="3">
{labels}
  </{tag}>
</View>
""".strip()


def apply_label_config(
    *,
    project_id: int,
    ls_url: str,
    api_key: str,
    classes: Sequence[str],
    shape: str = "bbox",
) -> int:
    client = make_client(ls_url, api_key)
    project = client.get_project(project_id)
    project.set_params(label_config=make_label_config_xml(classes, shape=shape))
    print(f"[ok] updated label_config for project ID={project.id}, classes={len(classes)}")
    return int(project.id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Label Studio bbox/polygon UI from a MMYOLO/COCO JSON.")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--mmyolo-json", required=True)
    parser.add_argument("--shape", default="bbox", choices=["bbox", "polygon"])
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    classes = collect_classes_mmyolo(args.mmyolo_json)
    print(f"[info] classes={len(classes)}")
    if args.dry_run:
        print(make_label_config_xml(classes[:5], shape=args.shape))
        return
    settings = settings_from_env(args.dotenv)
    apply_label_config(
        project_id=args.project_id,
        ls_url=settings.url,
        api_key=settings.api_key,
        classes=classes,
        shape=args.shape,
    )


if __name__ == "__main__":
    main()

