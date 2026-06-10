from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def collect_images(src_root: str | Path, recursive: bool = False) -> list[Path]:
    root = Path(src_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def local_file_url(doc_relative_path: str | Path) -> str:
    return f"/data/local-files/?d={quote(str(doc_relative_path), safe='/')}"


def resolve_local_file_url(image_url: str, doc_root: str | Path) -> Path:
    root = Path(doc_root)
    if image_url.startswith("/data/local-files/"):
        query = parse_qs(urlparse(image_url).query)
        rel = Path(unquote(query["d"][0]))
        return (root / rel).resolve()
    return Path(image_url).expanduser().resolve()


def doc_relative_path(
    src_path: str | Path,
    *,
    doc_root: str | Path,
    mirror_root: str | Path | None = None,
    link_root_name: str = "img_symlinks",
    create_link: bool = True,
) -> Path:
    src = Path(src_path).expanduser().resolve()
    doc = Path(doc_root).expanduser().resolve()

    try:
        return src.relative_to(doc)
    except ValueError:
        pass

    if mirror_root is None:
        raise RuntimeError(f"{src} is outside doc_root={doc}; mirror_root is required")

    mirror = Path(mirror_root).expanduser().resolve()
    try:
        rel = src.relative_to(mirror)
    except ValueError as exc:
        raise RuntimeError(f"mirror_root={mirror} is not a parent of src_path={src}") from exc

    link_path = doc / link_root_name / rel
    if create_link:
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink() and link_path.resolve() == src:
                return link_path.relative_to(doc)
            raise FileExistsError(f"Refusing to overwrite existing link target: {link_path}")
        link_path.symlink_to(src)

    return link_path.relative_to(doc)


def prefixed_project_title(project_id: int, title: str) -> str:
    prefix = f"{project_id}_"
    return title if title.startswith(prefix) else f"{prefix}{title}"

