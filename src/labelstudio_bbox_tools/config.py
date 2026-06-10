from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabelStudioSettings:
    url: str
    api_key: str
    doc_root: Path


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


def get_env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def settings_from_env(dotenv_path: str | Path | None = ".env") -> LabelStudioSettings:
    if dotenv_path:
        load_dotenv(dotenv_path)
    return LabelStudioSettings(
        url=get_env("LABEL_STUDIO_URL"),
        api_key=get_env("LABEL_STUDIO_API_KEY"),
        doc_root=Path(get_env("LABEL_STUDIO_DOC_ROOT", "/")).expanduser(),
    )

