from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from labelstudio_bbox_tools.video_inference.classes import color_for_class
from labelstudio_bbox_tools.video_inference.common import Detection

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def resolve_font_path(font_path: str | Path | None = None) -> Path | None:
    candidates = [font_path] if font_path else []
    candidates.extend(FONT_CANDIDATES)
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return path
    return None


def load_font(font_path: str | Path | None = None, font_size: int = 20) -> ImageFont.ImageFont:
    resolved = resolve_font_path(font_path)
    if resolved:
        return ImageFont.truetype(str(resolved), int(font_size))
    print("[warn] Nanum/Noto CJK font was not found; Korean labels may not render correctly.")
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _clamp_label_box(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x = max(0, min(x, max(0, image_width - width)))
    y = max(0, min(y, max(0, image_height - height)))
    return x, y, x + width, y + height


def _label_box_candidates(
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label_width: int,
    label_height: int,
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int]]:
    padding = 2
    raw = [
        (x1, y1 - label_height - padding),
        (x1, y1 + padding),
        (x1, y2 - label_height - padding),
        (x1, y2 + padding),
        (x2 - label_width, y1 - label_height - padding),
        (x2 - label_width, y1 + padding),
    ]
    return [
        _clamp_label_box(x, y, label_width, label_height, image_width, image_height)
        for x, y in raw
    ]


def _text_fill_for_background(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    red, green, blue = rgb
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)


def draw_detections_on_bgr(
    frame_bgr,
    detections: Sequence[Detection],
    *,
    color_map: dict[str, tuple[int, int, int]],
    font_path: str | Path | None = None,
    font_size: int = 20,
    line_width: int = 3,
    score_digits: int = 2,
):
    """Draw bbox and labels on an OpenCV BGR frame and return a BGR frame."""

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV and numpy are required for drawing video frames") from exc

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image)
    font = load_font(font_path, font_size)
    image_width, image_height = image.size
    used_label_boxes: list[tuple[int, int, int, int]] = []

    for det in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in det.xyxy]
        x1 = max(0, min(x1, image_width - 1))
        x2 = max(0, min(x2, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        y2 = max(0, min(y2, image_height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        color = color_for_class(det.class_name, color_map)
        for offset in range(max(1, int(line_width))):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)

        label = f"{det.class_name} {det.score:.{score_digits}f}"
        text_w, text_h = _text_size(draw, label, font)
        pad_x = max(4, font_size // 4)
        pad_y = max(3, font_size // 6)
        label_w = text_w + pad_x * 2
        label_h = text_h + pad_y * 2
        candidates = _label_box_candidates(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            label_width=label_w,
            label_height=label_h,
            image_width=image_width,
            image_height=image_height,
        )
        chosen = candidates[0]
        for candidate in candidates:
            if not any(_intersects(candidate, used) for used in used_label_boxes):
                chosen = candidate
                break
        used_label_boxes.append(chosen)
        draw.rectangle(chosen, fill=color)
        draw.text((chosen[0] + pad_x, chosen[1] + pad_y), label, font=font, fill=_text_fill_for_background(color))

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_title_bar_bgr(
    frame_bgr,
    *,
    title: str,
    font_path: str | Path | None = None,
    font_size: int = 28,
    bar_height: int = 48,
    background: tuple[int, int, int] = (30, 30, 30),
    foreground: tuple[int, int, int] = (255, 255, 255),
):
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV and numpy are required for drawing video frames") from exc

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    width, height = image.size
    out = Image.new("RGB", (width, height + bar_height), background)
    out.paste(image, (0, bar_height))
    draw = ImageDraw.Draw(out)
    font = load_font(font_path, font_size)
    text_w, text_h = _text_size(draw, title, font)
    draw.text((max(0, (width - text_w) // 2), max(0, (bar_height - text_h) // 2)), title, font=font, fill=foreground)
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
