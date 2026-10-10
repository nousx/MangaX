from __future__ import annotations

from PIL import Image
from manga_translator.utils.swallowed import note_ignored_error


def pil_image_has_alpha(image: Image.Image) -> bool:
    """Return whether a PIL image carries usable alpha, including nonstandard modes."""
    if image.mode == "P":
        return "transparency" in image.info

    try:
        return any(band.upper() == "A" for band in image.getbands())
    except Exception as ignored_error:
        note_ignored_error(ignored_error, "manga_translator/utils/image_modes.py:pil_image_has_alpha")
        return image.mode in {"RGBA", "LA"}


def normalize_rgb_image(
    image: Image.Image,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Return an RGB image, compositing any alpha channel over a solid background."""
    if image.mode == "RGB":
        return image

    if pil_image_has_alpha(image):
        rgba_image = image.convert("RGBA")
        background = Image.new("RGB", rgba_image.size, background_color)
        background.paste(rgba_image, mask=rgba_image.getchannel("A"))
        return background

    return image.convert("RGB")
