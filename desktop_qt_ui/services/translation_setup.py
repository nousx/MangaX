"""Turn the saved desktop settings into what the translation backend expects.

The desktop window and the command-line mode both start translations. They must
build the backend configuration and the save options the same way, otherwise the
same settings file produces different output depending on how a run was started.
"""
import os
from typing import Callable, Iterable, Optional

SECTION_KEYS = ('render', 'upscale', 'translator', 'detector', 'colorizer', 'inpainter', 'ocr')
# Stored value of the "do not upscale" choice in settings files written by older versions.
UPSCALE_DISABLED_LABEL = '不使用'
# Stored value of the "keep the original format" choice.
FORMAT_UNSPECIFIED_LABEL = '不指定'
MANGAJANAI_RATIOS = ('x2', 'x4', 'DAT2 x4')
DIRECTION_ALIASES = {'h': 'horizontal', 'v': 'vertical'}


def normalize_upscale_ratio(value):
    """Return the ratio as the backend wants it: None, an int, or a mangajanai label."""
    if value is None or value == UPSCALE_DISABLED_LABEL:
        return None
    if isinstance(value, str) and value in MANGAJANAI_RATIOS:
        return value
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def normalize_output_format(value) -> Optional[str]:
    """Return None when the original file format should be kept."""
    if not value or value == FORMAT_UNSPECIFIED_LABEL:
        return None
    return value


def build_backend_config(config_dict: dict, root_dir: str, warn: Optional[Callable[[str], None]] = None):
    """Build the backend Config from the desktop settings dictionary."""
    from manga_translator.config import (
        ColorizerConfig,
        Config,
        DetectorConfig,
        InpainterConfig,
        OcrConfig,
        RenderConfig,
        TranslatorConfig,
        UpscaleConfig,
    )

    render = dict(config_dict.get('render', {}))
    if render.get('direction') in DIRECTION_ALIASES:
        render['direction'] = DIRECTION_ALIASES[render['direction']]

    translator = dict(config_dict.get('translator', {}))
    prompt_path = translator.get('high_quality_prompt_path')
    if prompt_path and not os.path.isabs(prompt_path):
        full_prompt_path = os.path.join(root_dir, prompt_path)
        if os.path.exists(full_prompt_path):
            translator['high_quality_prompt_path'] = full_prompt_path
        elif warn:
            warn(f"--- WARNING: High quality prompt file not found at {full_prompt_path}")

    upscale = dict(config_dict.get('upscale', {}))
    if 'upscale_ratio' in upscale:
        upscale['upscale_ratio'] = normalize_upscale_ratio(upscale['upscale_ratio'])

    # Everything else the backend knows about, including the 'cli' section that
    # carries options such as PSD export.
    remaining = {
        key: value for key, value in config_dict.items()
        if key in Config.model_fields and key not in SECTION_KEYS
    }
    return Config(
        render=RenderConfig(**render),
        upscale=UpscaleConfig(**upscale),
        translator=TranslatorConfig(**translator),
        detector=DetectorConfig(**config_dict.get('detector', {})),
        colorizer=ColorizerConfig(**config_dict.get('colorizer', {})),
        inpainter=InpainterConfig(**config_dict.get('inpainter', {})),
        ocr=OcrConfig(**config_dict.get('ocr', {})),
        **remaining,
    )


def build_save_info(config_dict: dict, output_folder: str, input_folders: Iterable[str],
                    overwrite: Optional[bool] = None) -> dict:
    """Build the save options passed to MangaTranslator.translate_batch."""
    cli = config_dict.get('cli', {})
    return {
        'output_folder': output_folder,
        'format': normalize_output_format(cli.get('format')),
        'overwrite': cli.get('overwrite', True) if overwrite is None else overwrite,
        'input_folders': {os.path.normpath(folder) for folder in input_folders if folder},
        'save_to_source_dir': cli.get('save_to_source_dir', False),
    }
