"""
Automatic patching tool for the PaddleOCR-VL model files

Applies the necessary changes automatically before the model is loaded, so the model files do not have to be edited by hand
"""

import os
import sys
from pathlib import Path
from manga_translator.utils.swallowed import note_ignored_error


_PADDLEOCR_VL_KWARGS_DOCS = """    min_pixels (`int`, *optional*, defaults to 147456):
        Minimum number of pixels for the resized image.
    max_pixels (`int`, *optional*, defaults to 2359296):
        Maximum number of pixels for the resized image.
"""


def patch_transformers_paddleocr_vl_docs(module_file: str | None = None) -> bool:
    """Patch missing PaddleOCR-VL kwargs docs before Transformers imports the module."""
    if module_file is None:
        try:
            import transformers
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/ocr/paddleocr_vl_patcher.py:patch_transformers_paddleocr_vl_docs")
            return False
        module_file = str(
            Path(transformers.__file__).resolve().parent
            / "models"
            / "paddleocr_vl"
            / "image_processing_paddleocr_vl.py"
        )

    path = Path(module_file)
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False

    if "    min_pixels (`int`, *optional*, defaults to 147456):" in content:
        return False

    anchor = "    merge_size (`int`, *optional*, defaults to 2):\n        The merge size of the vision encoder to llm encoder.\n"
    if anchor not in content:
        return False

    path.write_text(content.replace(anchor, anchor + _PADDLEOCR_VL_KWARGS_DOCS, 1), encoding="utf-8")
    return True

def patch_paddleocr_vl_files(model_path: str):
    """
    Patch the PaddleOCR-VL model files automatically

    Args:
        model_path: path of the model folder
    """
    # 1. Create the __init__.py file (when it does not exist)
    init_file = os.path.join(model_path, '__init__.py')
    if not os.path.exists(init_file):
        init_content = '''# 将 ernie4_5 映射到当前模块，避免 transformers 查找不存在的模块
import sys
from pathlib import Path

# 获取当前模块路径
current_module_path = Path(__file__).parent

# 创建模块别名
if 'transformers.models.ernie4_5' not in sys.modules:
    # 将当前模块注册为 ernie4_5
    sys.modules['transformers.models.ernie4_5'] = sys.modules[__name__]
    sys.modules['transformers.models.ernie4_5.configuration_ernie4_5'] = sys.modules.get(f'{__name__}.configuration_paddleocr_vl')
    sys.modules['transformers.models.ernie4_5.modeling_ernie4_5'] = sys.modules.get(f'{__name__}.modeling_paddleocr_vl')

# 同时映射 ernie4_5_moe
if 'transformers.models.ernie4_5_moe' not in sys.modules:
    sys.modules['transformers.models.ernie4_5_moe'] = sys.modules[__name__]
    sys.modules['transformers.models.ernie4_5_moe.configuration_ernie4_5_moe'] = sys.modules.get(f'{__name__}.configuration_paddleocr_vl')
    sys.modules['transformers.models.ernie4_5_moe.modeling_ernie4_5_moe'] = sys.modules.get(f'{__name__}.modeling_paddleocr_vl')
'''
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(init_content)
    
    # 2. Patch modeling_paddleocr_vl.py (comment out the @check_model_inputs decorator)
    modeling_file = os.path.join(model_path, 'modeling_paddleocr_vl.py')
    if os.path.exists(modeling_file):
        with open(modeling_file, 'r', encoding='utf-8') as f:
            content = f.read()
        changed = False

        # Check whether a patch is needed
        if '@check_model_inputs' in content and '# @check_model_inputs' not in content:
            # Replace @check_model_inputs with a comment
            content = content.replace(
                '    @check_model_inputs',
                '    # @check_model_inputs  # 注释掉此装饰器以避免参数检查问题'
            )
            changed = True

        # In transformers 4.57+ the parameter of create_causal_mask was renamed from inputs_embeds to input_embeds
        legacy_mask_block = """        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=cache_position,
        )"""
        patched_mask_block = """        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=cache_position,
        )"""
        if legacy_mask_block in content:
            content = content.replace(legacy_mask_block, patched_mask_block)
            changed = True

        # The compatibility patch above must only apply to create_causal_mask; other forward and generation calls still need inputs_embeds
        accidental_replacements = (
            (
                """        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            input_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )""",
                """        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )""",
            ),
            (
                """        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            input_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )""",
                """        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )""",
            ),
            (
                """        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            input_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )""",
                """        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )""",
            ),
        )
        for bad_block, good_block in accidental_replacements:
            if bad_block in content:
                content = content.replace(bad_block, good_block)
                changed = True

        if changed:
            with open(modeling_file, 'w', encoding='utf-8') as f:
                f.write(content)


def register_ernie_modules(model_path: str):
    """
    Register the ernie4_5 module mapping

    Args:
        model_path: path of the model folder
    """
    # Import the module in advance to register the ernie4_5 mapping
    if os.path.exists(model_path):
        sys.path.insert(0, model_path)
        try:
            __import__('__init__')
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/ocr/paddleocr_vl_patcher.py:register_ernie_modules")
            pass
        finally:
            if model_path in sys.path:
                sys.path.remove(model_path)
