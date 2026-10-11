"""
Response utilities module

Responsible for image conversion, JSON conversion and temporary environment variable handling.
"""

import io
import json

from fastapi import HTTPException

from manga_translator import Config
from manga_translator.server.core.api_key_policy import get_effective_api_key_policy
from manga_translator.server.core.group_management_service import (
    get_group_management_service,
)
from manga_translator.server.core.middleware import get_services
from manga_translator.server.runtime_api import (
    apply_runtime_api_overrides,
    clear_runtime_api_overrides,
)
from manga_translator.server.to_json import to_translation


def transform_to_image(ctx):
    """
    Convert a translation context to image bytes

    Args:
        ctx: the translation context object

    Returns:
        The image byte data
    """
    # Check whether ctx.result exists
    if ctx.result is None:
        raise HTTPException(500, detail="Translation failed: no result image generated")
    
    # Check whether a placeholder is used (in web mode this flag is set after final.png is saved)
    if hasattr(ctx, 'use_placeholder') and ctx.use_placeholder:
        # ctx.result is already a 1x1 placeholder image, which transfers quickly
        img_byte_arr = io.BytesIO()
        ctx.result.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()

    # Return the full translation result
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def transform_to_json(ctx):
    """
    Convert a translation context to JSON bytes

    Args:
        ctx: the translation context object

    Returns:
        The JSON byte data
    """
    return to_translation(ctx).model_dump_json().encode("utf-8")


def transform_to_bytes(ctx):
    """
    Convert a translation context to the custom byte format

    Args:
        ctx: the translation context object

    Returns:
        The custom byte data
    """
    return to_translation(ctx).to_bytes()


async def apply_user_env_vars(user_env_vars_str: str, config: Config, admin_settings: dict, username: str = None):
    """
    Parse the environment variables the user provided, and check the policy.
    When the user provided no API keys, they are looked up in the preset the user selected

    **Important**: this function sets the user's API key on config.translator;
    the translator reads these values in parse_args, which gives per-user isolation of API keys.

    Args:
        user_env_vars_str: JSON string with the user's API keys
        config: the configuration object
        admin_settings: the administrator settings
        username: the user name (used to get the preset configuration)

    Returns:
        dict: dictionary of the environment variables the user provided, or None when there are none

    Raises:
        HTTPException: when the policy does not allow it
    """
    import logging
    logger = logging.getLogger('manga_translator.server')
    
    logger.info(f"[EnvVars] apply_user_env_vars called for user '{username}'")
    
    policy = get_effective_api_key_policy(username, admin_settings)
    require_user_keys = policy.get('require_user_keys', False)
    allow_server_keys = policy.get('allow_server_keys', True)
    
    # Try to parse the API keys the user provided directly first
    user_env_vars = None
    if user_env_vars_str and user_env_vars_str.strip() not in ('{}', ''):
        try:
            user_env_vars = json.loads(user_env_vars_str)
            user_env_vars = {k: v for k, v in user_env_vars.items() if v and k.isupper()}
            if user_env_vars:
                logger.info(f"[EnvVars] User '{username}' provided direct env vars: {list(user_env_vars.keys())}")
        except json.JSONDecodeError:
            pass
    
    # When the user did not provide a full configuration directly, try to merge with the preset
    logger.info(f"[EnvVars] Resolving preset env vars for user '{username}'")
    preset_state = await get_user_preset_env_state(username) if username else None
    preset_env_vars = (preset_state or {}).get('env_vars')

    merged_env_vars = {}
    if preset_env_vars:
        logger.info(
            f"[EnvVars] Using preset env vars for user '{username}'"
            f" from {preset_state.get('source')}: {list(preset_env_vars.keys())}"
        )
        merged_env_vars.update(preset_env_vars)
    if user_env_vars:
        logger.info(
            f"[EnvVars] Applying user env vars over preset/server for user '{username}':"
            f" {list(user_env_vars.keys())}"
        )
        merged_env_vars.update(user_env_vars)
    if merged_env_vars:
        config._allow_server_api_keys = allow_server_keys
        _apply_env_vars_to_config(
            merged_env_vars,
            config,
            logger,
            allow_server_api_keys=allow_server_keys,
        )
        apply_runtime_api_overrides(config, merged_env_vars)
        return merged_env_vars
    
    # No user API keys and no preset
    logger.info(f"[EnvVars] No preset env vars for user '{username}', using server defaults")
    if require_user_keys:
        # The user is required to provide API keys
        raise HTTPException(403, detail="User API keys are required")
    
    if not allow_server_keys:
        # Using the server's API keys is not allowed, and the user provided none either
        raise HTTPException(403, detail="Server API keys are not allowed, please provide your own")
    
    # Using the server's API keys is allowed (they are already in the environment variables)
    # Clear the per-user API key in config, so the server default is used
    config._allow_server_api_keys = True
    config.translator.user_api_key = None
    config.translator.user_api_base = None
    config.translator.user_api_model = None
    clear_runtime_api_overrides(config)
    return None


def _apply_env_vars_to_config(
    env_vars: dict,
    config: Config,
    logger,
    *,
    allow_server_api_keys: bool,
):
    """
    Map environment variables to the user-level fields of config.translator

    Supported environment variables:
    - OPENAI_API_KEY, GEMINI_API_KEY -> user_api_key
    - OPENAI_API_BASE, GEMINI_API_BASE -> user_api_base
    - OPENAI_MODEL, GEMINI_MODEL -> user_api_model

    Note: a preset may use the OPENAI_* variables to configure a third-party API (such as Gemini through an OpenAI-compatible interface),
    so these variables are all mapped to the user_api_* fields, and the translator uses the values according to its own type

    Args:
        env_vars: dictionary of environment variables
        config: the configuration object
        logger: the logger
    """
    logger.info(f"[EnvVars->Config] Processing env vars: {list(env_vars.keys())}")

    translator_value = getattr(config.translator, 'translator', '')
    translator_name = str(getattr(translator_value, 'value', translator_value) or '').strip().lower()
    provider_priority = {
        "openai": {
            "api_key": ["OPENAI_API_KEY"],
            "api_base": ["OPENAI_API_BASE"],
            "model": ["OPENAI_MODEL"],
        },
        "openai_hq": {
            "api_key": ["OPENAI_API_KEY"],
            "api_base": ["OPENAI_API_BASE"],
            "model": ["OPENAI_MODEL"],
        },
        "gemini": {
            "api_key": ["GEMINI_API_KEY"],
            "api_base": ["GEMINI_API_BASE"],
            "model": ["GEMINI_MODEL"],
        },
        "gemini_hq": {
            "api_key": ["GEMINI_API_KEY"],
            "api_base": ["GEMINI_API_BASE"],
            "model": ["GEMINI_MODEL"],
        },
    }
    priority = provider_priority.get(translator_name)

    config.translator.user_api_key = None
    config.translator.user_api_base = None
    config.translator.user_api_model = None

    if not priority:
        logger.info(
            f"[EnvVars->Config] Translator '{translator_name}' does not require"
            " provider API merge; keep existing translator config."
        )
        return

    def _pick_first(field_name: str, target_attr: str, mask_value: bool = False):
        for var in priority.get(field_name, []):
            value = env_vars.get(var)
            if not value:
                continue
            setattr(config.translator, target_attr, value)
            logger.info(f"[EnvVars->Config] Set {target_attr} from {var}")
            return

    _pick_first("api_key", "user_api_key", mask_value=True)
    _pick_first("api_base", "user_api_base")
    _pick_first("model", "user_api_model")

    if not config.translator.user_api_key:
        if allow_server_api_keys:
            logger.info(
                f"[EnvVars->Config] No matching runtime API key found for translator"
                f" '{translator_name}', falling back to server default key."
            )
            return
        expected_key_names = ", ".join(priority.get("api_key", []))
        raise HTTPException(
            403,
            detail=(
                f"Selected translator '{translator_name}' requires one of: {expected_key_names}. "
                "The API key you filled does not match the current translator."
            ),
        )
    
    # Final confirmation
    logger.info(f"[EnvVars->Config] Final config.translator: user_api_key={'SET' if config.translator.user_api_key else 'NOT SET'}, user_api_base={config.translator.user_api_base}, user_api_model={config.translator.user_api_model}")


async def get_user_preset_env_vars(username: str) -> dict:
    preset_state = await get_user_preset_env_state(username)
    return preset_state.get('env_vars') if preset_state else None


async def get_user_preset_env_state(username: str) -> dict:
    """
    Get the API keys in the preset currently in effect for a user, and where they come from.

    Args:
        username: the user name

    Returns:
        dict: {
            "preset_id": str,
            "preset_name": str,
            "source": "user_selected" | "group_default",
            "env_vars": {...},
        }
    """
    import logging
    logger = logging.getLogger('manga_translator.server')
    
    try:
        from manga_translator.server.core.config_management_service import (
            ConfigManagementService,
        )
        
        config_service = ConfigManagementService()
        
        # Get the user configuration
        user_config = config_service.get_user_config(username)
        logger.info(f"[Preset] User '{username}' config: {user_config}")

        preset_id = user_config.get('selected_preset_id') if user_config else None
        preset_source = 'user_selected'
        logger.info(f"[Preset] User '{username}' selected preset_id: {preset_id}")

        if not preset_id:
            account_service, _, _ = get_services()
            account = account_service.get_user(username)
            if account:
                group = get_group_management_service().get_group(account.group)
                preset_id = (group or {}).get('default_preset_id')
                if preset_id:
                    preset_source = 'group_default'
                    logger.info(
                        f"[Preset] User '{username}' has no selected preset,"
                        f" falling back to group default preset_id: {preset_id}"
                    )

        if not preset_id:
            logger.info(f"[Preset] No preset selected for user '{username}'")
            return None
        
        # Get the preset configuration (decrypted)
        preset = config_service.get_preset(preset_id, decrypt=True)
        logger.info(f"[Preset] Preset '{preset_id}' loaded: {preset is not None}")
        if not preset:
            logger.warning(f"[Preset] Preset '{preset_id}' not found")
            return None
        
        # Extract the API keys from the preset configuration
        preset_config = preset.get('config', {})
        logger.info(f"[Preset] Preset config keys: {list(preset_config.keys())}")
        if not preset_config:
            logger.info(f"[Preset] Preset '{preset_id}' has no config")
            return None
        
        # Extract the environment variables (upper-case keys)
        env_vars = {k: v for k, v in preset_config.items() if v and k.isupper()}
        
        # Only the names of the decrypted keys are logged, never a value or a prefix
        for key in env_vars:
            if 'API_KEY' in key:
                value = env_vars[key]
                if value:
                    logger.info(f"[Preset] {key} decrypted successfully")
                else:
                    logger.warning(f"[Preset] {key} is empty after decryption!")
        
        logger.info(f"[Preset] Extracted env vars for user '{username}': {list(env_vars.keys())}")
        if not env_vars:
            return None

        return {
            'preset_id': preset_id,
            'preset_name': preset.get('name'),
            'source': preset_source,
            'env_vars': env_vars,
        }
        
    except Exception as e:
        import logging
        logging.getLogger('manga_translator.server').warning(f"Failed to get user preset env vars: {e}")
        import traceback
        traceback.print_exc()
        return None
