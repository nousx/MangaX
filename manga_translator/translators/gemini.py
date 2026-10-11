# import re
import asyncio
import os

# import json
from typing import Any, Dict, List

from google.genai import types

from ..api_request_params import apply_gemini_sdk_generation_params
from ..api_key_rotation import APIRotationExhaustedError, run_with_api_candidates
from ..runtime_api_resolver import resolve_runtime_api_config
from ..utils.dotenv_utils import load_app_dotenv
from ..utils.curl_cffi_transport import GEMINI_CURL_HEADERS
from .common import (
    VALID_LANGUAGES,
    AsyncGeminiCurlCffi,
    CommonTranslator,
    extract_gemini_response_diagnostics,
    format_gemini_response_diagnostics,
    gemini_diagnostics_indicate_safety,
    gemini_error_message_indicates_safety,
    merge_glossary_to_file,
    parse_hq_response,
    validate_gemini_response,
)
from manga_translator.utils.swallowed import note_ignored_error

# The browser identity comes from curl_cffi's impersonate setting; only the request headers of the application are kept here.
BROWSER_HEADERS = GEMINI_CURL_HEADERS



class GeminiTranslator(CommonTranslator):
    """
    Gemini纯文本翻译器
    支持批量文本翻译，不包含图片处理
    """
    _LANGUAGE_CODE_MAP = VALID_LANGUAGES
    API_KEY_ENV = "GEMINI_API_KEY"
    API_BASE_ENV = "GEMINI_API_BASE"
    MODEL_ENV = "GEMINI_MODEL"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
    DEFAULT_MODEL_NAME = "gemini-1.5-flash"
    
    # Class variable: RPM limit timestamp shared across instances
    _GLOBAL_LAST_REQUEST_TS = {}  # {model_name: timestamp}
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.prev_context = ""  # Holds the multi-page context
        # Initial setup from environment variables
        # Reload the .env file only outside the web environment
        is_web_server = os.getenv('MANGA_TRANSLATOR_WEB_SERVER', 'false').lower() == 'true'
        if not is_web_server:
            load_app_dotenv(override=True)
        
        self.api_key = os.getenv(self.API_KEY_ENV, '')
        self.base_url = os.getenv(self.API_BASE_ENV, self.DEFAULT_BASE_URL) if self.API_BASE_ENV else self.DEFAULT_BASE_URL
        self.model_name = os.getenv(self.MODEL_ENV, self.DEFAULT_MODEL_NAME)
        self._runtime_api_settings = None
        self._refresh_runtime_api_settings(None)
        self.max_tokens = None  # No limit: use the model's default maximum
        self._MAX_REQUESTS_PER_MINUTE = 0  # No limit by default
        # Use the global timestamp, shared across instances
        if self.model_name not in type(self)._GLOBAL_LAST_REQUEST_TS:
            type(self)._GLOBAL_LAST_REQUEST_TS[self.model_name] = 0
        self._last_request_ts_key = self.model_name
        # Safety settings of the new SDK
        self.safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
        self._setup_client()
    
    def set_prev_context(self, context: str):
        """设置多页上下文（用于context_size > 0时）"""
        self.prev_context = context if context else ""
    
    def parse_args(self, args):
        """解析配置参数"""
        # Call the parent parse_args to set the common parameters (attempts, post_check and so on)
        super().parse_args(args)
        translator_args = self._resolve_translator_config(args)
        
        # Sync the retry count to the "total attempts" (first request + retries)
        self._max_total_attempts = self._resolve_max_total_attempts()
        
        # Read the RPM limit from the configuration
        max_rpm = self._get_config_value(translator_args, 'max_requests_per_minute', 0)
        if max_rpm > 0:
            self._MAX_REQUESTS_PER_MINUTE = max_rpm
            self.logger.info(f"Setting Gemini max requests per minute to: {max_rpm}")
        
        # Read the custom API parameter configuration
        self._configure_custom_api_params(args)
        
        # Read the per-user API key from the configuration (it takes precedence over the environment variable)
        # This lets the web server use a different API key for each user
        need_rebuild_client = False
        
        user_api_key = self._get_config_value(translator_args, 'user_api_key', None)
        if user_api_key and user_api_key != self.api_key:
            self.api_key = user_api_key
            need_rebuild_client = True
            self.logger.info("[UserAPIKey] Using user-provided API key for Gemini")
        
        user_api_base = self._get_config_value(translator_args, 'user_api_base', None)
        if user_api_base and user_api_base != self.base_url:
            self.base_url = user_api_base
            need_rebuild_client = True
            self.logger.info(f"[UserAPIKey] Using user-provided API base: {user_api_base}")
        
        user_api_model = self._get_config_value(translator_args, 'user_api_model', None)
        if user_api_model:
            self.model_name = user_api_model
            # Update the key of the global timestamp
            if self.model_name not in type(self)._GLOBAL_LAST_REQUEST_TS:
                type(self)._GLOBAL_LAST_REQUEST_TS[self.model_name] = 0
            self._last_request_ts_key = self.model_name
            self.logger.info(f"[UserAPIKey] Using user-provided model: {user_api_model}")

        if user_api_key or user_api_base or user_api_model:
            self._runtime_api_settings = None
        else:
            old_signature = (self.api_key or "", self.base_url, self.model_name)
            self._refresh_runtime_api_settings(args)
            if (self.api_key or "", self.base_url, self.model_name) != old_signature:
                need_rebuild_client = True

        # Rebuild the client when the API key or the base URL changed
        if need_rebuild_client:
            self.client = None
            self._setup_client()

    def _refresh_runtime_api_settings(self, config):
        settings = resolve_runtime_api_config(
            config,
            feature="translator",
            provider="gemini",
            api_key_env=self.API_KEY_ENV,
            api_base_env=self.API_BASE_ENV,
            model_env=self.MODEL_ENV,
            fallback_api_key_env=None,
            fallback_api_base_env=None,
            fallback_model_env=None,
            default_api_base=self.DEFAULT_BASE_URL,
            default_model=self.DEFAULT_MODEL_NAME,
            allow_empty_local_api_key=False,
        )
        self._runtime_api_settings = settings
        if settings.api_key:
            self.api_key = settings.api_key
            self.base_url = settings.base_url
            self.model_name = settings.model_name
        return settings

    async def _close_current_client(self):
        if not self.client:
            return
        close_fn = getattr(self.client, "close", None)
        try:
            if callable(close_fn):
                close_result = close_fn()
                if asyncio.iscoroutine(close_result):
                    await close_result
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/translators/gemini.py:GeminiTranslator._close_current_client")
            pass
        finally:
            self.client = None

    async def _reset_client_for_candidate(self, endpoint, error: Exception):
        del endpoint, error
        await self._close_current_client()

    def _apply_api_endpoint(self, endpoint):
        signature = (endpoint.api_key or "", endpoint.base_url, endpoint.model_name)
        if signature == (self.api_key or "", self.base_url, self.model_name) and self.client:
            return
        self.api_key = endpoint.api_key
        self.base_url = endpoint.base_url
        self.model_name = endpoint.model_name
        if self.model_name not in type(self)._GLOBAL_LAST_REQUEST_TS:
            type(self)._GLOBAL_LAST_REQUEST_TS[self.model_name] = 0
        self._last_request_ts_key = self.model_name
        self.client = None
        self._setup_client()

    async def _run_with_api_rotation(self, operation, operation_name: str):
        settings = self._runtime_api_settings
        if not settings or not settings.candidates:
            if not self.client:
                self._setup_client()
            return await operation()

        async def _with_endpoint(endpoint):
            self._apply_api_endpoint(endpoint)
            return await operation()

        return await run_with_api_candidates(
            endpoints=settings.candidates,
            strategy=settings.strategy,
            operation=_with_endpoint,
            provider_name=self._log_provider_name() if hasattr(self, "_log_provider_name") else "Gemini",
            operation_name=operation_name,
            logger=self.logger,
            retry_attempts=self.attempts,
            on_candidate_error=self._reset_client_for_candidate,
        )

    def _setup_client(self, system_instruction=None):
        """设置Gemini客户端"""
        if not self.client and self.api_key:
            # Check whether a custom API base is used
            is_custom_api = (
                self.base_url
                and self.base_url.strip()
                and self.base_url.strip() not in ["https://generativelanguage.googleapis.com", "https://generativelanguage.googleapis.com/"]
            )

            if is_custom_api:
                self.client = AsyncGeminiCurlCffi(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    default_headers=BROWSER_HEADERS,
                    impersonate="chrome",
                    timeout=600,
                    stream_timeout=300
                )
                self._use_curl_cffi = True
                self.logger.info(f"Gemini client initialized (forced curl_cffi with custom API base). Base URL: {self.base_url}")
            else:
                self.client = AsyncGeminiCurlCffi(
                    api_key=self.api_key,
                    default_headers=BROWSER_HEADERS,
                    impersonate="chrome",
                    timeout=600,
                    stream_timeout=300
                )
                self._use_curl_cffi = True
                self.logger.info("Gemini client initialized (forced curl_cffi mode)")

            self.logger.info("Safety settings policy: send OFF by default and fall back automatically on errors")

    async def _abort_inflight_request(self):
        """取消时尝试关闭当前客户端连接，尽快中断阻塞请求。"""
        if not self.client:
            return

        close_fn = getattr(self.client, "close", None)
        try:
            if callable(close_fn):
                close_result = close_fn()
                if asyncio.iscoroutine(close_result):
                    await close_result
        except Exception as e:
            self.logger.debug(f"Failed to close the client while interrupting a Gemini request (safe to ignore): {e}")
        finally:
            self.client = None
    


    def _build_user_prompt(self, texts: List[str], ctx: Any, retry_attempt: int = 0, retry_reason: str = "") -> str:
        """构建用户提示词（纯文本版）- 使用 JSON 格式以配合 HQ Prompt"""
        return self._build_user_prompt_for_texts(texts, ctx, "", retry_attempt=retry_attempt, retry_reason=retry_reason)
    
    def _get_system_instruction(self, source_lang: str, target_lang: str, custom_prompt_json: Dict[str, Any] = None, line_break_prompt_json: Dict[str, Any] = None, retry_attempt: int = 0, retry_reason: str = "", extract_glossary: bool = False) -> str:
        """获取完整的系统指令"""
        return self._build_system_prompt(source_lang, target_lang, custom_prompt_json=custom_prompt_json, line_break_prompt_json=line_break_prompt_json, retry_attempt=retry_attempt, retry_reason=retry_reason, extract_glossary=extract_glossary)

    async def _translate_batch(self, texts: List[str], source_lang: str, target_lang: str, custom_prompt_json: Dict[str, Any] = None, line_break_prompt_json: Dict[str, Any] = None, ctx: Any = None, split_level: int = 0) -> List[str]:
        """批量翻译方法（纯文本）"""
        if not texts:
            return []
        
        if not self.client:
            self._setup_client()
        
        if not self.client:
            raise RuntimeError("Gemini client initialization failed: check GEMINI_API_KEY / GEMINI_API_BASE / GEMINI_MODEL settings")
        
        # Initialise the retry information
        retry_attempt = 0
        retry_reason = ""
        
        # Keep the parameters for retries
        _source_lang = source_lang
        _target_lang = target_lang
        _custom_prompt_json = custom_prompt_json
        _line_break_prompt_json = line_break_prompt_json
        
        # Send the request
        max_retries = self._resolve_max_total_attempts()
        attempt = 0
        is_infinite = max_retries == -1
        last_exception = None
        local_attempt = 0  # Number of attempts for this batch
        
        # Whether to fall back (no safety settings are sent)
        should_retry_without_safety = False

        while is_infinite or attempt < max_retries:
            # Check whether the task was cancelled
            self._check_cancelled()
            
            # Check the global attempt count
            if not self._increment_global_attempt():
                self.logger.error("Reached global attempt limit. Stopping translation.")
                last_error_msg = str(last_exception) if last_exception else "Unknown error"
                raise Exception(f"Maximum attempts reached ({self._max_total_attempts}). Last error: {last_error_msg}")

            local_attempt += 1
            attempt += 1

            # Decide whether glossary extraction is on
            config_extract = False
            if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'translator'):
                config_extract = getattr(ctx.config.translator, 'extract_glossary', False)
            
            extract_glossary = bool(_custom_prompt_json) and config_extract

            # Get the system instruction (sent through systemInstruction)
            system_instruction = self._get_system_instruction(_source_lang, _target_lang, custom_prompt_json=_custom_prompt_json, line_break_prompt_json=_line_break_prompt_json, retry_attempt=retry_attempt, retry_reason=retry_reason, extract_glossary=extract_glossary)
            
            # Initialise the client (system_instruction is not passed in)
            if not self.client:
                self._setup_client(system_instruction=None)
            
            if not self.client:
                raise RuntimeError("Gemini client initialization failed: check GEMINI_API_KEY / GEMINI_API_BASE / GEMINI_MODEL settings")
            
            # Build the user prompt
            # When the HQ prompt is loaded, _build_user_prompt (that is, _build_user_prompt_for_texts) produces JSON input that matches the system prompt
            user_prompt = self._build_user_prompt(texts, ctx, retry_attempt=retry_attempt, retry_reason=retry_reason)
            contents = self._build_gemini_context_messages(self.prev_context)
            contents.append({"role": "user", "parts": [{"text": user_prompt}]})
            
            # Build the generation settings
            config_params = {
                "top_p": 0.95,
                "top_k": 64,
                "safety_settings": None if should_retry_without_safety else self.safety_settings,
            }
            # Set only when max_tokens is not None (for newer models)
            if self.max_tokens is not None:
                config_params["max_output_tokens"] = self.max_tokens
            
            try:
                # RPM limit
                if self._MAX_REQUESTS_PER_MINUTE > 0:
                    import time
                    now = time.time()
                    delay = 60.0 / self._MAX_REQUESTS_PER_MINUTE
                    elapsed = now - type(self)._GLOBAL_LAST_REQUEST_TS[self._last_request_ts_key]
                    if elapsed < delay:
                        sleep_time = delay - elapsed
                        self.logger.info(f'Ratelimit sleep: {sleep_time:.2f}s')
                        await self._sleep_with_cancel_polling(sleep_time)
                
                def _extract_gemini_stream_text(chunk):
                    return getattr(chunk, "text", "") or ""
                
                def _on_stream_chunk(delta_text, _full_text):
                    self._emit_stream_json_preview("[Gemini Stream]", delta_text, source_texts=texts)

                response = None
                streamed_text = None
                streamed_finish_reason = None
                streamed_diagnostics = None

                use_streaming = self._is_streaming_enabled(ctx)

                async def _send_gemini_request():
                    nonlocal response, streamed_text, streamed_finish_reason, streamed_diagnostics
                    response = None
                    streamed_text = None
                    streamed_finish_reason = None
                    streamed_diagnostics = None
                    generation_config = types.GenerateContentConfig(**config_params)
                    generation_config.system_instruction = system_instruction
                    custom_api_params = self._resolve_translator_custom_api_params(self.model_name)
                    apply_gemini_sdk_generation_params(generation_config, custom_api_params)
                    if custom_api_params:
                        self.logger.debug(f"Using translation model preset parameters: {custom_api_params}")
                    if use_streaming:
                        try:
                            self._reset_stream_json_preview()
                            # Try streaming automatically; fall back to an ordinary request when it is not supported
                            def _extract_stream_finish_reason(chunk):
                                nonlocal streamed_diagnostics
                                streamed_diagnostics = extract_gemini_response_diagnostics(chunk)
                                return streamed_diagnostics.get('finish_reason')

                            streamed_text, streamed_finish_reason = await self._run_unified_stream_transport(
                                create_stream=lambda: self.client.models.generate_content_stream(
                                    model=self.model_name,
                                    contents=contents,
                                    config=generation_config
                                ),
                                extract_text=_extract_gemini_stream_text,
                                extract_finish_reason=_extract_stream_finish_reason,
                                on_chunk=_on_stream_chunk,
                                on_cancel=self._abort_inflight_request,
                                poll_interval=0.2,
                                sync_iter_in_thread=not getattr(self, '_use_curl_cffi', False),
                            )
                            self._finish_stream_inline()
                        except Exception as stream_error:
                            self._finish_stream_inline()
                            streamed_text = None
                            streamed_finish_reason = None
                            streamed_diagnostics = None
                            self.logger.warning(f"Streaming request unavailable; fell back to a non-streaming request: {stream_error}")
                            # Use the standard SDK (the synchronous call is wrapped to be asynchronous)
                            if getattr(self, '_use_curl_cffi', False):
                                response = await self._await_with_cancel_polling(
                                    self.client.models.generate_content(
                                        model=self.model_name,
                                        contents=contents,
                                        generation_config=generation_config,
                                        safety_settings=None if should_retry_without_safety else self.safety_settings
                                    ),
                                    poll_interval=0.2,
                                    on_cancel=self._abort_inflight_request,
                                )
                            else:
                                response = await self._await_with_cancel_polling(
                                    asyncio.to_thread(
                                        self.client.models.generate_content,
                                        model=self.model_name,
                                        contents=contents,
                                        config=generation_config
                                    ),
                                    poll_interval=0.2,
                                    on_cancel=self._abort_inflight_request,
                                )
                    else:
                        self.logger.info("Streaming is disabled; using a non-streaming request.")
                        if getattr(self, '_use_curl_cffi', False):
                            response = await self._await_with_cancel_polling(
                                self.client.models.generate_content(
                                    model=self.model_name,
                                    contents=contents,
                                    generation_config=generation_config,
                                    safety_settings=None if should_retry_without_safety else self.safety_settings
                                ),
                                poll_interval=0.2,
                                on_cancel=self._abort_inflight_request,
                            )
                        else:
                            response = await self._await_with_cancel_polling(
                                asyncio.to_thread(
                                    self.client.models.generate_content,
                                    model=self.model_name,
                                    contents=contents,
                                    config=generation_config
                                ),
                                poll_interval=0.2,
                                on_cancel=self._abort_inflight_request,
                            )

                    if streamed_text is not None:
                        if not streamed_text.strip():
                            raise RuntimeError("Gemini returned empty content")
                    else:
                        validate_gemini_response(response, self.logger)
                        raw_text = getattr(response, "text", "")
                        if not (raw_text if isinstance(raw_text, str) else str(raw_text or "")).strip():
                            raise RuntimeError("Gemini returned empty content")

                await self._run_with_api_rotation(_send_gemini_request, "translation request")

                if self._MAX_REQUESTS_PER_MINUTE > 0:
                    import time
                    type(self)._GLOBAL_LAST_REQUEST_TS[self._last_request_ts_key] = time.time()

                if streamed_text is None:
                    # Check that the response object is valid
                    validate_gemini_response(response, self.logger)

                diagnostics = streamed_diagnostics or extract_gemini_response_diagnostics(
                    response,
                    fallback_finish_reason=streamed_finish_reason if streamed_text is not None else None,
                )
                diagnostics_text = format_gemini_response_diagnostics(diagnostics)
                finish_reason = diagnostics.get('finish_reason')
                finish_reason_str = diagnostics.get('finish_reason_str') or ""
                if finish_reason and "STOP" not in finish_reason_str.upper():  # Not a success
                    log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                    self.logger.warning(f"Gemini API failed ({log_attempt}): {diagnostics_text}")
                    if gemini_diagnostics_indicate_safety(diagnostics) and not should_retry_without_safety:
                        self.logger.warning("Gemini safety policy block detected; safety settings will be removed on the next retry.")
                        should_retry_without_safety = True
                    if not is_infinite and attempt >= max_retries:
                        break
                    await self._sleep_with_cancel_polling(1)
                    continue

                if streamed_text is not None:
                    result_text = streamed_text.strip()
                else:
                    raw_text = getattr(response, "text", "")
                    result_text = (raw_text if isinstance(raw_text, str) else str(raw_text or "")).strip()
                
                # Shared encoding clean-up (handles UTF-16-LE and similar problems)
                from .common import sanitize_text_encoding
                result_text = sanitize_text_encoding(result_text)

                if not result_text:
                    log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                    self.logger.warning(f"Gemini returned empty content ({diagnostics_text}) ({log_attempt}). Retrying...")
                    if gemini_diagnostics_indicate_safety(diagnostics) and not should_retry_without_safety:
                        self.logger.warning("Empty response includes Gemini safety policy information; safety settings will be removed on the next retry.")
                        should_retry_without_safety = True
                    raise Exception(f"Gemini returned empty content ({diagnostics_text})")
                
                self.logger.debug(f"--- Gemini Raw Response ---\n{result_text}\n---------------------------")

                # Parse with parse_hq_response (JSON object, array or text)
                translations, new_terms = parse_hq_response(result_text)
                
                # Handle glossary extraction
                if extract_glossary and new_terms:
                    self._emit_terms_from_list(new_terms)
                    prompt_path = None
                    if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'translator'):
                        prompt_path = getattr(ctx.config.translator, 'high_quality_prompt_path', None)
                    
                    if prompt_path:
                        merge_glossary_to_file(prompt_path, new_terms)
                    else:
                        self.logger.warning("Extracted new terms but prompt path not found in context.")
                
                # Strict validation: must match input count
                if len(translations) != len(texts):
                    retry_attempt += 1
                    retry_reason = f"Translation count mismatch: expected {len(texts)}, got {len(translations)}"
                    log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                    self.logger.warning(f"[{log_attempt}] {retry_reason}. Retrying...")
                    self.logger.warning(f"Expected texts: {texts}")
                    self.logger.warning(f"Got translations: {translations}")
                    
                    # Record the error so it can be shown when the maximum number of attempts is reached
                    last_exception = Exception(f"Translation count mismatch: expected {len(texts)}, got {len(translations)}")
                    
                    if not is_infinite and attempt >= max_retries:
                        raise Exception(f"Translation count mismatch after {max_retries} attempts: expected {len(texts)}, got {len(translations)}")
                    
                    await self._sleep_with_cancel_polling(2)
                    continue

                # Quality validation: empty translations, merged translations, suspicious symbols and so on
                is_valid, error_msg = self._validate_translation_quality(texts, translations)
                if not is_valid:
                    retry_attempt += 1
                    retry_reason = f"Quality check failed: {error_msg}"
                    log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                    self.logger.warning(f"[{log_attempt}] {retry_reason}. Retrying...")
                    
                    # Record the error so it can be shown when the maximum number of attempts is reached
                    last_exception = Exception(f"Quality check failed: {error_msg}")

                    if not is_infinite and attempt >= max_retries:
                        raise Exception(f"Quality check failed after {max_retries} attempts: {error_msg}")

                    await self._sleep_with_cancel_polling(2)
                    continue

                # Print each original text with its translation
                self._emit_final_translation_results(texts, translations)

                # BR check: whether the translation contains the required [BR] markers
                if not self._validate_br_markers(translations, queries=texts, ctx=ctx):
                    retry_attempt += 1
                    retry_reason = "BR markers missing in translations"
                    log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                    self.logger.warning(f"[{log_attempt}] {retry_reason}, retrying...")
                    
                    # Record the error so it can be shown when the maximum number of attempts is reached
                    last_exception = Exception("AI line break validation failed: BR markers missing in translations")
                    
                    # When the maximum number of retries is reached, raise a readable exception
                    if not is_infinite and attempt >= max_retries:
                        from .common import BRMarkersValidationException
                        self.logger.error("Gemini translation still failed after multiple retries: AI line break validation failed.")
                        raise BRMarkersValidationException(
                            missing_count=0,  # The exact numbers were logged in _validate_br_markers
                            total_count=len(texts),
                            tolerance=max(1, len(texts) // 10)
                        )
                    await self._sleep_with_cancel_polling(2)
                    continue

                return translations[:len(texts)]

            except APIRotationExhaustedError:
                raise
            except Exception as e:
                error_message = str(e)
                last_exception = e  # Keep the last error
                
                # Check whether the error is about the safety settings
                is_safety_error = any(keyword in error_message.lower() for keyword in [
                    'safety_settings', 'safetysettings', 'harm', 'block', 'safety'
                ]) or "400" in error_message or gemini_error_message_indicates_safety(error_message)
                
                # For a safety settings error with no fallback tried yet, flag the fallback
                if is_safety_error and not should_retry_without_safety:
                    self.logger.warning(f"Safety settings error detected; safety settings will be removed on the next retry: {error_message}")
                    should_retry_without_safety = True
                    # Retry at once without increasing the attempt count
                    await self._sleep_with_cancel_polling(1)
                    continue
                
                log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                self.logger.warning(f"Gemini translation failed ({log_attempt}): {e}")

                if gemini_error_message_indicates_safety(error_message):
                    self.logger.warning("Gemini safety policy block detected. Retrying...")
                
                # Check whether the maximum number of retries is reached
                if not is_infinite and attempt >= max_retries:
                    self.logger.error("Gemini translation still failed after multiple retries. Terminating.")
                    raise e
                
                await self._sleep_with_cancel_polling(1)
        
        raise last_exception if last_exception else RuntimeError("Gemini translation failed without a response")

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str], ctx=None) -> List[str]:
        """主翻译方法"""
        if not self.client:
            from .. import manga_translator
            if hasattr(manga_translator, 'config'):
                self.parse_args(manga_translator.config)

        if not queries:
            return []

        # Reset the global attempt counter
        self._reset_global_attempt_count()

        self.logger.info(f"Using Gemini text-only translation for {len(queries)} texts; maximum attempts: {self._max_total_attempts}")
        custom_prompt_json = getattr(ctx, 'custom_prompt_json', None) if ctx else None
        line_break_prompt_json = getattr(ctx, 'line_break_prompt_json', None) if ctx else None

        # Translate through the splitting wrapper
        translations = await self._translate_with_split(
            self._translate_batch,
            queries,
            split_level=0,
            source_lang=from_lang,
            target_lang=to_lang,
            custom_prompt_json=custom_prompt_json,
            line_break_prompt_json=line_break_prompt_json,
            ctx=ctx
        )

        # Apply text post-processing
        translations = [self._clean_translation_output(q, r, to_lang) for q, r in zip(queries, translations)]
        return translations
