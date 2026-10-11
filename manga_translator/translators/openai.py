import asyncio
import os
import re

# import json
from typing import Any, Dict, List

# import openai
from ..api_request_params import merge_openai_chat_request_params
from ..api_key_rotation import APIRotationExhaustedError, run_with_api_candidates
from ..runtime_api_resolver import resolve_runtime_api_config
from ..utils.dotenv_utils import load_app_dotenv
from ..utils.curl_cffi_transport import OPENAI_CURL_HEADERS
from .common import (
    VALID_LANGUAGES,
    AsyncOpenAICurlCffi,
    CommonTranslator,
    merge_glossary_to_file,
    parse_hq_response,
    validate_openai_response,
)
from .keys import OPENAI_API_KEY
from manga_translator.utils.swallowed import note_ignored_error

# The browser identity comes from curl_cffi's impersonate setting; only the request headers of the application are kept here.
BROWSER_HEADERS = OPENAI_CURL_HEADERS


class OpenAITranslator(CommonTranslator):
    """
    OpenAI text-only translator.
    Translates text in batches, without image handling
    """
    _LANGUAGE_CODE_MAP = VALID_LANGUAGES
    
    # Class variable: RPM limit timestamp shared across instances
    _GLOBAL_LAST_REQUEST_TS = {}  # {model_name: timestamp}
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.prev_context = ""  # Holds the multi-page context
        # Reload the .env file only outside the web environment
        is_web_server = os.getenv('MANGA_TRANSLATOR_WEB_SERVER', 'false').lower() == 'true'
        if not is_web_server:
            load_app_dotenv(override=True)
        
        self.api_key = os.getenv('OPENAI_API_KEY', OPENAI_API_KEY)
        self.base_url = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        self.model = os.getenv('OPENAI_MODEL', "gpt-4o")
        self._runtime_api_settings = None
        self._refresh_runtime_api_settings(None)
        self.max_tokens = None  # No limit: use the model's default maximum
        self._MAX_REQUESTS_PER_MINUTE = 0  # No limit by default
        # Use the global timestamp, shared across instances
        if self.model not in OpenAITranslator._GLOBAL_LAST_REQUEST_TS:
            OpenAITranslator._GLOBAL_LAST_REQUEST_TS[self.model] = 0
        self._last_request_ts_key = self.model
        self._setup_client()
    
    def set_prev_context(self, context: str):
        """Set the multi-page context (used when context_size > 0)"""
        self.prev_context = context if context else ""
    
    def parse_args(self, args):
        """Parse the configuration parameters"""
        # Call the parent parse_args to set the common parameters (attempts, post_check and so on)
        super().parse_args(args)
        translator_args = self._resolve_translator_config(args)
        
        # Sync the retry count to the "total attempts" (first request + retries)
        self._max_total_attempts = self._resolve_max_total_attempts()
        
        # Read the RPM limit from the configuration
        max_rpm = self._get_config_value(translator_args, 'max_requests_per_minute', 0)
        if max_rpm > 0:
            self._MAX_REQUESTS_PER_MINUTE = max_rpm
            self.logger.info(f"Setting OpenAI max requests per minute to: {max_rpm}")
        
        # Read the custom API parameter configuration
        self._configure_custom_api_params(args)
        
        # Read the per-user API key from the configuration (it takes precedence over the environment variable)
        need_rebuild_client = False
        
        user_api_key = self._get_config_value(translator_args, 'user_api_key', None)
        if user_api_key and user_api_key != self.api_key:
            self.api_key = user_api_key
            need_rebuild_client = True
            self.logger.info("[UserAPIKey] Using user-provided API key")
        
        user_api_base = self._get_config_value(translator_args, 'user_api_base', None)
        if user_api_base and user_api_base != self.base_url:
            self.base_url = user_api_base
            need_rebuild_client = True
            self.logger.info(f"[UserAPIKey] Using user-provided API base: {user_api_base}")
        
        user_api_model = self._get_config_value(translator_args, 'user_api_model', None)
        if user_api_model:
            self.model = user_api_model
            self.logger.info(f"[UserAPIKey] Using user-provided model: {user_api_model}")

        if user_api_key or user_api_base or user_api_model:
            self._runtime_api_settings = None
        else:
            old_signature = (self.api_key or "", self.base_url, self.model)
            self._refresh_runtime_api_settings(args)
            if (self.api_key or "", self.base_url, self.model) != old_signature:
                need_rebuild_client = True

        # Rebuild the client when the API key or the base URL changed
        if need_rebuild_client:
            self.client = None
            self._setup_client()

    def _refresh_runtime_api_settings(self, config):
        settings = resolve_runtime_api_config(
            config,
            feature="translator",
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            api_base_env="OPENAI_API_BASE",
            model_env="OPENAI_MODEL",
            fallback_api_key_env=None,
            fallback_api_base_env=None,
            fallback_model_env=None,
            default_api_base="https://api.openai.com/v1",
            default_model="gpt-4o",
            allow_empty_local_api_key=True,
        )
        self._runtime_api_settings = settings
        if settings.api_key:
            self.api_key = settings.api_key
            self.base_url = settings.base_url
            self.model = settings.model_name
        return settings

    async def _close_current_client(self):
        if not self.client:
            return
        try:
            await self.client.close()
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/translators/openai.py:OpenAITranslator._close_current_client")
            pass
        finally:
            self.client = None

    async def _reset_client_for_candidate(self, endpoint, error: Exception):
        del endpoint, error
        await self._close_current_client()

    def _apply_api_endpoint(self, endpoint):
        signature = (endpoint.api_key or "", endpoint.base_url, endpoint.model_name)
        if signature == (self.api_key or "", self.base_url, self.model) and self.client:
            return
        self.api_key = endpoint.api_key
        self.base_url = endpoint.base_url
        self.model = endpoint.model_name
        if self.model not in OpenAITranslator._GLOBAL_LAST_REQUEST_TS:
            OpenAITranslator._GLOBAL_LAST_REQUEST_TS[self.model] = 0
        self._last_request_ts_key = self.model
        self._setup_client(force_recreate=True)

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
            provider_name="OpenAI",
            operation_name=operation_name,
            logger=self.logger,
            retry_attempts=self.attempts,
            on_candidate_error=self._reset_client_for_candidate,
        )

    def _setup_client(self, force_recreate: bool = False):
        """Set up the OpenAI client

        Args:
            force_recreate: whether the client is rebuilt by force (used on retries, to drop the old connection)
        """
        if force_recreate and self.client:
            # Close the old client and drop the connection
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # When the event loop is running, create a task that closes it asynchronously
                    asyncio.create_task(self.client.close())
                else:
                    # Otherwise close it synchronously
                    loop.run_until_complete(self.client.close())
            except Exception as e:
                self.logger.debug(f"Error closing the previous client (safe to ignore): {e}")
            self.client = None

        if not self.client:
            # Always use the curl_cffi client (no fallback to the standard SDK)
            self.client = AsyncOpenAICurlCffi(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=BROWSER_HEADERS,
                impersonate="chrome",
                timeout=600.0,
                stream_timeout=300.0
            )
            self.logger.debug("Created a new OpenAI client connection (forced curl_cffi mode)")
    
    async def _cleanup(self):
        """Release the resources"""
        if self.client:
            try:
                await self.client.close()
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/translators/openai.py:OpenAITranslator._cleanup")
                pass  # Ignore errors during clean-up

    async def _abort_inflight_request(self):
        """On cancellation, break the connection of the current request, to avoid blocking for a long time."""
        if not self.client:
            return
        try:
            await self.client.close()
        except Exception as e:
            self.logger.debug(f"Failed to close the client while interrupting a request (safe to ignore): {e}")
        finally:
            self.client = None
    
    def __del__(self):
        """Destructor; makes sure the resources are released"""
        if self.client:
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running() and not loop.is_closed():
                    # When the event loop is not closed, clean up synchronously
                    loop.run_until_complete(self._cleanup())
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/translators/openai.py:OpenAITranslator.__del__")
                pass  # Ignore all clean-up errors

    def _build_user_prompt(self, texts: List[str], ctx: Any, retry_attempt: int = 0, retry_reason: str = "") -> str:
        """Build the user prompt (text-only version) - in JSON format, to go with the HQ prompt"""
        return self._build_user_prompt_for_texts(texts, ctx, "", retry_attempt=retry_attempt, retry_reason=retry_reason)

    def _get_system_prompt(self, source_lang: str, target_lang: str, custom_prompt_json: Dict[str, Any] = None, line_break_prompt_json: Dict[str, Any] = None, retry_attempt: int = 0, retry_reason: str = "", extract_glossary: bool = False) -> str:
        """Get the complete system prompt"""
        return self._build_system_prompt(source_lang, target_lang, custom_prompt_json=custom_prompt_json, line_break_prompt_json=line_break_prompt_json, retry_attempt=retry_attempt, retry_reason=retry_reason, extract_glossary=extract_glossary)

    async def _translate_batch(self, texts: List[str], source_lang: str, target_lang: str, custom_prompt_json: Dict[str, Any] = None, line_break_prompt_json: Dict[str, Any] = None, ctx: Any = None, split_level: int = 0) -> List[str]:
        """Batch translation method (text only)"""
        if not texts:
            return []
        
        if not self.client:
            self._setup_client()
        
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

        while is_infinite or attempt < max_retries:
            # Check whether the task was cancelled
            self._check_cancelled()
            
            # Check the global attempt count
            if not self._increment_global_attempt():
                self.logger.error("Reached global attempt limit. Stopping translation.")
                # Include the real cause of the last error
                last_error_msg = str(last_exception) if last_exception else "Unknown error"
                raise Exception(f"Maximum attempts reached ({self._max_total_attempts}). Last error: {last_error_msg}")

            local_attempt += 1
            attempt += 1
            
            # Decide whether glossary extraction is on
            config_extract = False
            if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'translator'):
                config_extract = getattr(ctx.config.translator, 'extract_glossary', False)
            
            extract_glossary = bool(_custom_prompt_json) and config_extract

            # Build the system and user prompts (with the retry information, to avoid cached answers)
            system_prompt = self._get_system_prompt(_source_lang, _target_lang, custom_prompt_json=_custom_prompt_json, line_break_prompt_json=_line_break_prompt_json, retry_attempt=retry_attempt, retry_reason=retry_reason, extract_glossary=extract_glossary)
            user_prompt = self._build_user_prompt(texts, ctx, retry_attempt=retry_attempt, retry_reason=retry_reason)
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self._build_openai_context_messages(self.prev_context))
            messages.append({"role": "user", "content": user_prompt})

            try:
                # RPM limit
                if self._MAX_REQUESTS_PER_MINUTE > 0:
                    import time
                    now = time.time()
                    delay = 60.0 / self._MAX_REQUESTS_PER_MINUTE
                    elapsed = now - OpenAITranslator._GLOBAL_LAST_REQUEST_TS[self._last_request_ts_key]
                    if elapsed < delay:
                        sleep_time = delay - elapsed
                        self.logger.info(f'Ratelimit sleep: {sleep_time:.2f}s')
                        await self._sleep_with_cancel_polling(sleep_time)
                
                # Build the API parameters; max_tokens is only passed when it has a value (newer models such as o1/gpt-4.1 reject null)
                api_params = {
                    "model": self.model,
                    "messages": messages,
                }
                if self.max_tokens is not None:
                    api_params["max_tokens"] = self.max_tokens
                
                def _extract_openai_stream_text(chunk):
                    if not (hasattr(chunk, 'choices') and chunk.choices):
                        return ""
                    choice = chunk.choices[0]
                    delta = getattr(choice, 'delta', None)
                    return getattr(delta, 'content', '') if delta else ""

                def _extract_openai_stream_finish_reason(chunk):
                    if not (hasattr(chunk, 'choices') and chunk.choices):
                        return None
                    return getattr(chunk.choices[0], 'finish_reason', None)
                
                def _on_stream_chunk(delta_text, _full_text):
                    self._emit_stream_json_preview("[OpenAI Stream]", delta_text, source_texts=texts)

                streamed_text = None
                streamed_finish_reason = None
                response = None
                use_streaming = self._is_streaming_enabled(ctx)

                async def _send_openai_request():
                    nonlocal response, streamed_text, streamed_finish_reason
                    response = None
                    streamed_text = None
                    streamed_finish_reason = None
                    custom_api_params = self._resolve_translator_custom_api_params(self.model)
                    request_params = merge_openai_chat_request_params(
                        {**api_params, "model": self.model},
                        custom_api_params,
                    )
                    if custom_api_params:
                        self.logger.debug(f"Using translation model preset parameters: {custom_api_params}")
                    if use_streaming:
                        try:
                            self._reset_stream_json_preview()
                            stream_params = dict(request_params)
                            stream_params["stream"] = True
                            streamed_text, streamed_finish_reason = await self._run_unified_stream_transport(
                                create_stream=lambda: self.client.chat.completions.create(**stream_params),
                                extract_text=_extract_openai_stream_text,
                                extract_finish_reason=_extract_openai_stream_finish_reason,
                                on_chunk=_on_stream_chunk,
                                on_cancel=self._abort_inflight_request,
                                poll_interval=0.2,
                                sync_iter_in_thread=False,
                            )
                            self._finish_stream_inline()
                        except Exception as stream_error:
                            self._finish_stream_inline()
                            streamed_text = None
                            streamed_finish_reason = None
                            self.logger.warning(f"Streaming request unavailable; fell back to a non-streaming request: {stream_error}")
                            response = await self._await_with_cancel_polling(
                                self.client.chat.completions.create(**request_params),
                                poll_interval=0.2,
                                on_cancel=self._abort_inflight_request,
                            )
                    else:
                        self.logger.info("Streaming is disabled; using a non-streaming request.")
                        response = await self._await_with_cancel_polling(
                            self.client.chat.completions.create(**request_params),
                            poll_interval=0.2,
                            on_cancel=self._abort_inflight_request,
                        )

                    if streamed_text is not None:
                        if not streamed_text.strip():
                            raise RuntimeError("OpenAI returned empty content")
                    else:
                        validate_openai_response(response, self.logger)
                        has_response_content = bool(
                            getattr(response, "choices", None)
                            and response.choices[0].message.content
                        )
                        if not has_response_content:
                            raise RuntimeError("OpenAI returned empty content")

                await self._run_with_api_rotation(_send_openai_request, "translation request")

                # Update the timestamp right after a successful API call, so every request (retries included) counts towards the rate limit
                if self._MAX_REQUESTS_PER_MINUTE > 0:
                    OpenAITranslator._GLOBAL_LAST_REQUEST_TS[self._last_request_ts_key] = time.time()

                if streamed_text is not None:
                    finish_reason = streamed_finish_reason
                    has_content = bool(streamed_text)
                else:
                    # Check that the response object is valid
                    validate_openai_response(response, self.logger)
                    # Success condition: any content is processed; quality checks follow later
                    finish_reason = response.choices[0].finish_reason if (hasattr(response, 'choices') and response.choices) else None
                    has_content = response.choices and response.choices[0].message.content
                 
                if has_content:
                    result_text = streamed_text.strip() if streamed_text is not None else response.choices[0].message.content.strip()
                    
                    # Shared encoding clean-up (handles UTF-16-LE and similar problems)
                    from .common import sanitize_text_encoding
                    result_text = sanitize_text_encoding(result_text)
                    
                    self.logger.debug(f"--- OpenAI Raw Response ---\n{result_text}\n---------------------------")
                    
                    # Remove <think>...</think> tags and their content (the reasoning of local models such as LM Studio)
                    result_text = re.sub(r'(</think>)?<think>.*?</think>', '', result_text, flags=re.DOTALL)
                    # Take the content of <answer>...</answer> (when present)
                    answer_match = re.search(r'<answer>(.*?)</answer>', result_text, flags=re.DOTALL)
                    if answer_match:
                        result_text = answer_match.group(1).strip()
                    
                    # Extra clean-up step: remove a Markdown code block, if any
                    if result_text.startswith("```") and result_text.endswith("```"):
                         # The regular expression here is safer than plain slicing
                         code_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', result_text, re.DOTALL)
                         if code_match:
                             result_text = code_match.group(1).strip()
                         elif result_text.startswith("```"): # Simple fallback
                             result_text = result_text.strip("`").strip()
                    
                    # Parse the response with the shared function (JSON and plain text)
                    translations, new_terms = parse_hq_response(result_text)
                    
                    # Handle the extracted glossary terms
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

                        # Drop the connection and rebuild the client before retrying
                        self.logger.info("Closing the previous connection and rebuilding the client before retrying...")
                        self._setup_client(force_recreate=True)
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

                        # Drop the connection and rebuild the client before retrying
                        self.logger.info("Closing the previous connection and rebuilding the client before retrying...")
                        self._setup_client(force_recreate=True)
                        await self._sleep_with_cancel_polling(2)
                        continue

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
                            self.logger.error("OpenAI translation still failed after multiple retries: AI line break validation failed.")
                            raise BRMarkersValidationException(
                                missing_count=0,  # The exact numbers were logged in _validate_br_markers
                                total_count=len(texts),
                                tolerance=max(1, len(texts) // 10)
                            )
                        
                        # Drop the connection and rebuild the client before retrying
                        self.logger.info("Closing the previous connection and rebuilding the client before retrying...")
                        self._setup_client(force_recreate=True)
                        await self._sleep_with_cancel_polling(2)
                        continue

                    return translations[:len(texts)]
                
                # When it did not succeed, record the reason and prepare to retry
                retry_attempt += 1
                log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                
                # finish_reason was read above; handle each case
                if finish_reason == 'content_filter':
                    retry_reason = "Content filter triggered"
                    self.logger.warning(f"OpenAI content blocked by safety policy ({log_attempt}). Retrying...")
                    last_exception = Exception("OpenAI content filter triggered")
                elif finish_reason == 'length':
                    retry_reason = "Response truncated due to length limit"
                    self.logger.warning(f"OpenAI response truncated (token limit reached) ({log_attempt}). Retrying...")
                    last_exception = Exception("OpenAI response truncated due to length limit")
                elif finish_reason == 'tool_calls':
                    retry_reason = "Tool calls instead of translation"
                    self.logger.warning(f"OpenAI attempted to call a tool instead of returning a translation ({log_attempt}). Retrying...")
                    last_exception = Exception("OpenAI attempted tool calls instead of translation")
                elif not has_content:
                    retry_reason = f"Empty content (finish_reason: {finish_reason})"
                    self.logger.warning(f"OpenAI returned empty content (finish_reason: '{finish_reason}') ({log_attempt}). Retrying...")
                    last_exception = Exception(f"OpenAI returned empty content (finish_reason: {finish_reason})")
                else:
                    retry_reason = f"Unexpected finish_reason: {finish_reason}"
                    self.logger.warning(f"OpenAI returned unexpected finish reason '{finish_reason}' ({log_attempt}). Retrying...")
                    last_exception = Exception(f"OpenAI returned unexpected finish_reason: {finish_reason}")

                if not is_infinite and attempt >= max_retries:
                    self.logger.error("OpenAI translation still failed after multiple retries. Terminating.")
                    raise last_exception
                
                # Drop the connection and rebuild the client before retrying
                self.logger.info("Closing the previous connection and rebuilding the client before retrying...")
                self._setup_client(force_recreate=True)
                await self._sleep_with_cancel_polling(1)

            except APIRotationExhaustedError:
                raise
            except Exception as e:
                log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                last_exception = e
                self.logger.warning(f"OpenAI translation failed ({log_attempt}): {e}")
                
                if not is_infinite and attempt >= max_retries:
                    self.logger.error("OpenAI translation still failed after multiple retries. Terminating.")
                    raise last_exception
                
                # Drop the connection and rebuild the client before retrying
                self.logger.info("Closing the previous connection and rebuilding the client before retrying...")
                self._setup_client(force_recreate=True)
                await self._sleep_with_cancel_polling(1)

        # Reached only after every retry has failed
        raise last_exception if last_exception else Exception("OpenAI translation failed after all retries")

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str], ctx=None) -> List[str]:
        """Main translation method"""
        if not queries:
            return []

        # Reset the global attempt counter
        self._reset_global_attempt_count()

        self.logger.info(f"Using OpenAI text-only translation for {len(queries)} texts; maximum attempts: {self._max_total_attempts}")
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
