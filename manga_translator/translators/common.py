import asyncio
import contextlib
import inspect
import json
import re
import shutil
import sys
import textwrap
import time
from abc import abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils import InfererModule, ModelWrapper, is_valuable_text, repeating_sequence
from ..utils.curl_cffi_transport import (
    CURL_CFFI_IMPERSONATE,
    create_curl_cffi_async_session,
    validate_api_key_for_http_header,
)
from ..utils.image_modes import normalize_rgb_image
from ..utils.log import get_logger
from ..utils.log_redaction import safe_host_for_log
from ..utils.openai_compat import resolve_openai_compatible_api_key
from ..utils.retry import (
    get_retry_attempts_from_config,
    normalize_retry_attempts,
    resolve_total_attempts,
    summarize_response_text,
)
from ..utils.system_proxy import system_proxy_request_kwargs
from manga_translator.utils.swallowed import note_ignored_error

_http_logger = get_logger('translator')


try:
    import readline
except Exception as ignored_error:
    note_ignored_error(ignored_error, "manga_translator/translators/common.py:<module>")
    readline = None

VALID_LANGUAGES = {
    'CHS': 'Chinese (Simplified)',
    'CHT': 'Chinese (Traditional)',
    'CSY': 'Czech',
    'NLD': 'Dutch',
    'ENG': 'English',
    'FRA': 'French',
    'DEU': 'German',
    'HUN': 'Hungarian',
    'ITA': 'Italian',
    'JPN': 'Japanese',
    'KOR': 'Korean',
    'POL': 'Polish',
    'PTB': 'Portuguese (Brazil)',
    'ROM': 'Romanian',
    'RUS': 'Russian',
    'ESP': 'Spanish',
    'TRK': 'Turkish',
    'UKR': 'Ukrainian',
    'VIN': 'Vietnamese',
    'ARA': 'Arabic',
    'PER': 'Persian',
    'CNR': 'Montenegrin',
    'SRP': 'Serbian',
    'HRV': 'Croatian',
    'THA': 'Thai',
    'IND': 'Indonesian',
    'FIL': 'Filipino (Tagalog)'
}

KEEP_LANGUAGES = {
    **VALID_LANGUAGES,
    'SWE': 'Swedish',
    'DAN': 'Danish',
    'NOR': 'Norwegian',
    'FIN': 'Finnish',
    'MSA': 'Malay',
    'CAT': 'Catalan',
}

ISO_639_1_TO_VALID_LANGUAGES = {
    'zh': 'CHS',
    'ja': 'JPN',
    'en': 'ENG',
    'ko': 'KOR',
    'vi': 'VIN',
    'cs': 'CSY',
    'nl': 'NLD',
    'fr': 'FRA',
    'de': 'DEU',
    'hu': 'HUN',
    'it': 'ITA',
    'pl': 'POL',
    'pt': 'PTB',
    'ro': 'ROM',
    'ru': 'RUS',
    'es': 'ESP',
    'tr': 'TRK',
    'uk': 'UKR',
    'ar': 'ARA',
    'fa': 'PER',
    'cnr': 'CNR',
    'sr': 'SRP',
    'hr': 'HRV',
    'th': 'THA',
    'id': 'IND',
    'tl': 'FIL'
}

# Keep Arabic-derived scripts in logical Unicode order. Qt handles shaping and
# bidi during rendering; stored translations must retain controls such as ZWNJ.
RTL_LANGUAGES = frozenset(('ARA', 'PER'))

ISO_639_1_TO_KEEP_LANGUAGES = {
    **ISO_639_1_TO_VALID_LANGUAGES,
    'sv': 'SWE',
    'da': 'DAN',
    'no': 'NOR',
    'nb': 'NOR',
    'nn': 'NOR',
    'fi': 'FIN',
    'ms': 'MSA',
    'ca': 'CAT',
}

_BR_EDGE_WHITESPACE_RE = re.compile(
    r"[^\S\r\n]*(\[BR\]|【BR】|<br\s*/?>)[^\S\r\n]*",
    re.IGNORECASE,
)


class InvalidServerResponse(Exception):
    pass


def _extract_http_error_details(response) -> str:
    raw_text = summarize_response_text(
        getattr(response, "text", ""),
        empty_placeholder="(empty response)",
    )
    try:
        payload = response.json()
    except Exception as ignored_error:
        note_ignored_error(ignored_error, "manga_translator/translators/common.py:_extract_http_error_details")
        return raw_text

    if not isinstance(payload, dict):
        return raw_text

    details: List[str] = []
    error_obj = payload.get("error")
    if isinstance(error_obj, dict):
        for key in ("code", "type", "param"):
            value = error_obj.get(key)
            if value not in (None, ""):
                details.append(f"{key}={value}")
        message = error_obj.get("message")
        if message:
            details.append(str(message))
    else:
        for key in ("code", "type"):
            value = payload.get(key)
            if value not in (None, ""):
                details.append(f"{key}={value}")
        for key in ("message", "msg", "detail"):
            value = payload.get(key)
            if value:
                details.append(str(value))
                break
        data_value = payload.get("data")
        if data_value not in (None, "", [], {}):
            try:
                data_text = json.dumps(data_value, ensure_ascii=False)
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/translators/common.py:_extract_http_error_details")
                data_text = str(data_value)
            details.append(f"data={summarize_response_text(data_text, limit=400)}")

    try:
        raw_json = json.dumps(payload, ensure_ascii=False)
    except Exception as ignored_error:
        note_ignored_error(ignored_error, "manga_translator/translators/common.py:_extract_http_error_details")
        raw_json = raw_text

    if details:
        return f"{'; '.join(details)} | raw={summarize_response_text(raw_json, limit=800)}"
    return summarize_response_text(raw_json, limit=800, empty_placeholder="(empty response)")


def _response_diagnostics(response, *, limit: int = 1200) -> str:
    """Return safe, bounded HTTP response details for parse errors."""
    headers = getattr(response, "headers", {}) or {}
    content_type = ""
    try:
        content_type = headers.get("content-type", "") or headers.get("Content-Type", "")
    except Exception as ignored_error:
        note_ignored_error(ignored_error, "manga_translator/translators/common.py:_response_diagnostics")
        content_type = ""
    status_code = getattr(response, "status_code", "unknown")
    raw_text = summarize_response_text(
        getattr(response, "text", ""),
        limit=limit,
        empty_placeholder="(empty response)",
    )
    return f"HTTP {status_code}, Content-Type: {content_type or '(missing)'}, raw response: {raw_text}"


class LanguageUnsupportedException(Exception):
    def __init__(self, language_code: str, translator: str = None, supported_languages: List[str] = None):
        error = 'Language not supported for %s: "%s"' % (translator if translator else 'chosen translator', language_code)
        if supported_languages:
            error += '. Supported languages: "%s"' % ','.join(supported_languages)
        super().__init__(error)

class BRMarkersValidationException(Exception):
    """AI断句检查失败异常"""
    def __init__(self, missing_count: int, total_count: int, tolerance: int):
        self.missing_count = missing_count
        self.total_count = total_count
        self.tolerance = tolerance
        super().__init__(
            f"AI line break validation failed: BR markers missing in {missing_count}/{total_count} translations (tolerance: {tolerance})"
        )




# ============================================================================
# AsyncOpenAI client wrapper - uses curl_cffi to get past TLS fingerprint checks
# ============================================================================

class AsyncOpenAICurlCffi:
    """
    异步 OpenAI 客户端包装器，使用 curl_cffi 绕过 TLS 指纹检测
    完全兼容 AsyncOpenAI 的接口，可直接替换使用

    用法:
        client = AsyncOpenAICurlCffi(
            api_key="your-api-key",
            base_url="https://api.openai.com/v1",
            default_headers=OPENAI_CURL_HEADERS,
            impersonate="chrome",
        )

        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """

    class ChatCompletions:
        """聊天完成接口"""

        def __init__(self, parent):
            self.parent = parent

        async def create(self, model, messages, temperature=None, max_tokens=None, **kwargs):
            """创建聊天完成请求"""
            url = f"{self.parent.base_url}/chat/completions"

            headers = {
                "Content-Type": "application/json"
            }
            if self.parent.api_key:
                headers["Authorization"] = f"Bearer {self.parent.api_key}"

            # Merge in the default request headers
            if self.parent.default_headers:
                headers.update(self.parent.default_headers)

            # Build the request data
            data = {
                "model": model,
                "messages": messages
            }

            if temperature is not None:
                data["temperature"] = temperature
            if max_tokens is not None:
                data["max_tokens"] = max_tokens

            # Add the other parameters
            data.update(kwargs)

            stream_mode = bool(data.get("stream"))
            # Some OpenAI-compatible sites wrongly default to SSE when the stream parameter is missing.
            # Ordinary requests pass false explicitly, so the response is non-streaming JSON.
            data["stream"] = stream_mode
            if stream_mode:
                return self._create_stream(url, data, headers)

            # Send the request asynchronously
            response = await self.parent.session.post(
                url,
                json=data,
                headers=headers,
                timeout=self.parent.timeout,
                **system_proxy_request_kwargs(url),
            )

            if response.status_code != 200:
                # Only the status and the host are logged: a URL or a response body can carry a credential.
                _http_logger.error(f"[AsyncOpenAICurlCffi] Error - Status: {response.status_code} Host: {safe_host_for_log(url)}")
                error_msg = (
                    f"API request failed with status {response.status_code}: "
                    f"{_extract_http_error_details(response)}"
                )
                raise Exception(error_msg)

            try:
                result = response.json()
            except Exception as e:
                raise Exception(
                    f"Failed to parse the API JSON response: {e}. {_response_diagnostics(response)}"
                ) from e

            # Convert to a response object like the one of the OpenAI SDK
            return _OpenAIResponse(result)

        def _create_stream(self, url, data, headers):
            """SSE 流式请求，返回异步可迭代对象。"""

            async def _gen():
                async with self.parent.session.stream(
                    "POST",
                    url,
                    json=data,
                    headers=headers,
                    timeout=self.parent.stream_timeout,
                    **system_proxy_request_kwargs(url),
                ) as response:
                    if response.status_code != 200:
                        text = await response.atext()
                        raise Exception(
                            f"API request failed with status {response.status_code}: "
                            f"{summarize_response_text(text)}"
                        )

                    async for raw_line in response.aiter_lines():
                        if isinstance(raw_line, (bytes, bytearray)):
                            raw_line = raw_line.decode("utf-8", errors="ignore")
                        line = str(raw_line or "").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(payload)
                        except Exception as ignored_error:
                            note_ignored_error(ignored_error, "manga_translator/translators/common.py:AsyncOpenAICurlCffi.ChatCompletions._create_stream._gen")
                            continue
                        yield _OpenAIStreamChunk(chunk_data)

            return _gen()

    class Chat:
        """聊天接口"""

        def __init__(self, parent):
            self.completions = AsyncOpenAICurlCffi.ChatCompletions(parent)

    class Models:
        """模型列表接口"""

        def __init__(self, parent):
            self.parent = parent

        async def list(self):
            """获取可用模型列表"""
            url = f"{self.parent.base_url}/models"

            headers = {
                "Content-Type": "application/json"
            }
            if self.parent.api_key:
                headers["Authorization"] = f"Bearer {self.parent.api_key}"

            # Merge in the default request headers
            if self.parent.default_headers:
                headers.update(self.parent.default_headers)

            # Send the request asynchronously
            response = await self.parent.session.get(
                url,
                headers=headers,
                timeout=self.parent.timeout,
                **system_proxy_request_kwargs(url),
            )

            if response.status_code != 200:
                # Only the status and the host are logged: a URL or a response body can carry a credential.
                _http_logger.error(f"[AsyncOpenAICurlCffi] List Error - Status: {response.status_code} Host: {safe_host_for_log(url)}")
                error_msg = (
                    f"API request failed with status {response.status_code}: "
                    f"{_extract_http_error_details(response)}"
                )
                raise Exception(error_msg)

            # Check the content type of the response
            content_type = response.headers.get('content-type', '')
            if 'application/json' not in content_type and 'text/json' not in content_type:
                # An HTML page may have come back, which means the API has no /models endpoint
                raise Exception(
                    "API does not support model listing (returned a non-JSON response). "
                    f"{_response_diagnostics(response)}. Enter the model name manually."
                )

            try:
                result = response.json()
            except Exception as e:
                raise Exception(
                    f"Failed to parse the API response: {str(e)}. "
                    f"{_response_diagnostics(response)}. Enter the model name manually."
                ) from e

            # Convert to a response object like the one of the OpenAI SDK
            return _ModelsResponse(result)

    def __init__(self, api_key, base_url="https://api.openai.com/v1",
                 default_headers=None, http_client=None, impersonate=CURL_CFFI_IMPERSONATE,
                 timeout=600, stream_timeout=300):
        """
        初始化异步客户端

        Args:
            api_key: OpenAI API 密钥
            base_url: API 基础 URL
            default_headers: 默认请求头
            http_client: 忽略此参数（为了兼容性）
            impersonate: 浏览器指纹；默认跟随 curl_cffi 的最新 Chrome 指纹
            timeout: 非流式请求超时时间（秒）
            stream_timeout: 流式 HTTP 请求超时时间（秒）
        """
        self.api_key = validate_api_key_for_http_header(
            resolve_openai_compatible_api_key(api_key, base_url)
        )
        self.base_url = base_url.rstrip('/')
        self.default_headers = default_headers or {}
        self.timeout = timeout
        self.stream_timeout = stream_timeout
        self.impersonate = impersonate

        try:
            self.session = create_curl_cffi_async_session(
                base_url=base_url,
                impersonate=impersonate,
            )
        except ImportError:
            raise ImportError(
                "curl_cffi is required for TLS fingerprint bypass. "
                "Install it with: pip install curl_cffi"
            )

        # Create the chat interface
        self.chat = self.Chat(self)
        # Create the model list interface
        self.models = self.Models(self)

    async def close(self):
        """关闭 session"""
        if hasattr(self.session, 'close'):
            await self.session.close()

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()


class _OpenAIResponse:
    """模拟 OpenAI SDK 的响应对象"""

    class Choice:
        class Message:
            def __init__(self, content):
                self.content = content

        def __init__(self, message_content, finish_reason):
            self.message = self.Message(message_content)
            self.finish_reason = finish_reason

    class Usage:
        def __init__(self, total_tokens, prompt_tokens=0, completion_tokens=0):
            self.total_tokens = total_tokens
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    def __init__(self, data):
        self.model = data.get('model', '')
        self.choices = [
            self.Choice(
                choice['message']['content'],
                choice.get('finish_reason', 'stop')
            )
            for choice in data.get('choices', [])
        ]
        usage_data = data.get('usage', {})
        self.usage = self.Usage(
            usage_data.get('total_tokens', 0),
            usage_data.get('prompt_tokens', 0),
            usage_data.get('completion_tokens', 0)
        )


class _OpenAIStreamChunk:
    """模拟 OpenAI SDK stream chunk 对象（最小字段集）"""

    class Choice:
        class Delta:
            def __init__(self, content):
                self.content = content

        def __init__(self, delta_content, finish_reason):
            self.delta = self.Delta(delta_content)
            self.finish_reason = finish_reason

    def __init__(self, data):
        choices = data.get("choices", []) if isinstance(data, dict) else []
        self.choices = []
        for c in choices:
            delta = c.get("delta", {}) if isinstance(c, dict) else {}
            delta_content = delta.get("content", "") if isinstance(delta, dict) else ""
            finish_reason = c.get("finish_reason") if isinstance(c, dict) else None
            self.choices.append(self.Choice(delta_content, finish_reason))


class _ModelsResponse:
    """模拟 OpenAI SDK 的模型列表响应对象"""

    class Model:
        def __init__(self, model_data):
            self.id = model_data.get('id', '')
            self.object = model_data.get('object', 'model')
            self.created = model_data.get('created', 0)
            self.owned_by = model_data.get('owned_by', '')

    def __init__(self, data):
        self.data = [
            self.Model(model_data)
            for model_data in data.get('data', [])
        ]
        self.object = data.get('object', 'list')


# ============================================================================
# AsyncGemini client wrapper - uses curl_cffi to get past TLS fingerprint checks
# ============================================================================

class AsyncGeminiCurlCffi:
    """
    异步 Gemini 客户端包装器，使用 curl_cffi 绕过 TLS 指纹检测
    兼容 Google genai SDK 的接口

    用法:
        client = AsyncGeminiCurlCffi(
            api_key="your-api-key",
            base_url="https://generativelanguage.googleapis.com",
            default_headers=GEMINI_CURL_HEADERS,
            impersonate="chrome",
        )

        response = await client.models.generate_content(
            model="gemini-1.5-flash",
            contents="Hello"
        )
    """

    class Models:
        """模型接口"""

        def __init__(self, parent):
            self.parent = parent

        async def generate_content(self, model, contents, generation_config=None, safety_settings=None, **kwargs):
            """生成内容请求"""
            # URL-encode the model name, for names that contain "/" (such as z-ai/glm4.7)
            import urllib.parse
            encoded_model = urllib.parse.quote(model, safe='')

            # Build the URL - Gemini API format
            url = f"{self.parent.base_url}/v1beta/models/{encoded_model}:generateContent"

            # The actual request uses the full API key
            request_headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.parent.api_key
            }

            # Merge in the default request headers
            if self.parent.default_headers:
                request_headers.update(self.parent.default_headers)

            # Build the request data
            data = {}

            def _normalize_system_instruction(value):
                if not value:
                    return None
                if isinstance(value, str):
                    return {"parts": [{"text": value}]}
                if isinstance(value, dict):
                    if "parts" in value:
                        return value
                    if "text" in value:
                        return {"parts": [{"text": str(value["text"])}]}
                    return {"parts": [{"text": json.dumps(value, ensure_ascii=False)}]}
                if isinstance(value, list):
                    parts = []
                    for item in value:
                        if isinstance(item, dict):
                            if any(k in item for k in ("text", "inlineData", "inline_data", "fileData", "file_data")):
                                parts.append(item)
                            else:
                                parts.append({"text": json.dumps(item, ensure_ascii=False)})
                        else:
                            parts.append({"text": str(item)})
                    return {"parts": parts}
                if hasattr(value, "model_dump"):
                    dumped = value.model_dump(mode="json", by_alias=True, exclude_none=True)
                    if isinstance(dumped, dict) and "parts" in dumped:
                        return dumped
                    return {"parts": [{"text": json.dumps(dumped, ensure_ascii=False)}]}
                return {"parts": [{"text": str(value)}]}

            # Handle the contents parameter
            if isinstance(contents, str):
                data["contents"] = [{"role": "user", "parts": [{"text": contents}]}]
            elif isinstance(contents, list):
                # For a list, check whether each item has a role field and add one when it is missing
                processed_contents = []
                for item in contents:
                    if isinstance(item, dict):
                        if "role" not in item:
                            # Add the default role
                            item = {"role": "user", **item}
                        processed_contents.append(item)
                    else:
                        processed_contents.append({"role": "user", "parts": [{"text": str(item)}]})
                data["contents"] = processed_contents
            else:
                data["contents"] = [{"role": "user", "parts": [{"text": str(contents)}]}]

            # Add the generation settings
            if generation_config:
                config_dict = {}
                if hasattr(generation_config, 'temperature'):
                    config_dict['temperature'] = generation_config.temperature
                if hasattr(generation_config, 'top_p'):
                    config_dict['topP'] = generation_config.top_p
                if hasattr(generation_config, 'top_k'):
                    config_dict['topK'] = generation_config.top_k
                if hasattr(generation_config, 'max_output_tokens'):
                    config_dict['maxOutputTokens'] = generation_config.max_output_tokens
                if config_dict:
                    data["generationConfig"] = config_dict

                system_instruction = _normalize_system_instruction(
                    getattr(generation_config, 'system_instruction', None)
                )
                if system_instruction:
                    data["systemInstruction"] = system_instruction

            kw_system_instruction = _normalize_system_instruction(kwargs.pop("system_instruction", None))
            if kw_system_instruction:
                data["systemInstruction"] = kw_system_instruction

            # Add the safety settings
            if safety_settings:
                safety_list = []
                for setting in safety_settings:
                    if hasattr(setting, 'category') and hasattr(setting, 'threshold'):
                        # Take the name of the enum value, without the class name prefix
                        # For example: "HarmCategory.HARM_CATEGORY_HARASSMENT" -> "HARM_CATEGORY_HARASSMENT"
                        # For example: "HarmBlockThreshold.OFF" -> "OFF"
                        category_str = str(setting.category)
                        threshold_str = str(setting.threshold)

                        # Remove the enum class name prefix
                        if '.' in category_str:
                            category_str = category_str.split('.')[-1]
                        if '.' in threshold_str:
                            threshold_str = threshold_str.split('.')[-1]

                        safety_list.append({
                            "category": category_str,
                            "threshold": threshold_str
                        })
                if safety_list:
                    data["safetySettings"] = safety_list

            # Add the other parameters
            data.update(kwargs)

            stream_mode = bool(data.pop("stream", False))
            if stream_mode:
                return self._generate_content_stream(url, data, request_headers)

            # Send the request asynchronously
            response = await self.parent.session.post(
                url,
                json=data,
                headers=request_headers,
                timeout=self.parent.timeout,
                **system_proxy_request_kwargs(url),
            )

            if response.status_code != 200:
                # Only the status and the host are logged: a URL or a response body can carry a credential.
                _http_logger.error(f"[AsyncGeminiCurlCffi] Error - Status: {response.status_code} Host: {safe_host_for_log(url)}")
                error_msg = f"Gemini API request failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = f"{error_msg}: {error_data['error'].get('message', '')}"
                except Exception as ignored_error:
                    note_ignored_error(ignored_error, "manga_translator/translators/common.py:AsyncGeminiCurlCffi.Models.generate_content")
                    error_msg = (
                        f"{error_msg}: "
                        f"{summarize_response_text(response.text, empty_placeholder='(empty response)')}"
                    )
                raise Exception(error_msg)

            # Check the content type and the content of the response
            content_type = response.headers.get('content-type', '')
            if 'application/json' not in content_type and 'text/json' not in content_type:
                raise Exception(
                    f"API returned a non-JSON response (Content-Type: {content_type}): "
                    f"{summarize_response_text(response.text)}"
                )

            try:
                result = response.json()
            except Exception as e:
                raise Exception(
                    f"Failed to parse the API response: {str(e)}. Response content: "
                    f"{summarize_response_text(response.text)}"
                )

            # Convert to a response object like the one of the Gemini SDK
            return _GeminiResponse(result)

        async def generate_content_stream(self, model, contents, config=None, **kwargs):
            """兼容 google-genai 的流式接口"""
            generation_config = config or kwargs.pop("generation_config", None)
            return await self.generate_content(
                model=model,
                contents=contents,
                generation_config=generation_config,
                stream=True,
                **kwargs
            )

        def _generate_content_stream(self, url, data, headers):
            async def _gen():
                stream_url = f"{url}?alt=sse" if "?" not in url else f"{url}&alt=sse"
                async with self.parent.session.stream(
                    "POST",
                    stream_url,
                    json=data,
                    headers=headers,
                    timeout=self.parent.stream_timeout,
                    **system_proxy_request_kwargs(stream_url),
                ) as response:
                    if response.status_code != 200:
                        text = await response.atext()
                        raise Exception(
                            f"Gemini API request failed with status {response.status_code}: "
                            f"{summarize_response_text(text)}"
                        )

                    async for raw_line in response.aiter_lines():
                        if isinstance(raw_line, (bytes, bytearray)):
                            raw_line = raw_line.decode("utf-8", errors="ignore")
                        line = str(raw_line or "").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        try:
                            chunk_data = json.loads(payload)
                        except Exception as ignored_error:
                            note_ignored_error(ignored_error, "manga_translator/translators/common.py:AsyncGeminiCurlCffi.Models._generate_content_stream._gen")
                            continue
                        yield _GeminiResponse(chunk_data)

            return _gen()

        async def list(self):
            """获取可用模型列表"""
            url = f"{self.parent.base_url}/v1beta/models"

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.parent.api_key
            }

            # Merge in the default request headers
            if self.parent.default_headers:
                headers.update(self.parent.default_headers)

            # Send the request asynchronously
            response = await self.parent.session.get(
                url,
                headers=headers,
                timeout=self.parent.timeout,
                **system_proxy_request_kwargs(url),
            )

            if response.status_code != 200:
                # Only the status and the host are logged: a URL or a response body can carry a credential.
                _http_logger.error(f"[AsyncGeminiCurlCffi] List Error - Status: {response.status_code} Host: {safe_host_for_log(url)}")
                error_msg = f"Gemini API request failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = f"{error_msg}: {error_data['error'].get('message', '')}"
                except Exception as ignored_error:
                    note_ignored_error(ignored_error, "manga_translator/translators/common.py:AsyncGeminiCurlCffi.Models.list")
                    error_msg = f"{error_msg}: {summarize_response_text(response.text)}"
                raise Exception(error_msg)

            # Check the content type of the response
            content_type = response.headers.get('content-type', '')
            if 'application/json' not in content_type and 'text/json' not in content_type:
                # An HTML page may have come back, which means the API has no /models endpoint
                raise Exception("API does not support model listing (returned a non-JSON response). Enter the model name manually.")

            try:
                result = response.json()
            except Exception as e:
                raise Exception(f"Failed to parse the API response: {str(e)}. Enter the model name manually.")

            # Return the model list
            return _GeminiModelsResponse(result)

    def __init__(self, api_key, base_url="https://generativelanguage.googleapis.com",
                 default_headers=None, impersonate=CURL_CFFI_IMPERSONATE,
                 timeout=600, stream_timeout=300):
        """
        初始化异步客户端

        Args:
            api_key: Gemini API 密钥
            base_url: API 基础 URL
            default_headers: 默认请求头
            impersonate: 浏览器指纹；默认跟随 curl_cffi 的最新 Chrome 指纹
            timeout: 非流式请求超时时间（秒）
            stream_timeout: 流式 HTTP 请求超时时间（秒）
        """
        self.api_key = validate_api_key_for_http_header(api_key)
        self.base_url = base_url.rstrip('/')
        self.default_headers = default_headers or {}
        self.timeout = timeout
        self.stream_timeout = stream_timeout
        self.impersonate = impersonate

        try:
            self.session = create_curl_cffi_async_session(
                base_url=base_url,
                impersonate=impersonate,
            )
        except ImportError:
            raise ImportError(
                "curl_cffi is required for TLS fingerprint bypass. "
                "Install it with: pip install curl_cffi"
            )

        # Create the models interface
        self.models = self.Models(self)

    async def close(self):
        """关闭 session"""
        if hasattr(self.session, 'close'):
            await self.session.close()

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()


class _GeminiResponse:
    """模拟 Gemini SDK 的响应对象"""

    class Candidate:
        class Content:
            class Part:
                def __init__(self, text):
                    self.text = text if isinstance(text, str) else (str(text) if text is not None else "")

            def __init__(self, parts_data):
                parts_data = parts_data or []
                self.parts = [self.Part(p.get('text', '') if p else '') for p in parts_data]

        def __init__(self, candidate_data):
            content_data = candidate_data.get('content') or {}
            parts_data = content_data.get('parts') or []
            self.content = self.Content(parts_data)
            self.finish_reason = candidate_data.get('finishReason')
            self.safety_ratings = candidate_data.get('safetyRatings') or []

    class PromptFeedback:
        def __init__(self, feedback_data):
            feedback_data = feedback_data or {}
            self.block_reason = feedback_data.get('blockReason')
            self.safety_ratings = feedback_data.get('safetyRatings') or []

    def __init__(self, data):
        self.raw = data or {}
        candidates_data = data.get('candidates') or [] if data else []
        self.candidates = [self.Candidate(c) for c in candidates_data]
        self.prompt_feedback = self.PromptFeedback(self.raw.get('promptFeedback') or {})

        # Convenience text property
        if self.candidates and self.candidates[0].content.parts:
            self._text = self.candidates[0].content.parts[0].text
        else:
            self._text = ""

    @property
    def text(self):
        return self._text


class _GeminiModelsResponse:
    """模拟 Gemini SDK 的模型列表响应对象"""

    class Model:
        def __init__(self, model_data):
            self.name = model_data.get('name', '')
            self.display_name = model_data.get('displayName', '')
            self.description = model_data.get('description', '')
            # Take the model ID from name (format: models/gemini-1.5-flash)
            if '/' in self.name:
                self.id = self.name.split('/')[-1]
            else:
                self.id = self.name

    def __init__(self, data):
        # "or []" keeps this working when models is None
        models_data = data.get('models') or [] if data else []
        self._models = [self.Model(m) for m in models_data]

    def __iter__(self):
        return iter(self._models)


def validate_openai_response(response, logger=None) -> bool:
    """
    验证OpenAI API响应对象的有效性
    
    Args:
        response: API返回的响应对象
        logger: 日志记录器（可选）
    
    Returns:
        bool: 响应是否有效
    
    Raises:
        Exception: 如果响应对象无效
    """
    # Check whether the response object has a choices attribute
    if not hasattr(response, 'choices'):
        error_msg = f"API returned an invalid response object: {type(response).__name__}, content: {str(response)[:200]}"
        if logger:
            logger.error(error_msg)
        raise Exception(f"API returned an invalid response object, type: {type(response).__name__}")
    
    return True

def validate_gemini_response(response, logger=None) -> bool:
    """
    验证Gemini API响应对象的有效性
    
    Args:
        response: API返回的响应对象
        logger: 日志记录器（可选）
    
    Returns:
        bool: 响应是否有效
    
    Raises:
        Exception: 如果响应对象无效
    """
    # Check whether the response object has a candidates attribute
    if not hasattr(response, 'candidates'):
        error_msg = f"Gemini API returned an invalid response object: {type(response).__name__}, content: {str(response)[:200]}"
        if logger:
            logger.error(error_msg)
        raise Exception(f"Gemini API returned an invalid response object, type: {type(response).__name__}")
    
    # Check for a text attribute (some error responses have none)
    if not hasattr(response, 'text'):
        diagnostics = extract_gemini_response_diagnostics(response)
        error_msg = f"Gemini API response is missing the text attribute: {format_gemini_response_diagnostics(diagnostics)}"
        if logger:
            logger.error(error_msg)
        raise Exception("Gemini API response is missing the text attribute")
    
    # text may exist but be None (for example a safety block or an empty reply); a later .strip() would crash
    if getattr(response, 'text', None) is None:
        diagnostics = extract_gemini_response_diagnostics(response)
        error_msg = f"Gemini returned empty content ({format_gemini_response_diagnostics(diagnostics)})"
        if logger:
            logger.error(error_msg)
        raise Exception(error_msg)
    
    return True


def _get_gemini_field(obj: Any, *names: str) -> Any:
    """兼容 SDK 对象 / 自定义对象 / dict 的 Gemini 字段读取。"""
    if obj is None:
        return None
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return None


def _normalize_gemini_enum(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, 'name'):
        try:
            return str(value.name)
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/translators/common.py:_normalize_gemini_enum")
            pass
    text = str(value)
    if '.' in text:
        text = text.split('.')[-1]
    return text


def _normalize_gemini_safety_ratings(ratings: Any) -> List[Dict[str, Any]]:
    normalized = []
    for item in ratings or []:
        category = _normalize_gemini_enum(_get_gemini_field(item, 'category'))
        probability = _normalize_gemini_enum(_get_gemini_field(item, 'probability'))
        severity = _normalize_gemini_enum(_get_gemini_field(item, 'severity'))
        blocked = _get_gemini_field(item, 'blocked')
        normalized.append({
            'category': category,
            'probability': probability,
            'severity': severity,
            'blocked': bool(blocked) if blocked is not None else None,
        })
    return normalized


def extract_gemini_response_diagnostics(response: Any, fallback_finish_reason: Any = None) -> Dict[str, Any]:
    """提取 Gemini 响应诊断信息，供日志与重试逻辑复用。"""
    candidate = None
    candidates = _get_gemini_field(response, 'candidates')
    if candidates:
        try:
            candidate = candidates[0]
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/translators/common.py:extract_gemini_response_diagnostics")
            candidate = None

    prompt_feedback = _get_gemini_field(response, 'prompt_feedback', 'promptFeedback')
    finish_reason = fallback_finish_reason
    if finish_reason is None:
        finish_reason = _get_gemini_field(candidate, 'finish_reason', 'finishReason')
    block_reason = _get_gemini_field(prompt_feedback, 'block_reason', 'blockReason')

    prompt_safety_ratings = _normalize_gemini_safety_ratings(
        _get_gemini_field(prompt_feedback, 'safety_ratings', 'safetyRatings')
    )
    candidate_safety_ratings = _normalize_gemini_safety_ratings(
        _get_gemini_field(candidate, 'safety_ratings', 'safetyRatings')
    )

    finish_reason_str = _normalize_gemini_enum(finish_reason)
    block_reason_str = _normalize_gemini_enum(block_reason)
    text_value = _get_gemini_field(response, 'text')

    return {
        'finish_reason': finish_reason,
        'finish_reason_str': finish_reason_str,
        'block_reason': block_reason,
        'block_reason_str': block_reason_str,
        'prompt_safety_ratings': prompt_safety_ratings,
        'candidate_safety_ratings': candidate_safety_ratings,
        'text': text_value,
    }


def _format_gemini_safety_ratings(ratings: List[Dict[str, Any]]) -> str:
    if not ratings:
        return "[]"

    parts = []
    for item in ratings:
        segments = []
        if item.get('category'):
            segments.append(item['category'])
        if item.get('probability'):
            segments.append(f"prob={item['probability']}")
        if item.get('severity'):
            segments.append(f"sev={item['severity']}")
        if item.get('blocked') is not None:
            segments.append(f"blocked={item['blocked']}")
        parts.append("(" + ", ".join(segments) + ")")
    return "[" + ", ".join(parts) + "]"


def format_gemini_response_diagnostics(diagnostics: Dict[str, Any]) -> str:
    """格式化 Gemini 诊断信息，便于日志输出。"""
    parts = [
        f"finish_reason={diagnostics.get('finish_reason_str') or 'None'}",
        f"block_reason={diagnostics.get('block_reason_str') or 'None'}",
    ]

    prompt_ratings = diagnostics.get('prompt_safety_ratings') or []
    candidate_ratings = diagnostics.get('candidate_safety_ratings') or []
    if prompt_ratings:
        parts.append(f"prompt_safety_ratings={_format_gemini_safety_ratings(prompt_ratings)}")
    if candidate_ratings:
        parts.append(f"candidate_safety_ratings={_format_gemini_safety_ratings(candidate_ratings)}")

    return ", ".join(parts)


def gemini_diagnostics_indicate_safety(diagnostics: Dict[str, Any]) -> bool:
    """根据 Gemini 响应诊断判断是否属于安全策略拦截。"""
    values = [
        (diagnostics.get('finish_reason_str') or "").upper(),
        (diagnostics.get('block_reason_str') or "").upper(),
    ]
    safety_keywords = (
        'SAFETY',
        'BLOCKLIST',
        'PROHIBITED_CONTENT',
        'SPII',
        'RECITATION',
    )
    if any(any(keyword in value for keyword in safety_keywords) for value in values):
        return True

    for rating in (diagnostics.get('prompt_safety_ratings') or []) + (diagnostics.get('candidate_safety_ratings') or []):
        if rating.get('blocked') is True:
            return True

    return False


def gemini_diagnostics_should_disable_images(diagnostics: Dict[str, Any]) -> bool:
    """根据 Gemini 诊断判断 HQ 重试时是否应去掉图片。"""
    if gemini_diagnostics_indicate_safety(diagnostics):
        return True
    finish_reason = (diagnostics.get('finish_reason_str') or "").upper()
    return 'OTHER' in finish_reason


def gemini_error_message_indicates_safety(error_message: str) -> bool:
    upper = (error_message or "").upper()
    return any(token in upper for token in (
        'SAFETY',
        'BLOCKLIST',
        'PROHIBITED_CONTENT',
        'SPII',
        'RECITATION',
        'BLOCKED=TRUE',
        'FINISH_REASON: 2',
        'FINISH_REASON=2',
        'FINISH_REASON IS 2',
    ))

def draw_text_boxes_on_image(image, text_regions: List[Any], text_order: List[int], 
                             upscaled_size: Tuple[int, int] = None):
    """
    在图片上绘制带编号的文本框
    
    Args:
        image: 原始图片 (numpy array 或 PIL Image)
        text_regions: 文本区域列表，每个区域应该有 xyxy 或 min_rect 属性
        text_order: 文本顺序列表，对应每个文本框的编号
        upscaled_size: 超分后的图片尺寸 (height, width)，用于坐标转换。如果为None则不转换
    
    Returns:
        绘制了文本框的图片（与输入类型相同）
    """
    if image is None or len(text_regions) == 0:
        return image
    
    # Convert a PIL Image to a numpy array
    from PIL import Image as PILImage
    is_pil = isinstance(image, PILImage.Image)
    if is_pil:
        canvas = np.array(normalize_rgb_image(image))
    else:
        canvas = image.copy()
    
    h, w = canvas.shape[:2]
    
    # Coordinate scale (upscaled coordinates -> original image coordinates)
    scale_x, scale_y = 1.0, 1.0
    if upscaled_size is not None:
        upscaled_h, upscaled_w = upscaled_size
        if upscaled_w > 0 and upscaled_h > 0:
            scale_x = w / upscaled_w
            scale_y = h / upscaled_h
    
    # Work out the line width
    lw = max(round(sum(canvas.shape[:2]) / 2 * 0.003), 2)
    
    # Several colours (RGB)
    colors = [
        (255, 0, 0),     # red
        (0, 255, 0),     # green
        (0, 0, 255),     # blue
        (255, 165, 0),   # orange
        (128, 0, 128),   # purple
        (0, 255, 255),   # cyan
        (255, 0, 255),   # magenta
        (255, 255, 0),   # yellow
        (0, 128, 0),     # dark green
        (128, 0, 0),     # dark red
    ]
    
    # Collect the bounds of every box first
    all_boxes = []
    for region in text_regions:
        if hasattr(region, 'xyxy'):
            x1, y1, x2, y2 = region.xyxy
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            all_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        elif hasattr(region, 'min_rect'):
            pts = region.min_rect.astype(np.float64)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            bx1, by1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            bx2, by2 = int(pts[:, 0].max()), int(pts[:, 1].max())
            all_boxes.append((bx1, by1, bx2, by2))
    
    def check_overlap(lx, ly, lw_size, lh_size, exclude_idx):
        """检查标签区域是否与其他框重叠"""
        label_rect = (lx, ly - lh_size, lx + lw_size, ly)
        for i, (bx1, by1, bx2, by2) in enumerate(all_boxes):
            if i == exclude_idx:
                continue
            # Check whether the rectangles overlap
            if not (label_rect[2] < bx1 or label_rect[0] > bx2 or label_rect[3] < by1 or label_rect[1] > by2):
                return True
        return False
    
    # Go through the text regions and draw each one
    for idx, region in enumerate(text_regions):
        if idx >= len(text_order):
            break
            
        order_num = text_order[idx]
        color = colors[idx % len(colors)]
        
        # Get the box coordinates and convert them
        # Move the border outwards, so a thick border does not cover the text
        expand = lw  # Number of pixels to move outwards (equal to the line width)
        
        if hasattr(region, 'xyxy'):
            x1, y1, x2, y2 = region.xyxy
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            # Move the border outwards
            box_x1, box_y1 = int(x1) - expand, int(y1) - expand
            box_x2, box_y2 = int(x2) + expand, int(y2) + expand
            cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), color, lw)
        elif hasattr(region, 'min_rect'):
            pts = region.min_rect.astype(np.float64)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            # Work out the centre and expand the polygon outwards
            center_x = pts[:, 0].mean()
            center_y = pts[:, 1].mean()
            for i in range(len(pts)):
                dx = pts[i, 0] - center_x
                dy = pts[i, 1] - center_y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 0:
                    pts[i, 0] += (dx / dist) * expand
                    pts[i, 1] += (dy / dist) * expand
            pts = pts.astype(np.int32)
            cv2.polylines(canvas, [pts], True, color, lw)
            box_x1, box_y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            box_x2, box_y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
        else:
            continue
        
        # Draw the number label
        label_text = str(order_num)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(lw / 2, 0.6)
        font_thickness = max(lw, 2)
        
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        margin = 3
        
        # Four candidate positions: above, below, left, right
        candidates = [
            (box_x1, box_y1 - margin),                          # above
            (box_x1, box_y2 + text_height + margin),            # below
            (box_x1 - text_width - margin, box_y1 + text_height), # left
            (box_x2 + margin, box_y1 + text_height),            # right
        ]
        
        # Pick a position that does not overlap and is inside the image
        label_x, label_y = candidates[0]  # Above by default
        for cx, cy in candidates:
            # Check whether it is inside the image
            if cx < 0 or cy - text_height < 0 or cx + text_width > w or cy > h:
                continue
            # Check whether it overlaps another box
            if not check_overlap(cx, cy, text_width, text_height, idx):
                label_x, label_y = cx, cy
                break
        
        # Final bounds check
        label_x = max(0, min(label_x, w - text_width))
        label_y = max(text_height, min(label_y, h))
        
        # Draw the number (with a black outline)
        cv2.putText(canvas, label_text, (label_x, label_y), font, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
        cv2.putText(canvas, label_text, (label_x, label_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    
    # If the input was a PIL Image, convert back to PIL
    if is_pil:
        return PILImage.fromarray(canvas)
    return canvas


class MTPEAdapter():
    async def dispatch(self, queries: List[str], translations: List[str]) -> List[str]:
        # TODO: Make it work in windows (e.g. through os.startfile)
        if not readline:
            print('MTPE is currently only supported on linux')
            return translations
        new_translations = []
        print('Running Machine Translation Post Editing (MTPE)')
        for i, (query, translation) in enumerate(zip(queries, translations)):
            print(f'\n[{i + 1}/{len(queries)}] {query}:')
            readline.set_startup_hook(lambda: readline.insert_text(translation.replace('\n', '\\n')))
            new_translation = ''
            try:
                new_translation = input(' -> ').replace('\\n', '\n')
            finally:
                readline.set_startup_hook()
            new_translations.append(new_translation)
        print()
        return new_translations


def _flatten_prompt_data(data, indent: int = 0) -> str:
    """Recursively flattens a dictionary or list into a formatted string.
    
    Used to convert custom prompt JSON/YAML data into a readable text block
    for inclusion in system prompts.
    """
    prompt_parts = []
    prefix = "  " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                prompt_parts.append(f"{prefix}- {key}:")
                prompt_parts.append(_flatten_prompt_data(value, indent + 1))
            else:
                prompt_parts.append(f"{prefix}- {key}: {value}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                prompt_parts.append(_flatten_prompt_data(item, indent + 1))
            else:
                prompt_parts.append(f"{prefix}- {item}")
    
    return "\n".join(prompt_parts)


class CommonTranslator(InfererModule):
    # Translator has to support all languages listed in here. The language codes will be resolved into
    # _LANGUAGE_CODE_MAP[lang_code] automatically if _LANGUAGE_CODE_MAP is a dict.
    # If it is a list it will simply return the language code as is.
    _LANGUAGE_CODE_MAP = {}

    # The amount of repeats upon detecting an invalid translation.
    # Use with _is_translation_invalid and _modify_invalid_translation_query.
    _INVALID_REPEAT_COUNT = 0

    # Will sleep for the rest of the minute if the request count is over this number.
    _MAX_REQUESTS_PER_MINUTE = -1

    def __init__(self):
        super().__init__()
        self.mtpe_adapter = MTPEAdapter()
        self._last_request_ts = 0
        self.enable_post_translation_check = False
        self.post_check_repetition_threshold = 5
        self.post_check_max_retry_attempts = 2
        self.attempts = -1
        self._MAX_SPLIT_ATTEMPTS = 3  # Maximum split depth
        self._SPLIT_THRESHOLD = 2  # Split after N retries
        self._global_attempt_count = 0  # Global attempt counter
        self._max_total_attempts = -1  # Global maximum number of attempts
        self._cancel_check_callback = None  # Cancel-check callback
        self._custom_api_params_config = None
        self._enable_streaming = True
        self._stream_inline_last_len = 0
        self._stream_inline_buffer = ""
        self._stream_json_seen: Dict[int, str] = {}
        self._stream_term_seen: Dict[Tuple[str, str, str], str] = {}
        self._stream_preview_buffer = ""
        self._stream_preview_scan_pos = 0
        self._stream_preview_object_starts: List[int] = []
        self._stream_preview_in_string = False
        self._stream_preview_escape = False
        self._stream_preview_tag_tail = ""
        self._stream_preview_in_think = False
        self._stream_result_header_printed = False
        self._stream_result_pairs_printed = False

    def _normalize_retry_attempts(self, attempts: Any) -> int:
        return normalize_retry_attempts(attempts, logger=self.logger, default=-1)

    def _resolve_max_total_attempts(self) -> int:
        return resolve_total_attempts(self.attempts)

    def _resolve_translator_config(self, config: Any) -> Any:
        if isinstance(config, dict):
            return config.get('translator', config)
        return getattr(config, 'translator', config)

    def _get_config_value(self, config: Any, key: str, default: Any = None) -> Any:
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _is_streaming_enabled(self, ctx: Any = None) -> bool:
        if ctx and hasattr(ctx, 'config') and ctx.config is not None:
            translator_config = self._resolve_translator_config(ctx.config)
            value = self._get_config_value(translator_config, 'enable_streaming', None)
            if value is not None:
                return bool(value)
        return bool(getattr(self, '_enable_streaming', True))
    
    def _configure_custom_api_params(self, args) -> bool:
        """Remember the runtime config; model matching happens per API request."""
        from ..custom_api_params import is_custom_api_params_enabled

        use_custom_params = is_custom_api_params_enabled(args)
        self._custom_api_params_config = args if use_custom_params else None
        return use_custom_params

    def _resolve_translator_custom_api_params(self, model_name: str | None) -> dict[str, Any]:
        from ..custom_api_params import resolve_custom_api_params

        if self._custom_api_params_config is None:
            return {}
        return resolve_custom_api_params(
            self._custom_api_params_config,
            self.logger,
            model_name=model_name,
            section="translator",
        )
    
    def set_cancel_check_callback(self, callback):
        """设置取消检查回调"""
        self._cancel_check_callback = callback
    
    def _check_cancelled(self):
        """检查任务是否被取消"""
        if self._cancel_check_callback and self._cancel_check_callback():
            raise asyncio.CancelledError("Translation cancelled by user")

    async def _await_with_cancel_polling(
        self,
        awaitable: Awaitable,
        poll_interval: float = 0.2,
        on_cancel: Optional[Callable[[], Awaitable[None] | None]] = None,
    ):
        """
        等待一个长耗时 awaitable，并定期轮询取消状态。
        在收到取消时，尝试取消内部任务并执行 on_cancel 清理回调。
        """
        task = asyncio.create_task(awaitable)
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=poll_interval)
                if done:
                    return task.result()
                self._check_cancelled()
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
                with contextlib.suppress(Exception):
                    done, _ = await asyncio.wait({task}, timeout=max(poll_interval, 0.2))
                    if not done:
                        self.logger.debug("Timed out cancelling the request task; stopping the wait")
            if on_cancel:
                try:
                    cleanup_result = on_cancel()
                    if asyncio.iscoroutine(cleanup_result):
                        await asyncio.wait_for(cleanup_result, timeout=max(poll_interval, 0.3))
                except asyncio.TimeoutError:
                    self.logger.debug("Timed out cleaning up the request during cancellation; stopping the wait")
                except Exception as cleanup_error:
                    self.logger.debug(f"Request cleanup failed during cancellation (safe to ignore): {cleanup_error}")
            raise

    async def _sleep_with_cancel_polling(self, seconds: float, poll_interval: float = 0.2):
        """可取消的 sleep，避免等待期间无法响应停止。"""
        if seconds <= 0:
            self._check_cancelled()
            return
        await self._await_with_cancel_polling(
            asyncio.sleep(seconds),
            poll_interval=min(poll_interval, max(seconds, 0.05)),
        )

    async def _run_unified_stream_transport(
        self,
        *,
        create_stream: Callable[[], Any],
        extract_text: Callable[[Any], Any],
        extract_finish_reason: Optional[Callable[[Any], Any]] = None,
        on_chunk: Optional[Callable[[str, str], None]] = None,
        on_cancel: Optional[Callable[[], Awaitable[None] | None]] = None,
        poll_interval: float = 0.2,
        sync_iter_in_thread: bool = False,
        first_chunk_timeout: float = 300.0,
        idle_timeout: float = 300.0,
    ) -> Tuple[str, Any]:
        """
        通用流式传输层：
        - OpenAI async stream（异步迭代）
        - Gemini stream（同步迭代，放入 to_thread 消费）
        返回：(聚合后的完整文本, 最后一次 finish_reason)
        """
        def _normalize_stream_piece(piece_text: str, current_text: str) -> str:
            """
            兼容“增量块/累计块/重复块”三种常见流格式，尽量只返回新增部分。
            """
            if not piece_text:
                return ""
            if not current_text:
                return piece_text

            # Standard cumulative chunk: piece = current + delta
            if piece_text.startswith(current_text):
                return piece_text[len(current_text):]

            # Rolled back or truncated chunk: piece is only a prefix of current (and not a very short token), ignore it
            if len(piece_text) >= 16 and current_text.startswith(piece_text):
                return ""

            # Some services put current in the middle of piece; take the tail after its last occurrence
            pos = piece_text.rfind(current_text)
            if pos != -1:
                return piece_text[pos + len(current_text):]

            # Plain resend: a longer fragment that current already ends with, ignore it
            if len(piece_text) >= 16 and current_text.endswith(piece_text):
                return ""

            # Partial overlap: the end of current + the start of piece
            max_overlap = min(len(piece_text), len(current_text))
            for overlap in range(max_overlap, 0, -1):
                if current_text.endswith(piece_text[:overlap]):
                    return piece_text[overlap:]

            # When the relation cannot be decided, treat it as a delta (the cautious choice)
            return piece_text

        text_parts: List[str] = []
        last_finish_reason = None

        if sync_iter_in_thread:
            def _consume_sync_stream():
                local_parts: List[str] = []
                local_finish = None
                stream_obj = create_stream()
                for chunk in stream_obj:
                    piece = extract_text(chunk)
                    if piece:
                        piece_text = str(piece)
                        current_text = ''.join(local_parts)
                        normalized_piece = _normalize_stream_piece(piece_text, current_text)
                        if normalized_piece:
                            local_parts.append(normalized_piece)
                        if on_chunk:
                            on_chunk(normalized_piece, ''.join(local_parts))
                    if extract_finish_reason:
                        finish = extract_finish_reason(chunk)
                        if finish is not None:
                            local_finish = finish
                return ''.join(local_parts), local_finish

            return await self._await_with_cancel_polling(
                asyncio.to_thread(_consume_sync_stream),
                poll_interval=poll_interval,
                on_cancel=on_cancel,
            )

        stream_obj = create_stream()
        while inspect.isawaitable(stream_obj):
            stream_obj = await self._await_with_cancel_polling(
                stream_obj,
                poll_interval=poll_interval,
                on_cancel=on_cancel,
            )

        try:
            aiter = stream_obj.__aiter__()
            got_first_chunk = False
            while True:
                chunk_timeout = first_chunk_timeout if not got_first_chunk else idle_timeout
                try:
                    chunk = await self._await_with_cancel_polling(
                        asyncio.wait_for(aiter.__anext__(), timeout=chunk_timeout),
                        poll_interval=poll_interval,
                        on_cancel=on_cancel,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as timeout_error:
                    timeout_type = "first chunk" if not got_first_chunk else "idle"
                    raise TimeoutError(f"Streaming {timeout_type} timeout ({chunk_timeout:.0f}s)") from timeout_error

                got_first_chunk = True
                self._check_cancelled()
                piece = extract_text(chunk)
                if piece:
                    piece_text = str(piece)
                    current_text = ''.join(text_parts)
                    normalized_piece = _normalize_stream_piece(piece_text, current_text)
                    if normalized_piece:
                        text_parts.append(normalized_piece)
                    if on_chunk:
                        on_chunk(normalized_piece, ''.join(text_parts))
                if extract_finish_reason:
                    finish = extract_finish_reason(chunk)
                    if finish is not None:
                        last_finish_reason = finish
        except asyncio.CancelledError:
            if on_cancel:
                cleanup_result = on_cancel()
                if asyncio.iscoroutine(cleanup_result):
                    with contextlib.suppress(Exception):
                        await cleanup_result
            raise
        finally:
            close_fn = getattr(stream_obj, "aclose", None)
            if callable(close_fn):
                with contextlib.suppress(Exception):
                    close_ret = close_fn()
                    if asyncio.iscoroutine(close_ret):
                        await close_ret

        return ''.join(text_parts), last_finish_reason

    def _get_retry_hint(self, attempt: int, reason: str = "") -> str:
        """
        生成重试提示信息，用于避免模型服务器缓存导致的重复错误
        
        Args:
            attempt: 当前尝试次数
            reason: 重试原因（可选）
            
        Returns:
            重试提示字符串
        """
        hints = [
            f"[Retry attempt #{attempt}]",
            f"[This is attempt #{attempt}, please provide a different response]",
            f"[Attempt {attempt}: Previous response had issues, please try again]",
            f"[Retry #{attempt}: Please ensure quality this time]",
            f"[Attempt {attempt} - Previous attempts failed quality check]"
        ]
        
        # Choose a hint by attempt number (cycling through them)
        base_hint = hints[(attempt - 1) % len(hints)]
        
        # When a reason was given, add it to the hint
        if reason:
            return f"{base_hint} Reason: {reason}\n\n"
        else:
            return f"{base_hint}\n\n"

    # Detailed fallback prompt for the HQ translators (when system_prompt_hq.yaml/json does not exist)
    _HQ_FALLBACK_PROMPT = """You are an expert manga translator. Your task is to accurately translate manga text from the source language into **{{{target_lang}}}**. You will be given the full manga page for context.

**CRITICAL INSTRUCTIONS (FOLLOW STRICTLY):**

1.  **TRANSLATE EVERYTHING**: Translate all text provided, including sound effects and single characters. Do not leave any line untranslated.

2.  **ACCURACY AND TONE**:
    -   Preserve the original tone, emotion, and character's voice.
    -   Ensure consistent translation of names, places, and special terms.
    -   For onomatopoeia (sound effects), provide the equivalent sound in {{{target_lang}}} or a brief description (e.g., '(rumble)', '(thud)').

3.  **ANTI-HALLUCINATION**:
    -   Do not add information that is not present in the original text.
    -   If OCR appears wrong, prefer the visible image context over broken OCR text.

---

**FINAL INSTRUCTION:** Translate the provided text regions faithfully and follow the separate output-format requirements appended below."""
    def _parse_prev_context_turns(self, prev_context: str) -> List[Dict[str, str]]:
        """解析历史上下文，只接受新的 user/assistant JSON 轮次。"""
        payload = (prev_context or "").strip()
        if not payload:
            return []

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []

        turns: List[Dict[str, str]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            user_content = item.get("user")
            assistant_content = item.get("assistant")
            if isinstance(user_content, str) and isinstance(assistant_content, str):
                turns.append({"user": user_content, "assistant": assistant_content})
        return turns

    def _build_openai_context_messages(self, prev_context: str) -> List[Dict[str, Any]]:
        """将历史上下文转换为 OpenAI 多轮消息，不附带图片。"""
        turns = self._parse_prev_context_turns(prev_context)
        if not turns:
            self.logger.info("[Context] None")
            return []

        total_chars = sum(len(turn["user"]) + len(turn["assistant"]) for turn in turns)
        self.logger.info(f"[Context] Turns: {len(turns)}, Length: {total_chars} chars")
        messages: List[Dict[str, Any]] = []
        for turn in turns:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        return messages

    def _build_gemini_context_messages(self, prev_context: str) -> List[Dict[str, Any]]:
        """将历史上下文转换为 Gemini 多轮消息，不附带图片。"""
        turns = self._parse_prev_context_turns(prev_context)
        if not turns:
            self.logger.info("[Context] None")
            return []

        total_chars = sum(len(turn["user"]) + len(turn["assistant"]) for turn in turns)
        self.logger.info(f"[Context] Turns: {len(turns)}, Length: {total_chars} chars")
        messages: List[Dict[str, Any]] = []
        for turn in turns:
            messages.append({"role": "user", "parts": [{"text": turn["user"]}]})
            messages.append({"role": "model", "parts": [{"text": turn["assistant"]}]})
        return messages

    def _build_system_prompt_prefix(
        self,
        line_break_prompt_str: str,
        custom_prompt_str: str,
        retry_attempt: int = 0,
        retry_reason: str = "",
    ) -> str:
        """构建系统提示词前缀：[重试提示] → [断句提示] → [自定义提示]"""
        prompt_prefix = ""

        if retry_attempt > 0:
            prompt_prefix += self._get_retry_hint(retry_attempt, retry_reason) + "\n"

        if line_break_prompt_str:
            prompt_prefix += f"{line_break_prompt_str}\n\n---\n\n"

        if custom_prompt_str:
            prompt_prefix += f"{custom_prompt_str}\n\n---\n\n"

        return prompt_prefix

    def _build_system_prompt_without_glossary(
        self,
        prompt_prefix: str,
        base_prompt: str,
        target_lang_full: str,
    ) -> str:
        """普通翻译模式：基础系统提示 + 标准 translations 输出格式。"""
        final_prompt = prompt_prefix + base_prompt
        output_format_prompt = get_system_prompt_hq_format_prompt(target_lang_full, extract_glossary=False)
        if output_format_prompt:
            final_prompt += "\n\n---\n\n" + output_format_prompt
        else:
            self.logger.info("Automatic term extraction is disabled, but no standard output format prompt was loaded.")
        return final_prompt

    def _build_system_prompt_with_glossary(
        self,
        prompt_prefix: str,
        base_prompt: str,
        target_lang_full: str,
    ) -> str:
        """术语提取模式：基础系统提示 + 术语提取规则 + 扩展输出格式。"""
        final_prompt = prompt_prefix + base_prompt
        extraction_prompt = get_glossary_extraction_prompt(target_lang_full)
        output_format_prompt = get_system_prompt_hq_format_prompt(target_lang_full, extract_glossary=True)
        glossary_sections = [p for p in (extraction_prompt, output_format_prompt) if p]

        if glossary_sections:
            final_prompt += "\n\n---\n\n" + "\n\n---\n\n".join(glossary_sections)
            self.logger.info("Automatic term extraction is enabled; using the system prompt with the new_terms output format.")
        else:
            self.logger.info("Automatic term extraction is enabled, but no supplemental term extraction prompt was loaded.")

        return final_prompt

    def _build_system_prompt(
        self,
        source_lang: str,
        target_lang: str,
        custom_prompt_json: dict = None,
        line_break_prompt_json: dict = None,
        retry_attempt: int = 0,
        retry_reason: str = "",
        extract_glossary: bool = False,
    ) -> str:
        """
        构建完整的系统提示词（统一实现，所有翻译器共享）。

        不开启自动术语提取时：
        [重试提示] → [断句提示] → [自定义提示] → [基础系统提示] → [标准输出格式]

        开启自动术语提取时：
        [重试提示] → [断句提示] → [自定义提示] → [基础系统提示] → [术语提取规则] → [扩展输出格式]
        """
        target_lang_full = VALID_LANGUAGES.get(target_lang, target_lang)

        # --- Custom prompt ---
        custom_prompt_str = ""
        if custom_prompt_json:
            custom_prompt_str = _flatten_prompt_data(custom_prompt_json)

        # --- Line-breaking prompt ---
        line_break_prompt_str = ""
        if line_break_prompt_json and line_break_prompt_json.get('line_break_prompt'):
            line_break_prompt_str = line_break_prompt_json['line_break_prompt']

        # --- Load the HQ system prompt (YAML preferred, JSON still accepted) ---
        import os

        from ..utils import BASE_PATH
        from .prompt_loader import load_system_prompt_hq
        dict_dir = os.path.join(BASE_PATH, 'dict')
        base_prompt = load_system_prompt_hq(dict_dir)

        # Fallback
        if not base_prompt:
            base_prompt = self._HQ_FALLBACK_PROMPT

        # --- Replace the placeholders ---
        base_prompt = base_prompt.replace("{{{target_lang}}}", target_lang_full)
        if custom_prompt_str:
            custom_prompt_str = custom_prompt_str.replace("{{{target_lang}}}", target_lang_full)

        prompt_prefix = self._build_system_prompt_prefix(
            line_break_prompt_str=line_break_prompt_str,
            custom_prompt_str=custom_prompt_str,
            retry_attempt=retry_attempt,
            retry_reason=retry_reason,
        )

        if extract_glossary:
            return self._build_system_prompt_with_glossary(
                prompt_prefix=prompt_prefix,
                base_prompt=base_prompt,
                target_lang_full=target_lang_full,
            )

        return self._build_system_prompt_without_glossary(
            prompt_prefix=prompt_prefix,
            base_prompt=base_prompt,
            target_lang_full=target_lang_full,
        )

    def _build_unified_user_prompt(self, batch_data: List[Dict], ctx=None, prev_context: str = "", retry_attempt: int = 0, retry_reason: str = "", is_image_mode: bool = True) -> str:
        """
        统一的用户提示词构建方法（支持多模态和纯文本）
        Unified user prompt builder for both multimodal and text-only modes.

        Args:
            batch_data: List of dicts, each containing 'original_texts' and optional 'text_regions'.
            ctx: Context object.
            prev_context: 保留兼容；历史上下文现在作为独立消息注入，不再拼进当前用户提示词。
            retry_attempt: Retry attempt count.
            retry_reason: Reason for retry.
            is_image_mode: Whether to include image-specific descriptions.

        Returns:
            Constructed user prompt string.
        """
        import json
        
        # Check whether AI line breaking is on
        enable_ai_break = False
        if ctx and hasattr(ctx, 'config') and ctx.config and hasattr(ctx.config, 'render'):
            enable_ai_break = getattr(ctx.config.render, 'disable_auto_wrap', False)

        prompt = ""

        # Put the retry hint first (when this is a retry)
        if retry_attempt > 0:
            prompt += self._get_retry_hint(retry_attempt, retry_reason) + "\n"

        if is_image_mode:
            prompt += "Please translate the following manga text regions. I'm providing multiple images with their text regions in reading order:\n\n"
            # Add the image information
            for i, data in enumerate(batch_data):
                prompt += f"=== Image {i+1} ===\n"
                prompt += f"Text regions ({len(data['original_texts'])} regions):\n"
                text_order = data.get('text_order', [])
                for j, text in enumerate(data['original_texts']):
                    if text is not None:
                        display_id = text_order[j] if j < len(text_order) else (j + 1)
                        prompt += f"  {display_id}. {text}\n"
                prompt += "\n"
        else:
            prompt += "Please translate the following manga text regions:\n\n"

        prompt += "All texts to translate (JSON Array):\n"
        input_data = []
        text_index = 1
        for img_idx, data in enumerate(batch_data):
            # Get text_regions for AI line breaking
            text_regions = data.get('text_regions', [])
            text_order = data.get('text_order', [])
            
            for region_idx, text in enumerate(data['original_texts']):
                # Skip None values
                if text is None:
                    self.logger.warning(f"Skipping None text (img_idx={img_idx}, region_idx={region_idx})")
                    continue
                
                # Preprocess the text: remove line breaks
                text_clean = text.replace('\n', ' ').replace('\ufffd', '')
                
                # HQ mode prefers text_order, so the numbers match the ones on the image
                item_id = text_order[region_idx] if region_idx < len(text_order) else text_index
                item = {
                    "id": item_id,
                    "text": text_clean
                }
                
                # AI line breaking: get original_region_count
                if enable_ai_break:
                    region_count = 1
                    # Try to get it from text_regions
                    if text_regions and region_idx < len(text_regions):
                        region = text_regions[region_idx]
                        if hasattr(region, 'lines') and region.lines is not None:
                            region_count = len(region.lines)
                        elif isinstance(region, dict) and 'lines' in region:
                            region_count = len(region['lines'])
                    
                    # When that fails (for example text_regions is empty in plain-text mode), fall back to counting line breaks
                    if region_count == 1 and text:
                        newline_count = text.count('\n')
                        if newline_count > 0:
                            region_count = newline_count + 1
                    
                    item["original_region_count"] = region_count
                
                input_data.append(item)
                text_index += 1

        prompt += json.dumps(input_data, ensure_ascii=False, indent=2)
        prompt += "\n\nCRITICAL: Provide translations in the exact same order as the input array. Follow the OUTPUT FORMAT specified in the System Prompt."

        return prompt

    def _build_user_prompt_for_hq(self, batch_data: List, ctx=None, prev_context: str = "", retry_attempt: int = 0, retry_reason: str = "") -> str:
        """Alias for backward compatibility (HQ mode)"""
        return self._build_unified_user_prompt(batch_data, ctx, prev_context, retry_attempt, retry_reason, is_image_mode=True)

    def _build_user_prompt_for_texts(self, texts: List[str], ctx=None, prev_context: str = "", retry_attempt: int = 0, retry_reason: str = "") -> str:
        """Alias for text mode: wraps texts into batch_data"""
        # Build a stand-in batch_data
        batch_data = [{
            'original_texts': texts,
            'text_regions': getattr(ctx, 'text_regions', []) if ctx else []
        }]
        return self._build_unified_user_prompt(batch_data, ctx, prev_context, retry_attempt, retry_reason, is_image_mode=False)

    def _validate_br_markers(self, translations: List[str], queries: List[str] = None, ctx=None, batch_indices: List[int] = None, batch_data: List = None, split_level: int = 0) -> bool:
        """
        检查翻译结果是否包含必要的[BR]标记
        Check if translations contain necessary [BR] markers

        同时清理单区域（region_count == 1）翻译中多余的断句标记：
        该清理只依赖 AI 断句（disable_auto_wrap）开关，与「AI断句检查」
        （check_br_and_retry）是否开启无关。

        Args:
            translations: 翻译结果列表
            queries: 原始查询列表（可选）
            ctx: 上下文（用于获取配置和区域信息）
            batch_indices: 批次索引列表（可选，用于定位text_regions）
            batch_data: 批次数据列表（可选，HQ翻译器使用）
            split_level: 分割级别（可选，用于跳过深度分割时的检查）

        Returns:
            True if validation passes, False if BR markers are missing
        """
        import re

        # When the split depth is too large (>=3), skip the BR check to avoid endless retries
        if split_level >= 3:
            self.logger.info(f"[AI Line Break Check] Split level is too deep (split_level={split_level}); skipping the BR marker check")
            return True

        # Check whether AI line breaking is on
        ai_break_enabled = False
        if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'render'):
            ai_break_enabled = getattr(ctx.config.render, 'disable_auto_wrap', False)

        if not ai_break_enabled:
            return True  # AI line breaking is off, so BR needs no handling

        # Get the number of regions for each translation
        region_counts = []
        single_region_indices = []  # Indexes of translations confirmed to be a single region (region_count < 2)
        if ctx and hasattr(ctx, 'text_regions') and ctx.text_regions:
            for idx in range(len(translations)):
                # Work out the actual region index
                if batch_indices and idx < len(batch_indices):
                    region_idx = batch_indices[idx]
                else:
                    region_idx = idx

                if region_idx < len(ctx.text_regions):
                    region = ctx.text_regions[region_idx]
                    region_count = len(region.lines) if hasattr(region, 'lines') else 1
                    region_counts.append(region_count)
                    if region_count < 2:
                        single_region_indices.append(idx)
                else:
                    region_counts.append(1)  # 1 by default
        elif batch_data:
            # The HQ translators use batch_data
            for idx in range(len(translations)):
                region_idx = idx
                for data in batch_data:
                    if 'text_regions' in data and data['text_regions'] and region_idx < len(data['text_regions']):
                        region = data['text_regions'][region_idx]
                        region_count = len(region.lines) if hasattr(region, 'lines') else 1
                        region_counts.append(region_count)
                        if region_count < 2:
                            single_region_indices.append(idx)
                        break
                else:
                    region_counts.append(1)
        else:
            region_counts = [1] * len(translations)  # All 1 by default

        # Single-region clean-up (independent of the "AI line-breaking check" switch; it only needs AI line breaking to be on):
        # when the model ignores the N=1 rule and returns [BR]/<br>/【BR】, reduce it to one line,
        # so a single region is not rendered as several lines.
        for idx in single_region_indices:
            translation = translations[idx]
            if translation:
                cleaned = re.sub(r'\s*(\[BR\]|【BR】|<br\s*/?>)\s*', '', translation, flags=re.IGNORECASE)
                if cleaned != translation:
                    self.logger.info(
                        f"[AI Line Breaking] Automatically removed extra line break markers from single-region translation #{idx+1}: {translation[:50]!r} -> {cleaned[:50]!r}"
                    )
                    translations[idx] = cleaned

        # Check whether the BR check is on
        check_enabled = False
        if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'render'):
            check_enabled = getattr(ctx.config.render, 'check_br_and_retry', False)

        if not check_enabled:
            return True  # The check is off: pass

        # Check each translation and count the missing BRs
        needs_check_count = 0
        missing_br_count = 0
        missing_indices = []

        for idx, (translation, region_count) in enumerate(zip(translations, region_counts)):
            # Only translations with 2 or more regions are checked
            if region_count >= 2:
                needs_check_count += 1
                # Check whether it contains a BR marker
                has_br = bool(re.search(r'(\[BR\]|【BR】|<br>)', translation, flags=re.IGNORECASE))
                if not has_br:
                    missing_br_count += 1
                    missing_indices.append(idx + 1)
                    self.logger.warning(
                        f"Translation {idx+1} missing [BR] markers (expected for {region_count} regions): {translation[:50]}..."
                    )

        # Tolerated number of errors: one tenth, at least 1
        if needs_check_count > 0:
            tolerance = max(1, needs_check_count // 10)

            if missing_br_count > tolerance:
                # Above the tolerance: validation fails
                self.logger.warning(
                    f"[AI Line Break Check] Translations missing BR markers ({missing_br_count}/{needs_check_count}) exceed the tolerance ({tolerance}); retry required"
                )
                return False
            elif missing_br_count > 0:
                # Within the tolerance: warn but pass
                self.logger.warning(
                    f"[AI Line Break Check] ⚠ {missing_br_count}/{needs_check_count} translations are missing BR markers, but this is within the tolerance ({tolerance}); continuing"
                )
                return True
            else:
                # All passed
                self.logger.info(f"[AI Line Break Check] ✓ All translations for multiline regions contain [BR] markers (checked {needs_check_count}/{len(translations)} translations)")
                return True

        return True  # No translation needs checking: pass

    def _validate_translation_quality(self, queries: List[str], translations: List[str]) -> Tuple[bool, str]:
        """
        验证翻译质量，检测常见问题

        Args:
            queries: 原文列表
            translations: 译文列表

        Returns:
            (is_valid, error_message)
        """
        # 1. Check that the counts match (required, cannot be skipped)
        if len(translations) != len(queries):
            return False, f"Translation count mismatch: expected {len(queries)}, got {len(translations)}"

        # 2. Check for empty translations (the original is not empty but the translation is) - disabled
        # empty_translation_errors = []
        # for i, (source, translation) in enumerate(zip(queries, translations)):
        #     if source.strip() and not translation.strip():
        #         empty_translation_errors.append(i + 1)
        # 
        # if empty_translation_errors:
        #     return False, f"Empty translation detected at positions: {empty_translation_errors}"

        # 3. Check for merged translations (the original is normal text but the translation is only punctuation) - disabled
        # for i, (source, translation) in enumerate(zip(queries, translations)):
        #     is_source_simple = all(char in string.punctuation or char.isspace() for char in source)
        #     is_translation_simple = all(char in string.punctuation or char.isspace() for char in translation)
        #
        #     if is_translation_simple and not is_source_simple:
        #         return False, f"Detected potential merged translation at position {i+1}"

        # 4. Check for suspicious symbols (model hallucination) - disabled
        # SUSPICIOUS_SYMBOLS = ["ହ", "ି", "ഹ"]
        # for symbol in SUSPICIOUS_SYMBOLS:
        #     for translation in translations:
        #         if symbol in translation:
        #             return False, f"Suspicious symbol '{symbol}' detected in translation"

        return True, ""

    def _reset_global_attempt_count(self):
        """重置全局尝试计数器（每次新的翻译任务开始时调用）"""
        self._global_attempt_count = 0
        self._max_total_attempts = self._resolve_max_total_attempts()

    def _increment_global_attempt(self) -> bool:
        """
        增加全局尝试计数，返回是否还可以继续尝试

        Returns:
            True: 还可以继续尝试
            False: 已达到总次数上限
        """
        self._global_attempt_count += 1

        # Unlimited retry mode
        if self._max_total_attempts == -1:
            return True

        # Check whether the limit is exceeded (note: the request that reaches the limit is still allowed to run)
        if self._global_attempt_count > self._max_total_attempts:
            self.logger.warning(f"Exceeded max total attempts: {self._global_attempt_count}/{self._max_total_attempts}")
            return False

        return True

    class SplitException(Exception):
        """用于触发分割的特殊异常"""
        def __init__(self, attempt_count, texts):
            self.attempt_count = attempt_count
            self.texts = texts
            super().__init__(f"Split triggered after {attempt_count} attempts")

    async def _translate_with_split(self, translator_func, texts: List[str], split_level: int = 0, **kwargs) -> List[str]:
        """
        带分割重试的翻译包装器（新逻辑）

        Args:
            translator_func: 实际的翻译函数（async callable）
            texts: 要翻译的文本列表
            split_level: 当前分割层级
            **kwargs: 传递给translator_func的其他参数

        Returns:
            翻译结果列表
        """
        # Check whether the global attempt count is exceeded
        if self._max_total_attempts != -1 and self._global_attempt_count >= self._max_total_attempts:
            self.logger.error(f"Global attempt limit reached before translation: {self._global_attempt_count}/{self._max_total_attempts}")
            raise Exception(f"Translation failed: reached max total attempts ({self._max_total_attempts})")

        try:
            # Try to translate (whether to split is checked inside)
            translations = await translator_func(texts, split_level=split_level, **kwargs)
            return translations

        except self.SplitException as split_ex:
            # Start a split
            if split_level < self._MAX_SPLIT_ATTEMPTS and len(texts) > 1:
                self.logger.warning(
                    f"Splitting after {split_ex.attempt_count} attempts at split_level={split_level}, "
                    f"batch size {len(texts)} → splitting into two halves"
                )

                # Split in two (only texts is split, not batch_data or the other parameters)
                mid = len(texts) // 2
                left_texts = texts[:mid]
                right_texts = texts[mid:]

                self.logger.info(f"Split: left={len(left_texts)}, right={len(right_texts)}, global_attempts={self._global_attempt_count}/{self._max_total_attempts}")

                # Translate the two halves concurrently (kwargs is passed on unchanged)
                try:
                    left_translations, right_translations = await asyncio.gather(
                        self._translate_with_split(translator_func, left_texts, split_level + 1, **kwargs),
                        self._translate_with_split(translator_func, right_texts, split_level + 1, **kwargs),
                        return_exceptions=False
                    )
                except Exception as split_error:
                    # When the concurrent run fails, fall back to one after the other
                    self.logger.warning(f"Concurrent split failed, falling back to sequential: {split_error}")
                    left_translations = await self._translate_with_split(translator_func, left_texts, split_level + 1, **kwargs)
                    right_translations = await self._translate_with_split(translator_func, right_texts, split_level + 1, **kwargs)

                # Join the results
                return left_translations + right_translations

            else:
                # It cannot be split any further: stop the translation
                if len(texts) == 1:
                    self.logger.error(f"Single text translation failed at split_level={split_level}: {texts[0][:50]}...")
                    raise Exception(f"Translation failed for single text after {split_ex.attempt_count} attempts")
                else:
                    self.logger.error(f"Max split level ({self._MAX_SPLIT_ATTEMPTS}) reached, batch size={len(texts)}")
                    raise Exception(f"Translation failed: max split level reached with batch size {len(texts)}")

        except Exception as e:
            # Any other exception (not caused by a split) stops it at once
            self.logger.error(f"Translation failed with exception at split_level={split_level}: {e}")
            raise e

    def parse_args(self, config):
        translator_config = self._resolve_translator_config(config)
        self._enable_streaming = self._get_config_value(
            translator_config,
            'enable_streaming',
            self._enable_streaming,
        )
        self.enable_post_translation_check = getattr(
            translator_config,
            'enable_post_translation_check',
            self.enable_post_translation_check,
        )
        self.post_check_repetition_threshold = getattr(
            translator_config,
            'post_check_repetition_threshold',
            self.post_check_repetition_threshold,
        )
        self.post_check_max_retry_attempts = getattr(
            translator_config,
            'post_check_max_retry_attempts',
            self.post_check_max_retry_attempts,
        )
        self.attempts = get_retry_attempts_from_config(config, logger=self.logger, fallback=-1)
        self._max_total_attempts = self._resolve_max_total_attempts()

    def _emit_stream_lines(self, prefix: str, text: str, width: int = 100) -> None:
        """将流式增量按固定宽度换行输出，避免命令行单行过长被截断。"""
        content = (text or "").strip()
        if not content:
            return
        for line in textwrap.wrap(content, width=width, break_long_words=True, break_on_hyphens=False):
            self.logger.info(f"{prefix} {line}")

    def _reset_stream_json_preview(self) -> None:
        self._stream_json_seen = {}
        self._stream_term_seen = {}
        self._stream_preview_buffer = ""
        self._stream_preview_scan_pos = 0
        self._stream_preview_object_starts = []
        self._stream_preview_in_string = False
        self._stream_preview_escape = False
        self._stream_preview_tag_tail = ""
        self._stream_preview_in_think = False
        self._stream_result_header_printed = False
        self._stream_result_pairs_printed = False

    def _has_stream_result_pairs(self) -> bool:
        return bool(self._stream_result_pairs_printed)

    def _emit_final_translation_results(self, source_texts: List[str], translations: List[str]) -> None:
        """
        输出最终翻译结果。
        - 若流式预览没有完整覆盖最终结果，则补打一份最终快照
        - 若预览内容与最终结果一致，则只输出结尾分隔线，避免重复刷屏
        """
        should_log_full = not self._has_stream_result_pairs()

        if not should_log_full:
            if len(source_texts) != len(translations):
                should_log_full = True
            else:
                for tid, translated in enumerate(translations, start=1):
                    if self._stream_json_seen.get(tid) != translated:
                        should_log_full = True
                        break

        if should_log_full:
            if self._stream_result_header_printed:
                self.logger.info("--- Final Translation Results ---")
            else:
                self.logger.info("--- Translation Results ---")
                self._stream_result_header_printed = True
            for original, translated in zip(source_texts, translations):
                self.logger.info(f"{original} -> {translated}")
            self._stream_result_pairs_printed = True

        self.logger.info("---------------------------")

    def _emit_terms_from_list(self, new_terms: List[Dict[str, Any]]) -> None:
        """统一输出术语提取结果；按正式名称、叫法和译文去重。"""
        if not new_terms:
            return
        for term in new_terms:
            if not isinstance(term, dict):
                continue
            term_o = str(term.get("original") or term.get("src") or "").strip()
            term_c = str(term.get("category") or "").strip()
            aliases = _auto_alias_deltas(term)
            if not term_o or not aliases:
                continue

            for alias in aliases:
                alias_o = alias["original"]
                for translation in alias["translations"]:
                    term_t = translation["text"]
                    key = (term_o, alias_o, term_t)
                    prev_category = self._stream_term_seen.get(key)
                    if prev_category is not None:
                        if prev_category == term_c:
                            continue
                        if prev_category and not term_c:
                            continue
                    # Without a category, buffer it instead of printing; print later if a category arrives, so there are not two lines
                    if not term_c and prev_category is None:
                        self._stream_term_seen[key] = ""
                        continue
                    self._stream_term_seen[key] = term_c

                    relation = term_o if term_o == alias_o else f"{term_o} [{alias_o}]"
                    if term_c:
                        self.logger.info(f"[TERM] {relation} -> {term_t} ({term_c})")
                    else:
                        self.logger.info(f"[TERM] {relation} -> {term_t}")

    def _filter_stream_preview_delta(self, delta_text: str) -> str:
        """
        以流式方式剥离 <think>/<answer> 标签，仅返回应参与 JSON 预览解析的新增正文。
        """
        if not delta_text:
            return ""

        known_tags = ("<think>", "</think>", "<answer>", "</answer>")
        data = f"{self._stream_preview_tag_tail}{delta_text}"
        self._stream_preview_tag_tail = ""
        out_parts: List[str] = []
        i = 0

        while i < len(data):
            if self._stream_preview_in_think:
                close_idx = data.lower().find("</think>", i)
                if close_idx == -1:
                    keep = min(len("</think>") - 1, len(data) - i)
                    self._stream_preview_tag_tail = data[-keep:] if keep > 0 else ""
                    return "".join(out_parts)
                i = close_idx + len("</think>")
                self._stream_preview_in_think = False
                continue

            lt_idx = data.find("<", i)
            if lt_idx == -1:
                out_parts.append(data[i:])
                break

            if lt_idx > i:
                out_parts.append(data[i:lt_idx])

            lower_rem = data[lt_idx:].lower()
            matched = False
            for tag in known_tags:
                if lower_rem.startswith(tag):
                    matched = True
                    if tag == "<think>":
                        self._stream_preview_in_think = True
                    i = lt_idx + len(tag)
                    break
                if tag.startswith(lower_rem):
                    self._stream_preview_tag_tail = data[lt_idx:]
                    return "".join(out_parts)

            if matched:
                continue

            out_parts.append("<")
            i = lt_idx + 1

        return "".join(out_parts)

    def _consume_completed_stream_preview_objects(self) -> List[str]:
        """
        从 preview_buffer 的 scan_pos 开始增量扫描，提取新闭合的 JSON 对象。
        """
        completed_objects: List[str] = []
        i = self._stream_preview_scan_pos

        while i < len(self._stream_preview_buffer):
            ch = self._stream_preview_buffer[i]

            if not self._stream_preview_object_starts:
                if ch == "{":
                    self._stream_preview_object_starts.append(i)
                    self._stream_preview_in_string = False
                    self._stream_preview_escape = False
                i += 1
                continue

            if self._stream_preview_in_string:
                if self._stream_preview_escape:
                    self._stream_preview_escape = False
                elif ch == "\\":
                    self._stream_preview_escape = True
                elif ch == '"':
                    self._stream_preview_in_string = False
            else:
                if ch == '"':
                    self._stream_preview_in_string = True
                elif ch == "{":
                    self._stream_preview_object_starts.append(i)
                elif ch == "}":
                    start = self._stream_preview_object_starts.pop()
                    completed_objects.append(self._stream_preview_buffer[start:i + 1])
                    if not self._stream_preview_object_starts:
                        self._stream_preview_in_string = False
                        self._stream_preview_escape = False

            i += 1

        self._stream_preview_scan_pos = i
        return completed_objects

    def _compact_stream_preview_buffer(self) -> None:
        """
        丢弃已经扫描且不再需要的前缀，避免 buffer 无界增长。
        """
        if not self._stream_preview_buffer:
            return

        trim_to = (
            min(self._stream_preview_object_starts)
            if self._stream_preview_object_starts
            else self._stream_preview_scan_pos
        )
        if trim_to <= 0:
            return

        self._stream_preview_buffer = self._stream_preview_buffer[trim_to:]
        self._stream_preview_scan_pos = max(0, self._stream_preview_scan_pos - trim_to)
        self._stream_preview_object_starts = [pos - trim_to for pos in self._stream_preview_object_starts]

    def _emit_stream_preview_translation_item(
        self,
        prefix: str,
        tid: int,
        translated: str,
        source_texts: Optional[List[str]] = None,
    ) -> None:
        translated = str(translated)
        if self._stream_json_seen.get(tid) == translated:
            return

        self._stream_json_seen[tid] = translated
        if not self._stream_result_header_printed:
            self.logger.info("--- Translation Results ---")
            self._stream_result_header_printed = True

        if source_texts and 1 <= tid <= len(source_texts):
            self.logger.info(f"{source_texts[tid - 1]} -> {translated}")
        else:
            self.logger.info(f"{prefix} #{tid}: {translated}")
        self._stream_result_pairs_printed = True

    def _process_stream_preview_object(
        self,
        prefix: str,
        obj_text: str,
        source_texts: Optional[List[str]] = None,
    ) -> None:
        try:
            parsed = json.loads(obj_text)
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/translators/common.py:CommonTranslator._process_stream_preview_object")
            return

        if not isinstance(parsed, dict):
            return

        translation_items: List[Tuple[int, str]] = []
        if "id" in parsed and "translation" in parsed:
            try:
                translation_items.append((int(parsed["id"]), str(parsed["translation"])))
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/translators/common.py:CommonTranslator._process_stream_preview_object")
                pass

        trans_list = parsed.get("translations")
        if not trans_list and "t" in parsed:
            trans_list = parsed.get("t")

        if isinstance(trans_list, list):
            if trans_list and isinstance(trans_list[0], dict):
                for item in trans_list:
                    if not isinstance(item, dict):
                        continue
                    if "id" not in item:
                        continue
                    text = item.get("translation")
                    if text is None:
                        text = item.get("text")
                    if text is None:
                        continue
                    try:
                        translation_items.append((int(item["id"]), str(text)))
                    except Exception as ignored_error:
                        note_ignored_error(ignored_error, "manga_translator/translators/common.py:CommonTranslator._process_stream_preview_object")
                        continue
            else:
                for idx, text in enumerate(trans_list, start=1):
                    translation_items.append((idx, str(text)))

        for tid, translated in translation_items:
            self._emit_stream_preview_translation_item(prefix, tid, translated, source_texts=source_texts)

        term_items: List[Dict[str, Any]] = []
        if "original" in parsed and "aliases" in parsed:
            term_items.append(dict(parsed))

        nested_terms = parsed.get("new_terms") or parsed.get("glossary")
        if isinstance(nested_terms, list):
            for item in nested_terms:
                if not isinstance(item, dict):
                    continue
                term_items.append(dict(item))

        if term_items:
            if not self._stream_result_header_printed:
                self.logger.info("--- Translation Results ---")
                self._stream_result_header_printed = True
            self._emit_terms_from_list(term_items)

    def _emit_stream_json_preview(self, prefix: str, delta_text: str, source_texts: Optional[List[str]] = None) -> None:
        """
        仅消费新增 delta_text，通过 preview_buffer + scan_pos 增量提取新闭合的 JSON 对象。
        """
        if not delta_text:
            return

        filtered_delta = self._filter_stream_preview_delta(str(delta_text))
        if filtered_delta:
            self._stream_preview_buffer += filtered_delta

        completed_objects = self._consume_completed_stream_preview_objects()
        for obj_text in completed_objects:
            self._process_stream_preview_object(prefix, obj_text, source_texts=source_texts)

        self._compact_stream_preview_buffer()

        if not filtered_delta and not completed_objects:
            return

    def _update_stream_inline(self, prefix: str, delta_text: str) -> None:
        """按增量流内容刷新；遇到换行符时真正换行输出。"""
        if not delta_text:
            return
        self._stream_inline_buffer += str(delta_text)

        # Print the complete line (with the line breaks the model returned)
        while "\n" in self._stream_inline_buffer:
            line_text, rest = self._stream_inline_buffer.split("\n", 1)
            self._stream_inline_buffer = rest
            line = f"{prefix} {line_text}"
            pad = max(0, self._stream_inline_last_len - len(line))
            try:
                sys.stdout.write("\r" + line + (" " * pad) + "\n")
                sys.stdout.flush()
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/translators/common.py:CommonTranslator._update_stream_inline")
                self._emit_stream_lines(prefix, line_text)
            self._stream_inline_last_len = 0

        # Without a line break, refresh on the same line
        text = self._stream_inline_buffer
        if not text:
            return
        try:
            term_width = shutil.get_terminal_size(fallback=(120, 24)).columns
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/translators/common.py:CommonTranslator._update_stream_inline")
            term_width = 120
        # Leave a small margin, so the text does not jitter at the edge
        available = max(20, term_width - len(prefix) - 2)
        tail = text[-available:]
        line = f"{prefix} {tail}"
        pad = max(0, self._stream_inline_last_len - len(line))
        try:
            sys.stdout.write("\r" + line + (" " * pad))
            sys.stdout.flush()
            self._stream_inline_last_len = len(line)
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/translators/common.py:CommonTranslator._update_stream_inline")
            # When the terminal is not writable, fall back to ordinary logging
            self._emit_stream_lines(prefix, tail)

    def _finish_stream_inline(self) -> None:
        """结束同一行刷新，补一个换行。"""
        if self._stream_inline_last_len > 0 or self._stream_inline_buffer:
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/translators/common.py:CommonTranslator._finish_stream_inline")
                pass
            self._stream_inline_last_len = 0
            self._stream_inline_buffer = ""

    def supports_languages(self, from_lang: str, to_lang: str, fatal: bool = False) -> bool:
        supported_src_languages = ['auto'] + list(self._LANGUAGE_CODE_MAP)
        supported_tgt_languages = list(self._LANGUAGE_CODE_MAP)

        if from_lang not in supported_src_languages:
            if fatal:
                raise LanguageUnsupportedException(from_lang, self.__class__.__name__, supported_src_languages)
            return False
        if to_lang not in supported_tgt_languages:
            if fatal:
                raise LanguageUnsupportedException(to_lang, self.__class__.__name__, supported_tgt_languages)
            return False
        return True

    def parse_language_codes(self, from_lang: str, to_lang: str, fatal: bool = False) -> Tuple[str, str]:
        if not self.supports_languages(from_lang, to_lang, fatal):
            return None, None
        if type(self._LANGUAGE_CODE_MAP) is list:
            return from_lang, to_lang

        _from_lang = self._LANGUAGE_CODE_MAP.get(from_lang) if from_lang != 'auto' else 'auto'
        _to_lang = self._LANGUAGE_CODE_MAP.get(to_lang)
        return _from_lang, _to_lang

    async def translate(self, from_lang: str, to_lang: str, queries: List[str], use_mtpe: bool = False, ctx=None) -> List[str]:
        """
        Translates list of queries of one language into another.
        """
        if to_lang not in VALID_LANGUAGES:
            raise ValueError('Invalid language code: "%s". Choose from the following: %s' % (to_lang, ', '.join(VALID_LANGUAGES)))
        if from_lang not in VALID_LANGUAGES and from_lang != 'auto':
            raise ValueError('Invalid language code: "%s". Choose from the following: auto, %s' % (from_lang, ', '.join(VALID_LANGUAGES)))
        self.logger.info(f'Translating into {VALID_LANGUAGES[to_lang]}')

        if from_lang == to_lang:
            # Text clean-up (such as replacing full-width periods) applies even when source and target language are the same
            return [self._clean_translation_output(q, q, to_lang) for q in queries]

        # Dont translate queries without text
        query_indices = []
        final_translations = []
        for i, query in enumerate(queries):
            if not is_valuable_text(query):
                final_translations.append(queries[i])
            else:
                final_translations.append(None)
                query_indices.append(i)

        queries = [queries[i] for i in query_indices]

        translations = [''] * len(queries)
        untranslated_indices = list(range(len(queries)))
        for i in range(1 + self._INVALID_REPEAT_COUNT): # Repeat until all translations are considered valid
            if i > 0:
                self.logger.warning(f'Repeating because of invalid translation. Attempt: {i+1}')
                await asyncio.sleep(0.1)

            # Sleep if speed is over the ratelimit
            await self._ratelimit_sleep()

            # Translate
            _translations = await self._translate(*self.parse_language_codes(from_lang, to_lang, fatal=True), queries, ctx=ctx)

            # Strict validation: translation count must match query count
            if len(_translations) != len(queries):
                error_msg = f"Translation count mismatch: expected {len(queries)}, got {len(_translations)}"
                self.logger.error(error_msg)
                self.logger.error(f"Queries: {queries}")
                self.logger.error(f"Translations: {_translations}")
                raise InvalidServerResponse(error_msg)

            # Only overwrite yet untranslated indices
            for j in untranslated_indices:
                translations[j] = _translations[j]

            if self._INVALID_REPEAT_COUNT == 0:
                break

            new_untranslated_indices = []
            for j in untranslated_indices:
                q, t = queries[j], translations[j]
                # Repeat invalid translations with slightly modified queries
                if self._is_translation_invalid(q, t):
                    new_untranslated_indices.append(j)
                    queries[j] = self._modify_invalid_translation_query(q, t)
            untranslated_indices = new_untranslated_indices

            if not untranslated_indices:
                break

        translations = [self._clean_translation_output(q, r, to_lang) for q, r in zip(queries, translations)]

        if use_mtpe:
            translations = await self.mtpe_adapter.dispatch(queries, translations)

        # Merge with the queries without text
        for i, trans in enumerate(translations):
            final_translations[query_indices[i]] = trans
            self.logger.info(f'{i}: {queries[i]} => {trans}')

        return final_translations

    @abstractmethod
    async def _translate(self, from_lang: str, to_lang: str, queries: List[str], ctx=None) -> List[str]:
        pass

    async def _ratelimit_sleep(self):
        if self._MAX_REQUESTS_PER_MINUTE > 0:
            now = time.time()
            ratelimit_timeout = self._last_request_ts + 60 / self._MAX_REQUESTS_PER_MINUTE
            if ratelimit_timeout > now:
                self.logger.info(f'Ratelimit sleep: {(ratelimit_timeout-now):.2f}s')
                await asyncio.sleep(ratelimit_timeout-now)
            self._last_request_ts = time.time()

    def _is_translation_invalid(self, query: str, trans: str) -> bool:
        if not trans and query:
            return True
        if not query or not trans:
            return False

        query_symbols_count = len(set(query))
        trans_symbols_count = len(set(trans))
        if query_symbols_count > 6 and trans_symbols_count < 6 and trans_symbols_count < 0.25 * len(trans):
            return True
        return False

    def _modify_invalid_translation_query(self, query: str, trans: str) -> str:
        """
        Can be overwritten if _INVALID_REPEAT_COUNT was set. It modifies the query
        for the next translation attempt.
        """
        return query

    def _clean_translation_output(self, query: str, trans: str, to_lang: str) -> str:
        """
        Tries to spot and skim down invalid translations.
        """
        if not query or not trans:
            return ''

        # Remove the internal marker: 【Original regions: X】 or [Original regions: X]
        # Remove internal markers: 【Original regions: X】 or [Original regions: X]
        trans = re.sub(r'【Original regions:\s*\d+】\s*', '', trans, flags=re.IGNORECASE)
        trans = re.sub(r'\[Original regions:\s*\d+\]\s*', '', trans, flags=re.IGNORECASE)
        # Only whitespace next to BR in text returned by the translator is cleaned; text typed in the editor does not pass through here.
        trans = _BR_EDGE_WHITESPACE_RE.sub(r"\1", trans)

        # Replace runs of full-width periods (．．． or ．．) with an ellipsis
        trans = trans.replace('．．．', '…')
        trans = trans.replace('．．', '…')

        # '  ' -> ' '
        trans = re.sub(r'\s+', r' ', trans)
        # 'text.text' -> 'text. text'
        trans = re.sub(r'(?<![.,;!?])([.,;!?])(?=\w)', r'\1 ', trans)
        # ' ! ! . . ' -> ' !!.. '
        trans = re.sub(r'([.,;!?])\s+(?=[.,;!?]|$)', r'\1', trans)

        if to_lang not in RTL_LANGUAGES:
            # 'text .' -> 'text.'
            trans = re.sub(r'(?<=[.,;!?\w])\s+([.,;!?])', r'\1', trans)
            # ' ... text' -> ' ...text'
            trans = re.sub(r'((?:\s|^)\.+)\s+(?=\w)', r'\1', trans)

        seq = repeating_sequence(trans.lower())

        # 'aaaaaaaaaaaaa' -> 'aaaaaa'
        if len(trans) < len(query) and len(seq) < 0.5 * len(trans):
            # Shrink sequence to length of original query
            trans = seq * max(len(query) // len(seq), 1)
            # Transfer capitalization of query to translation
            nTrans = ''
            for i in range(min(len(trans), len(query))):
                nTrans += trans[i].upper() if query[i].isupper() else trans[i]
            trans = nTrans

        # words = text.split()
        # elements = list(set(words))
        # if len(elements) / len(words) < 0.1:
        #     words = words[:int(len(words) / 1.75)]
        #     text = ' '.join(words)

        #     # For words that appear more then four times consecutively, remove the excess
        #     for el in elements:
        #         el = re.escape(el)
        #         text = re.sub(r'(?: ' + el + r'){4} (' + el + r' )+', ' ', text)

        return trans

class OfflineTranslator(CommonTranslator, ModelWrapper):
    _MODEL_SUB_DIR = 'translators'

    async def _translate(self, *args, **kwargs):
        return await self.infer(*args, **kwargs)

    @abstractmethod
    async def _infer(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        pass

    async def load(self, from_lang: str, to_lang: str, device: str):
        return await super().load(device, *self.parse_language_codes(from_lang, to_lang))

    @abstractmethod
    async def _load(self, from_lang: str, to_lang: str, device: str):
        pass

    async def reload(self, from_lang: str, to_lang: str, device: str):
        return await super().reload(device, from_lang, to_lang)


    async def unload(self, device: str):
        return await super().unload()

def sanitize_text_encoding(text: str) -> str:
    """
    统一的文本编码清理函数，处理各种编码问题
    Unified text encoding sanitization to handle various encoding issues
    
    Args:
        text: 输入文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return text
    
    try:
        # 1. Try to detect and repair UTF-16-LE encoding problems
        # When the text has a UTF-16-LE BOM or its signature, try to decode it again
        if isinstance(text, bytes):
            # For bytes, try several encodings
            for encoding in ['utf-8', 'utf-16-le', 'utf-16-be', 'latin-1']:
                try:
                    text = text.decode(encoding, errors='ignore')
                    break
                except (UnicodeDecodeError, AttributeError):
                    continue
        
        # 2. Remove invisible control characters and damaged characters
        # Keep the common control characters: line feed (\n), carriage return (\r), tab (\t)
        import unicodedata
        cleaned = []
        for char in text:
            # Skip control characters (other than \n, \r, \t)
            if unicodedata.category(char)[0] == 'C' and char not in '\n\r\t':
                continue
            # Skip private-use characters (possibly a damaged encoding)
            if '\uE000' <= char <= '\uF8FF':  # private use area
                continue
            # Skip the replacement character (it marks a decoding failure)
            if char == '\ufffd':
                continue
            cleaned.append(char)
        
        text = ''.join(cleaned)
        
        # 3. Repair common encoding mix-ups
        # The pattern produced when UTF-16-LE is read as UTF-8,
        # for example every character followed by \x00
        if '\x00' in text:
            text = text.replace('\x00', '')
        
        # 4. Make sure the text is valid UTF-8
        # Validate and clean by encoding and decoding again
        text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        
        return text
        
    except Exception as e:
        import logging
        logger = logging.getLogger('manga_translator')
        logger.warning(f"Text encoding cleanup failed: {e}; returning the original text")
        # When cleaning fails, at least remove the obviously bad characters
        if isinstance(text, str):
            return text.replace('\ufffd', '').replace('\x00', '')
        return str(text)


def normalize_model_output_text(text: str, preview: bool = False) -> str:
    """
    清理模型在正文外包裹的控制标签，避免思考区和回答区交叉污染解析。

    preview=True 时会额外移除未闭合的 <think> 尾部，避免流式中途把思考区当正文。
    """
    if not text:
        return text

    cleaned = str(text)
    cleaned = re.sub(r'(?:</think>)?<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    if preview:
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

    answer_match = re.search(r'<answer>(.*?)</answer>', cleaned, flags=re.DOTALL | re.IGNORECASE)
    if answer_match:
        cleaned = answer_match.group(1).strip()

    cleaned = re.sub(r'</?answer>', '', cleaned, flags=re.IGNORECASE)
    return cleaned



def extract_json_payload_from_mixed_text(text: str) -> Tuple[str, bool]:
    """
    通用 JSON 提取器：从混杂文本中优先提取最可能的 JSON 负载。
    返回 (payload, extracted)，extracted=True 表示确实抽取到了 JSON 片段。
    """
    import json
    import re

    if text is None:
        return "", False

    raw = str(text).strip()
    if not raw:
        return raw, False

    def _extract_balanced_json_candidates(src: str) -> List[str]:
        candidates = []
        stack = []
        start = -1
        in_str = False
        escape = False
        for i, ch in enumerate(src):
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch in '{[':
                if not stack:
                    start = i
                stack.append(ch)
            elif ch in '}]' and stack:
                top = stack[-1]
                if (top == '{' and ch == '}') or (top == '[' and ch == ']'):
                    stack.pop()
                    if not stack and start >= 0:
                        candidates.append(src[start:i + 1].strip())
                        start = -1
                else:
                    stack = []
                    start = -1
        return candidates

    def _is_json_parseable(candidate: str) -> bool:
        try:
            json.loads(candidate)
            return True
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/translators/common.py:extract_json_payload_from_mixed_text._is_json_parseable")
            try:
                import json5
                json5.loads(candidate)
                return True
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/translators/common.py:extract_json_payload_from_mixed_text._is_json_parseable")
                return False

    candidates: List[str] = []

    # 1) fenced code block candidates
    fenced = re.findall(r'`{3,}\s*(?:json)?\s*([\s\S]*?)`{3,}', raw, flags=re.IGNORECASE)
    for c in fenced:
        c = c.strip()
        if c:
            candidates.append(c)

    # 2) balanced-json candidates from full text
    candidates.extend(_extract_balanced_json_candidates(raw))

    # 3) fallback: from first bracket onward
    first_bracket = raw.find('[')
    first_brace = raw.find('{')
    json_start = -1
    if first_bracket != -1 and first_brace != -1:
        json_start = min(first_bracket, first_brace)
    elif first_bracket != -1:
        json_start = first_bracket
    elif first_brace != -1:
        json_start = first_brace
    if json_start >= 0:
        candidates.append(raw[json_start:].strip())

    # dedup keep-order
    dedup: List[str] = []
    seen = set()
    for c in candidates:
        key = c.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(key)

    if not dedup:
        return raw, False

    preferred_markers = ('"translations"', "'translations'", '"translation"', '"id"', '"new_terms"', '"glossary"')

    best_candidate = None
    best_score = (-1, -1)
    for c in dedup:
        low = c.lower()
        marker_score = sum(1 for m in preferred_markers if m in low)
        parseable_score = 1 if _is_json_parseable(c) else 0
        score = (parseable_score * 10 + marker_score, len(c))
        if score > best_score:
            best_score = score
            best_candidate = c

    if best_candidate is None:
        return raw, False
    return best_candidate, True

def parse_hq_response(result_text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    专门解析HQ翻译器的响应，支持提取翻译和新术语
    Parse HQ translator response, supporting extraction of translations and new terms
    
    Returns:
        (translations, new_terms)
    """
    import json
    import logging
    import re
    
    logger = logging.getLogger('manga_translator')
    
    # Shared encoding clean-up
    result_text = sanitize_text_encoding(result_text)
    result_text = normalize_model_output_text(result_text)
    extracted_payload, extracted = extract_json_payload_from_mixed_text(result_text)
    if extracted:
        result_text = extracted_payload
        logger.info("Extracted JSON payload from mixed response text")
    
    _original_text = result_text # Keep for logging
    result_text = result_text.strip()
    if not result_text:
        return [], []

    # 1. Collect candidate JSON fragments from the raw text (tolerates repeated streaming chunks)
    def _extract_balanced_json_candidates(text: str) -> List[str]:
        candidates = []
        stack = []
        start = -1
        in_str = False
        escape = False
        for i, ch in enumerate(text):
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch in '{[':
                if not stack:
                    start = i
                stack.append(ch)
            elif ch in '}]' and stack:
                top = stack[-1]
                if (top == '{' and ch == '}') or (top == '[' and ch == ']'):
                    stack.pop()
                    if not stack and start >= 0:
                        candidates.append(text[start:i + 1].strip())
                        start = -1
                else:
                    stack = []
                    start = -1
        return candidates

    raw_text = result_text
    candidate_texts: List[str] = []

    if "```" in raw_text:
        fenced = re.findall(r'```(?:json)?\s*\n(.*?)\n```', raw_text, flags=re.DOTALL)
        candidate_texts.extend([x.strip() for x in fenced if x and x.strip()])
        lines = raw_text.split('\n')
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped_text = "\n".join(lines).strip()
        if stripped_text:
            candidate_texts.append(stripped_text)
        result_text = stripped_text or raw_text

    # 2. Find where the JSON starts (drops a prefix)
    first_bracket = result_text.find('[')
    first_brace = result_text.find('{')
    json_start = -1
    if first_bracket != -1 and first_brace != -1: json_start = min(first_bracket, first_brace)
    elif first_bracket != -1: json_start = first_bracket
    elif first_brace != -1: json_start = first_brace
    
    if json_start > 0:
        result_text = result_text[json_start:].strip()
    if result_text:
        candidate_texts.append(result_text)

    # Every balanced JSON fragment in the raw text is a candidate too
    candidate_texts.extend(_extract_balanced_json_candidates(raw_text))

    # Remove duplicates, keeping the order
    dedup_candidates = []
    seen_candidates = set()
    for c in candidate_texts:
        if not c:
            continue
        key = c.strip()
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        dedup_candidates.append(key)

    translations = []
    new_terms = []
    
    def _parse_candidate(candidate_text: str):
        parsed = None
        try:
            parsed = json.loads(candidate_text)
        except json.JSONDecodeError:
            try:
                import json5
                parsed = json5.loads(candidate_text)
                logger.info("Using json5 for parsing")
            except (ImportError, Exception) as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/translators/common.py:parse_hq_response._parse_candidate")
                parsed = None
        if parsed is None:
            return None

        out_trans = []
        out_terms = []
        try:
            if isinstance(parsed, dict):
                trans_list = parsed.get("translations")
                if not trans_list and "t" in parsed:
                    trans_list = parsed.get("t")
                if not trans_list:
                    trans_list = []

                if isinstance(trans_list, list):
                    if trans_list and isinstance(trans_list[0], dict):
                        for item in trans_list:
                            text = item.get('translation') or item.get('text') or list(item.values())[0]
                            out_trans.append(str(text) if text is not None else "")
                    else:
                        out_trans = [str(x) for x in trans_list]

                terms_list = parsed.get("new_terms") or parsed.get("glossary")
                if isinstance(terms_list, list):
                    out_terms = terms_list
            elif isinstance(parsed, list):
                if parsed:
                    if isinstance(parsed[0], dict):
                        for item in parsed:
                            text = item.get('translation') or item.get('text') or list(item.values())[0]
                            out_trans.append(str(text) if text is not None else "")
                    else:
                        out_trans = [str(x) for x in parsed]
            return out_trans, out_terms
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/translators/common.py:parse_hq_response._parse_candidate")
            return None

    # Prefer the candidate with the most translation entries; on a tie, the one with more glossary terms
    best = None
    best_score = (-1, -1)
    for c in dedup_candidates:
        parsed_result = _parse_candidate(c)
        if not parsed_result:
            continue
        cand_trans, cand_terms = parsed_result
        score = (len(cand_trans), len(cand_terms))
        if score > best_score:
            best_score = score
            best = (cand_trans, cand_terms)

    if best is not None:
        return best[0], best[1]

    # === Strategy 3: brute-force extraction with regular expressions ===
    logger.warning("JSON parsing failed, falling back to Regex extraction")
    
    # 3.1 Try to extract objects with an ID: {"id": 1, "translation": "..."}
    object_pattern = r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"translation"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
    matches = re.findall(object_pattern, result_text)
    
    if matches:
        logger.info(f"Regex extracted {len(matches)} translations with IDs")
        translations = [match[1].replace('\\"', '"').replace('\\n', '\n') for match in matches]
        
        return translations, new_terms

    # 3.2 Try to extract only the translation field
    translation_pattern = r'"translation"\s*:\s*"((?:[^"\\]|\\.)*)"'
    matches = re.findall(translation_pattern, result_text)
    if matches:
         logger.warning(f"Regex extracted {len(matches)} translations (no IDs)")
         translations = [match.replace('\\"', '"').replace('\\n', '\n') for match in matches]
         return translations, []

    # 3.3 Last resort: split by lines (only when it does not look like JSON)
    if not result_text.startswith('{') and not result_text.startswith('['):
         for line in result_text.split('\n'):
            line = line.strip()
            if line:
                line = re.sub(r'^\d+\.\s*', '', line)
                line = line.replace('\\n', '\n').replace('↵', '\n')
                translations.append(line)

    return translations, new_terms

def parse_json_or_text_response(result_text: str) -> List[str]:
    """
    解析LLM返回的文本，支持JSON列表格式或按行分割格式
    Wrapper around parse_hq_response for backward compatibility
    """
    translations, _ = parse_hq_response(result_text)
    return translations





def _glossary_match_key(value: Any) -> str:
    return str(value or "").strip()


def _auto_alias_deltas(term: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read a sanitized ``aliases[].translations[].text`` AI delta."""
    grouped: List[Dict[str, Any]] = []
    indexes: Dict[str, int] = {}
    raw_aliases = term.get("aliases")

    if isinstance(raw_aliases, list):
        for raw_alias in raw_aliases:
            if not isinstance(raw_alias, dict):
                continue
            alias_original = str(raw_alias.get("original") or "").strip()
            raw_translations = raw_alias.get("translations")
            if (
                not alias_original
                or not isinstance(raw_translations, list)
                or len(raw_translations) != 1
                or not isinstance(raw_translations[0], dict)
            ):
                continue

            text = str(raw_translations[0].get("text") or "").strip()
            if not text:
                continue

            alias_key = _glossary_match_key(alias_original)
            if alias_key in indexes:
                continue
            indexes[alias_key] = len(grouped)
            grouped.append(
                {"original": alias_original, "translations": [{"text": text}]}
            )

    return grouped


def _merge_auto_glossary_aliases(
    entry: Dict[str, Any], incoming_aliases: List[Dict[str, Any]]
) -> bool:
    """Append AI alias deltas without changing any authored data."""
    canonical_original = str(entry.get("original") or "").strip()
    aliases = entry.get("aliases")

    if not isinstance(aliases, list):
        legacy_translation = str(entry.get("translation") or "").strip()
        canonical_key = _glossary_match_key(canonical_original)
        has_new_content = any(
            _glossary_match_key(alias["original"]) != canonical_key
            for alias in incoming_aliases
        )
        if not has_new_content:
            return False

        translations = []
        if legacy_translation:
            legacy_item = {"text": legacy_translation}
            legacy_condition = str(entry.get("condition") or "").strip()
            if legacy_condition:
                legacy_item["condition"] = legacy_condition
            translations.append(legacy_item)
        aliases = [{"original": canonical_original, "translations": translations}]
        entry.pop("translation", None)
        entry.pop("condition", None)
        entry["aliases"] = aliases

    modified = False
    for incoming_alias in incoming_aliases:
        alias_original = incoming_alias["original"]
        alias_key = _glossary_match_key(alias_original)
        target_alias = next(
            (
                alias
                for alias in aliases
                if isinstance(alias, dict)
                and _glossary_match_key(alias.get("original")) == alias_key
            ),
            None,
        )
        if target_alias is None:
            aliases.append(
                {
                    "original": alias_original,
                    "translations": [
                        {"text": item["text"]}
                        for item in incoming_alias["translations"]
                    ],
                }
            )
            modified = True
            continue
        # AI may create a new alias, but never append another translation to
        # an alias that already exists. Conditional variants remain manual.
        continue
    return modified


def merge_glossary_to_file(file_path: str, new_terms: List[Dict[str, Any]]) -> bool:
    """
    将新提取的术语合并到提示词文件中
    Merge newly extracted terms into the prompt file
    
    支持 JSON (.json) 和 YAML (.yaml/.yml) 格式，根据文件扩展名自动选择。
    
    Args:
        file_path: 提示词文件路径
        new_terms: 统一别名结构的新增或追加增量
    """
    import json
    import os

    from .prompt_loader import load_prompt_file
    
    if not new_terms:
        return False

    ext = os.path.splitext(file_path)[1].lower()
    is_yaml = ext in ('.yaml', '.yml')

    try:
        # Read the existing file (always through prompt_loader, which handles JSON and YAML)
        data = {}
        if os.path.exists(file_path):
            loaded = load_prompt_file(file_path)
            if loaded is not None:
                data = loaded
        
        # Make sure the structure is complete
        if "glossary" not in data or not isinstance(data["glossary"], dict):
            # When the old format is a list, or there is no glossary, start a new categorised structure
            data["glossary"] = {
                "Person": [], "Location": [], "Org": [], "Item": [], "Skill": [], "Creature": []
            }
        
        glossary = data["glossary"]
        # Make sure every standard category key exists
        valid_keys_map = {
            "person": "Person", 
            "location": "Location", 
            "org": "Org", 
            "organization": "Org",
            "item": "Item", 
            "skill": "Skill", 
            "creature": "Creature"
        }
        
        # Make sure the standard keys exist in the glossary
        for key in set(valid_keys_map.values()):
            if key not in glossary:
                glossary[key] = []

        modified = False
        
        for term in new_terms:
            if not isinstance(term, dict):
                continue
            raw_category = term.get("category")
            original = str(term.get("original") or "").strip()
            incoming_aliases = _auto_alias_deltas(term)
            
            if not original or not incoming_aliases:
                continue

            # Map the category to its standard key
            normalized_cat = str(raw_category or "").strip().lower()
            target_key = valid_keys_map.get(normalized_cat)
            if target_key is None:
                continue
            
            if not isinstance(glossary.get(target_key), list):
                glossary[target_key] = []

            original_key = _glossary_match_key(original)
            target_entry = next(
                (
                    entry
                    for entry in glossary[target_key]
                    if isinstance(entry, dict)
                    and _glossary_match_key(entry.get("original")) == original_key
                ),
                None,
            )
            if target_entry is not None:
                if target_entry.get("overwrite") is True:
                    modified = (
                        _merge_auto_glossary_aliases(target_entry, incoming_aliases)
                        or modified
                    )
                continue

            # A newly created entry must contain its canonical source form.
            if (
                _glossary_match_key(incoming_aliases[0]["original"])
                != _glossary_match_key(original)
            ):
                continue

            new_entry: Dict[str, Any] = {
                "original": original,
                "aliases": incoming_aliases,
                "overwrite": False,
            }
            glossary[target_key].append(new_entry)
            modified = True
        
        if modified:
            # Make sure the storage folder exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                if is_yaml:
                    from .prompt_loader import _get_yaml
                    yaml = _get_yaml()
                    if yaml is not None:
                        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                    else:
                        # YAML is not available: fall back to JSON
                        json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            return True
            
    except Exception as e:
        _http_logger.error(f"Error merging glossary: {e}")
        return False
    
    return False

def get_glossary_extraction_prompt(target_lang: str) -> str:
    """
    获取术语提取的追加提示词
    Get the additional prompt for glossary extraction
    """
    import os

    from ..utils import BASE_PATH
    from .prompt_loader import load_glossary_extraction_prompt as _load_glossary
    
    dict_dir = os.path.join(BASE_PATH, 'dict')
    return _load_glossary(dict_dir, target_lang)


def get_system_prompt_hq_format_prompt(target_lang: str, extract_glossary: bool = False) -> str:
    """
    获取 HQ 通用输出格式提示词。
    extract_glossary=False: 仅要求 translations
    extract_glossary=True: 要求 translations + new_terms
    """
    import os

    from ..utils import BASE_PATH
    from .prompt_loader import load_system_prompt_hq_format as _load_hq_format

    dict_dir = os.path.join(BASE_PATH, 'dict')
    return _load_hq_format(dict_dir, target_lang, extract_glossary=extract_glossary)


def get_glossary_output_format_prompt(target_lang: str) -> str:
    """
    兼容旧调用：获取开启术语提取时的 HQ 输出格式提示词。
    """
    return get_system_prompt_hq_format_prompt(target_lang, extract_glossary=True)
