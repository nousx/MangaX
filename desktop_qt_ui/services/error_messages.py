"""Turn a raw translation error into a message the user can act on.

Kept apart from the task logic so the classification rules can be read and
tested on their own.
"""
import logging
import os
import re
import textwrap


def build_friendly_error_message(error_message: str, error_traceback: str, i18n=None) -> str:
    """
    Build a friendly error hint from the error message, in the current interface language.
    """
    def _wrap_error_text(text: str, width: int = 88) -> str:
        wrapped_lines = []
        for line in (text or "").splitlines():
            if not line:
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                or [""]
            )
        return "\n".join(wrapped_lines)

    def _current_log_file_path() -> str:
        for handler in reversed(logging.getLogger().handlers):
            if isinstance(handler, logging.FileHandler):
                path = str(getattr(handler, "baseFilename", "") or "").strip()
                if path:
                    return os.path.normpath(os.path.abspath(path))
        return os.path.normpath(os.path.abspath(os.path.join("result", "log_*.txt")))

    # The caller passes the interface language manager; without one the keys are returned as they are.

    def _translate(key: str, **kwargs) -> str:
        return i18n.translate(key, **kwargs) if i18n else key

    friendly_msg = ""

    # Unwrap retry exhaustion messages before classifying the underlying error.
    real_error = error_message
    if "maximum attempts reached" in error_message.lower():
        marker = "last error:"
        marker_index = error_message.lower().find(marker)
        if marker_index >= 0:
            real_error = error_message[marker_index + len(marker):].strip()

    lower_error = real_error.lower()
    local_model_missing = any(
        marker in lower_error
        for marker in (
            "model file not found",
            "model file does not exist",
            "model file is missing",
            "missing model file",
        )
    )
    http_404 = bool(re.search(
        r"\b(?:http(?:/\d(?:\.\d)?)?(?:\s+error)?|status(?:[\s_-]*code)?|error\s+code)"
        r"[\s\"']*[:=]?[\s\"']*404\b"
        r"|\b404\s+(?:not[\s_]found|client\s+error)\b",
        lower_error,
    ))

    def _has_status_code(code: str) -> bool:
        # A bare substring test also matches digits inside hashes, sizes
        # and request IDs, so require the code to stand on its own.
        return bool(re.search(rf"(?<![0-9a-z]){code}(?![0-9a-z])", lower_error))

    def _is_image_output_unsupported_error(*section_markers: str) -> bool:
        if not any(marker in lower_error for marker in section_markers):
            return False
        return any(
            marker in lower_error
            for marker in (
                "image_url",
                "unknown variant",
                "expected `text`",
                "expected 'text'",
                "did not contain an image",
                "did not contain image data",
                "compatible image output interface",
                "only support text chat",
                "not image generation/editing output",
            )
        )
    def _is_candidate_exhausted_error(*feature_markers: str) -> bool:
        candidate_exhausted = (
            "no available api candidates" in lower_error
            or (
                "exhausting " in lower_error
                and " api candidate" in lower_error
            )
        )
        return candidate_exhausted and any(
            marker in lower_error for marker in feature_markers
        )

    def _is_model_unsupported_error() -> bool:
        return (
            "code=20012" in lower_error
            or "model does not exist" in lower_error
            or ("does not exist" in lower_error and "model" in lower_error)
            or "model not found" in lower_error
            or "invalid model" in lower_error
            or "no such model" in lower_error
            or "supported api model names" in lower_error
            or "supported model names" in lower_error
            or ("you passed" in lower_error and "model" in lower_error)
            or ("unsupported" in lower_error and "model" in lower_error)
        )

    def _is_feature_model_unsupported_error(*feature_markers: str) -> bool:
        return _is_model_unsupported_error() and any(
            marker in lower_error for marker in feature_markers
        )

    renderer_markers = ("renderer", "render request")
    colorizer_markers = ("colorizer", "colorization", "colorize")
    ocr_markers = ("ocr", "optical character recognition")

    # Local model paths must not be classified as API/model-name errors.
    if local_model_missing:
        friendly_msg = _translate("friendly_error_local_model_missing")

    # Check whether the AI line-breaking check failed
    elif ("br markers missing" in lower_error or
        "ai line break validation failed" in lower_error or
        "BRMarkersValidationException" in error_traceback or
        "_validate_br_markers" in error_traceback):
        friendly_msg = _translate("friendly_error_br_markers")

    # Check whether it is a translation count mismatch
    elif "translation count mismatch" in lower_error:
        friendly_msg = _translate("friendly_error_translation_count")

    # Check whether the translation quality check failed
    elif "quality check failed" in lower_error:
        friendly_msg = _translate("friendly_error_translation_quality")

    # Check whether it is an empty response from OpenAI/Gemini (handled together)
    elif (
        (("NoneType" in real_error or "NoneType" in error_traceback) and
         ("strip" in real_error.lower() or "strip" in error_traceback.lower()))
        or ("returned empty content" in real_error.lower())
        or ("returned empty text" in real_error.lower())
        or ("response text is empty" in lower_error)
    ):
        friendly_msg = _translate("friendly_error_empty_ai_response")

    # Check whether the render/colorize model or candidate is unavailable
    elif (
        _is_candidate_exhausted_error(*renderer_markers)
        or _is_feature_model_unsupported_error(*renderer_markers)
        or _is_image_output_unsupported_error(*renderer_markers)
    ):
        friendly_msg = _translate("friendly_error_renderer_unsupported")

    elif (
        _is_candidate_exhausted_error(*colorizer_markers)
        or _is_feature_model_unsupported_error(*colorizer_markers)
        or _is_image_output_unsupported_error(*colorizer_markers)
    ):
        friendly_msg = _translate("friendly_error_colorizer_unsupported")

    elif (
        _is_candidate_exhausted_error(*ocr_markers)
        or _is_feature_model_unsupported_error(*ocr_markers)
    ):
        friendly_msg = _translate("friendly_error_ocr_unavailable")

    # Check whether the model or the API endpoint does not support image input
    elif (
        "does not support image input" in lower_error
        or "no endpoints found that support image input" in lower_error
        or ("support image input" in lower_error and "endpoint" in lower_error)
        or ("multimodal" in lower_error and "renderer" not in lower_error)
        or ("vision" in lower_error and "renderer" not in lower_error)
        or ("image_url" in lower_error and "renderer" not in lower_error)
        or ("expected `text`" in lower_error and "renderer" not in lower_error)
        or ("unknown variant" in lower_error and "renderer" not in lower_error)
    ):
        friendly_msg = _translate("friendly_error_multimodal_unsupported")

    # Check whether the model does not exist or its name is wrong
    elif _is_model_unsupported_error():
        friendly_msg = _translate("friendly_error_model_unsupported")

    # Check whether it is a 404 error (wrong API address or model setting)
    elif "api_404_error" in lower_error or (http_404 and "html error page" in lower_error):
        friendly_msg = _translate("friendly_error_api_404_html")

    elif http_404:
        friendly_msg = _translate("friendly_error_http_404")

    # Check whether it is an API key error
    elif (
        "api key" in real_error.lower()
        or "authentication" in real_error.lower()
        or "unauthorized" in real_error.lower()
        or _has_status_code("401")
        or "no available api candidates" in real_error.lower()
        or "exhausting api candidates" in real_error.lower()
        or "api candidates" in real_error.lower()
    ):
        friendly_msg = _translate("friendly_error_api_credentials")

    # Check whether it is a network connection error
    elif (
        "connection" in real_error.lower()
        or "connect" in real_error.lower()
        or "failed to connect" in real_error.lower()
        or "could not connect to server" in real_error.lower()
        or "connection timed out" in real_error.lower()
        or "timed out" in lower_error
        or "timeout" in real_error.lower()
        or "network" in real_error.lower()
        or "curl: (7)" in real_error.lower()
        or "curl: (28)" in real_error.lower()
        or "host" in real_error.lower()
        or "hostname" in real_error.lower()
        or "dns" in real_error.lower()
        or "getaddrinfo" in real_error.lower()
        or "failed to resolve" in real_error.lower()
        or "temporary failure in name resolution" in real_error.lower()
        or "name or service not known" in real_error.lower()
        or "no address associated with hostname" in real_error.lower()
        or "nodename nor servname provided" in real_error.lower()
    ):
        friendly_msg = _translate("friendly_error_network")

    # Check whether it is a rate limit error
    elif "rate limit" in real_error.lower() or _has_status_code("429") or "too many requests" in real_error.lower():
        friendly_msg = _translate("friendly_error_http_429")

    # Check whether it is a 403 forbidden error
    elif _has_status_code("403") or "forbidden" in real_error.lower():
        friendly_msg = _translate("friendly_error_http_403")

    # Check whether it is a 500 server error
    elif _has_status_code("500") or "internal server error" in real_error.lower():
        friendly_msg = _translate("friendly_error_http_500")

    # Check whether it is a 502/503/504 gateway error
    elif any(_has_status_code(code) for code in ["502", "503", "504"]) or "bad gateway" in real_error.lower() or "service unavailable" in real_error.lower() or "gateway timeout" in real_error.lower():
        error_code = "502/503/504"
        if _has_status_code("502"):
            error_code = "502"
        elif _has_status_code("503"):
            error_code = "503"
        elif _has_status_code("504"):
            error_code = "504"

        friendly_msg = _translate("friendly_error_http_gateway", code=error_code)

    # Check whether it is a content filter error
    elif "content filter" in real_error.lower() or "content_filter" in real_error:
        friendly_msg = _translate("friendly_error_content_filter")

    # Check whether it is an unsupported language error
    elif "language not supported" in real_error.lower() or "LanguageUnsupportedException" in error_traceback:
        friendly_msg = _translate("friendly_error_language_unsupported")

    # Check whether the request was blocked
    elif "blocked" in real_error.lower() or "request was blocked" in real_error.lower():
        friendly_msg = _translate("friendly_error_request_blocked")

    # General error
    else:
        friendly_msg = _translate(
            "friendly_error_generic",
            error=error_message,
            log_path=_current_log_file_path(),
        )

    friendly_msg += _translate(
        "friendly_error_raw_details",
        error=_wrap_error_text(error_message),
    )
    if error_traceback and "Traceback" in error_traceback:
        # Keep only the detailed API error information (no code paths)
        lines = error_traceback.split('\n')
        api_error_lines = []

        for line in lines:
            # Keep only the API error lines (the ones with the detailed error content)
            if line.strip() and any(keyword in line for keyword in ['BadRequest', 'Error code:', "'error':", "'message':", "{'error':"]):
                api_error_lines.append(line.strip())

        if api_error_lines:
            friendly_msg += "\n"
            friendly_msg += _wrap_error_text('\n'.join(api_error_lines)) + "\n"


    return friendly_msg
