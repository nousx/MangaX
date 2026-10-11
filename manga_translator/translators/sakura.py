import os
import re

try:
    import openai
except ImportError:
    openai = None
import asyncio
import logging
from typing import Callable, List, Tuple

from ..runtime_api_resolver import get_runtime_api_override
from ..utils.system_proxy import openai_http_client_kwargs
from .common import CommonTranslator, validate_openai_response
from .keys import SAKURA_API_BASE, SAKURA_DICT_PATH

DEFAULT_SAKURA_API_BASE = 'http://127.0.0.1:8080/v1'
DEFAULT_SAKURA_DICT_PATH = './dict/sakura_dict.txt'


class SakuraDict():
    def __init__(self, path: str, logger: logging.Logger) -> None:
        self.logger = logger
        self.dict_str = ""
        self.path = path
        if not os.path.exists(path):
            self.logger.warning(f"Dictionary file not found: {path}")
            return
        self.dict_str = self.get_dict_from_file(path)

    def load_galtransl_dic(self, dic_path: str):
        """
        Load a Galtransl dictionary.
        """

        with open(dic_path, encoding="utf8") as f:
            dic_lines = f.readlines()
        if len(dic_lines) == 0:
            return
        dic_path = os.path.abspath(dic_path)
        dic_name = os.path.basename(dic_path)
        normalDic_count = 0

        gpt_dict = []
        for line in dic_lines:
            if line.startswith("\n"):
                continue
            elif line.startswith("\\\\") or line.startswith("//"):  # Skip comment lines
                continue

            # Replace four spaces with a tab
            line = line.replace("    ", "\t")

            sp = line.rstrip("\r\n").split("\t")  # Strip extra line breaks and split on tabs
            len_sp = len(sp)

            if len_sp < 2:  # At least 2 elements
                continue

            src = sp[0]
            dst = sp[1]
            info = sp[2] if len_sp > 2 else None
            gpt_dict.append({"src": src, "dst": dst, "info": info})
            normalDic_count += 1

        gpt_dict_text_list = []
        for gpt in gpt_dict:
            src = gpt['src']
            dst = gpt['dst']
            info = gpt['info'] if "info" in gpt.keys() else None
            if info:
                single = f"{src}->{dst} #{info}"
            else:
                single = f"{src}->{dst}"
            gpt_dict_text_list.append(single)

        gpt_dict_raw_text = "\n".join(gpt_dict_text_list)
        self.dict_str = gpt_dict_raw_text
        self.logger.info(
            f"Loaded Galtransl dictionary: {dic_name}, {normalDic_count} standard entries"
        )

    def load_sakura_dict(self, dic_path: str):
        """
        Load a standard Sakura dictionary directly.
        """

        with open(dic_path, encoding="utf8") as f:
            dic_lines = f.readlines()

        if len(dic_lines) == 0:
            return
        dic_path = os.path.abspath(dic_path)
        dic_name = os.path.basename(dic_path)
        normalDic_count = 0

        gpt_dict_text_list = []
        for line in dic_lines:
            if line.startswith("\n"):
                continue
            elif line.startswith("\\\\") or line.startswith("//"):  # Skip comment lines
                continue

            sp = line.rstrip("\r\n").split("->")  # Strip extra line breaks and split on ->
            len_sp = len(sp)

            if len_sp < 2:  # At least 2 elements
                continue

            src = sp[0]
            dst_info = sp[1].split("#")  # Split the target and the note on #
            dst = dst_info[0].strip()
            info = dst_info[1].strip() if len(dst_info) > 1 else None
            if info:
                single = f"{src}->{dst} #{info}"
            else:
                single = f"{src}->{dst}"
            gpt_dict_text_list.append(single)
            normalDic_count += 1

        gpt_dict_raw_text = "\n".join(gpt_dict_text_list)
        self.dict_str = gpt_dict_raw_text
        self.logger.info(
            f"Loaded standard Sakura dictionary: {dic_name}, {normalDic_count} standard entries"
        )

    def detect_type(self, dic_path: str):
        """
        Detect the dictionary type.
        """
        with open(dic_path, encoding="utf8") as f:
            dic_lines = f.readlines()
        self.logger.debug(f"Detecting dictionary type: {dic_path}")
        if len(dic_lines) == 0:
            return "unknown"

        # Whether it is a Galtransl dictionary
        is_galtransl = True
        for line in dic_lines:
            if line.startswith("\n"):
                continue
            elif line.startswith("\\\\") or line.startswith("//"):
                continue

            if "\t" not in line and "    " not in line:
                is_galtransl = False
                break

        if is_galtransl:
            return "galtransl"

        # Whether it is a Sakura dictionary
        is_sakura = True
        for line in dic_lines:
            if line.startswith("\n"):
                continue
            elif line.startswith("\\\\") or line.startswith("//"):
                continue

            if "->" not in line:
                is_sakura = False
                break

        if is_sakura:
            return "sakura"

        return "unknown"

    def get_dict_str(self):
        """
        Get the dictionary content.
        """
        if self.dict_str == "":
            if not os.path.exists(self.path):
                return ""
            try:
                self.dict_str = self.get_dict_from_file(self.path)
                return self.dict_str
            except Exception as e:
                self.logger.warning(f"Failed to load dictionary: {e}")
                return ""
        return self.dict_str

    def get_dict_from_file(self, dic_path: str):
        """
        Load the dictionary from a file.
        """
        dic_type = self.detect_type(dic_path)
        if dic_type == "galtransl":
            self.load_galtransl_dic(dic_path)
        elif dic_type == "sakura":
            self.load_sakura_dict(dic_path)
        else:
            self.logger.warning(f"Unknown dictionary type: {dic_path}")
        return self.dict_str


class SakuraTranslator(CommonTranslator):

    _TIMEOUT = 999  # Timeout for the server response (seconds)
    _RETRY_ATTEMPTS = 3  # Number of retries when a request fails
    _TIMEOUT_RETRY_ATTEMPTS = 3  # Number of retries when a request times out
    _RATELIMIT_RETRY_ATTEMPTS = 3  # Number of retries when a request is rate-limited
    _REPEAT_DETECT_THRESHOLD = 20  # Threshold for repetition detection

    _CHAT_SYSTEM_TEMPLATE = (
        '你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，注意不要擅自添加原文中没有的代词，也不要擅自增加或减少换行。'
    )

    _LANGUAGE_CODE_MAP = {
        'CHS': 'Simplified Chinese',
        'JPN': 'Japanese'
    }

    def __init__(self):
        super().__init__()
        self.api_base = self._normalize_api_base(os.getenv('SAKURA_API_BASE') or SAKURA_API_BASE)
        self.dict_path = os.getenv('SAKURA_DICT_PATH') or SAKURA_DICT_PATH or DEFAULT_SAKURA_DICT_PATH
        self.client = None
        self._setup_client()
        self.temperature = 0.3
        self.top_p = 0.3
        self.frequency_penalty = 0.1
        self._current_style = "precise"
        self._emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]')
        self._heart_pattern = re.compile(r'❤')
        self.sakura_dict = SakuraDict(self.get_dict_path(), self.logger)

    @staticmethod
    def _normalize_api_base(api_base: str) -> str:
        normalized = (
            str(api_base or SAKURA_API_BASE or DEFAULT_SAKURA_API_BASE).strip()
            or DEFAULT_SAKURA_API_BASE
        )
        normalized = normalized.rstrip("/")
        if "/v1" not in normalized:
            normalized = f"{normalized}/v1"
        return normalized

    def _setup_client(self):
        self.client = openai.AsyncOpenAI(
            api_key="sk-114514",
            base_url=self.api_base,
            **openai_http_client_kwargs(self.api_base),
        )

    def parse_args(self, config):
        super().parse_args(config)
        translator_args = self._resolve_translator_config(config)
        user_env_vars = getattr(config, "_user_env_vars", None) or {}
        runtime_override = get_runtime_api_override(config, "translator", "sakura")

        api_base = (
            self._get_config_value(translator_args, "user_api_base", None)
            or runtime_override.get("api_base")
            or user_env_vars.get("SAKURA_API_BASE")
            or os.getenv("SAKURA_API_BASE")
            or SAKURA_API_BASE
            or DEFAULT_SAKURA_API_BASE
        )
        api_base = self._normalize_api_base(api_base)
        if api_base != self.api_base:
            self.api_base = api_base
            self._setup_client()
            self.logger.info(f"Sakura API base updated: {self.api_base}")

        dict_path = (
            user_env_vars.get("SAKURA_DICT_PATH")
            or os.getenv("SAKURA_DICT_PATH")
            or SAKURA_DICT_PATH
            or DEFAULT_SAKURA_DICT_PATH
        )
        dict_path = str(dict_path or DEFAULT_SAKURA_DICT_PATH).strip() or DEFAULT_SAKURA_DICT_PATH
        if dict_path != self.dict_path:
            self.dict_path = dict_path
            self.sakura_dict = SakuraDict(self.get_dict_path(), self.logger)

    def get_dict_path(self):
        return getattr(self, 'dict_path', None) or SAKURA_DICT_PATH or DEFAULT_SAKURA_DICT_PATH

    def detect_and_caculate_repeats(self, s: str, threshold: int = _REPEAT_DETECT_THRESHOLD, remove_all=True) -> Tuple[bool, str, int, str]:
        """
        Detect whether the text contains a repeating pattern and count the repetitions.
        Returns: (whether it repeats, the text with the repetition removed, the repetition count, the repeating pattern)
        """
        repeated = False
        counts = []
        for pattern_length in range(1, len(s) // 2 + 1):
            i = 0
            while i < len(s) - pattern_length:
                pattern = s[i:i + pattern_length]
                count = 1
                j = i + pattern_length
                while j <= len(s) - pattern_length:
                    if s[j:j + pattern_length] == pattern:
                        count += 1
                        j += pattern_length
                    else:
                        break
                counts.append(count)
                if count >= threshold:
                    self.logger.warning(f"Repeated pattern detected: {pattern}, repetitions: {count}")
                    repeated = True
                    if remove_all:
                        s = s[:i + pattern_length] + s[j:]
                    break
                i += 1
            if repeated:
                break

        # Mode of the repetition counts
        if counts:
            mode_count = max(set(counts), key=counts.count)
        else:
            mode_count = 0

        # Work out the actual threshold from the default threshold and the mode
        actual_threshold = max(threshold, mode_count)

        return repeated, s, count, pattern, actual_threshold

    @staticmethod
    def enlarge_small_kana(text, ignore=''):
        """Convert small hiragana or katakana to their normal size

        Parameters
        ----------
        text : str
            A string of full-width hiragana or katakana.
        ignore : str, optional
            Characters to leave alone during the conversion.

        Returns
        -------
        str
            The hiragana or katakana string, with small kana converted to full size

        Examples
        --------
        >>> print(enlarge_small_kana('さくらきょうこ'))
        さくらきようこ
        >>> print(enlarge_small_kana('キュゥべえ'))
        キユウべえ
        """
        SMALL_KANA = list('ぁぃぅぇぉゃゅょっァィゥェォヵヶャュョッ')
        SMALL_KANA_NORMALIZED = list('あいうえおやゆよつアイウエオカケヤユヨツ')
        SMALL_KANA2BIG_KANA = dict(zip(map(ord, SMALL_KANA), SMALL_KANA_NORMALIZED))

        def _exclude_ignorechar(ignore, conv_map):
            for character in map(ord, ignore):
                del conv_map[character]
            return conv_map

        def _convert(text, conv_map):
            return text.translate(conv_map)

        def _translate(text, ignore, conv_map):
            if ignore:
                _conv_map = _exclude_ignorechar(ignore, conv_map.copy())
                return _convert(text, _conv_map)
            return _convert(text, conv_map)

        return _translate(text, ignore, SMALL_KANA2BIG_KANA)

    def _format_prompt_log(self, prompt: str) -> str:
        """
        Format the prompt text for log output.
        """
        gpt_dict_raw_text = self.sakura_dict.get_dict_str()
        if gpt_dict_raw_text:
            return '\n'.join([
                'System:',
                self._CHAT_SYSTEM_TEMPLATE,
                'User:',
                "根据以下术语表：",
                gpt_dict_raw_text,
                "将下面的日文文本根据上述术语表的对应关系和注释翻译成中文：",
                prompt,
            ])
        return '\n'.join([
            'System:',
            self._CHAT_SYSTEM_TEMPLATE,
            'User:',
            '将下面的日文文本翻译成中文：',
            prompt,
        ])

    def _split_text(self, text: str) -> List[str]:
        """
        Split a string into a list at line breaks.
        """
        if isinstance(text, list):
            return text
        return text.split('\n')

    def _normalize_response_text(self, response) -> str:
        """
        Turn a model response into a string in one place, to avoid type errors in the log and in later processing.
        """
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            normalized = []
            for item in response:
                if isinstance(item, str):
                    normalized.append(item)
                elif isinstance(item, dict) and "text" in item:
                    normalized.append(str(item["text"]))
                else:
                    normalized.append(str(item))
            return "\n".join(normalized)
        return str(response)

    def _preprocess_queries(self, queries: List[str]) -> List[str]:
        """
        Preprocess the query text: remove emoji, replace special characters and add 「」 marks.
        """
        queries = [self.enlarge_small_kana(query) for query in queries]
        queries = [self._emoji_pattern.sub('', query) for query in queries]
        queries = [self._heart_pattern.sub('♥', query) for query in queries]
        queries = [f'「{query}」' for query in queries]
        self.logger.debug(f'Preprocessed query text: {queries}')
        return queries

    async def _check_translation_quality(self, queries: List[str], response: str) -> List[str]:
        """
        Check the quality of the translation result, including repetition and line count alignment; when there is a problem, try to translate again or return the original text.
        """
        async def _retry_translation(queries: List[str], check_func: Callable[[str], bool], error_message: str) -> str:
            styles = ["precise", "normal", "aggressive", ]
            for i in range(self._RETRY_ATTEMPTS):
                self._set_gpt_style(styles[i])
                self.logger.warning(f'{error_message} Attempts: {i + 1}. Current parameter style: {self._current_style}.')
                response = await self._handle_translation_request(queries)
                if not check_func(response):
                    return response
            return None

        # Check whether the request contains repetition above the default threshold
        if self._detect_repeats(''.join(queries), self._REPEAT_DETECT_THRESHOLD):
            self.logger.warning(f'The request itself contains repeated content exceeding the default threshold of {self._REPEAT_DETECT_THRESHOLD}.')

        # Work out the actual threshold from the mode of the translation and the default threshold
        actual_threshold = max(max(self._get_repeat_count(query) for query in queries), self._REPEAT_DETECT_THRESHOLD)

        if self._detect_repeats(response, actual_threshold):
            response = await _retry_translation(queries, lambda r: self._detect_repeats(r, actual_threshold), f'Excessive repetition detected (threshold: {actual_threshold}); possible model degradation. Retrying translation.')
            if response is None:
                self.logger.warning(f'Suspected model degradation persists after {self._RETRY_ATTEMPTS} attempts; switching to single-line translation.')
                return await self._translate_single_lines(queries)

        if not self._check_align(queries, response):
            response = await _retry_translation(queries, lambda r: not self._check_align(queries, r), 'Source and translation line counts do not match. Retrying translation.')
            if response is None:
                self.logger.warning(f'Original and translated line counts still differ after {self._RETRY_ATTEMPTS} attempts; switching to single-line translation.')
                return await self._translate_single_lines(queries)

        return self._split_text(response)

    def _detect_repeats(self, text: str, threshold: int = _REPEAT_DETECT_THRESHOLD) -> bool:
        """
        Detect whether the text contains a repeating pattern.
        """
        is_repeated, text, count, pattern, actual_threshold = self.detect_and_caculate_repeats(text, threshold, remove_all=False)
        return is_repeated

    def _get_repeat_count(self, text: str, threshold: int = _REPEAT_DETECT_THRESHOLD) -> bool:
        """
        Count how many times the repeating pattern occurs in the text.
        """
        is_repeated, text, count, pattern, actual_threshold = self.detect_and_caculate_repeats(text, threshold, remove_all=False)
        return count

    def _check_align(self, queries: List[str], response: str) -> bool:
        """
        Check whether the original text and the translation have the same number of lines.
        """
        translations = self._split_text(response)
        is_aligned = len(queries) == len(translations)
        if not is_aligned:
            self.logger.warning(f"Line count mismatch - original: {len(queries)}, translated: {len(translations)}")
        return is_aligned

    async def _translate_single_lines(self, queries: List[str]) -> List[str]:
        """
        Translate the query text line by line.
        """
        translations = []
        for query in queries:
            response = await self._handle_translation_request(query)
            if self._detect_repeats(response):
                self.logger.warning(f"Single-line translation contains repeated content: {response}; returning the original text.")
                translations.append(query)
            else:
                translations.append(response)
        return translations

    def _delete_quotation_mark(self, texts: List[str]) -> List[str]:
        """
        Remove the 「」 marks from the text.
        """
        new_texts = []
        for text in texts:
            text = text.strip('「」')
            new_texts.append(text)
        return new_texts

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str], ctx=None) -> List[str]:
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')
        self.logger.info(f'Sakura current endpoint: {self.api_base}')
        self.logger.debug(f'Original text: {queries}')
        text_prompt = '\n'.join(queries)
        self.logger.debug('-- Sakura Prompt --\n' + self._format_prompt_log(text_prompt) + '\n\n')

        # Preprocess the query text
        queries = self._preprocess_queries(queries)

        # Send the translation request
        response = await self._handle_translation_request(queries)
        response = self._normalize_response_text(response)
        self.logger.debug(f'-- Sakura Response --\n{response}\n\n')

        # Check the translation for repetition or a line count mismatch
        translations = await self._check_translation_quality(queries, response)

        return self._delete_quotation_mark(translations)

    async def _handle_translation_request(self, prompt) -> str:
        """
        Handle a translation request, including error handling and the retry logic.
        """
        ratelimit_attempt = 0
        server_error_attempt = 0
        timeout_attempt = 0
        while True:
            try:
                request_task = asyncio.create_task(self._request_translation(prompt))
                response = await asyncio.wait_for(request_task, timeout=self._TIMEOUT)
                break
            except asyncio.TimeoutError:
                timeout_attempt += 1
                if timeout_attempt >= self._TIMEOUT_RETRY_ATTEMPTS:
                    raise Exception('Sakura request timed out.')
                self.logger.warning(f'Sakura retrying after a timeout. Attempts: {timeout_attempt}')
            except openai.RateLimitError:
                ratelimit_attempt += 1
                if ratelimit_attempt >= self._RATELIMIT_RETRY_ATTEMPTS:
                    raise
                self.logger.warning(f'Sakura retrying due to rate limiting. Attempt: {ratelimit_attempt}')
                await asyncio.sleep(2)
            except (openai.APIError, openai.APIConnectionError) as e:
                server_error_attempt += 1
                if server_error_attempt >= self._RETRY_ATTEMPTS:
                    self.logger.error(f'Sakura API request failed. URL: {self.api_base}, error: {e}')
                    raise Exception(f'Sakura API request failed (URL: {self.api_base}): {e}') from e
                self.logger.warning(f'Sakura retrying due to a server error. URL: {self.api_base}, attempt: {server_error_attempt}, error: {e}')

        return response

    async def _request_translation(self, input_text_list) -> str:
        """
        Send a translation request to the Sakura API.
        """
        if isinstance(input_text_list, list):
            raw_text = "\n".join(input_text_list)
        else:
            raw_text = input_text_list
        raw_lenth = len(raw_text)
        max_lenth = 512
        max_token_num = max(raw_lenth*2, max_lenth)
        extra_query = {
            'do_sample': False,
            'num_beams': 1,
            'repetition_penalty': 1.0,
        }
        gpt_dict_raw_text = self.sakura_dict.get_dict_str()
        self.logger.debug(f"Sakura Dict: {gpt_dict_raw_text}")
        if gpt_dict_raw_text:
            user_prompt = f"根据以下术语表：\n{gpt_dict_raw_text}\n将下面的日文文本根据上述术语表的对应关系和注释翻译成中文：{raw_text}"
        else:
            user_prompt = f"将下面的日文文本翻译成中文：{raw_text}"
        messages = [
            {
                "role": "system",
                "content": f"{self._CHAT_SYSTEM_TEMPLATE}"
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        response = await self.client.chat.completions.create(
            model="sukinishiro",
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_token_num,
            frequency_penalty=self.frequency_penalty,
            seed=-1,
            extra_query=extra_query,
        )
        
        # Check that the response object is valid
        validate_openai_response(response, self.logger)
        
        # Extract and return the response text
        for choice in response.choices:
            if 'text' in choice:
                return choice.text

        return response.choices[0].message.content

    def _set_gpt_style(self, style_name: str):
        """
        Set the generation style of GPT.
        """
        if self._current_style == style_name:
            return
        self._current_style = style_name
        if style_name == "precise":
            temperature, top_p = 0.1, 0.3
            frequency_penalty = 0.05
        elif style_name == "normal":
            temperature, top_p = 0.3, 0.3
            frequency_penalty = 0.2
        elif style_name == "aggressive":
            temperature, top_p = 0.3, 0.3
            frequency_penalty = 0.3

        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
