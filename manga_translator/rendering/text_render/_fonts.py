"""Qt font runtime.

Bootstraps the off-screen QGuiApplication, registers font files and cleans foundry square brackets, and holds the thread-local
FontState (font selection and the caches at each level), the construction of QFont/QTextLayout and the choice of hyphenator.
"""
import logging
import os
import sys
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field

from hyphen import Hyphenator
from hyphen.dictools import LANGUAGES as HYPHENATOR_LANGUAGES
from langcodes import standardize_tag
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QFont, QFontDatabase, QGuiApplication, QRawFont, QTextLayout

from ...utils import BASE_PATH
from ..rich_text import TextStyle
from ._shared import _QFONT_CACHE_MAX, _QT_FONT_PROBE_SIZE, _RAW_FONT_CACHE_MAX, _cache_get, _cache_put
from manga_translator.utils.swallowed import note_ignored_error

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    HYPHENATOR_LANGUAGES.remove('fr')
    HYPHENATOR_LANGUAGES.append('fr_FR')
except Exception as ignored_error:
    note_ignored_error(ignored_error, "manga_translator/rendering/text_render/_fonts.py:<module>")
    pass

DEFAULT_FONT_FAMILY = 'Microsoft YaHei UI'
FONT_STYLE_SEPARATOR = '::'
_thread_state = threading.local()
_qt_runtime_lock = threading.Lock()
_qt_runtime_app = None
_font_descriptor_cache = {}
_font_families_cache = {}
_font_registration_ids = {}
_font_file_signatures = {}
_font_original_names = {}
_font_family_aliases = {}
_font_registry_revision = 0
_project_font_paths = set()
_hyphenator_cache = {}


@dataclass(frozen=True)
class LayoutFontDescriptor:
    family: str
    style: str = ''


@dataclass
class FontState:
    registry_revision: int = field(default_factory=lambda: _font_registry_revision)
    font_family: str = ''
    font_style: str = ''
    bold: bool = False
    raw_fonts: dict = field(default_factory=OrderedDict)
    qfonts: dict = field(default_factory=OrderedDict)
    glyph_specs: dict = field(default_factory=OrderedDict)
    glyphs: dict = field(default_factory=OrderedDict)
    measures: dict = field(default_factory=dict)
    vertical: dict = field(default_factory=OrderedDict)


def _bootstrap_qt_fontdir_for_offscreen() -> None:
    """Help Qt's offscreen/freetype font database find bundled fonts.

    Qt's offscreen plugin may rely on QT_QPA_FONTDIR or Qt6/lib/fonts. Our
    packaged fonts live under the project fonts/ directory, so expose that as a
    default font directory when running in offscreen mode and the caller did not
    already provide one.
    """
    if os.environ.get('QT_QPA_FONTDIR'):
        return
    if os.environ.get('QT_QPA_PLATFORM') != 'offscreen':
        return
    font_dir = os.path.join(BASE_PATH, 'fonts')
    if os.path.isdir(font_dir):
        os.environ['QT_QPA_FONTDIR'] = font_dir
        logger.info('Using bundled fonts for Qt offscreen mode: %s', font_dir)


def _ensure_qt_runtime():
    global _qt_runtime_app
    app = QGuiApplication.instance()
    if app is not None:
        return app
    with _qt_runtime_lock:
        app = QGuiApplication.instance()
        if app is None:
            os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
            _bootstrap_qt_fontdir_for_offscreen()
            _qt_runtime_app = QGuiApplication([])
            return _qt_runtime_app
        return app


# The bundled Thai fonts were first shipped with this prefix in their family
# and file names. Settings and projects saved by those versions still resolve.
_LEGACY_FAMILY_PREFIX = 'MangaX '
_LEGACY_FILE_PREFIX = 'MangaX-'


def legacy_font_family(family: str) -> str:
    """The current family name for a legacy prefixed one, or '' when it is not one."""
    family = str(family or '')
    if family.casefold().startswith(_LEGACY_FAMILY_PREFIX.casefold()):
        return family[len(_LEGACY_FAMILY_PREFIX):]
    return ''


def _normalize_font_path(path: str) -> str:
    return path.replace('\\', '/')


def _resolve_existing_font_path(path: str) -> str:
    if not path:
        return ''
    path = _normalize_font_path(path)
    candidates = [path]
    if not os.path.isabs(path):
        candidates.extend([
            _normalize_font_path(os.path.join(BASE_PATH, 'fonts', os.path.basename(path))),
            _normalize_font_path(os.path.join(BASE_PATH, path)),
        ])
    name = os.path.basename(path)
    if name.casefold().startswith(_LEGACY_FILE_PREFIX.casefold()):
        candidates.append(_normalize_font_path(
            os.path.join(BASE_PATH, 'fonts', name[len(_LEGACY_FILE_PREFIX):])))
    return next((candidate for candidate in candidates if candidate and os.path.exists(candidate)), '')


# Qt parses the bracketed part of "Family [Foundry]" as the foundry name (parseFontName in qfontdatabase);
# a family name that starts with "[" is split into "empty family + foundry", and matching degrades to any font of that foundry,
# unrelated to the font that was asked for (for example, a whole series whose names start with a bracketed tag hits the same file).
_QT_FOUNDRY_SENSITIVE_NAME_IDS = (1, 3, 4, 16, 21)


def qt_family_is_ambiguous(family: str) -> bool:
    """Return True when Qt's "Family [Foundry]" parsing yields an empty family."""
    name = (family or '').strip()
    return name.startswith('[') and name.rfind(']') > 0


def strip_qt_foundry_brackets(family: str) -> str:
    return (family or '').replace('[', '').replace(']', '').strip()


def _font_registration_key(path: str) -> str:
    return _normalize_font_path(os.path.normcase(os.path.abspath(path)))


def font_registry_revision() -> int:
    return _font_registry_revision


def _font_registry_changed() -> None:
    global _font_registry_revision
    _font_registry_revision += 1
    _font_descriptor_cache.clear()
    _font_family_aliases.clear()
    for path, names in sorted(_font_original_names.items()):
        for name in names:
            for variant in (name, strip_qt_foundry_brackets(name)):
                if variant:
                    _font_family_aliases.setdefault(variant.casefold(), path)


def unregister_font_file(path: str) -> bool:
    """Remove a registered file and invalidate selectors/physical-font caches."""
    key = _font_registration_key(path)
    font_id = _font_registration_ids.get(key)
    if font_id is None and key not in _font_families_cache:
        return True
    if font_id is not None and not QFontDatabase.removeApplicationFont(font_id):
        logger.warning('Could not unregister font: %s', path)
        return False
    _font_registration_ids.pop(key, None)
    _font_file_signatures.pop(key, None)
    _font_families_cache.pop(key, None)
    _font_original_names.pop(key, None)
    _font_registry_changed()
    return True


def _sanitized_font_bytes(path: str):
    """Return ``(font data with the square brackets removed from the name table, list of original family names)``.

    The data is None when nothing needs doing or the rewrite fails. The original family names (the nameID 1/4/16 records of every language)
    are used to build the "old name -> file" alias mapping.
    """
    if not path.lower().endswith(('.ttf', '.otf')):
        return None, []
    try:
        import io
        from fontTools.ttLib import TTFont
    except ImportError:
        logger.warning('fontTools unavailable; cannot rewrite bracketed font names: %s', path)
        return None, []
    try:
        font = TTFont(path, lazy=True)
        try:
            name_table = font['name']
            changed = False
            original_names = []
            for record in name_table.names:
                if record.nameID not in _QT_FOUNDRY_SENSITIVE_NAME_IDS:
                    continue
                try:
                    value = record.toUnicode()
                except UnicodeDecodeError:
                    continue
                if record.nameID in (1, 4, 16) and value:
                    original_names.append(value)
                if '[' in value or ']' in value:
                    record.string = strip_qt_foundry_brackets(value)
                    changed = True
            if not changed:
                return None, []
            # Add an English preferred family name (nameID 16) when it is missing: the freetype font database of the offscreen platform
            # prefers nameID 16 when choosing a name, and without an English record it picks the Chinese name and turns it into garbage.
            en_family = name_table.getName(1, 3, 1, 0x409)
            if en_family is not None and name_table.getName(16, 3, 1, 0x409) is None:
                name_table.setName(en_family.toUnicode(), 16, 3, 1, 0x409)
            if 'DSIG' in font:
                del font['DSIG']  # After the name table changes, the original signature is bound to be invalid
            buffer = io.BytesIO()
            font.save(buffer)
            return buffer.getvalue(), original_names
        finally:
            font.close()
    except Exception:
        logger.exception('Failed to sanitize font name table: %s', path)
        return None, []


def register_font_file(path: str) -> list:
    """Register a font file and return the list of families that are safe to use for QFont matching.

    A font whose family name starts with "[" (a bracketed tag in front of the name) is registered as an in-memory copy
    without the square brackets, to get around Qt's foundry syntax; empty names and names with square brackets are filtered out of the returned list.
    """
    key = _font_registration_key(path)
    if QGuiApplication.instance() is None:
        return []
    try:
        stat = os.stat(path)
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError:
        if key in _font_registration_ids:
            unregister_font_file(path)
        return []
    cached = _font_families_cache.get(key)
    if cached is not None and _font_file_signatures.get(key) == signature:
        return cached
    if key in _font_registration_ids and not unregister_font_file(path):
        return cached or []

    families = []
    font_id = -1
    original_names = []
    try:
        font_id = QFontDatabase.addApplicationFont(path)
        families = list(QFontDatabase.applicationFontFamilies(font_id)) if font_id >= 0 else []
        if any(qt_family_is_ambiguous(name) for name in families):
            sanitized, original_names = _sanitized_font_bytes(path)
            if sanitized is not None:
                QFontDatabase.removeApplicationFont(font_id)
                font_id = QFontDatabase.addApplicationFontFromData(sanitized)
                families = list(QFontDatabase.applicationFontFamilies(font_id)) if font_id >= 0 else []
                # Old configurations and rich-text styles may still store the original name (with its Chinese variant);
                # record an "original name / name without brackets -> file" mapping as a fallback for set_font.
                logger.info(
                    'Registered bracketed font with sanitized families: %s -> %s',
                    os.path.basename(path), families,
                )
            else:
                logger.warning(
                    'Font family uses Qt foundry brackets and could not be rewritten; '
                    'QFont matching may pick a wrong font: %s', path,
                )
    except Exception:
        logger.exception('Failed to register font: %s', path)

    families = [name for name in families if name and not qt_family_is_ambiguous(name)]
    if font_id >= 0:
        _font_registration_ids[key] = font_id
        _font_file_signatures[key] = signature
        _font_families_cache[key] = families
        _font_original_names[key] = tuple(original_names)
        _font_registry_changed()
    return families


def _register_project_fonts() -> None:
    """Register custom project fonts in Qt; rendering still addresses them by family."""
    font_dir = os.path.join(BASE_PATH, 'fonts')
    current_paths = set()
    for root, directories, filenames in os.walk(font_dir):
        directories.sort()
        for filename in sorted(filenames):
            if not filename.lower().endswith(('.ttf', '.otf', '.ttc', '.pfb')):
                continue
            path = _font_registration_key(os.path.join(root, filename))
            current_paths.add(path)
            register_font_file(path)
    for path in _project_font_paths - current_paths:
        unregister_font_file(path)
    _project_font_paths.clear()
    _project_font_paths.update(current_paths)


def _system_font_dirs() -> list:
    if sys.platform == 'win32':
        dirs = [os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')]
        local_appdata = os.environ.get('LOCALAPPDATA')
        if local_appdata:
            # Windows now puts fonts installed by the user here by default
            dirs.append(os.path.join(local_appdata, 'Microsoft', 'Windows', 'Fonts'))
    elif sys.platform == 'darwin':
        dirs = ['/System/Library/Fonts', '/Library/Fonts', os.path.expanduser('~/Library/Fonts')]
    else:
        dirs = ['/usr/share/fonts', '/usr/local/share/fonts',
                os.path.expanduser('~/.local/share/fonts'), os.path.expanduser('~/.fonts')]
    return [font_dir for font_dir in dirs if os.path.isdir(font_dir)]


_system_fonts_registered = False


def _register_system_fonts() -> bool:
    """The font database of the offscreen/minimal platforms does not enumerate system fonts (it only scans QT_QPA_FONTDIR),
    so a system font chosen in desktop mode would not be found in the CLI or on the server; on the first family name that misses, the whole system
    font folder is registered with Qt, scanned only once per process. Returns whether looking the family name up again is worthwhile.
    """
    global _system_fonts_registered
    if _system_fonts_registered:
        return False
    _system_fonts_registered = True
    app = QGuiApplication.instance()
    if app is None or app.platformName() not in ('offscreen', 'minimal'):
        return False
    count = 0
    for font_dir in _system_font_dirs():
        for root, _, filenames in os.walk(font_dir):
            for filename in filenames:
                if not filename.lower().endswith(('.ttf', '.otf', '.ttc', '.otc')):
                    continue
                register_font_file(os.path.join(root, filename))
                count += 1
    logger.info('Registered system fonts for headless Qt: %d files', count)
    return count > 0


def _state() -> FontState:
    state = getattr(_thread_state, 'value', None)
    if state is None:
        state = FontState()
        _thread_state.value = state
        set_font(DEFAULT_FONT_FAMILY)
    if state.registry_revision != _font_registry_revision:
        for cache in (state.raw_fonts, state.qfonts, state.glyph_specs, state.glyphs,
                      state.measures, state.vertical):
            cache.clear()
        state.registry_revision = _font_registry_revision
    return _thread_state.value


def _raw_font(path: str, pixel_size: float) -> QRawFont:
    state = _state()
    _ensure_qt_runtime()
    norm_path = _normalize_font_path(path)
    pixel_size = float(max(pixel_size, 1.0))
    key = (norm_path, pixel_size)
    font = _cache_get(state.raw_fonts, key)
    if font is not None:
        return font
    # Reuse an existing instance: find a font of any size with the same path, copy it, then setPixelSize
    for cached_key in reversed(state.raw_fonts):
        if cached_key[0] == norm_path:
            base = state.raw_fonts[cached_key]
            font = QRawFont(base)
            font.setPixelSize(pixel_size)
            return _cache_put(state.raw_fonts, key, font, _RAW_FONT_CACHE_MAX)
    # First load: create from the file
    font = QRawFont(norm_path, pixel_size)
    if not font.isValid():
        raise RuntimeError(f'Could not load Qt font: {norm_path}')
    return _cache_put(state.raw_fonts, key, font, _RAW_FONT_CACHE_MAX)


def _font_descriptor(path: str) -> LayoutFontDescriptor:
    _ensure_qt_runtime()
    path = _normalize_font_path(path)
    registered_families = register_font_file(path)
    descriptor = _font_descriptor_cache.get(path)
    if descriptor:
        return descriptor

    family = ''
    style = ''
    try:
        raw = _raw_font(path, _QT_FONT_PROBE_SIZE)
        if raw.isValid():
            family = raw.familyName() or ''
            style = raw.styleName() or ''
    except Exception as ignored_error:
        note_ignored_error(ignored_error, "manga_translator/rendering/text_render/_fonts.py:_font_descriptor")
        pass

    # A family name with square brackets read from the file cannot be handed to QFont matching as it is (foundry syntax);
    # use the cleaned name returned when it was registered instead.
    if (not family or qt_family_is_ambiguous(family)) and registered_families:
        family = registered_families[0]


    if not family:
        raise RuntimeError(f'Could not resolve Qt font family: {path}')
    if qt_family_is_ambiguous(family):
        logger.warning('Bracketed font family may not match correctly in Qt: %s (%s)', family, path)
    descriptor = LayoutFontDescriptor(family=family, style=style)
    _font_descriptor_cache[path] = descriptor
    return descriptor


def _split_font_value(value: str) -> tuple[str, str]:
    """Parse the optional style suffix used by desktop font selectors."""
    family, separator, style = str(value or '').rpartition(FONT_STYLE_SEPARATOR)
    if separator and family and style:
        return family, style
    return str(value or ''), ''


def _set_family(state: FontState, family: str, style: str = ''):
    if state.font_family == family and state.font_style == style:
        return
    state.font_family = family
    state.font_style = style
    state.qfonts.clear()
    # The keys of glyph_specs/glyphs include the font family, so nothing needs clearing when the font changes;
    # the cache is kept to avoid parsing the glyph data of large font files again
    state.measures.clear()
    state.vertical.clear()


def _match_family(requested: str):
    """Resolve a family name in the Qt font database; returns a usable family name or None."""
    if not requested:
        return None
    available = {name.casefold(): name for name in QFontDatabase.families()}
    family = available.get(requested.casefold())
    if family is None or qt_family_is_ambiguous(family):
        # A family name with a bracketed prefix, from an old configuration or a system install, cannot go through QFont matching (foundry syntax);
        # map it to the bracket-free name created when it was registered.
        stripped = strip_qt_foundry_brackets(requested)
        stripped_family = available.get(stripped.casefold()) if stripped else None
        if stripped_family and not qt_family_is_ambiguous(stripped_family):
            return stripped_family
        # The font database of offscreen and similar environments may not provide the Chinese family name; fall back to registering the file and using its family name
        alias_path = _font_family_aliases.get(requested.casefold()) or (
            _font_family_aliases.get(stripped.casefold()) if stripped else None)
        if alias_path and os.path.exists(alias_path):
            try:
                return _font_descriptor(alias_path).family
            except Exception:
                logger.exception('Could not load font file: %s', alias_path)
    return family


def set_font(font: str):
    """Select a Qt family/style; a legacy font file path is mapped to its face."""
    state = getattr(_thread_state, 'value', None) or FontState()
    _thread_state.value = state
    requested = str(font or '').strip()
    resolved = _resolve_existing_font_path(requested)
    if resolved:
        try:
            descriptor = _font_descriptor(resolved)
            _set_family(state, descriptor.family, descriptor.style)
            return
        except Exception:
            logger.exception('Could not load font file: %s', resolved)

    _ensure_qt_runtime()
    _register_project_fonts()
    requested_family, requested_style = _split_font_value(requested)
    family = _match_family(requested_family)
    if family is None and legacy_font_family(requested_family):
        family = _match_family(legacy_font_family(requested_family))
    if family is None and requested and _register_system_fonts():
        family = _match_family(requested_family)
    if family is not None and qt_family_is_ambiguous(family):
        logger.warning('Bracketed font family may not match correctly in Qt: %s', family)
    if family is None:
        # Headless, QGuiApplication.font() is the logical font "Sans Serif", whose match is arbitrary;
        # fall back directly to a default family that is certain to exist in the bundled fonts folder.
        family = DEFAULT_FONT_FAMILY
        if requested:
            logger.warning('Qt font family not found: %s; using %s', requested, family)
    _set_family(state, family, requested_style)


def load_font_file(path: str) -> str:
    """Register a font file and return its Qt family without persisting the path."""
    resolved = _resolve_existing_font_path(path)
    if not resolved:
        raise FileNotFoundError(path)
    return _font_descriptor(resolved).family


def set_bold(bold: bool):
    state = _state()
    value = bool(bold)
    if state.bold == value:
        return
    state.bold = value
    state.measures.clear()
    state.vertical.clear()


@contextmanager
def _bold_scope(bold: bool):
    state = _state()
    previous = state.bold
    state.bold = bool(bold)
    try:
        yield
    finally:
        state.bold = previous


@contextmanager
def _style_font_scope(style: TextStyle):
    state = _state()
    previous_family = state.font_family
    previous_style = state.font_style
    previous_bold = state.bold
    requested_font = getattr(style, 'font_family', None)
    if requested_font:
        set_font(requested_font)
    state.bold = bool(getattr(style, 'bold', False))
    try:
        yield
    finally:
        _set_family(state, previous_family or DEFAULT_FONT_FAMILY, previous_style)
        state.bold = previous_bold


def _layout_font(font_size: int, letter_spacing: float) -> QFont:
    state = _state()
    family = state.font_family or DEFAULT_FONT_FAMILY
    style = state.font_style
    key = (family, style, bool(state.bold), int(max(font_size, 1)), round(float(letter_spacing), 4))
    qfont = _cache_get(state.qfonts, key)
    if qfont is None:
        qfont = QFontDatabase.font(family, style, 12) if style else QFont()
        if not style:
            qfont.setFamilies([family])
        if state.bold or not style:
            qfont.setBold(state.bold)
        qfont.setPixelSize(key[3])
        qfont.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        qfont.setStyleStrategy(QFont.StyleStrategy.PreferOutline)
        qfont.setKerning(True)
        qfont.setLetterSpacing(QFont.SpacingType.PercentageSpacing, float(letter_spacing) * 100.0)
        _cache_put(state.qfonts, key, qfont, _QFONT_CACHE_MAX)
    return QFont(qfont)


def _create_text_layout(text: str, font_size: int, letter_spacing: float = 1.0):
    qfont = _layout_font(font_size, letter_spacing)
    if not text:
        return text, qfont, None, None
    layout = QTextLayout(text, qfont)
    layout.beginLayout()
    line = layout.createLine()
    if not line.isValid():
        layout.endLayout()
        return text, qfont, None, None
    line.setLineWidth(1_000_000.0)
    line.setPosition(QPointF(0.0, 0.0))
    layout.endLayout()
    return text, qfont, layout, line


def select_hyphenator(lang: str):
    lang = standardize_tag(lang or 'en_US')
    if lang not in HYPHENATOR_LANGUAGES:
        lang = next((avail for avail in reversed(HYPHENATOR_LANGUAGES) if avail.startswith(lang)), '')
    if not lang:
        return None
    if lang not in _hyphenator_cache:
        try:
            _hyphenator_cache[lang] = Hyphenator(lang)
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/rendering/text_render/_fonts.py:select_hyphenator")
            _hyphenator_cache[lang] = None
    return _hyphenator_cache[lang]
