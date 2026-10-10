"""Durable chapter work, reviewed translation memory and measured quality checks.

Artifacts are isolated from source/editor files. Applying a result is explicit.
"""
import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import tempfile
import threading
import time
import unicodedata
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from manga_translator.translators.manga_context import (
    context_path, fingerprint, glossary_entries, load_chapter_context,
)

logger = logging.getLogger(__name__)

STAGES = ('ocr', 'translation', 'clean', 'layout')
RESUMABLE = ('paused', 'failed', 'interrupted', 'cancelled')
# Per-region keys owned by the assistant. They never influence pixels.
METADATA_KEYS = ('speaker', 'listener', 'text_kind', 'locked', '_mangax_review')
STYLE_KITS = {
    'dialogue': {'font_family': 'MangaX Itim', 'line_spacing': 1.3, 'direction': 'h', 'alignment': 'center'},
    'thought': {'font_family': 'MangaX Mali', 'line_spacing': 1.4, 'direction': 'h', 'alignment': 'center'},
    'narration': {'font_family': 'MangaX Pridi', 'line_spacing': 1.4, 'direction': 'h', 'alignment': 'left'},
    'shout': {'font_family': 'MangaX Kanit::Bold', 'line_spacing': 1.2, 'direction': 'h', 'alignment': 'center'},
    'sfx': {'font_family': 'MangaX Chonburi', 'line_spacing': 1.1, 'direction': 'h', 'alignment': 'center'},
}


def tr(key, **kwargs):
    """Translate a user-facing message through the desktop i18n service when it is running."""
    manager = None
    try:
        from services import get_i18n_manager
        manager = get_i18n_manager()
    except Exception:
        manager = None
    if manager is not None:
        text = manager.translate(key, **kwargs)
        if text != key:
            return text
    return key.format(**kwargs) if kwargs else key


def jsonable(value):
    """Detach editor data (numpy arrays, tuples) into plain JSON values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.mangax-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _direction(value):
    return {'horizontal': 'h', 'vertical': 'v'}.get(value, value)


def _color(value):
    """One representation for '#rrggbb' and [r, g, b]; the editor save converts between them."""
    if isinstance(value, str) and value.startswith('#') and len(value) == 7:
        try:
            return [int(value[i:i + 2], 16) for i in (1, 3, 5)]
        except ValueError:
            return value
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3:
        try:
            return [int(round(float(channel))) for channel in value]
        except (TypeError, ValueError):
            return jsonable(value)
    return jsonable(value)


def _number(value, digits=3):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def world_lines(region):
    """Region polygons as they sit on the page: stored lines are unrotated around 'center'."""
    lines = np.asarray(region.get('lines', []), dtype=np.float64).reshape(-1, 4, 2)
    try:
        angle = float(region.get('angle') or 0)
    except (TypeError, ValueError):
        angle = 0.0
    if not len(lines) or abs(angle) < 1e-6:
        return lines
    center = region.get('center')
    if not isinstance(center, (list, tuple, np.ndarray)) or len(center) != 2:
        points = lines.reshape(-1, 2)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
    center = np.asarray(center, dtype=np.float64)
    radians = math.radians(angle)
    rotation = np.array([[math.cos(radians), math.sin(radians)], [-math.sin(radians), math.cos(radians)]])
    return (lines - center) @ rotation + center


def geometry_stamp(region):
    """What mask refinement reads from a region: its page polygons, nothing else."""
    try:
        return np.round(world_lines(region), 1).tolist()
    except (TypeError, ValueError):
        return jsonable(region.get('lines'))


def review_stamp(region):
    """Fingerprint of everything a reviewer approved, stable across the editor's save format."""
    plain = ('text', 'translation', 'translation_rich', 'speaker', 'listener', 'text_kind',
             'font_family', 'alignment', 'white_frame')
    value = {key: jsonable(region.get(key)) for key in plain}
    for key in ('font_size', 'stroke_width', 'line_spacing', 'letter_spacing', 'angle'):
        value[key] = _number(region.get(key))
    value['direction'] = _direction(region.get('direction'))
    foreground = region.get('font_color')
    if foreground is None:
        foreground = region.get('fg_colors', region.get('fg_color'))
    value['fg'] = _color(foreground)
    value['bg'] = _color(region.get('bg_colors', region.get('bg_color')))
    value['lines'] = geometry_stamp(region)
    return fingerprint(value)


def reviewed(region):
    return bool(region.get('_mangax_review')) and region['_mangax_review'] == review_stamp(region)


def approved_memory_key(region, context, target):
    return fingerprint([region.get('text', ''), region.get('speaker', ''),
                        region.get('listener', ''), target, context.get('story', ''),
                        context.get('characters', []), context.get('glossary', [])])


def region_block(region, config=None):
    """Build a TextBlock the way the backend loads an editor JSON region (colors, stroke, lines)."""
    from manga_translator.utils import TextBlock
    args = {key: value for key, value in copy.deepcopy(region).items() if key not in METADATA_KEYS}
    text = str(args.pop('text', '') or '')
    if not args.get('texts'):
        args['texts'] = [text]
    foreground = args.pop('font_color', None)
    colors = args.pop('fg_colors', None)
    if isinstance(foreground, str) and foreground.startswith('#') and len(foreground) == 7:
        args['fg_color'] = tuple(_color(foreground))
    elif isinstance(colors, (list, tuple)) and len(colors) == 3:
        args['fg_color'] = tuple(colors)
    background = args.pop('bg_colors', None)
    if isinstance(background, (list, tuple)) and len(background) == 3:
        args['bg_color'] = tuple(background)
    if 'stroke_width' in args:
        args['default_stroke_width'] = args.pop('stroke_width')
    if 'direction' in args:
        args['direction'] = _direction(args['direction'])
    if not args.get('target_lang') and config is not None:
        args['target_lang'] = config.translator.target_lang
    lines = np.asarray(args.get('lines', []), dtype=np.float64)
    if lines.shape == (4, 2):
        lines = lines.reshape(1, 4, 2)
    if lines.ndim != 3 or lines.shape[1:] != (4, 2) or not len(lines):
        raise ValueError(tr('A text box has no valid coordinates'))
    args['lines'] = lines
    for key in ('font_size', 'angle'):
        if args.get(key) is None:
            args.pop(key, None)
    args['adjust_bg_color'] = False  # Colors were confirmed in the editor.
    return TextBlock(**args)


def straighten(image, quad, curvature=0):
    """Rectify ordered TL/TR/BR/BL corners into a detached RGB crop."""
    points = np.asarray(quad, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError(tr('Enter four corners in order: top-left, top-right, bottom-right, bottom-left'))
    h, w = image.shape[:2]
    if (points[:, 0] < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] < 0).any() or (points[:, 1] >= h).any():
        raise ValueError(tr('Corners must be inside the image'))
    if not cv2.isContourConvex(points.astype(np.int32)) or abs(cv2.contourArea(points)) < 16:
        raise ValueError(tr('The four-corner frame must not cross itself or be too small'))
    width = max(4, int(max(np.linalg.norm(points[1]-points[0]), np.linalg.norm(points[2]-points[3]))))
    height = max(4, int(max(np.linalg.norm(points[3]-points[0]), np.linalg.norm(points[2]-points[1]))))
    if width * height > 40_000_000:
        raise ValueError(tr('The OCR frame is too large'))
    dest = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, dest), (width, height), borderMode=cv2.BORDER_REPLICATE)
    if curvature:
        yy, xx = np.indices((height, width), dtype=np.float32)
        yy += float(curvature) * height * (1 - (2 * xx / max(width-1, 1) - 1) ** 2)
        crop = cv2.remap(crop, xx, yy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return crop


def _font_problem(value):
    """Select `value` exactly like the renderer does and report what it could not honor.

    Region fonts are saved as 'Family', 'Family::Style' or (legacy) a font file path.
    Returns (problem message or None, QRawFont of the face that will really be used).
    """
    from manga_translator.rendering import text_render
    from PyQt6.QtGui import QFont, QFontDatabase, QRawFont
    text_render.set_font(value)  # Registers project fonts; resolves paths, case and aliases.
    state = text_render._state()
    family, style = state.font_family, state.font_style
    requested, separator, requested_style = str(value).rpartition('::')
    if not (separator and requested and requested_style):
        requested, requested_style = str(value), ''
    problem = None
    if os.path.splitext(str(value))[1].lower() in ('.ttf', '.otf', '.ttc', '.pfb'):
        # Legacy projects stored a font file path; the renderer maps it to that file's face.
        if not text_render._fonts._resolve_existing_font_path(str(value).strip()):
            problem = tr('Font not found: {font}', font=os.path.basename(str(value)))
    elif family.casefold() not in (requested.casefold(), text_render.strip_qt_foundry_brackets(requested).casefold()):
        problem = tr('Font not found: {font}', font=requested)
    elif requested_style and requested_style.casefold() not in {name.casefold() for name in QFontDatabase.styles(family)}:
        problem = tr('Font not found: {font}', font=f'{requested} ({requested_style})')
    qfont = QFontDatabase.font(family, style, 12) if style else QFont(family)
    return problem, QRawFont.fromFont(qfont)


def quality_issues(path, regions, context, config):
    """Measured checks for one page.

    Font work goes through the renderer's own registry (`text_render.set_font`), whose state is
    thread-local and whose QFontDatabase calls are serialized by Qt; the backend export already
    renders on worker threads the same way, so this is safe off the GUI thread. `config` must be a
    snapshot the caller owns.
    """
    from manga_translator.rendering import calc_box_from_font
    issues = []
    def add(index, kind, message):
        issues.append({'path': str(path), 'index': index, 'kind': kind, 'message': message})
    if not regions:
        add(-1, 'no_regions', tr('No text boxes yet: run whole-page OCR or add a box manually'))
    terms = glossary_entries(context)
    for index, region in enumerate(regions):
        source, translation = str(region.get('text', '')).strip(), str(region.get('translation', '')).strip()
        if not source:
            add(index, 'source', tr('No source text'))
        if source and not translation:
            add(index, 'translation', tr('No translation yet'))
        probability = region.get('prob', region.get('confidence'))
        if isinstance(probability, (int, float)) and probability < .65:
            add(index, 'confidence', tr('OCR should be re-checked (this score is not translation accuracy)'))
        if region.get('_mangax_review') and not reviewed(region):
            add(index, 'review_changed', tr('Text or style changed after review'))
        elif translation and not reviewed(region):
            add(index, 'review', tr('Translation has not been reviewed'))
        for term in terms:
            if term['original'].casefold() in source.casefold() and translation and term['translation'] not in translation:
                add(index, 'glossary', tr('Term does not match the glossary: {original} → {translation}',
                                          original=term['original'], translation=term['translation']))
        if not translation:
            continue
        try:
            value = str(region.get('font_family') or config.render.font_family or 'MangaX Itim')
            problem, raw = _font_problem(value)
            if problem:
                add(index, 'font', problem)
            missing = sorted({c for c in translation if not c.isspace() and c not in '[]'
                              and unicodedata.category(c)[0] != 'C' and not raw.supportsCharacter(c)})
            if missing:
                add(index, 'glyph', tr('Font is missing characters: {characters}', characters=''.join(missing)[:30]))
            lines = np.asarray(region.get('lines', []), dtype=float).reshape(-1, 2)
            if not len(lines):
                raise ValueError(tr('A text box has no valid coordinates'))
            width, height = lines[:, 0].max()-lines[:, 0].min(), lines[:, 1].max()-lines[:, 1].min()
            metrics = calc_box_from_font(max(1, int(region.get('font_size') or 24)),
                region.get('translation_rich') or translation, _direction(region.get('direction')) != 'v',
                float(region.get('line_spacing') or 1), config, region.get('target_lang') or config.translator.target_lang,
                letter_spacing=float(region.get('letter_spacing') or 1), stroke_width=float(region.get('stroke_width', .07) or 0))
            # Report source text-frame overflow; the editor may deliberately grow its white frame.
            if metrics[0] > width + 2 or metrics[1] > height + 2:
                add(index, 'overflow', tr('Text overflows the source box: {text_size} / {box_size} px',
                                          text_size=f'{int(metrics[0])}×{int(metrics[1])}', box_size=f'{int(width)}×{int(height)}'))
        except Exception as exc:
            add(index, 'geometry', tr('Could not measure the text: {error}', error=str(exc)[:140]))
    return issues


class ChapterStudio:
    """One worker-owned queue. No Qt widgets or live editor mutation here.

    `self.lock` guards the job list and every job dict: the worker mutates them only through
    locked helpers and other threads read them through `snapshot()`.
    """
    def __init__(self, root, file_service, translation_service):
        self.app_root = Path(root)  # One root for artifacts and chapter context alike.
        self.root = self.app_root / 'result' / 'chapter-studio'
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal = self.root / 'jobs.json'
        self.file_service, self.translation_service = file_service, translation_service
        self.pause = threading.Event()
        self.cancel = threading.Event()
        self.interrupt = threading.Event()  # Application shutdown, not a user decision.
        self.lock = threading.RLock()
        self._loop = self._task = None
        self.recovered_journal = None
        self.jobs = self._load_journal()
        for job in self.jobs:
            if job.get('status') == 'running':
                job['status'] = 'interrupted'
        self._save()

    def _load_journal(self):
        """Read the job journal; an unreadable one is moved aside instead of blocking the editor."""
        if not self.journal.exists():
            return []
        try:
            value = json.loads(self.journal.read_text(encoding='utf-8'))
            if not isinstance(value, list):
                raise ValueError('journal is not a list')
            jobs = []
            for job in value:
                if not (isinstance(job, dict) and isinstance(job.get('id'), str)
                        and isinstance(job.get('paths'), list) and all(isinstance(p, str) for p in job['paths'])
                        and isinstance(job.get('stages'), list) and all(s in STAGES for s in job['stages'])
                        and isinstance(job.get('config'), dict)):
                    raise ValueError('journal entry is malformed')
                job['status'] = job.get('status') if isinstance(job.get('status'), str) else 'interrupted'
                for key in ('completed', 'failures'):
                    if not isinstance(job.get(key), list):
                        job[key] = []
                job['failures'] = [f for f in job['failures'] if isinstance(f, dict)]
                jobs.append(job)
            return jobs
        except Exception as exc:
            backup = self.journal.with_name(self.journal.name + '.corrupt')
            if backup.exists():
                backup = self.journal.with_name(f'{self.journal.name}.{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:6]}.corrupt')
            try:
                os.replace(self.journal, backup)
                self.recovered_journal = backup
                logger.error('Chapter job journal is unreadable (%s); moved to %s and starting empty', exc, backup)
            except OSError as move_error:
                logger.error('Chapter job journal is unreadable (%s) and could not be moved aside: %s', exc, move_error)
            return []

    def _save(self):
        with self.lock:
            atomic_json(self.journal, self.jobs)

    def _update(self, job, save=True, **fields):
        with self.lock:
            job.update(fields)
            if save:
                self._save()

    def _complete(self, job, unit, save=True):
        with self.lock:
            if unit not in job['completed']:
                job['completed'].append(unit)
            if save:
                self._save()

    def _fail(self, job, path, stage, message):
        with self.lock:
            job['failures'].append({'path': path, 'stage': stage, 'message': str(message)[:1000]})
            self._save()

    def snapshot(self, limit=30):
        """Detached, display-sized view of recent jobs for other threads."""
        with self.lock:
            return [{'id': job['id'], 'status': job['status'], 'pages': len(job['paths']),
                     'stages': list(job['stages']), 'completed': len(job['completed']),
                     'failures': [dict(failure) for failure in job['failures']]}
                    for job in self.jobs[-limit:]]

    def job(self, job_id):
        with self.lock:
            return next((job for job in self.jobs if job['id'] == job_id), None)

    def next_job(self, statuses=('queued',), newest=False):
        with self.lock:
            jobs = reversed(self.jobs) if newest else self.jobs
            return next((job for job in jobs if job['status'] in statuses), None)

    def page_dir(self, path):
        # A short digest keeps artifact paths below Windows' 260-character limit.
        return self.root / 'pages' / fingerprint(os.path.normcase(os.path.abspath(path)))[:20]

    def source_identity(self, path):
        from manga_translator.utils.path_manager import find_json_path
        native = find_json_path(path)
        return {'image': file_hash(path), 'editor': file_hash(native) if native else None}

    def _editor_regions(self, path):
        regions, _, _, _ = self.file_service.load_translation_json(str(path))
        return jsonable(regions)

    def read_page(self, path):
        identity = self.source_identity(path)
        checkpoint = self.page_dir(path) / 'page.json'
        data = None
        if checkpoint.is_file():
            try:
                data = json.loads(checkpoint.read_text(encoding='utf-8'))
                if not (isinstance(data, dict) and isinstance(data.get('regions'), list)
                        and isinstance(data.get('source'), dict)):
                    raise ValueError('checkpoint is malformed')
            except Exception as exc:
                logger.warning('Ignoring unreadable chapter checkpoint %s: %s', checkpoint, exc)
                data = None
        if data and data['source'].get('image') == identity['image']:
            if not isinstance(data.get('stages'), dict):
                data['stages'] = {}
            if data['source'] != identity:
                # The editor saved this page since the checkpoint: its text is the truth now.
                # Artifacts stay; each stage key decides whether they are still valid.
                data.update(source=identity, regions=self._editor_regions(path))
                data.pop('editor_seed', None)
            return data
        return {'path': str(path), 'source': identity, 'regions': self._editor_regions(path), 'stages': {}, 'mask': None}

    def save_page(self, path, data):
        atomic_json(self.page_dir(path) / 'page.json', data)

    def seed_editor(self, path, regions):
        """Take the open editor page as input unless the checkpoint is already ahead of it.

        The checkpoint is ahead when the editor still shows exactly what was seeded last time,
        the state an applied result was undone to, or nothing at all on first contact. Results
        of earlier assistant stages are kept in those cases.
        """
        regions = jsonable(copy.deepcopy(regions))
        data = self.read_page(path)
        seed = fingerprint(regions)
        ahead = self.editor_untouched(data, regions)
        if not ahead and fingerprint(data['regions']) != seed:
            data['regions'] = regions  # Derived artifacts stay; stage keys decide their validity.
        data['editor_seed'] = seed
        data.pop('editor_undo_seed', None)
        self.save_page(path, data)
        return data

    @staticmethod
    def editor_untouched(data, regions):
        """True when the editor holds no work newer than this checkpoint."""
        if not data.get('editor_seed'):
            return not regions
        return fingerprint(jsonable(copy.deepcopy(regions))) in (data['editor_seed'], data.get('editor_undo_seed'))

    def mark_seed(self, path, regions, before=None):
        """Record that the editor now shows `regions` after an explicit apply, and what an
        undo of that apply would bring back (`before`)."""
        data = self.read_page(path)
        data['editor_seed'] = fingerprint(jsonable(copy.deepcopy(regions)))
        if before is not None:
            data['editor_undo_seed'] = fingerprint(jsonable(copy.deepcopy(before)))
        self.save_page(path, data)

    def enqueue(self, paths, stages, config, output_dir=''):
        if not paths or any(stage not in STAGES for stage in stages) or not stages:
            raise ValueError(tr('Select an image and the stages to run first'))
        # Configuration is captured at enqueue; unrelated UI changes cannot change a queued job.
        payload = config.model_dump(mode='json')
        job = {'id': uuid.uuid4().hex, 'paths': list(dict.fromkeys(map(str, paths))),
               'stages': list(stages), 'config': payload, 'output_dir': str(output_dir or ''),
               'status': 'queued', 'completed': [], 'failures': []}
        with self.lock:
            if sum(job.get('status') == 'queued' for job in self.jobs) >= 10:
                raise ValueError(tr('The queue is full (10 jobs)'))
            self.jobs.append(job)
            self._save()
        return job

    def stage_key(self, data, stage, config, context):
        """Fingerprint of a stage's true inputs; unrelated edits must not invalidate its result."""
        image = data['source'].get('image')
        regions = data['regions']
        if stage == 'ocr':
            # OCR results are outputs, not inputs.
            inputs = [config.detector.model_dump(mode='json'), config.ocr.model_dump(mode='json')]
        elif stage == 'translation':
            inputs = [config.translator.model_dump(mode='json'),
                      {key: context.get(key) for key in ('story', 'characters', 'glossary')},
                      [[r.get('text', ''), r.get('speaker', ''), r.get('listener', ''), bool(r.get('locked')),
                        bool(str(r.get('translation', '')).strip())] for r in regions]]
        elif stage == 'clean':
            # Inpainting sees the image, the raw text mask and region polygons only;
            # translations, fonts and review marks never reach it.
            mask = self.page_dir(data.get('path') or '') / data['mask'] if data.get('mask') and data.get('path') else None
            inputs = [config.inpainter.model_dump(mode='json'), config.mask_dilation_offset, config.kernel_size,
                      file_hash(mask) if mask and mask.is_file() else config.detector.model_dump(mode='json'),
                      [geometry_stamp(r) for r in regions]]
        else:
            clean = data.get('stages', {}).get('clean', {})
            inputs = [config.render.model_dump(mode='json'), clean.get('artifact_hash'),
                      [{key: value for key, value in r.items() if key not in METADATA_KEYS and key != 'prob'}
                       for r in regions]]
        return fingerprint([image, stage, inputs])

    def export_target(self, path, config, output_dir=''):
        """Where a finished page goes, following the main app: the chosen output folder plus the
        chapter folder name, or `manga_translator_work/result` beside the source. The source
        folder itself is never used, so an original can not be overwritten."""
        from manga_translator.utils.path_manager import WORK_DIR_NAME
        source = Path(os.path.abspath(path))
        beside = source.parent / WORK_DIR_NAME / 'result'
        folder = beside
        if output_dir and not getattr(config.cli, 'save_to_source_dir', False):
            folder = Path(os.path.abspath(output_dir)) / source.parent.name
            if os.path.normcase(str(folder)) == os.path.normcase(str(source.parent)):
                folder = beside
        extension = str(getattr(config.cli, 'format', None) or '').strip().lower().lstrip('.')
        name = f'{source.stem}.{extension}' if extension and extension not in ('none', '不指定') else source.name
        return folder / name

    def export_page(self, path, rendered, config, output_dir=''):
        """Write the finished page under its original file name; returns the written path."""
        target = self.export_target(path, config, output_dir)
        with Image.open(rendered) as image:
            image = image.convert('RGB')
            try:
                self._save_image(target, image, quality=int(getattr(config.cli, 'save_quality', 100) or 100))
            except (KeyError, ValueError, OSError) as exc:
                if target.suffix.lower() == '.png':
                    raise
                # Pillow cannot write this source format here; a PNG keeps the page lossless.
                logger.warning('Could not write %s (%s); exporting PNG instead', target.name, exc)
                target = target.with_suffix('.png')
                self._save_image(target, image)
        return target

    def export_dir(self, path):
        """Folder holding this page's exported result, or None before the first export."""
        try:
            exported = self.read_page(path).get('stages', {}).get('layout', {}).get('export')
        except Exception:
            return None
        return Path(exported).parent if exported and os.path.isfile(exported) else None

    def reset_halt(self):
        """Forget earlier pause/stop requests; call right before submitting the next job."""
        self.pause.clear()
        self.cancel.clear()

    def request_cancel(self):
        """User cancel: stop between units and interrupt the unit that is awaiting."""
        self.cancel.set()
        self._interrupt_task()

    def request_shutdown(self):
        self.interrupt.set()
        self._interrupt_task()

    def _interrupt_task(self):
        loop, task = self._loop, self._task
        if loop is not None and task is not None and not task.done():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # Loop already closed.

    def _halted(self):
        return self.cancel.is_set() or self.pause.is_set() or self.interrupt.is_set()

    def _halt_status(self):
        return 'cancelled' if self.cancel.is_set() else 'interrupted' if self.interrupt.is_set() else 'paused'

    async def run(self, job, notify=lambda value: None):
        from manga_translator.config import Config
        # Pause/stop requests are NOT cleared here: one made between submitting and starting
        # this coroutine must still be honored. The starter calls reset_halt() first.
        self._loop, self._task = asyncio.get_running_loop(), asyncio.current_task()
        with self.lock:
            job.update(status='running', failures=[])
            job.setdefault('completed', [])
            self._save()
        path = ''
        try:
            config = Config.model_validate(job['config'])
            for path in job['paths']:
                if self._halted():
                    break
                data = self.read_page(path)
                data['path'] = str(path)
                context = load_chapter_context(path, self.app_root)
                for stage in job['stages']:
                    if self._halted():
                        break
                    unit = path + '::' + stage
                    key = self.stage_key(data, stage, config, context)
                    saved = data.get('stages', {}).get(stage, {})
                    artifact = self.page_dir(path) / ('clean.png' if stage == 'clean' else 'translated.png')
                    artifact_ok = stage not in ('clean', 'layout') or (artifact.is_file() and file_hash(artifact) == saved.get('artifact_hash'))
                    if stage == 'layout' and artifact_ok:
                        artifact_ok = bool(saved.get('export')) and os.path.isfile(saved['export'])
                    if saved.get('key') == key and artifact_ok:
                        self._complete(job, unit, save=False)
                        notify({'status': 'skipped', 'path': path, 'stage': stage})
                        continue
                    notify({'status': 'running', 'path': path, 'stage': stage})
                    try:
                        await self._stage(path, data, stage, config, context)
                        saved = {}
                        if stage in ('clean', 'layout'):
                            saved['artifact_hash'] = file_hash(artifact)
                        if stage == 'layout':
                            saved['export'] = str(self.export_page(path, artifact, config, job.get('output_dir', '')))
                        data.setdefault('stages', {})[stage] = saved
                        # Key after a successful mutation makes translation/layout idempotent.
                        saved['key'] = self.stage_key(data, stage, config, context)
                        self.save_page(path, data)  # Durable result BEFORE journal completion.
                        self._complete(job, unit)
                        notify({'status': 'completed', 'path': path, 'stage': stage, 'export': saved.get('export', '')})
                    except Exception as exc:
                        # Expected user-facing refusals are ValueErrors; anything else keeps its traceback.
                        logger.error('Chapter stage %s failed for %s: %s', stage, path, exc,
                                     exc_info=not isinstance(exc, ValueError))
                        self._fail(job, path, stage, exc)
                        notify({'status': 'failed', 'path': path, 'stage': stage, 'message': str(exc)})
                        break  # A failed prerequisite must not run dependent stages on stale data.
                    await asyncio.sleep(0)
            status = self._halt_status() if self._halted() else 'failed' if job['failures'] else 'completed'
            self._update(job, save=False, status=status)
        except asyncio.CancelledError:
            # Completed units are already durable; report whose decision stopped the job.
            self._update(job, save=False, status='cancelled' if self.cancel.is_set() else 'interrupted')
            raise
        except Exception as exc:
            logger.exception('Chapter job %s failed', job.get('id'))
            self._update(job, save=False, status='failed')
            self._fail(job, path, 'input', exc)
            notify({'status': 'failed', 'path': path, 'message': str(exc)})
        finally:
            self._task = None
            if job['status'] == 'running':
                self._update(job, save=False, status='failed')
            self._save()
            notify({'status': job['status'], 'job': job['id']})

    async def _stage(self, path, data, stage, config, context):
        folder = self.page_dir(path)
        folder.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as source:
            image = np.array(source.convert('RGB'))
        import torch
        device = 'cuda' if config.cli.use_gpu and torch.cuda.is_available() else 'cpu'
        if stage == 'ocr':
            if data['regions'] and any(r.get('translation') or r.get('_mangax_review') or r.get('locked') for r in data['regions']):
                raise ValueError(tr('This page already has translations or locked boxes. Use OCR on the selected box to keep that work'))
            from manga_translator.detection import dispatch as detect
            from manga_translator.ocr import dispatch as ocr
            from manga_translator.textline_merge import dispatch as merge
            d = config.detector
            lines, mask, _ = await detect(d.detector, image, d.detection_size, d.text_threshold,
                d.box_threshold, d.unclip_ratio, device=device,
                use_yolo_obb=d.use_yolo_obb, yolo_obb_conf=d.yolo_obb_conf,
                yolo_obb_overlap_threshold=d.yolo_obb_overlap_threshold,
                min_box_area_ratio=d.min_box_area_ratio,
                det_rearrange_min_effective_short_side=d.det_rearrange_min_effective_short_side)
            if lines:
                lines = await ocr(config.ocr.ocr, image, lines, config.ocr, device=device, runtime_config=config)
            blocks = await merge(lines, image.shape[1], image.shape[0], config) if lines else []
            # Boxes whose text could not be read are kept: the reviewer fixes them and the
            # repair stage still needs their position.
            regions = [jsonable(block.to_dict()) for block in blocks]
            for region in regions:
                region['translation'] = region['translation_raw'] = ''
                region['font_family'] = config.render.font_family or 'MangaX Itim'
                region['target_lang'] = config.translator.target_lang
                region['direction'] = 'h' if config.translator.target_lang == 'THA' else region.get('direction', 'auto')
            data['regions'] = regions
            if mask is not None:
                self._save_image(folder / 'mask.png', np.asarray(mask, dtype=np.uint8))
                data['mask'] = 'mask.png'
            else:
                data['mask'] = None
            # New boxes replace the old ones; the image did not change, so 'clean' stays valid
            # exactly when its key (mask + polygons) still matches.
        elif stage == 'translation':
            if not data['regions']:
                raise ValueError(tr('No text yet: run OCR first'))
            memory = context.get('memory', {})
            pending = []
            for index, region in enumerate(data['regions']):
                if region.get('locked') or reviewed(region) or str(region.get('translation') or '').strip():
                    continue
                if not str(region.get('text', '')).strip():
                    continue
                key = approved_memory_key(region, context, config.translator.target_lang)
                accepted = memory.get(key) if isinstance(memory, dict) else None
                if isinstance(accepted, str) and accepted:
                    region['translation'] = accepted
                    region['translation_raw'] = accepted
                    region['_mangax_review'] = review_stamp(region)
                else:
                    pending.append(index)
            cli_batch_size = (config.translator.claude_batch_size
                              if str(config.translator.translator) == "claude"
                              else config.translator.codex_batch_size)
            batch_size = max(1, min(30, cli_batch_size))
            from manga_translator.config import Translator
            for start in range(0, len(pending), batch_size):
                if self._halted():
                    raise ValueError(tr('Stopped between translation batches: finished results are saved, press Resume to continue'))
                indices = pending[start:start+batch_size]
                texts = [data['regions'][i]['text'] for i in indices]
                results = await self.translation_service.translate_text_batch(texts,
                    translator=Translator(config.translator.translator), target_lang=config.translator.target_lang,
                    config=config, regions=data['regions'], chapter_context=context)
                if len(results) != len(indices) or any(result is None or not str(result.translated_text or '').strip() for result in results):
                    raise ValueError(tr('Translation failed. Check the account, quota and connection, then retry'))
                for index, result in zip(indices, results):
                    data['regions'][index]['translation'] = result.translated_text
                    data['regions'][index]['translation_raw'] = result.translated_text
                    data['regions'][index]['translation_rich'] = None
                self.save_page(path, data)  # Retain successful batches even if the next batch fails.
        elif stage == 'clean':
            from manga_translator.inpainting import dispatch as inpaint
            from manga_translator.mask_refinement import dispatch as refine
            if not data['regions']:
                raise ValueError(tr('No text boxes yet: run OCR first'))
            mask = None
            if data.get('mask') and (folder / data['mask']).is_file():
                with Image.open(folder / data['mask']) as mask_source:
                    mask = np.array(mask_source.convert('L'))
                if mask.shape != image.shape[:2]:
                    mask = None  # Stale mask from another image size.
            if mask is None:
                from manga_translator.detection import dispatch as detect
                d = config.detector
                _, mask, _ = await detect(d.detector, image, d.detection_size, d.text_threshold,
                    d.box_threshold, d.unclip_ratio, device=device,
                    det_rearrange_min_effective_short_side=d.det_rearrange_min_effective_short_side)
                if mask is None:
                    raise ValueError(tr('Could not detect a text mask, so the boxes were not painted over'))
                mask = np.asarray(mask, dtype=np.uint8)
            # Refinement matches mask strokes against the polygons as they sit on the page.
            blocks = []
            for region in data['regions']:
                block = region_block({**region, 'lines': world_lines(region).tolist(), 'angle': 0}, config)
                blocks.append(block)
            mask = await refine(blocks, image, mask, method='fit_text',
                                dilation_offset=config.mask_dilation_offset, kernel_size=config.kernel_size)
            clean = await inpaint(config.inpainter.inpainter, image, mask, config.inpainter,
                                  config.inpainter.inpainting_size, device=device)
            self._save_image(folder / 'refined_mask.png', mask)
            self._save_image(folder / 'clean.png', clean)
        else:
            from manga_translator.rendering import dispatch as render
            if any(str(r.get('text', '')).strip() and not str(r.get('translation', '')).strip() for r in data['regions']):
                raise ValueError(tr('Some boxes have no translation yet. Check the whole chapter before exporting'))
            clean_path = folder / 'clean.png'
            if not data.get('stages', {}).get('clean') or not clean_path.is_file():
                raise ValueError(tr('Repair the image before laying out text, so text does not cover the original'))
            clean_record = data['stages']['clean']
            if clean_record.get('key') != self.stage_key(data, 'clean', config, context) or clean_record.get('artifact_hash') != file_hash(clean_path):
                raise ValueError(tr('Boxes moved or the repaired image changed. Run the image repair stage again before exporting'))
            with Image.open(clean_path) as clean_source:
                base = np.array(clean_source.convert('RGB'))
            fallback_font = config.render.font_family or ''
            blocks = []
            for region in data['regions']:
                block = region_block(region, config)
                if not block.font_family:
                    block.font_family = fallback_font
                blocks.append(block)
            rendered_image = await render(base, blocks, config, original_img=image, skip_font_scaling=True)
            self._save_image(folder / 'translated.png', rendered_image)

    @staticmethod
    def _save_image(path, pixels, quality=100):
        """Atomically write an image; the format follows the target extension."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + '.' + uuid.uuid4().hex[:8] + '.tmp')
        try:
            image = pixels if isinstance(pixels, Image.Image) else Image.fromarray(np.asarray(pixels, dtype=np.uint8))
            extension = path.suffix.lower()
            image_format = Image.registered_extensions().get(extension)
            if image_format is None:
                raise ValueError(f'Unsupported image format: {extension}')
            options = {'quality': quality} if image_format in ('JPEG', 'WEBP', 'AVIF') else {}
            if image_format == 'WEBP' and quality >= 100:
                options = {'lossless': True}
            image.save(temporary, format=image_format, **options)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
