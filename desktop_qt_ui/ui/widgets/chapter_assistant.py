"""Editor chapter assistant, built on existing controls and undo boundaries."""
import asyncio
import copy
import logging
import os
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QImage, QPixmap, QUndoCommand
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, ComboBox, LineEdit, PlainTextEdit, PushButton, ScrollArea

from services.async_service import AsyncService
from services.chapter_studio import (
    ChapterStudio, RESUMABLE, STYLE_KITS, approved_memory_key, atomic_json, context_path,
    fingerprint, jsonable, load_chapter_context, quality_issues, review_stamp, reviewed, straighten, tr,
)

logger = logging.getLogger('manga_translator')

STATUS_KEYS = {
    'queued': 'Status: queued', 'running': 'Status: running', 'paused': 'Status: paused',
    'failed': 'Status: failed', 'interrupted': 'Status: interrupted', 'cancelled': 'Status: cancelled',
    'completed': 'Status: completed', 'skipped': 'Status: reused saved result',
}
STAGE_KEYS = {'ocr': 'Stage: OCR', 'translation': 'Stage: translation', 'clean': 'Stage: image repair',
              'layout': 'Stage: layout and export', 'input': 'Stage: input'}


class ReplaceChapterRegions(QUndoCommand):
    def __init__(self, model, regions):
        super().__init__(tr('Apply assistant result to editor'))
        self.model = model
        self.before = copy.deepcopy(model.get_regions())
        self.after = copy.deepcopy(regions)
    def redo(self):
        self.model.replace_regions(copy.deepcopy(self.after))
    def undo(self):
        self.model.replace_regions(copy.deepcopy(self.before))


class ChapterAssistant(ScrollArea):
    updated = pyqtSignal(object)
    issues_ready = pyqtSignal(object)
    deskew_ready = pyqtSignal(object)

    def __init__(self, editor, parent=None):
        super().__init__(parent)
        self.editor = editor
        self.controller, self.model = editor.controller, editor.model
        self.config_service = self.controller.config_service
        self.studio = ChapterStudio(self.config_service.root_dir, self.controller.file_service,
                                    self.controller.translation_service)
        self.worker = AsyncService()  # Independent queue survives editor page switches.
        self.future = None
        self.current_job = None
        self._context_folder = None
        self._pending_issue = None
        self._texts = []  # (setter, key) pairs re-applied on language change.
        self._combos = []  # (combo, [(key, value), ...])
        self._context_timer = QTimer(self)
        self._context_timer.setSingleShot(True)
        self._context_timer.setInterval(350)
        self._context_timer.timeout.connect(self._load_context)
        self.setWidgetResizable(True)
        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.setWidget(content)
        self.status = BodyLabel(self)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._label(layout, 'OCR → check source → translate → repair image → lay out text → export')
        self.scope = self._combo(layout, [('This page', 'page'),
                                          ('Whole chapter (images in the list on the right)', 'chapter')])
        self.actions = []
        for title, stages in [('OCR whole page', ['ocr']), ('Translate text', ['translation']),
                              ('Repair image locally', ['clean']), ('Lay out text and export', ['layout'])]:
            self.actions.append(self._button(layout, title, lambda checked=False, value=stages: self.enqueue(value)))
        self.stop_after = self._combo(layout, [('Run every stage', 'all'), ('Stop for review after OCR', 'ocr'),
                                               ('Stop for review after translation', 'translation')])
        self.actions.append(self._button(layout, 'Queue the selected stages', self.enqueue_all))
        controls = QHBoxLayout()
        for title, callback in [('Pause', self.pause), ('Stop', self.cancel), ('Resume / Retry', self.resume)]:
            self._button(controls, title, callback)
        layout.addLayout(controls)
        self.jobs = QListWidget(self)
        self.jobs.setMaximumHeight(150)
        self._text(self.jobs.setAccessibleName, 'Job queue and history')
        layout.addWidget(self.jobs)
        for title, callback in [("Apply this page's result to the editor (undoable)", self.apply_result),
                                ("Open this page's result folder", self.open_results),
                                ('Draw a box for selected-box OCR', self.controller.enter_drawing_mode)]:
            self._button(layout, title, callback)
        self._label(layout, 'Story context and character voices (saved per chapter folder)')
        self.story = self._editor(layout, 'Story context, e.g. who is talking to whom and what happens in this scene', 95)
        self.characters = self._editor(
            layout, 'One character per line: name | voice | pronouns\nYuna | polite, without a particle on every sentence | I–you', 95)
        self.terms = self._editor(layout, 'Chapter terms, one per line: source = translation\nYuna = ยูนะ', 90)
        self._button(layout, 'Save context and terms for Codex', self.save_context)
        self._label(layout, 'Selected box: speaker / listener / text type')
        self.speaker, self.listener = LineEdit(self), LineEdit(self)
        self._text(self.speaker.setPlaceholderText, 'Speaker')
        self._text(self.listener.setPlaceholderText, 'Listener')
        layout.addWidget(self.speaker)
        layout.addWidget(self.listener)
        self.kind = self._combo(layout, [('Dialogue', 'dialogue'), ('Thought', 'thought'), ('Narration', 'narration'),
                                         ('Shout', 'shout'), ('SFX', 'sfx')])
        for title, callback in [('Save speaker and type', self.annotate),
                                ('Apply the style kit to the selected box', self.apply_style),
                                ('Apply the style kit to this type on the whole page', self.apply_style_page),
                                ('Approve the selected box and remember its translation', self.accept_selected),
                                ('Lock / unlock the selected box', self.toggle_lock)]:
            self._button(layout, title, callback)
        self._label(layout, 'Check the whole chapter before exporting')
        self.audit_button = self._button(layout, 'Check OCR / translation / fonts / text overflow', self.audit)
        self.issues = QListWidget(self)
        self.issues.setMinimumHeight(140)
        self.issues.itemActivated.connect(self.go_to_issue)
        self.issues.itemClicked.connect(self.go_to_issue)
        layout.addWidget(self.issues)
        self._label(layout, 'Skewed-text OCR: select one box, adjust the four corners, then preview')
        self.corners = LineEdit(self)
        self._text(self.corners.setPlaceholderText, 'x,y; x,y; x,y; x,y (top-left → clockwise)')
        layout.addWidget(self.corners)
        self.curvature = LineEdit(self)
        self.curvature.setText('0')
        self._text(self.curvature.setPlaceholderText, 'Curvature -0.5 to 0.5 (0 = straight)')
        layout.addWidget(self.curvature)
        self.deskew_preview = QLabel(self)
        self.deskew_preview.setMinimumHeight(90)
        self.deskew_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.deskew_preview)
        self._button(layout, 'Preview the straightened image', self.preview_deskew)
        self.deskew_button = self._button(layout, 'Read OCR from the preview (undoable)', self.run_deskew)
        layout.addStretch()
        self.updated.connect(self.on_update)
        self.issues_ready.connect(self.on_issues)
        self.deskew_ready.connect(self.on_deskew)
        self.model.regions_changed.connect(self.on_model_changed)
        self.model.selection_changed.connect(self.selection_changed)
        self.model.display_layers_changed.connect(self.on_model_changed)
        editor.logic.file_snapshot_changed.connect(self.on_model_changed)
        # Registered last: a half-built assistant must never be reachable from the main page.
        editor.app_logic.chapter_assistant = self
        self.refresh()
        if self.studio.recovered_journal:
            self.message(self._t('The job history file was damaged and has been moved to {path}. The queue starts empty.',
                                 path=self.studio.recovered_journal.name))

    # ---- i18n helpers -------------------------------------------------
    def _t(self, key, **kwargs):
        return tr(key, **kwargs)

    def _text(self, setter, key):
        self._texts.append((setter, key))
        setter(self._t(key))

    def _label(self, layout, key):
        label = CaptionLabel(self)
        label.setWordWrap(True)
        self._text(label.setText, key)
        layout.addWidget(label)
        return label

    def _button(self, layout, key, callback):
        button = PushButton(self)
        self._text(button.setText, key)
        button.clicked.connect(lambda checked=False: callback())
        layout.addWidget(button)
        return button

    def _combo(self, layout, items):
        combo = ComboBox(self)
        for key, value in items:
            combo.addItem(self._t(key), userData=value)
        self._combos.append((combo, items))
        layout.addWidget(combo)
        return combo

    def _editor(self, layout, placeholder_key, height):
        editor = PlainTextEdit(self)
        self._text(editor.setPlaceholderText, placeholder_key)
        editor.setMaximumHeight(height)
        layout.addWidget(editor)
        return editor

    def refresh_ui_texts(self):
        """Re-apply every static text after a language change."""
        for setter, key in self._texts:
            setter(self._t(key))
        for combo, items in self._combos:
            for index, (key, _) in enumerate(items):
                combo.setItemText(index, self._t(key))
        self.refresh()

    def _status(self, status):
        return self._t(STATUS_KEYS[status]) if status in STATUS_KEYS else str(status or '')

    def _stage(self, stage):
        return self._t(STAGE_KEYS[stage]) if stage in STAGE_KEYS else str(stage or '')

    # ---- state --------------------------------------------------------
    def busy(self):
        return bool(self.future and not self.future.done())

    def refresh(self):
        path = self.model.get_source_image_path()
        regions = self.model.get_regions()
        source_count = sum(bool(str(r.get('text') or '').strip()) for r in regions)
        translated = sum(bool(str(r.get('translation') or '').strip()) for r in regions)
        accepted = sum(reviewed(r) for r in regions)
        self.status.setText('\n'.join([
            Path(path).name if path else self._t('Open an image from the list first'),
            self._t('Source {source}/{total} · translated {translated}/{total} · reviewed {reviewed}',
                    source=source_count, translated=translated, total=len(regions), reviewed=accepted),
            self._t('OCR / LaMa run on this computer · Codex uses the internet')]))
        for button in self.actions:
            button.setEnabled(bool(path))
            button.setToolTip(self._t('Open an image before starting') if not path
                              else self._t('Jobs are queued; results are saved apart from the originals'))
        selected = self.model.get_selection()
        self.deskew_button.setEnabled(bool(path) and len(selected) == 1)
        selected_id = self.jobs.currentItem().data(Qt.ItemDataRole.UserRole) if self.jobs.currentItem() else None
        self.jobs.clear()
        for job in self.studio.snapshot():  # Detached copies: the worker thread owns the live jobs.
            item = QListWidgetItem('\n'.join([
                self._t('{status} · {pages} page(s) · {stages}', status=self._status(job['status']), pages=job['pages'],
                        stages=' → '.join(self._stage(stage) for stage in job['stages'])),
                self._t('Done {done} step(s) · {failed} error(s)', done=job['completed'], failed=len(job['failures']))]))
            if job['failures']:
                item.setToolTip('\n'.join(f"{Path(f.get('path', '')).name} · {self._stage(f.get('stage'))}: {f.get('message', '')}"
                                          for f in job['failures'][-5:]))
            item.setData(Qt.ItemDataRole.UserRole, job['id'])
            self.jobs.addItem(item)
            if job['id'] == selected_id:
                self.jobs.setCurrentItem(item)

    def on_model_changed(self, *_):
        self.refresh()
        self._context_timer.start()
        if self._pending_issue:
            path, index = self._pending_issue
            if self.model.get_source_image_path() == path and index < len(self.model.get_regions()):
                self._pending_issue = None
                self.model.set_selection([index] if index >= 0 else [])

    def selection_changed(self, indices):
        self.refresh()
        if len(indices) == 1 and indices[0] < len(self.model.get_regions()):
            region = self.model.get_regions()[indices[0]]
            self.speaker.setText(region.get('speaker', ''))
            self.listener.setText(region.get('listener', ''))
            self.kind.setCurrentIndex(max(0, self.kind.findData(region.get('text_kind', 'dialogue'))))
            points = np.asarray(region.get('lines', []), dtype=float).reshape(-1, 2)
            if len(points):
                x0, y0 = points.min(axis=0)
                x1, y1 = points.max(axis=0)
                self.corners.setText('; '.join(f'{x:.0f},{y:.0f}' for x, y in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]))

    def _load_context(self):
        path = self.model.get_source_image_path()
        if not path or os.path.dirname(path) == self._context_folder:
            return
        try:
            context = load_chapter_context(path, self.config_service.root_dir)
        except Exception as exc:
            logger.warning('Could not read chapter context for %s: %s', path, exc)
            return self.message(self._t('Could not read the saved chapter context: {error}', error=exc))
        self._context_folder = os.path.dirname(path)
        self.story.setPlainText(str(context.get('story', '')))
        self.characters.setPlainText('\n'.join(' | '.join(str(c.get(k, '')) for k in ('name','voice','pronouns'))
                                               for c in context.get('characters', []) if isinstance(c, dict)))
        self.terms.setPlainText('\n'.join(f"{t.get('original', '')} = {t.get('translation', '')}"
                                          for t in context.get('glossary', []) if isinstance(t, dict)))

    def save_context(self):
        path = self.model.get_source_image_path()
        if not path:
            return self.message(self._t('Open an image before saving context'))
        if os.path.dirname(path) != self._context_folder:
            self._load_context()
        try:
            context = load_chapter_context(path, self.config_service.root_dir)
            characters = []
            for line in self.characters.toPlainText().splitlines():
                parts = [part.strip() for part in line.split('|')]
                if parts[0]:
                    characters.append(dict(zip(('name','voice','pronouns'), (parts + ['', ''])[:3])))
            terms = []
            for line in self.terms.toPlainText().splitlines():
                original, separator, translation = line.partition('=')
                if separator and original.strip() and translation.strip():
                    terms.append({'original': original.strip(), 'translation': translation.strip()})
            context.update(story=self.story.toPlainText()[:4000], characters=characters[:30], glossary=terms[:500])
            atomic_json(context_path(path, self.config_service.root_dir), context)
        except Exception as exc:
            logger.exception('Could not save chapter context')
            return self.message(str(exc))
        self.message(self._t('Context saved for Codex'))

    def selected_paths(self):
        current = self.model.get_source_image_path()
        return [item.path for item in self.editor.logic.file_model.files] if self.scope.currentData() == 'chapter' else [current] if current else []

    def enqueue_all(self):
        stages = ['ocr', 'translation', 'clean', 'layout']
        stop = self.stop_after.currentData()
        self.enqueue(stages if stop == 'all' else stages[:stages.index(stop)+1])

    def enqueue(self, stages):
        try:
            if self.editor.app_logic.state_manager.is_translating():
                raise ValueError(self._t('The main page is working: stop that task before starting the assistant queue'))
            self.controller.commit_pending_edits()
            path = self.model.get_source_image_path()
            if path and not self.busy():
                self.studio.seed_editor(path, self.model.get_regions())
            from manga_translator.config import Config
            settings = self.config_service.get_config()
            config = Config.model_validate(settings.model_dump(mode='json'))
            panel = self.editor.property_panel
            config.ocr.ocr = panel.get_selected_ocr_model()
            config.translator.translator = panel.get_selected_translator()
            config.translator.target_lang = panel.get_selected_target_language()
            output_dir = getattr(getattr(settings, 'app', None), 'last_output_path', '') or ''
            self.studio.enqueue(self.selected_paths(), stages, config, output_dir)
            self.start_next()
        except Exception as exc:
            logger.warning('Chapter assistant could not queue %s: %s', stages, exc)
            self.message(str(exc))

    def start_next(self, job=None):
        """Start `job`, or the oldest queued job, unless one is already running."""
        if self.busy():
            return
        job = job or self.studio.next_job()
        if job:
            self.current_job = job['id']
            self.studio.reset_halt()
            self.future = self.worker.submit_task(self.studio.run(job, self.updated.emit))
        self.refresh()

    def on_update(self, update):
        self.refresh()
        status = update.get('status')
        if 'job' in update:
            lines = [self._t('Job finished: {status}', status=self._status(status))]
            job = next((j for j in self.studio.snapshot() if j['id'] == update['job']), None)
            if job and job['failures']:
                failure = job['failures'][-1]
                lines.append(f"{Path(failure.get('path', '')).name} · {self._stage(failure.get('stage'))}: {failure.get('message', '')}")
            self.message('\n'.join(lines))
            # A finished or failed job hands over to the next queued one. Pause, stop and
            # shutdown are decisions to halt the queue, so nothing starts on its own then.
            if status in ('completed', 'failed'):
                QTimer.singleShot(100, self.start_next)
            return
        lines = [' · '.join(part for part in (self._status(status), Path(update.get('path') or '').name,
                                              self._stage(update.get('stage'))) if part)]
        if update.get('message'):
            lines.append(str(update['message']))
        if update.get('export'):
            lines.append(self._t('Exported to {path}', path=update['export']))
        self.message('\n'.join(lines))

    def pause(self):
        self.studio.pause.set()
        self.message(self._t('Will pause after the current unit is saved'))

    def cancel(self):
        # The job reports 'cancelled' itself; its future completes only when it really stopped.
        self.studio.request_cancel()
        self.message(self._t('Stopping. Results saved so far are kept'))

    def resume(self):
        if self.busy():
            return self.message(self._t('A job is already running'))
        item = self.jobs.currentItem()
        job = self.studio.job(item.data(Qt.ItemDataRole.UserRole)) if item else None
        if job is None or job['status'] not in RESUMABLE:
            job = self.studio.next_job(RESUMABLE, newest=True)
        if job is None and self.studio.next_job() is None:
            return self.message(self._t('Nothing to resume'))
        self.start_next(job)  # With no halted job left, this restarts the queue.

    def apply_result(self):
        path = self.model.get_source_image_path()
        if not path:
            return
        if self.busy():
            return self.message(self._t('Pause the queue and let it save before applying results'))
        try:
            self.controller.commit_pending_edits()
            data = self.studio.read_page(path)
            before = self.model.get_regions()
            current = fingerprint(jsonable(copy.deepcopy(before)))
            if not self.studio.editor_untouched(data, before) and current != fingerprint(data['regions']):
                return self.message(self._t('This page was edited after the job started, so the result was not applied over it. Save or review your work, then start again'))
            self.controller.execute_command(ReplaceChapterRegions(self.model, data['regions']))
            self.studio.mark_seed(path, self.model.get_regions(), before)
        except Exception as exc:
            logger.exception('Could not apply the assistant result')
            return self.message(str(exc))
        self.message(self._t('Result applied to the editor. Undo with Ctrl+Z'))

    def open_results(self):
        path = self.model.get_source_image_path()
        if path:
            # The exported page when there is one, otherwise the working artifacts.
            folder = self.studio.export_dir(path) or self.studio.page_dir(path)
            folder.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def edit_selected(self, change, indices=None):
        self.controller.commit_pending_edits()
        old = self.model.get_regions()
        new = copy.deepcopy(old)
        for index in self.model.get_selection() if indices is None else indices:
            if 0 <= index < len(new):
                change(new[index])
        from editor.commands import MultiRegionUpdateCommand
        command = MultiRegionUpdateCommand(self.model, old, new, self._t('Assistant: edit text boxes'), source='chapter-assistant')
        if command.has_changes():
            self.controller.execute_command(command)

    def annotate(self):
        self.edit_selected(lambda r: r.update(speaker=self.speaker.text().strip(), listener=self.listener.text().strip(), text_kind=self.kind.currentData()))

    def apply_style(self):
        kit = STYLE_KITS[self.kind.currentData()]
        speaker = self.speaker.text().strip()
        self.edit_selected(lambda r: r.update(kit) if not r.get('locked') and (not speaker or r.get('speaker') == speaker) else None)

    def apply_style_page(self):
        kind = self.kind.currentData()
        self.edit_selected(lambda r: r.update(STYLE_KITS[kind]) if not r.get('locked') else None,
                           [i for i,r in enumerate(self.model.get_regions()) if r.get('text_kind','dialogue') == kind and (not self.speaker.text().strip() or r.get('speaker') == self.speaker.text().strip())])

    def toggle_lock(self):
        self.edit_selected(lambda r: r.update(locked=not r.get('locked', False)))

    def accept_selected(self):
        path = self.model.get_source_image_path()
        if not path:
            return
        try:
            context = load_chapter_context(path, self.config_service.root_dir)
            memory = context.setdefault('memory', {})
            examples = context.setdefault('approved_examples', [])
            target = self.editor.property_panel.get_selected_target_language()
            def accept(region):
                if str(region.get('text') or '').strip() and str(region.get('translation') or '').strip():
                    region['_mangax_review'] = review_stamp(region)
                    memory[approved_memory_key(region, context, target)] = region['translation']
                    examples.append({'source': region['text'], 'translation': region['translation']})
            self.edit_selected(accept)
            context['memory'] = dict(list(memory.items())[-5000:])
            context['approved_examples'] = examples[-200:]
            atomic_json(context_path(path, self.config_service.root_dir), context)
            # Review stamps also travel in the editor JSON; the checkpoint keeps them for queued work.
            if not self.busy():
                self.studio.seed_editor(path, self.model.get_regions())
        except Exception as exc:
            logger.exception('Could not approve the selected boxes')
            return self.message(str(exc))
        self.message(self._t('Approved and remembered. Later text or style edits will ask for a new review'))

    def audit(self):
        if self.busy():
            return self.message(self._t('Pause the queue and let it save before checking'))
        self.controller.commit_pending_edits()
        current_path = self.model.get_source_image_path()
        current_regions = copy.deepcopy(self.model.get_regions())
        paths = [item.path for item in self.editor.logic.file_model.files] or ([current_path] if current_path else [])
        # Snapshot: the worker must not read settings the GUI thread may be editing.
        config = self.config_service.get_config().model_copy(deep=True)
        root = self.config_service.root_dir
        self.audit_button.setEnabled(False)
        async def check():
            issues = []
            try:
                for path in paths:
                    regions = current_regions if path == current_path else self.studio.read_page(path)['regions']
                    context = load_chapter_context(path, root)
                    issues.extend(quality_issues(path, regions, context, config))
                    await asyncio.sleep(0)
            except Exception as exc:
                logger.exception('Chapter check failed')
                issues.append({'path': current_path or '', 'index': -1, 'kind': 'error', 'message': str(exc)})
            self.issues_ready.emit(issues)
        if self.worker.submit_task(check()) is None:
            self.audit_button.setEnabled(True)

    def on_issues(self, issues):
        self.audit_button.setEnabled(True)
        self.issues.clear()
        for issue in issues:
            item = QListWidgetItem(self._t('{file} · box {number}', file=Path(issue['path']).name, number=issue['index']+1)
                                   + '\n' + issue['message'])
            item.setData(Qt.ItemDataRole.UserRole, issue)
            self.issues.addItem(item)
        self.message(self._t('Chapter checked: {count} item(s) need attention', count=len(issues)) if issues
                     else self._t('Chapter checked: no problems found by these checks'))

    def go_to_issue(self, item):
        issue = item.data(Qt.ItemDataRole.UserRole)
        if issue['path'] == self.model.get_source_image_path():
            if issue['index'] < len(self.model.get_regions()):
                self.model.set_selection([issue['index']] if issue['index'] >= 0 else [])
        elif issue['path']:
            self._pending_issue = (issue['path'], issue['index'])
            self.editor.logic.load_image_into_editor(issue['path'])

    def preview_deskew(self):
        try:
            points = [[float(value) for value in pair.split(',')] for pair in self.corners.text().split(';')]
            curvature = float(self.curvature.text())
            if not -.5 <= curvature <= .5:
                raise ValueError(self._t('Curvature must be between -0.5 and 0.5'))
            page = self.model.get_image()
            if page is None:
                raise ValueError(self._t('Open an image before starting'))
            source = np.array(page.convert('RGB'))
            crop = np.ascontiguousarray(straighten(source, points, curvature))
            self._crop = crop
            self._crop_identity = self.model.get_document_identity()
            indices = self.model.get_selection()
            self._crop_region_id = self.model.get_region_id(indices[0]) if len(indices) == 1 else None
            self._crop_controls = (self.corners.text(), self.curvature.text())
            image = QImage(crop.data, crop.shape[1], crop.shape[0], crop.strides[0], QImage.Format.Format_RGB888).copy()
            self.deskew_preview.setPixmap(QPixmap.fromImage(image).scaled(320, 130, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            return crop
        except Exception as exc:
            self.message(str(exc))
            return None

    def run_deskew(self):
        indices = self.model.get_selection()
        if len(indices) != 1 or not hasattr(self, '_crop'):
            return self.message(self._t('Select one box and preview the straightened image first'))
        if self._crop_identity != self.model.get_document_identity():
            return self.message(self._t('The image changed. Preview again before OCR'))
        if self.model.get_region_id(indices[0]) != self._crop_region_id or self._crop_controls != (self.corners.text(), self.curvature.text()):
            return self.message(self._t('The box or its corners changed. Preview again before OCR'))
        crop = self._crop.copy()
        identity = self._crop_identity
        region_id = self.model.get_region_id(indices[0])
        previous_text = self.model.get_regions()[indices[0]].get('text', '')
        h,w = crop.shape[:2]
        from manga_translator.config import OcrConfig
        config = OcrConfig.model_validate(self.config_service.get_config().ocr.model_dump())
        config.ocr = self.editor.property_panel.get_selected_ocr_model()
        async def recognize():
            try:
                result = await self.controller.ocr_service.recognize_region(crop,
                    {'lines': [[[0,0],[w-1,0],[w-1,h-1],[0,h-1]]]}, config)
                self.deskew_ready.emit({'identity': identity, 'region_id': region_id, 'previous_text': previous_text, 'text': result.text if result else ''})
            except Exception as exc:
                logger.exception('Skewed-text OCR failed')
                self.updated.emit({'status':'failed','message':str(exc)})
        self.worker.submit_task(recognize())

    def on_deskew(self, result):
        index = self.model.find_region_index(result['region_id'])
        if result['identity'] != self.model.get_document_identity() or index is None:
            return self.message(self._t('The box or page changed, so the OCR result was not applied'))
        if self.model.get_regions()[index].get('text', '') != result['previous_text']:
            return self.message(self._t('The source text was edited during OCR, so the result was not applied'))
        if result['text']:
            self.controller.update_original_text(index, result['text'])
            self.message(self._t('Skewed-text OCR finished. Undo with Ctrl+Z'))
        else:
            self.message(self._t('OCR could not read the text. Adjust the corners or pick another model'))

    def message(self, text):
        self.status.setText(text)

    def shutdown(self):
        # Never let the assistant block the editor's own shutdown sequence.
        try:
            self.studio.request_shutdown()
            self.worker.shutdown()
        except Exception:
            logger.exception('Chapter assistant shutdown failed')
