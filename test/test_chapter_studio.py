import _bootstrap  # noqa: F401  (torch before PyQt6, offscreen Qt)

import asyncio
import copy
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'desktop_qt_ui'))
from services.chapter_studio import (
    ChapterStudio, approved_memory_key, atomic_json, file_hash, quality_issues, region_block,
    review_stamp, reviewed, straighten,
)
from manga_translator.config import Config
from manga_translator.translators.manga_context import bounded_context, context_path
from manga_translator.utils import TextBlock

METADATA = ('speaker', 'listener', 'text_kind', 'locked', '_mangax_review')


def box(x=10, y=10, w=60, h=30, **extra):
    return {'lines': [[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], 'text': 'Hello', 'translation': '',
            'font_size': 20, 'angle': 0, **extra}


class ChapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = str(self.root / 'page.png')
        Image.new('RGB', (100,100), 'white').save(self.path)
        self.file_service = SimpleNamespace(load_translation_json=lambda path: ([],None,None,{}))
        self.studio = ChapterStudio(self.root, self.file_service, None)
        self.config = Config()
        self.config.translator.translator = 'codex'
        self.config.translator.target_lang = 'THA'
    def tearDown(self):
        self.temp.cleanup()

    async def test_resume_uses_saved_stage_and_invalidates_changed_source(self):
        calls = []
        async def stage(path, data, name, config, context):
            calls.append(name)
            data['regions'] = [{'text': 'Hello', 'translation': ''}]
        with patch.object(self.studio, '_stage', side_effect=stage):
            job = self.studio.enqueue([self.path], ['ocr'], self.config)
            await self.studio.run(job)
            reopened = ChapterStudio(self.root, self.file_service, None)
            with patch.object(reopened, '_stage', side_effect=stage):
                await reopened.run(reopened.jobs[0])
                self.assertEqual(calls, ['ocr'])
                Image.new('RGB', (100,100), 'black').save(self.path)
                await reopened.run(reopened.jobs[0])
                self.assertEqual(calls, ['ocr','ocr'])

    async def test_durable_result_precedes_completed_journal(self):
        observed = []
        async def stage(path, data, *args):
            data['regions'] = [{'text':'saved'}]
        def notify(event):
            if event.get('status') == 'completed' and 'stage' in event:
                result = json.loads((self.studio.page_dir(self.path)/'page.json').read_text())
                journal = json.loads(self.studio.journal.read_text())
                observed.append((result['regions'][0]['text'], len(journal[0]['completed'])))
        with patch.object(self.studio, '_stage', side_effect=stage):
            await self.studio.run(self.studio.enqueue([self.path], ['ocr'], self.config), notify)
        self.assertEqual(observed, [('saved',1)])

    async def test_failed_prerequisite_stops_dependent_stages(self):
        calls = []
        async def fail(path,data,stage,*args):
            calls.append(stage)
            raise ValueError('OCR failed')
        with patch.object(self.studio, '_stage', side_effect=fail):
            job = self.studio.enqueue([self.path], ['ocr','translation','clean','layout'], self.config)
            await self.studio.run(job)
        self.assertEqual(calls,['ocr'])
        self.assertEqual(job['status'],'failed')
        self.assertEqual(len(job['failures']),1)

    async def test_partial_translation_is_saved_and_retry_only_pending(self):
        regions = [{'text':str(i), 'translation':'', 'lines':[[[1,1],[90,1],[90,90],[1,90]]]} for i in range(4)]
        self.studio.seed_editor(self.path, regions)
        self.config.translator.codex_batch_size = 2
        requests = []
        async def translate(texts, **kwargs):
            requests.append(texts)
            if len(requests) == 2:
                return [None] * len(texts)
            return [SimpleNamespace(translated_text='ไทย'+t) for t in texts]
        self.studio.translation_service = SimpleNamespace(translate_text_batch=translate)
        job = self.studio.enqueue([self.path], ['translation'], self.config)
        await self.studio.run(job)
        self.assertEqual(job['status'],'failed')
        self.assertEqual([r['translation'] for r in self.studio.read_page(self.path)['regions']], ['ไทย0','ไทย1','',''])
        await self.studio.run(job)
        self.assertEqual(job['status'],'completed')
        self.assertEqual(requests,[['0','1'],['2','3'],['2','3']])

    async def test_corrupt_artifact_is_not_reused(self):
        calls = []
        async def stage(path,data,name,*args):
            calls.append(name)
            self.studio._save_image(self.studio.page_dir(path)/'clean.png', np.zeros((10,10,3),np.uint8))
        with patch.object(self.studio, '_stage', side_effect=stage):
            job = self.studio.enqueue([self.path], ['clean'], self.config)
            await self.studio.run(job)
            artifact = self.studio.page_dir(self.path)/'clean.png'
            artifact.write_bytes(b'corrupt')
            await self.studio.run(job)
        self.assertEqual(calls,['clean','clean'])

    async def test_pause_retains_completed_unit(self):
        async def stage(path,data,name,*args):
            data['regions'] = [{'text':'Hello'}]
            self.studio.pause.set()
        with patch.object(self.studio, '_stage', side_effect=stage):
            job = self.studio.enqueue([self.path],['ocr','translation'],self.config)
            await self.studio.run(job)
        self.assertEqual(job['status'],'paused')
        self.assertEqual(job['completed'],[self.path+'::ocr'])

    # ---- journal recovery -------------------------------------------------
    def test_corrupt_journal_is_moved_aside_and_queue_starts_empty(self):
        samples = (b'{"not": "a list"', b'{"a": 1}', b'[{"id": 5}]', bytes([255, 254, 0]) + b'garbage')
        for content in samples:
            with self.subTest(content=content):
                self.studio.journal.write_bytes(content)
                reopened = ChapterStudio(self.root, self.file_service, None)
                self.assertEqual(reopened.jobs, [])
                self.assertIsNotNone(reopened.recovered_journal)
                self.assertTrue(reopened.recovered_journal.name.endswith('.corrupt'))
                self.assertEqual(reopened.recovered_journal.read_bytes(), content)
                self.assertEqual(json.loads(reopened.journal.read_text(encoding='utf-8')), [])
                reopened.enqueue([self.path], ['ocr'], self.config)  # Still usable.
        # Earlier backups are never overwritten.
        self.assertEqual(len(list(self.studio.root.glob('jobs.json*.corrupt'))), len(samples))

    def test_healthy_journal_survives_reopen_and_running_becomes_interrupted(self):
        job = self.studio.enqueue([self.path], ['ocr'], self.config)
        job['status'] = 'running'
        self.studio._save()
        reopened = ChapterStudio(self.root, self.file_service, None)
        self.assertIsNone(reopened.recovered_journal)
        self.assertEqual([(j['id'], j['status']) for j in reopened.jobs], [(job['id'], 'interrupted')])

    def test_corrupt_page_checkpoint_falls_back_to_editor_data(self):
        self.studio.seed_editor(self.path, [box()])
        (self.studio.page_dir(self.path) / 'page.json').write_text('{broken', encoding='utf-8')
        self.assertEqual(self.studio.read_page(self.path)['regions'], [])

    # ---- stage keys depend on true inputs only ---------------------------
    async def _clean_then(self, mutate):
        """Run a mocked clean, apply `mutate` to the checkpoint regions, run clean again."""
        calls = []
        async def stage(path, data, name, *args):
            calls.append(name)
            self.studio._save_image(self.studio.page_dir(path) / 'clean.png', np.zeros((10, 10, 3), np.uint8))
        self.studio.seed_editor(self.path, [box(), box(x=20, y=60)])
        with patch.object(self.studio, '_stage', side_effect=stage):
            await self.studio.run(self.studio.enqueue([self.path], ['clean'], self.config))
            data = self.studio.read_page(self.path)
            mutate(data['regions'])
            self.studio.save_page(self.path, data)
            await self.studio.run(self.studio.enqueue([self.path], ['clean'], self.config))
        return calls

    async def test_translating_or_restyling_after_clean_keeps_clean(self):
        def edit(regions):
            regions[0].update(translation='สวัสดี', translation_raw='สวัสดี', font_family='MangaX Kanit::Bold',
                              font_size=31, font_color='#ff0000', line_spacing=1.4, alignment='left',
                              speaker='Yuna', text_kind='shout', locked=True)
            regions[0]['_mangax_review'] = review_stamp(regions[0])
            regions[1]['text'] = 'Edited source'
        self.assertEqual(await self._clean_then(edit), ['clean'])

    async def test_moving_or_resizing_a_region_invalidates_clean(self):
        def move(regions):
            regions[0]['lines'] = box(x=14)['lines']
        def resize(regions):
            regions[1]['lines'] = box(x=20, y=60, w=90)['lines']
        def rotate(regions):
            regions[0]['angle'] = 15
        def remove(regions):
            del regions[1]
        for mutate in (move, resize, rotate, remove):
            with self.subTest(mutate=mutate.__name__):
                self.assertEqual(await self._clean_then(mutate), ['clean', 'clean'])
                (self.studio.page_dir(self.path) / 'page.json').unlink()

    async def test_clean_before_translation_still_allows_layout_and_moved_box_refuses(self):
        rendered = []
        async def fake_render(base, blocks, config, **kwargs):
            rendered.append([block.translation for block in blocks])
            return base
        real_stage = self.studio._stage
        async def stage(path, data, name, *args):
            if name == 'clean':
                self.studio._save_image(self.studio.page_dir(path) / 'clean.png', np.full((100, 100, 3), 255, np.uint8))
            else:
                await real_stage(path, data, name, *args)
        async def translate(texts, **kwargs):
            return [SimpleNamespace(translated_text='ไทย') for _ in texts]
        self.studio.translation_service = SimpleNamespace(translate_text_batch=translate)
        self.studio.seed_editor(self.path, [box()])
        with patch.object(self.studio, '_stage', side_effect=stage), patch('manga_translator.rendering.dispatch', side_effect=fake_render):
            for stages in (['clean'], ['translation']):
                job = self.studio.enqueue([self.path], stages, self.config)
                await self.studio.run(job)
                self.assertEqual(job['status'], 'completed', job['failures'])
            data = self.studio.read_page(self.path)
            data['regions'][0].update(font_family='MangaX Mali', font_size=28)  # Restyle after inpainting.
            self.studio.save_page(self.path, data)
            job = self.studio.enqueue([self.path], ['layout'], self.config)
            await self.studio.run(job)
            self.assertEqual(job['status'], 'completed', job['failures'])
            self.assertEqual(rendered, [['ไทย']])
            data = self.studio.read_page(self.path)
            data['regions'][0]['lines'] = box(x=30)['lines']
            self.studio.save_page(self.path, data)
            job = self.studio.enqueue([self.path], ['layout'], self.config)
            await self.studio.run(job)
            self.assertEqual(job['status'], 'failed')
            self.assertEqual(job['failures'][0]['stage'], 'layout')
            self.assertEqual(len(rendered), 1)

    def test_seeding_keeps_results_until_the_editor_really_changes(self):
        self.studio.seed_editor(self.path, [])
        data = self.studio.read_page(self.path)
        data['regions'] = [box(translation='ผล')]  # An OCR/translation job produced this.
        data['stages'] = {'ocr': {'key': 'k'}}
        self.studio.save_page(self.path, data)
        kept = self.studio.seed_editor(self.path, [])  # Editor unchanged: the next stage sees the result.
        self.assertEqual(kept['regions'][0]['translation'], 'ผล')
        edited = self.studio.seed_editor(self.path, [box(translation='แก้เอง')])  # Editor wins.
        self.assertEqual(edited['regions'][0]['translation'], 'แก้เอง')
        self.assertEqual(edited['stages'], {'ocr': {'key': 'k'}})  # Stage keys decide validity.
        numpy_regions = [dict(box(), lines=np.array(box()['lines'], dtype=np.float64), fg_colors=(1, 2, 3))]
        self.studio.seed_editor(self.path, numpy_regions)  # Editor data is not plain JSON.
        self.assertEqual(self.studio.read_page(self.path)['regions'][0]['lines'],
                         [[[10.0, 10.0], [70.0, 10.0], [70.0, 40.0], [10.0, 40.0]]])

    # ---- cancel / shutdown status ---------------------------------------
    async def _halt(self, request):
        started = asyncio.Event()
        async def stage(path, data, name, *args):
            started.set()
            await asyncio.Event().wait()
        with patch.object(self.studio, '_stage', side_effect=stage):
            job = self.studio.enqueue([self.path], ['ocr', 'translation'], self.config)
            events = []
            task = asyncio.ensure_future(self.studio.run(job, events.append))
            await started.wait()
            request()
            with self.assertRaises(asyncio.CancelledError):
                await task
        return job, events

    async def test_user_cancel_ends_as_cancelled(self):
        job, events = await self._halt(self.studio.request_cancel)
        self.assertEqual(job['status'], 'cancelled')
        self.assertEqual(events[-1], {'status': 'cancelled', 'job': job['id']})
        self.assertEqual(json.loads(self.studio.journal.read_text(encoding='utf-8'))[0]['status'], 'cancelled')

    async def test_shutdown_ends_as_interrupted(self):
        job, _ = await self._halt(self.studio.request_shutdown)
        self.assertEqual(job['status'], 'interrupted')

    async def test_cancel_between_units_keeps_finished_work(self):
        async def stage(path, data, name, *args):
            data['regions'] = [box()]
            self.studio.cancel.set()
        with patch.object(self.studio, '_stage', side_effect=stage):
            job = self.studio.enqueue([self.path], ['ocr', 'translation'], self.config)
            await self.studio.run(job)
        self.assertEqual(job['status'], 'cancelled')
        self.assertEqual(job['completed'], [self.path + '::ocr'])
        self.assertEqual(self.studio.snapshot()[0]['status'], 'cancelled')

    # ---- export ----------------------------------------------------------
    async def test_layout_exports_under_original_name_without_touching_source(self):
        chapter = self.root / 'My Chapter'
        chapter.mkdir()
        source = chapter / 'page-001.png'
        Image.new('RGB', (40, 40), 'white').save(source)
        before = file_hash(source)
        calls = []
        async def stage(path, data, name, *args):
            calls.append(name)
            self.studio._save_image(self.studio.page_dir(path) / 'translated.png', np.full((40, 40, 3), 7, np.uint8))
        output = self.root / 'out'
        with patch.object(self.studio, '_stage', side_effect=stage):
            events = []
            job = self.studio.enqueue([str(source)], ['layout'], self.config, str(output))
            await self.studio.run(job, events.append)
            exported = output / 'My Chapter' / 'page-001.png'
            self.assertEqual(job['status'], 'completed', job['failures'])
            self.assertTrue(exported.is_file())
            with Image.open(exported) as image:
                self.assertEqual(np.array(image)[0, 0].tolist(), [7, 7, 7])
            self.assertEqual(self.studio.export_dir(str(source)), exported.parent)
            self.assertEqual([e.get('export') for e in events if e.get('status') == 'completed' and 'stage' in e],
                             [str(exported)])
            self.assertEqual(file_hash(source), before)
            await self.studio.run(self.studio.enqueue([str(source)], ['layout'], self.config, str(output)))
            self.assertEqual(calls, ['layout'])  # Reused.
            exported.unlink()  # A missing export is produced again.
            await self.studio.run(self.studio.enqueue([str(source)], ['layout'], self.config, str(output)))
            self.assertEqual(calls, ['layout', 'layout'])
            self.assertTrue(exported.is_file())

    def test_export_never_targets_the_source_folder(self):
        chapter = self.root / 'Chapter'
        chapter.mkdir()
        source = chapter / 'p.webp'
        beside = chapter / 'manga_translator_work' / 'result' / 'p.webp'
        # No output folder chosen, or one that maps back onto the chapter folder itself.
        self.assertEqual(self.studio.export_target(str(source), self.config, ''), beside)
        self.assertEqual(self.studio.export_target(str(source), self.config, str(self.root)), beside)
        self.assertEqual(self.studio.export_target(str(source), self.config, str(self.root / 'out')),
                         self.root / 'out' / 'Chapter' / 'p.webp')
        self.config.cli.format = 'jpg'
        self.assertEqual(self.studio.export_target(str(source), self.config, str(self.root / 'out')).name, 'p.jpg')
        self.config.cli.save_to_source_dir = True
        self.assertEqual(self.studio.export_target(str(source), self.config, str(self.root / 'out')),
                         beside.with_suffix('.jpg'))

    # ---- one root for context ------------------------------------------
    async def test_run_reads_chapter_context_from_the_studio_root(self):
        atomic_json(context_path(self.path, self.root), {'story': 'ฉากทะเล'})
        seen = []
        async def stage(path, data, name, config, context):
            seen.append(context.get('story'))
        with patch.object(self.studio, '_stage', side_effect=stage):
            await self.studio.run(self.studio.enqueue([self.path], ['ocr'], self.config))
        self.assertEqual(seen, ['ฉากทะเล'])


class RoundtripTests(unittest.TestCase):
    """Assistant annotations must survive the editor's save -> reopen cycle."""
    def test_metadata_and_review_survive_editor_save_and_reopen(self):
        from editor.document_state import RegionCollection
        from services.export_service import ExportService
        from services.file_service import FileService
        with tempfile.TemporaryDirectory() as temp:
            image = os.path.join(temp, 'page.png')
            Image.new('RGB', (200, 200), 'white').save(image)
            region = box(translation='สวัสดี', translation_raw='สวัสดี', texts=['Hello'], font_family='MangaX Itim',
                         direction='h', alignment='center', fg_colors=[10, 20, 30], bg_colors=[255, 255, 255],
                         stroke_width=0.2, line_spacing=1.3, letter_spacing=1.0, target_lang='THA',
                         center=[40.0, 25.0], speaker='Yuna', listener='Ren', text_kind='shout', locked=True)
            region['_mangax_review'] = review_stamp(region)
            # The model normalizes regions on the way in and out.
            in_editor = RegionCollection([region]).snapshot()
            ExportService().save_editor_project(image, in_editor, None, {'render': {}})
            files = FileService.__new__(FileService)
            files.logger = SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None, error=print)
            files.config_service = SimpleNamespace(get_config=lambda: None)
            loaded, _, _, _ = files.load_translation_json(image)
            reopened = RegionCollection(loaded).snapshot()[0]
        for key in METADATA:
            self.assertEqual(reopened[key], region[key], key)
        self.assertEqual(reopened['direction'], 'horizontal')  # The save format really differs...
        self.assertEqual(reopened['font_color'], '#0a141e')
        self.assertTrue(reviewed(reopened))  # ...and the approval still holds.
        reopened['translation'] = 'แก้แล้ว'
        self.assertFalse(reviewed(reopened))

    def test_backend_textblock_write_back_keeps_annotations(self):
        region = box(translation='ไทย', texts=['Hello'], speaker='Yuna', text_kind='sfx', locked=False,
                     _mangax_review='stamp')
        written = TextBlock(**region).to_dict()
        self.assertEqual({key: written[key] for key in METADATA if key in written},
                         {'speaker': 'Yuna', 'text_kind': 'sfx', 'locked': False, '_mangax_review': 'stamp'})
        plain = TextBlock(**box(texts=['Hello'])).to_dict()
        self.assertFalse(set(plain) & set(METADATA))  # Nothing is invented for other tools.

    def test_region_block_uses_editor_colors_and_stroke(self):
        block = region_block(box(translation='ไทย', font_color='#102030', bg_colors=[250, 251, 252],
                                 stroke_width=0.11, direction='horizontal', speaker='Yuna'))
        self.assertEqual(tuple(block.fg_colors), (16, 32, 48))
        self.assertEqual(tuple(block.bg_colors), (250, 251, 252))
        self.assertEqual(block.default_stroke_width, 0.11)
        self.assertEqual(block.text, 'Hello')
        self.assertEqual(block.editor_metadata, {})


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.translator.target_lang = 'THA'

    def kinds(self, **fields):
        region = box(w=400, h=120, translation='สวัสดี', **fields)
        return {issue['kind'] for issue in quality_issues('page.png', [region], {}, self.config)}

    def test_font_forms_used_by_the_editor_are_not_false_positives(self):
        for value in ('MangaX Itim', 'MangaX Itim::Regular', 'mangax itim', 'MangaX Kanit::Bold',
                      'fonts/MangaX-itim-400-normal.ttf'):
            with self.subTest(font=value):
                self.assertFalse(self.kinds(font_family=value) & {'font', 'glyph', 'geometry'})

    def test_unknown_font_style_and_file_are_reported(self):
        for value in ('No Such Family 123', 'MangaX Itim::NoSuchStyle', 'fonts/no-such-file.ttf'):
            with self.subTest(font=value):
                self.assertIn('font', self.kinds(font_family=value))

    def test_runs_on_a_worker_thread(self):
        result = []
        worker = threading.Thread(target=lambda: result.append(self.kinds(font_family='MangaX Mali')))
        worker.start()
        worker.join(60)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0] & {'font', 'glyph', 'geometry'})


class ContextTests(unittest.TestCase):
    def test_glossary_is_relevant_and_voice_context_is_bounded(self):
        context = bounded_context({'custom_prompt_json': {'glossary': {'characters': [
            {'original':'Yuna','translation':'ยูนะ'}, {'original':'Other','translation':'อื่น'}]}},
            'chapter_context': {'story':'ฉากทะเล', 'characters':[{'name':'Yuna','voice':'a'*20000}]},
            'manga_regions':[{'text':'Yuna!','speaker':'hero','listener':'Yuna'}]}, ['Yuna!'])
        self.assertEqual(context['glossary'],[{'original':'Yuna','translation':'ยูนะ'}])
        self.assertEqual(context['speakers'][0]['listener'],'Yuna')
        self.assertLessEqual(len(json.dumps(context,ensure_ascii=False).encode()),24000)

    def test_edit_invalidates_review_and_memory_context(self):
        region = {'text':'Hello', 'translation':'สวัสดี', 'font_size':24}
        region['_mangax_review'] = review_stamp(region)
        self.assertTrue(reviewed(region))
        changed = copy.deepcopy(region)
        changed['font_size'] = 25
        self.assertFalse(reviewed(changed))
        self.assertNotEqual(approved_memory_key(region, {'glossary':[]},'THA'),
                            approved_memory_key(region, {'glossary':[{'original':'Hello','translation':'หวัดดี'}]},'THA'))

    def test_deskew_preserves_source_and_rejects_crossed_corners(self):
        image = np.full((100,100,3), 200, dtype=np.uint8)
        original = image.copy()
        crop = straighten(image, [[10,10],[90,20],[80,90],[20,80]], .1)
        self.assertEqual(crop.ndim,3)
        self.assertTrue(np.array_equal(image,original))
        with self.assertRaises(ValueError):
            straighten(image,[[10,10],[90,90],[90,10],[10,90]])


if __name__ == '__main__':
    unittest.main()
