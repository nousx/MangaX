"""Continuing an unfinished queue must skip finished pages for that run only."""
import _bootstrap  # noqa: F401, I001

from pathlib import Path
from types import SimpleNamespace

from desktop_qt_ui.app_logic import MainAppLogic
from manga_translator.config import Config, Translator, TranslatorConfig
from manga_translator.manga_translator import MangaTranslator
from manga_translator.utils.batch_skip import input_path, plan_batch_inputs


def _logic(tmp_path, overwrite=True):
    """A stand-in with just what start_backend_task touches."""
    started = []
    settings = SimpleNamespace(
        app=SimpleNamespace(last_output_path=str(tmp_path)),
        model_dump=lambda: {"cli": {"overwrite": overwrite}, "app": {"last_output_path": str(tmp_path)}},
    )
    logic = SimpleNamespace(
        chapter_assistant=None,
        state_manager=SimpleNamespace(is_translating=lambda: False),
        _stop_requested=False,
        _skip_existing_once=False,
        _scan_future=None,
        _translate_future=None,
        _cleanup_future=None,
        config_service=SimpleNamespace(get_config=lambda: settings),
        source_files=["page.png"],
        _validate_runtime_api_requirements=lambda config: True,
        _ui_log=lambda *args, **kwargs: None,
        _t=lambda text, **kwargs: text,
        start_file_scanning=started.append,
    )
    logic.start_backend_task = lambda: MainAppLogic.start_backend_task(logic)
    return logic, started


def test_should_skip_existing_pages_when_continuing(tmp_path):
    logic, started = _logic(tmp_path, overwrite=True)

    MainAppLogic.resume_backend_task(logic)

    assert started[0]["cli"]["overwrite"] is False


def test_should_keep_the_saved_setting_for_a_normal_start(tmp_path):
    logic, started = _logic(tmp_path, overwrite=True)

    MainAppLogic.start_backend_task(logic)

    assert started[0]["cli"]["overwrite"] is True


def test_should_apply_to_one_run_only(tmp_path):
    logic, started = _logic(tmp_path, overwrite=True)

    MainAppLogic.resume_backend_task(logic)
    MainAppLogic.start_backend_task(logic)

    assert [config["cli"]["overwrite"] for config in started] == [False, True]


def test_should_forget_the_request_when_the_start_is_refused(tmp_path):
    logic, started = _logic(tmp_path, overwrite=True)
    logic.state_manager = SimpleNamespace(is_translating=lambda: True)

    MainAppLogic.resume_backend_task(logic)
    logic.state_manager = SimpleNamespace(is_translating=lambda: False)
    MainAppLogic.start_backend_task(logic)

    assert [config["cli"]["overwrite"] for config in started] == [True]


def test_should_skip_pages_already_saved_beside_the_source(tmp_path):
    """The planner must look in <chapter>/manga_translator_work/result when that setting is on."""
    chapter = tmp_path / "Chapter 1"
    chapter.mkdir()
    pages = [chapter / f"{index:03}.webp" for index in range(1, 4)]
    for page in pages:
        page.write_bytes(b"source")
    result = chapter / "manga_translator_work" / "result"
    result.mkdir(parents=True)
    (result / "001.webp").write_bytes(b"done")
    (result / "002.webp").write_bytes(b"done")
    translator = MangaTranslator(params={"translator": "none", "use_gpu": False, "filter_text_enabled": False})
    config = Config(translator=TranslatorConfig(translator=Translator.none))
    save_info = {
        "output_folder": str(tmp_path / "elsewhere"),
        "format": None,
        "overwrite": False,
        "input_folders": {str(chapter)},
        "save_to_source_dir": True,
    }

    plan = plan_batch_inputs(translator, [(str(page), config) for page in pages], save_info)

    assert [Path(input_path(item)).name for item in plan.pending_items] == ["003.webp"]
    assert plan.skipped_count == 2
