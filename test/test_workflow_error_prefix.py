import _bootstrap  # noqa: F401, I001

from desktop_qt_ui.services import workflow_service


# MangaTranslator decides whether a TXT import succeeded by checking that the result
# does not start with "Error", so every failure message has to keep that prefix.
ERROR_PREFIX = "Error"


def test_should_return_error_prefix_when_input_files_are_missing(tmp_path):
    result = workflow_service.safe_update_large_json_from_text(
        str(tmp_path / "missing.txt"),
        str(tmp_path / "missing.json"),
        str(tmp_path / "missing_template.txt"),
    )

    assert result.startswith(ERROR_PREFIX)


def test_should_return_error_prefix_when_template_cannot_be_parsed(tmp_path):
    text_path = tmp_path / "page.txt"
    json_path = tmp_path / "page.json"
    template_path = tmp_path / "template.txt"
    text_path.write_text("hello", encoding="utf-8")
    json_path.write_text("{}", encoding="utf-8")
    template_path.write_text("", encoding="utf-8")

    result = workflow_service.safe_update_large_json_from_text(
        str(text_path), str(json_path), str(template_path)
    )

    assert result.startswith(ERROR_PREFIX)
