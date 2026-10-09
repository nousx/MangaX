"""API validation must follow the stages that the selected workflow actually runs."""


def required_api_sections(config):
    cli = config.cli
    if cli.export_from_local_json and (cli.generate_and_export or (cli.template and cli.save_text)):
        return set()
    if cli.template and cli.save_text:
        # Original-text export performs local preprocessing, not translation or rendering.
        return {"ocr", "colorizer"}
    sections = {"translator", "ocr", "colorizer", "render"}
    if cli.generate_and_export:
        sections.discard("render")
    return sections
