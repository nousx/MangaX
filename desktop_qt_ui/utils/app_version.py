"""
Helper functions for the application version.
One place for reading the version and formatting it for display, in development and in a PyInstaller build.
"""
from __future__ import annotations

from utils.resource_helper import iter_existing_resource_paths


def get_app_version(default: str = "unknown") -> str:
    """Read the application version from the single version file, packaging/VERSION."""
    for version_path in iter_existing_resource_paths(("packaging/VERSION",)):
        try:
            with open(version_path, "r", encoding="utf-8") as version_file:
                version = version_file.read().strip()
        except OSError:
            continue
        if version:
            return version.lstrip("v")
    return default


def format_app_title(base_title: str, version: str | None) -> str:
    """Build the window title with the version number."""
    normalized_version = (version or "").strip()
    if not normalized_version or normalized_version == "unknown":
        return base_title
    return f"{base_title} v{normalized_version}"


def format_version_label(version: str | None) -> str:
    """Build the version label shown in the sidebar."""
    normalized_version = (version or "").strip()
    if not normalized_version or normalized_version == "unknown":
        return ""
    return f"v{normalized_version}"
