"""Filesystem helpers to keep hardcoded paths out of the rest of the app.

All generated files (images / PDFs) are written under a configurable base
directory. Helper functions here guarantee the directories exist and return
absolute, safe paths.
"""

from __future__ import annotations

from pathlib import Path

from app.config import get_settings


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as absolute."""
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def images_dir() -> Path:
    """Return the (created) absolute directory for generated images."""
    return ensure_dir(get_settings().images_dir)


def pdfs_dir() -> Path:
    """Return the (created) absolute directory for generated PDFs."""
    return ensure_dir(get_settings().pdfs_dir)