"""Image service.

Coordinates generating one image per panel through the configured
:class:`ImageGenerator`. Returns absolute file paths; the comic service is
responsible for turning those into public URLs and updating the store.
"""

from __future__ import annotations

from pathlib import Path

from app.ai.image_generator import ImageGenerator
from app.models.panel import PanelRecord
from app.utils.errors import ImageGenerationError


async def generate_panel_image(
    generator: ImageGenerator,
    *,
    final_prompt: str,
    style: str,
    panel: PanelRecord,
) -> Path:
    """Generate and save a single panel's image."""
    try:
        path = await generator.generate(prompt=final_prompt, style=style)
    except ImageGenerationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ImageGenerationError(str(exc)) from exc
    panel.image_path = str(path)
    return path


async def check_provider(generator: ImageGenerator) -> dict:
    """Return provider availability info (used by the images test route)."""
    available = await generator.is_available()
    return {
        "provider": generator.__class__.__name__,
        "available": available,
    }