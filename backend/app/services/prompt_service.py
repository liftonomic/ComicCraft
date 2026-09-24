"""Prompt service.

Assembles the final image generation prompt for every panel by combining the
model's raw panel description with the character-consistency block and the
art style. This is the single place where character consistency is applied
to image prompts, so every panel renders the same character.
"""

from __future__ import annotations

from app.ai.prompts import image_prompt_with_character
from app.models.character import CharacterRecord


def build_image_prompt(
    *,
    panel_image_prompt: str,
    character: CharacterRecord,
    art_style: str,
) -> str:
    """Return the final text prompt for the image provider."""
    return image_prompt_with_character(
        panel_image_prompt=panel_image_prompt,
        character_block=character.consistency_block(),
        art_style=art_style,
    )