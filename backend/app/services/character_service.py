"""Character service.

Responsible for turning raw user input + AI output into a consistent
:class:`CharacterRecord`, and for producing the natural-language consistency
block that is injected into every image prompt.
"""

from __future__ import annotations

from typing import Any

from app.ai.gemini import BaseGemini
from app.models.character import CharacterRecord
from app.utils.errors import StoryGenerationError
from app.utils.security import new_id


def build_character_record(
    *,
    comic_id: str,
    raw: dict[str, Any],
) -> CharacterRecord:
    """Create a CharacterRecord from a Gemini profile dict (or user fields)."""
    return CharacterRecord(
        id=new_id(),
        comic_id=comic_id,
        name=str(raw.get("name", "")).strip(),
        species=str(raw.get("species", "")).strip(),
        appearance=str(raw.get("appearance", "")).strip(),
        personality=str(raw.get("personality", "")).strip(),
        clothing=str(raw.get("clothing", "")).strip(),
        colors=str(raw.get("colors", "")).strip(),
        physical_traits=str(raw.get("physical_traits", "")).strip(),
        description=str(raw.get("description", "")).strip(),
    )


def consistency_block(character: CharacterRecord) -> str:
    """Return the character identity text for image prompts."""
    return character.consistency_block()


async def generate_character_profile(
    gemini: BaseGemini,
    *,
    comic_id: str,
    character_name: str,
    character_description: str,
    prompt: str,
) -> CharacterRecord:
    """Ask Gemini for a structured profile, falling back to user fields."""
    try:
        raw = await gemini.generate_character_profile(
            character_name=character_name,
            character_description=character_description,
            prompt=prompt,
        )
    except StoryGenerationError:
        # If the user already supplied a name/description, keep it.
        raw = {"name": character_name, "description": character_description}

    if not raw.get("name") and character_name:
        raw["name"] = character_name
    if not raw.get("description") and character_description:
        raw["description"] = character_description

    return build_character_record(comic_id=comic_id, raw=raw)