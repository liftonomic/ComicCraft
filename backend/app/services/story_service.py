"""Story service.

The story service coordinates Gemini calls with the character profile and
produces the structured outline: title + character + panels. Panels carry a
raw ``image_prompt`` produced by the model; the final image prompt (with
character consistency) is assembled by :mod:`app.services.prompt_service`.
"""

from __future__ import annotations

from typing import Any

from app.ai.gemini import BaseGemini
from app.models.panel import DialogueLineRecord, PanelRecord
from app.utils.security import new_id


def _dialogue(value: Any) -> list[DialogueLineRecord]:
    if not isinstance(value, list):
        return []
    lines: list[DialogueLineRecord] = []
    for item in value:
        if isinstance(item, dict):
            lines.append(
                DialogueLineRecord(
                    character=str(item.get("character", "")),
                    text=str(item.get("text", "")),
                )
            )
    return lines


def panels_from_outline(
    *,
    comic_id: str,
    outline: dict[str, Any],
) -> list[PanelRecord]:
    """Convert the Gemini outline's ``panels`` list into PanelRecords."""
    raw_panels = outline.get("panels") or []
    panels: list[PanelRecord] = []
    for index, item in enumerate(raw_panels, start=1):
        if not isinstance(item, dict):
            continue
        number = int(item.get("number") or index)
        panels.append(
            PanelRecord(
                id=new_id(),
                comic_id=comic_id,
                panel_number=number,
                title=str(item.get("title", f"Panel {number}")),
                scene=str(item.get("scene", "")),
                narration=str(item.get("narration", "")),
                dialogue=_dialogue(item.get("dialogue")),
                image_prompt=str(item.get("image_prompt", "")),
            )
        )
    return panels


async def generate_outline(
    gemini: BaseGemini,
    *,
    prompt: str,
    setting: str,
    tone: str,
    art_style: str,
    panel_count: int,
    character_name: str,
    character_description: str,
) -> dict[str, Any]:
    """Return the full structured comic outline dict from Gemini."""
    return await gemini.generate_comic_outline(
        prompt=prompt,
        setting=setting,
        tone=tone,
        art_style=art_style,
        panel_count=panel_count,
        character_name=character_name,
        character_description=character_description,
    )


def summarize_comic(panels: list[PanelRecord]) -> str:
    """Short prose summary used as context when continuing a comic."""
    if not panels:
        return "No prior story yet."
    scenes = [p.scene for p in panels[:6] if p.scene]
    return " ".join(scenes) if scenes else "The story has started."