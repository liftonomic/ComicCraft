"""Prompt templates for Gemini.

Templates live here so prompt engineering (wording, styling) is separate
from the call-side code. Every Gemini call in this application returns
structured JSON; the templates below instruct the model to do so.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# JSON schema contract shared with the model.
# ---------------------------------------------------------------------------

OUTLINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "synopsis": {"type": "string"},
        "character": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "species": {"type": "string"},
                "appearance": {"type": "string"},
                "personality": {"type": "string"},
                "clothing": {"type": "string"},
                "colors": {"type": "string"},
                "physical_traits": {"type": "string"},
            },
            "required": ["name"],
        },
        "panels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "title": {"type": "string"},
                    "scene": {"type": "string"},
                    "narration": {"type": "string"},
                    "dialogue": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "character": {"type": "string"},
                                "text": {"type": "string"},
                            },
                            "required": ["character", "text"],
                        },
                    },
                    "image_prompt": {"type": "string"},
                },
                "required": ["number", "title", "scene", "narration", "image_prompt"],
            },
        },
    },
    "required": ["title", "character", "panels"],
}

CHARACTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "species": {"type": "string"},
        "appearance": {"type": "string"},
        "personality": {"type": "string"},
        "clothing": {"type": "string"},
        "colors": {"type": "string"},
        "physical_traits": {"type": "string"},
    },
    "required": ["name"],
}

#: System default appended to every user story prompt so Gemini keeps the
#: story (and therefore the image prompts) short and on-topic.
STORY_DIRECTIVE = (
    "Give simple too short story for this. No explanation or anything just "
    "short simple, within 2 lines"
)


def with_story_directive(prompt: str) -> str:
    """Wrap the user's prompt with :data:`STORY_DIRECTIVE`."""
    return f'"{prompt}"\n\n{STORY_DIRECTIVE}'


def _system_instruction() -> str:
    return (
        "You are a professional comic book writer and artist. You write "
        "engaging, well-paced comic scripts and produce detailed image "
        "prompts for each panel. Always respond with valid JSON matching the "
        "schema provided. Do not include markdown fences or commentary."
    )


def character_profile_prompt(
    *,
    character_name: str,
    character_description: str,
    prompt: str,
) -> str:
    """Ask Gemini for a detailed, structured character profile."""
    return (
        "Create a detailed character profile for the comic described below.\n\n"
        f"Story prompt: {with_story_directive(prompt)}\n"
        f"Character name: {character_name or 'not specified (invent one)'}\n"
        f"Character description: {character_description or 'not specified'}\n\n"
        "Return JSON with fields: name, species, appearance, personality, "
        "clothing, colors, physical_traits. Be specific so the character "
        "renders consistently across panels."
    )


def comic_outline_prompt(
    *,
    prompt: str,
    setting: str,
    tone: str,
    art_style: str,
    panel_count: int,
    character_name: str,
    character_description: str,
) -> str:
    """Ask Gemini for the full structured comic (title, character, panels)."""
    return (
        f"Write a {panel_count}-panel comic script.\n\n"
        f"Story prompt: {with_story_directive(prompt)}\n"
        f"Setting: {setting or 'invent a fitting setting'}\n"
        f"Tone: {tone or 'adventure'}\n"
        f"Art style: {art_style or 'anime'}\n"
        f"Character: {character_name or 'invent one'} — {character_description or ''}\n\n"
        "Rules:\n"
        "- Provide a catchy 'title'.\n"
        "- Provide a 'synopsis'.\n"
        "- Provide a consistent 'character' object with a stable appearance.\n"
        "- Provide exactly the requested number of 'panels'. Each panel has: "
        "number, title, scene (what is drawn), narration, dialogue (array of "
        "{character, text}), and a short 'image_prompt' (under 40 words) "
        "that directly depicts the story prompt as a standalone illustration.\n"
        "- Make the story engaging and coherent across panels."
    )
def panel_regeneration_prompt(
    *,
    instruction: str,
    panel: dict[str, Any],
    character_block: str,
) -> str:
    """Ask Gemini to rewrite one panel based on an instruction."""
    snapshot = json.dumps(panel, ensure_ascii=False)
    return (
        "Reimagine the single comic panel below according to the user's "
        "instruction, keeping the rest of the comic unchanged.\n\n"
        f"Instruction: {instruction}\n"
        f"Current panel JSON: {snapshot}\n"
        f"Character (keep consistent): {character_block}\n\n"
        "Return the panel as JSON with exactly these fields: number, title, "
        "scene, narration, dialogue (array of {character, text}), image_prompt."
    )


def continue_comic_prompt(
    *,
    instruction: str,
    prior_summary: str,
    character_block: str,
    panel_count: int,
) -> str:
    """Ask Gemini to append a new chapter to an existing comic."""
    return (
        "Continue the comic by writing the next chapter.\n\n"
        f"Continuation instruction: {instruction}\n"
        f"Prior story summary: {prior_summary}\n"
        f"Character (keep consistent): {character_block}\n"
        f"Number of new panels: {panel_count}\n\n"
        "Return JSON with a 'title' (the chapter title), 'synopsis', and "
        "'panels' (an array of new panels). Each panel has: number, title, "
        "scene, narration, dialogue (array of {character, text}), image_prompt. "
        "Advance the story; make the new chapter engage readers who read the "
        "previous one."
    )


def image_prompt_with_character(
    *,
    panel_image_prompt: str,
    character_block: str,
    art_style: str,
) -> str:
    """Assemble the final image prompt by injecting character consistency.

    The character block is included verbatim so appearance, clothing, colors,
    species, traits and personality stay identical across every panel.
    """
    style_note = f"Art style: {art_style}." if art_style else ""
    return (
        f"{panel_image_prompt} "
        f"Character consistency: {character_block} "
        f"{style_note} "
        "No text, no speech bubbles, no watermarks."
    ).strip()