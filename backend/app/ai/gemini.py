"""Thin wrapper around the Google GenAI SDK.

This module is responsible ONLY for talking to Gemini and returning parsed
Python structures (dicts / lists). No comic business logic lives here — that
is the job of the service layer.

Two implementations share a common interface:

* ``GeminiClient``  — real calls to the Gemini API (requires GEMINI_API_KEY).
* ``MockGeminiClient`` — deterministic offline stand-in for tests / dev.

Structured output is requested via ``response_mime_type="application/json"``
plus a ``response_schema``; as a fallback the text result is parsed as JSON
so the pipeline degrades gracefully.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.utils.errors import StoryGenerationError

logger = logging.getLogger(__name__)


class BaseGemini:
    """Interface shared by the real client and the mock."""

    async def generate_character_profile(
        self, *, character_name: str, character_description: str, prompt: str
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def generate_comic_outline(
        self,
        *,
        prompt: str,
        setting: str,
        tone: str,
        art_style: str,
        panel_count: int,
        character_name: str,
        character_description: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

class GeminiClient(BaseGemini):
    """Real implementation backed by ``google.genai.Client``."""

    def __init__(self, settings) -> None:
        self._settings = settings
        self._client = None
        self._model = settings.gemini_model

    def _lazy_client(self):
        # Import here so the SDK is only loaded when actually used.
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    def _generate_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        client = self._lazy_client()
        cfg = {
            "system_instruction": (
                "You are a professional comic book writer and artist. "
                "Always respond with valid JSON only. No markdown fences."
            ),
            "response_mime_type": "application/json",
            "response_schema": schema,
        }
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=cfg,
            )
            text = response.text or ""
        except Exception as exc:  # noqa: BLE001 - normalize any provider error
            logger.warning("Gemini structured call failed, falling back: %s", exc)
            text = self._generate_text(prompt)

        return _parse_json(text, schema, context="gemini")

    def _generate_text(self, prompt: str) -> str:
        """Fallback: text generation then JSON parse (no schema coupling)."""
        client = self._lazy_client()
        response = client.models.generate_content(
            model=self._model,
            contents=prompt,
            config={"system_instruction": "Always respond with valid JSON only."},
        )
        return response.text or ""

    # --- Public API --------------------------------------------------------
    async def generate_character_profile(
        self, *, character_name: str, character_description: str, prompt: str
    ) -> dict[str, Any]:
        from app.ai.prompts import CHARACTER_SCHEMA, character_profile_prompt

        user_prompt = character_profile_prompt(
            character_name=character_name,
            character_description=character_description,
            prompt=prompt,
        )
        return self._generate_json(user_prompt, CHARACTER_SCHEMA)
    async def generate_comic_outline(
        self,
        *,
        prompt: str,
        setting: str,
        tone: str,
        art_style: str,
        panel_count: int,
        character_name: str,
        character_description: str,
    ) -> dict[str, Any]:
        from app.ai.prompts import OUTLINE_SCHEMA, comic_outline_prompt

        user_prompt = comic_outline_prompt(
            prompt=prompt,
            setting=setting,
            tone=tone,
            art_style=art_style,
            panel_count=panel_count,
            character_name=character_name,
            character_description=character_description,
        )
        return self._generate_json(user_prompt, OUTLINE_SCHEMA)

    async def regenerate_panel(
        self, *, instruction: str, panel: dict[str, Any], character_block: str
    ) -> dict[str, Any]:
        from app.ai.prompts import CHARACTER_SCHEMA, panel_regeneration_prompt

        user_prompt = panel_regeneration_prompt(
            instruction=instruction,
            panel=panel,
            character_block=character_block,
        )
        return self._generate_json(user_prompt, CHARACTER_SCHEMA)

    async def continue_comic(
        self,
        *,
        instruction: str,
        prior_summary: str,
        character_block: str,
        panel_count: int,
    ) -> dict[str, Any]:
        from app.ai.prompts import OUTLINE_SCHEMA, continue_comic_prompt

        user_prompt = continue_comic_prompt(
            instruction=instruction,
            prior_summary=prior_summary,
            character_block=character_block,
            panel_count=panel_count,
        )
        return self._generate_json(user_prompt, OUTLINE_SCHEMA)


def _parse_json(
    text: str, schema: Optional[dict[str, Any]], context: str
) -> dict[str, Any]:
    """Parse model output into a dict, stripping markdown fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[len("json"):].lstrip()
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("response was not a JSON object")
        return data
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Could not parse Gemini output for %s: %s", context, exc)
        raise StoryGenerationError(
            f"Gemini returned invalid JSON for {context.replace('_', ' ')}"
        ) from exc
        from app.ai.prompts import CHARACTER_SCHEMA, character_profile_prompt

        user_prompt = character_profile_prompt(
            character_name=character_name,
            character_description=character_description,
            prompt=prompt,
        )
        return self._generate_json(user_prompt, CHARACTER_SCHEMA)
    async def regenerate_panel(
        self, *, instruction: str, panel: dict[str, Any], character_block: str
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def continue_comic(
        self,
        *,
        instruction: str,
        prior_summary: str,
        character_block: str,
        panel_count: int,
    ) -> dict[str, Any]:
        raise NotImplementedError
class MockGeminiClient(BaseGemini):
    """Deterministic offline stand-in for Gemini (dev/tests).

    Mirrors the async interface of :class:`GeminiClient` but returns canned,
    well-formed structured output so the full pipeline can be exercised
    without network or API keys.
    """

    def __init__(self) -> None:
        self._calls = 0

    async def generate_character_profile(
        self, *, character_name: str, character_description: str, prompt: str
    ) -> dict[str, Any]:
        self._calls += 1
        name = character_name or "Arlo"
        return {
            "name": name,
            "species": "fox",
            "appearance": "bright orange fur with a white tail tip",
            "personality": "brave, curious and kind",
            "clothing": "a blue scarf",
            "colors": "orange, white, blue",
            "physical_traits": "large emerald eyes, bushy tail",
        }

    async def generate_comic_outline(
        self,
        *,
        prompt: str,
        setting: str,
        tone: str,
        art_style: str,
        panel_count: int,
        character_name: str,
        character_description: str,
    ) -> dict[str, Any]:
        self._calls += 1
        name = character_name or "Arlo"
        title = f"{name} and the Hidden Path"
        panels = []
        acts = [
            ("A Quiet Beginning", "A new day starts in an ordinary place."),
            ("A Mysterious Sign", "The hero discovers something unusual."),
            ("Crossing Over", "The true adventure begins."),
            ("An Unexpected Friend", "A companion joins the journey."),
            ("The Way Forward", "The path ahead becomes clear."),
        ]
        for i in range(1, max(1, panel_count) + 1):
            act_title, scene = acts[(i - 1) % len(acts)]
            panels.append(
                {
                    "number": i,
                    "title": f"{act_title} · Panel {i}",
                    "scene": f"{scene} {name} moves deeper into {setting or 'the story'}.",
                    "narration": f"Step by step, {name} pressed onward.",
                    "dialogue": [
                        {"character": name, "text": "What lies beyond?"}
                    ],
                    "image_prompt": (
                        f"{name} standing in {setting or 'a vivid landscape'}, "
                        "looking onward with curiosity, cinematic lighting"
                    ),
                }
            )
        return {
            "title": title,
            "synopsis": f"{name} embarks on a {tone or 'thrilling'} adventure.",
            "character": {
                "name": name,
                "species": "fox",
                "appearance": "bright orange fur with a white tail tip",
                "personality": "brave, curious and kind",
                "clothing": "a blue scarf",
                "colors": "orange, white, blue",
                "physical_traits": "large emerald eyes, bushy tail",
            },
            "panels": panels,
        }

    async def regenerate_panel(
        self, *, instruction: str, panel: dict[str, Any], character_block: str
    ) -> dict[str, Any]:
        self._calls += 1
        return {
            "number": panel.get("number", 1),
            "title": panel.get("title", "Panel"),
            "scene": f"Reimagined: {instruction}",
            "narration": (panel.get("narration") or "") + " The scene shifts.",
            "dialogue": panel.get("dialogue", []),
            "image_prompt": (
                f"{instruction} — {panel.get('image_prompt', 'a dramatic scene')}"
            ),
        }

    async def continue_comic(
        self,
        *,
        instruction: str,
        prior_summary: str,
        character_block: str,
        panel_count: int,
    ) -> dict[str, Any]:
        self._calls += 1
        return {
            "title": "Chapter II — New Horizons",
            "synopsis": f"The story continues: {instruction}",
            "character": {
                "name": "Arlo",
                "appearance": "bright orange fur",
                "personality": "brave",
            },
            "panels": [
                {
                    "number": i,
                    "title": f"Continuing {i}",
                    "scene": f"{instruction} unfolds over panel {i}.",
                    "narration": "The journey continues.",
                    "dialogue": [{"character": "Arlo", "text": "Onward!"}],
                    "image_prompt": f"{instruction}, scene {i}, consistent character",
                }
                for i in range(1, max(1, panel_count) + 1)
            ],
        }