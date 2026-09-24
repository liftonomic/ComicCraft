"""AI-layer tests: prompt builders and the offline providers.

These are fast, dependency-free checks that the prompt contract and the
mock providers behave as the pipeline expects (valid JSON shapes, a real
image file on disk, verbatim character-consistency injection).

Run from the repository root:
    python -m unittest discover -s backend/tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ai import prompts  # noqa: E402
from app.ai.gemini import MockGeminiClient  # noqa: E402
from app.ai.image_generator import MockImageGenerator, get_image_generator  # noqa: E402
from app.config import get_settings  # noqa: E402


class PromptBuilderTests(unittest.TestCase):
    """The prompt builders must produce non-empty, on-contract strings."""

    def test_outline_schema_shape(self) -> None:
        schema = prompts.OUTLINE_SCHEMA
        self.assertEqual(schema["type"], "object")
        self.assertIn("panels", schema["properties"])
        self.assertIn("character", schema["properties"])
        panel_props = schema["properties"]["panels"]["items"]["properties"]
        for field in ("number", "title", "scene", "narration", "image_prompt"):
            self.assertIn(field, panel_props)
        self.assertIn("dialogue", panel_props)

    def test_comic_outline_prompt_mentions_panel_count(self) -> None:
        text = prompts.comic_outline_prompt(
            prompt="A fox adventure",
            setting="forest",
            tone="adventure",
            art_style="anime",
            panel_count=4,
            character_name="Leo",
            character_description="orange fox",
        )
        self.assertIn("4-panel", text)
        self.assertIn("A fox adventure", text)
        self.assertIn("Leo", text)

    def test_character_profile_prompt_has_fields(self) -> None:
        text = prompts.character_profile_prompt(
            character_name="Leo",
            character_description="orange fox",
            prompt="A fox adventure",
        )
        for field in (
            "name",
            "species",
            "appearance",
            "personality",
            "clothing",
            "colors",
            "physical_traits",
        ):
            self.assertIn(field, text)

    def test_image_prompt_injects_character_block_verbatim(self) -> None:
        block = "orange fur, blue scarf, emerald eyes"
        text = prompts.image_prompt_with_character(
            panel_image_prompt="a fox in a forest",
            character_block=block,
            art_style="anime",
        )
        self.assertIn(block, text)
        self.assertIn("a fox in a forest", text)
        self.assertIn("anime", text)
        self.assertIn("No text", text)

    def test_regeneration_prompt_embeds_panel_json(self) -> None:
        panel = {
            "number": 1,
            "title": "Start",
            "scene": "a forest",
            "narration": "Once.",
            "dialogue": [{"character": "Leo", "text": "Hi"}],
            "image_prompt": "fox in forest",
        }
        text = prompts.panel_regeneration_prompt(
            instruction="make it night",
            panel=panel,
            character_block="orange fur",
        )
        self.assertIn("make it night", text)
        # The panel must be embedded as valid JSON we could round-trip.
        self.assertIn("a forest", text)
        self.assertIn(json.dumps(panel, ensure_ascii=False), text)


class MockGeminiClientTests(unittest.IsolatedAsyncioTestCase):
    """The offline Gemini stand-in must honour the JSON contract."""

    async def test_comic_outline_contract(self) -> None:
        client = MockGeminiClient()
        outline = await client.generate_comic_outline(
            prompt="A fox adventure",
            setting="forest",
            tone="adventure",
            art_style="anime",
            panel_count=3,
            character_name="Leo",
            character_description="orange fox",
        )
        self.assertTrue(outline["title"])
        self.assertEqual(outline["character"]["name"], "Leo")
        self.assertEqual(len(outline["panels"]), 3)
        numbers = [p["number"] for p in outline["panels"]]
        self.assertEqual(numbers, [1, 2, 3])
        for panel in outline["panels"]:
            self.assertTrue(panel["image_prompt"])
            self.assertIsInstance(panel["dialogue"], list)

    async def test_character_profile_contract(self) -> None:
        client = MockGeminiClient()
        profile = await client.generate_character_profile(
            character_name="", character_description="", prompt="x"
        )
        # Defaults to a name when none is supplied.
        self.assertTrue(profile["name"])
        for field in ("species", "appearance", "personality", "clothing"):
            self.assertTrue(profile[field])

    async def test_regenerate_panel_contract(self) -> None:
        client = MockGeminiClient()
        panel = {
            "number": 2,
            "title": "Middle",
            "scene": "old scene",
            "narration": "old",
            "dialogue": [],
            "image_prompt": "old prompt",
        }
        result = await client.regenerate_panel(
            instruction="make it rain", panel=panel, character_block="orange"
        )
        self.assertEqual(result["number"], 2)
        self.assertIn("make it rain", result["scene"])
        self.assertIsInstance(result["dialogue"], list)
        self.assertTrue(result["image_prompt"])

    async def test_continue_comic_contract(self) -> None:
        client = MockGeminiClient()
        chapter = await client.continue_comic(
            instruction="more adventure",
            prior_summary="they explored",
            character_block="orange",
            panel_count=2,
        )
        self.assertTrue(chapter["title"])
        self.assertEqual(len(chapter["panels"]), 2)


class MockImageGeneratorTests(unittest.IsolatedAsyncioTestCase):
    """The mock provider must write a real, readable PNG file."""

    async def test_generate_writes_png(self) -> None:
        settings = get_settings()
        generator = MockImageGenerator(settings)
        path = await generator.generate(prompt="a fox in a forest", style="anime")
        self.assertTrue(path.exists(), f"missing file: {path}")
        self.assertEqual(path.suffix, ".png")
        with path.open("rb") as handle:
            self.assertEqual(handle.read(8), b"\x89PNG\r\n\x1a\n")

    async def test_is_available(self) -> None:
        generator = MockImageGenerator(get_settings())
        self.assertTrue(await generator.is_available())


class ImageGeneratorFactoryTests(unittest.TestCase):
    """The factory must return the provider named by settings."""

    def test_factory_returns_configured_provider(self) -> None:
        settings = get_settings().model_copy(update={"image_provider": "mock"})
        generator = get_image_generator(settings)
        self.assertIsInstance(generator, MockImageGenerator)


if __name__ == "__main__":
    unittest.main(verbosity=2)