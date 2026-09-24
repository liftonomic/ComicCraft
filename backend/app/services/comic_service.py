"""Comic service.

The orchestrator for the whole generation pipeline. It owns the comic state
machine and coordinates the story, character, prompt, image and export
services. Route handlers stay thin and simply call methods here.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.ai.gemini import BaseGemini
from app.ai.image_generator import ImageGenerator
from app.config import Settings
from app.database.database import ComicStore
from app.models.comic import ComicRecord
from app.models.panel import PanelRecord
from app.schemas.comic import (
    Comic,
    ComicGenerationResponse,
    ComicStatusResponse,
)
from app.schemas.enums import GenerationStatus, STAGE_PROGRESS
from app.utils.errors import InvalidStateError, NotFoundError
from app.utils.security import new_id

from . import character_service, prompt_service, story_service
from .export_service import ExportService

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _error_payload() -> dict:
    return {
        "code": "INTERNAL_ERROR",
        "message": "Comic generation failed. Please try again.",
    }


def _public_image_url(image_path: str) -> str:
    """Convert a stored filesystem image path into a browser-loadable URL.

    Panel images are written under the configured storage base, which is
    mounted at ``/generated``. Exposing that relative URL (instead of a raw
    Windows path like ``C:\\...\\generated\\images\\x.png``) lets the frontend
    load the artwork directly and keeps the API host-agnostic.
    """
    if not image_path:
        return ""
    name = Path(image_path).name
    return f"/generated/images/{name}" if name else ""


class ComicService:
    """Orchestrates comic generation and lifecycle."""

    def __init__(
        self,
        *,
        store: ComicStore,
        gemini: BaseGemini,
        image_generator: ImageGenerator,
        settings: Settings,
    ) -> None:
        self._store = store
        self._gemini = gemini
        self._image_generator = image_generator
        self._settings = settings
        self._export = ExportService(settings=settings, store=store)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _commit(self, comic: ComicRecord, **updates: Any) -> ComicRecord:
        """Produce an updated copy of ``comic`` and persist it."""
        updated = comic.model_copy(update={**updates, "updated_at": _utcnow()})
        self._store.update(updated)
        return updated

    def _require(self, comic_id: str) -> ComicRecord:
        comic = self._store.get(comic_id)
        if comic is None:
            raise NotFoundError(f"Comic '{comic_id}' was not found")
        return comic

    def _require_ready(self, comic: ComicRecord) -> ComicRecord:
        if comic.status != GenerationStatus.READY:
            raise InvalidStateError(
                f"Comic is in state '{comic.status.value}'; expected READY"
            )
        return comic

    # ------------------------------------------------------------------
    # Creation + background generation
    # ------------------------------------------------------------------
    def create_comic(self, request) -> ComicGenerationResponse:
        """Registers a comic and launches background generation."""
        comic_id = new_id()
        comic = ComicRecord(
            id=comic_id,
            user_id="anonymous",
            prompt=request.prompt,
            tone=request.tone,
            art_style=request.art_style,
            panel_count=request.panel_count,
            setting=request.setting,
            status=GenerationStatus.CREATED,
            progress=STAGE_PROGRESS[GenerationStatus.CREATED],
            current_step="Queued",
        )
        self._store.create(comic)
        asyncio.create_task(self._run_generation(comic_id, request))
        return ComicGenerationResponse(
            comic_id=comic_id,
            status=GenerationStatus.GENERATING_STORY,
        )

    async def _set_status(
        self,
        comic: ComicRecord,
        status: GenerationStatus,
        *,
        step: str,
        progress: Optional[int] = None,
    ) -> ComicRecord:
        progress = progress if progress is not None else STAGE_PROGRESS.get(status, 0)
        return self._commit(comic, status=status, current_step=step, progress=progress)

    async def _run_generation(
        self, comic_id: str, request
    ) -> None:
        """Full pipeline executed in the background task."""
        comic = self._require(comic_id)

        # 1) Story stage ------------------------------------------------
        comic = await self._set_status(
            comic, GenerationStatus.GENERATING_STORY, step="Finding the story arc"
        )
        logger.info("COMIC %s | story stage: character profile", comic_id)
        try:
            character = await character_service.generate_character_profile(
                self._gemini,
                comic_id=comic_id,
                character_name=request.character_name,
                character_description=request.character_description,
                prompt=request.prompt,
            )
            comic = self._commit(comic, character=character)

            logger.info("COMIC %s | story stage: comic outline", comic_id)
            outline = await story_service.generate_outline(
                self._gemini,
                prompt=request.prompt,
                setting=request.setting,
                tone=request.tone,
                art_style=request.art_style,
                panel_count=request.panel_count,
                character_name=request.character_name,
                character_description=request.character_description,
            )
            panels = story_service.panels_from_outline(comic_id=comic_id, outline=outline)
            if not panels:
                raise ValueError("Gemini did not produce any panels")
            title = str(outline.get("title") or f"{request.character_name or 'A'} Story")
            comic = self._commit(comic, title=title, panels=panels)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Comic %s failed during story generation", comic_id)
            self._commit(comic, status=GenerationStatus.FAILED, error=_error_payload())
            return

        # 2) Image stage -------------------------------------------------
        comic = await self._set_status(
            comic,
            GenerationStatus.GENERATING_IMAGES,
            step="Composing scenes and dialogue",
        )
        total = len(comic.panels)
        for idx, panel in enumerate(comic.panels, start=1):
            step = f"Generating panel {idx} of {total}"
            comic = self._commit(comic, current_step=step)
            logger.info("COMIC %s | image stage: %s", comic_id, step)
            await self._generate_one_panel_image(comic_id, panel)
            comic = self._store.get(comic_id) or comic
            if comic.status == GenerationStatus.FAILED:
                return

        # 3) Build stage --------------------------------------------------
        comic = await self._set_status(
            comic, GenerationStatus.BUILDING_COMIC, step="Building your comic"
        )
        comic = self._commit(
            comic,
            images=[p.image_path for p in comic.panels if p.image_path],
        )
        await asyncio.sleep(0)
        comic = await self._set_status(comic, GenerationStatus.READY, step="Ready")
        logger.info("COMIC %s | ready", comic_id)

    async def _generate_one_panel_image(
        self, comic_id: str, panel: PanelRecord
    ) -> None:
        comic = self._store.get(comic_id)
        if comic is None:
            return
        final_prompt = prompt_service.build_image_prompt(
            panel_image_prompt=panel.image_prompt,
            character=comic.character,
            art_style=comic.art_style,
        )
        try:
            path = await self._image_generator.generate(
                prompt=final_prompt, style=comic.art_style
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Comic %s panel image failed", comic_id)
            self._commit(
                comic,
                status=GenerationStatus.FAILED,
                error={
                    "code": "IMAGE_GENERATION_FAILED",
                    "message": f"Panel {panel.panel_number} image generation failed",
                },
            )
            return
        updated_panel = panel.model_copy(update={"image_path": str(path)})
        comic = self._store.get(comic_id) or comic
        comic = comic.model_copy(
            update={
                "panels": [
                    p if p.id != panel.id else updated_panel for p in comic.panels
                ]
            }
        )
        self._store.update(comic)

    # ------------------------------------------------------------------
    # Read + status
    # ------------------------------------------------------------------
    def get_comic(self, comic_id: str) -> Comic:
        """Return the public Comic schema for a comic."""
        comic = self._require(comic_id)
        return self._to_schema(comic)

    def get_status(self, comic_id: str) -> ComicStatusResponse:
        comic = self._require(comic_id)
        return ComicStatusResponse(
            comic_id=comic_id,
            status=comic.status,
            progress=comic.progress,
            current_step=comic.current_step,
        )

    # ------------------------------------------------------------------
    # Regeneration / continuation
    # ------------------------------------------------------------------
    async def regenerate_panel(
        self, comic_id: str, panel_id: str, instruction: str
    ) -> Comic:
        """Regenerate a single panel (story + image only for that panel)."""
        comic = self._require_ready(self._require(comic_id))
        panel = next((p for p in comic.panels if p.id == panel_id), None)
        if panel is None:
            raise NotFoundError(f"Panel '{panel_id}' was not found")

        panel_dict = {
            "number": panel.panel_number,
            "title": panel.title,
            "scene": panel.scene,
            "narration": panel.narration,
            "dialogue": [d.model_dump() for d in panel.dialogue],
            "image_prompt": panel.image_prompt,
        }
        character_block = (
            comic.character.consistency_block()
            if comic.character
            else "no character specified"
        )
        try:
            result = await self._gemini.regenerate_panel(
                instruction=instruction,
                panel=panel_dict,
                character_block=character_block,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Comic %s panel regeneration failed", comic_id)
            raise InvalidStateError(
                "Panel regeneration failed. Please try again."
            ) from exc

        updated_panel = panel.model_copy(
            update={
                "title": str(result.get("title") or panel.title),
                "scene": str(result.get("scene") or panel.scene),
                "narration": str(result.get("narration") or panel.narration),
                "image_prompt": str(result.get("image_prompt") or panel.image_prompt),
            }
        )
        comic = comic.model_copy(
            update={
                "panels": [
                    p if p.id != panel_id else updated_panel for p in comic.panels
                ]
            }
        )
        comic = self._commit(comic)

        # Regenerate the image for only this panel.
        await self._generate_one_panel_image(comic_id, updated_panel)
        return self.get_comic(comic_id)

    async def continue_comic(
        self, comic_id: str, instruction: str, panel_count: int
    ) -> Comic:
        """Append a new chapter to an existing comic."""
        comic = self._require_ready(self._require(comic_id))
        character_block = (
            comic.character.consistency_block()
            if comic.character
            else "no character specified"
        )
        prior_summary = story_service.summarize_comic(comic.panels)
        try:
            result = await self._gemini.continue_comic(
                instruction=instruction,
                prior_summary=prior_summary,
                character_block=character_block,
                panel_count=panel_count,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Comic %s continuation failed", comic_id)
            raise InvalidStateError("Continuation failed. Please try again.") from exc

        new_panels = story_service.panels_from_outline(comic_id=comic_id, outline=result)
        if not new_panels:
            raise InvalidStateError("Continuation produced no panels")

        comic = comic.model_copy(update={"panels": comic.panels + new_panels})
        comic = self._commit(comic, panel_count=len(comic.panels))
        for panel in new_panels:
            await self._generate_one_panel_image(comic_id, panel)
        return self.get_comic(comic_id)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    async def export_comic(self, comic_id: str) -> str:
        """Generate the PDF, remember its path on the record, and return it."""
        comic = self._require_ready(self._require(comic_id))
        pdf_path = await self._export.build_pdf(comic)
        self._commit(comic.model_copy(update={"pdf_path": pdf_path}))
        return pdf_path

    def pdf_path_for(self, comic_id: str) -> str:
        """Return the last exported PDF path for a comic, if any."""
        return self._require(comic_id).pdf_path

    def delete_comic(self, comic_id: str) -> None:
        """Remove a comic from the store (raises 404 if absent)."""
        self._require(comic_id)
        self._store.delete(comic_id)

    def _to_schema(self, comic: ComicRecord) -> Comic:
        from app.schemas.panel import DialogueLine, Panel

        character = (
            {
                "name": comic.character.name,
                "appearance": comic.character.appearance,
                "personality": comic.character.personality,
                "clothing": comic.character.clothing,
                "colors": comic.character.colors,
                "species": comic.character.species,
                "physical_traits": comic.character.physical_traits,
            }
            if comic.character
            else {}
        )

        panels = [
            Panel(
                id=p.id,
                number=p.panel_number,
                title=p.title,
                scene=p.scene,
                narration=p.narration,
                dialogue=[
                    DialogueLine(character=d.character, text=d.text)
                    for d in p.dialogue
                ],
                image_prompt=p.image_prompt,
                image_url=_public_image_url(p.image_path),
                status=p.status,
            )
            for p in comic.panels
        ]
        return Comic(
            id=comic.id,
            title=comic.title,
            prompt=comic.prompt,
            tone=comic.tone,
            style=comic.art_style,
            panel_count=comic.panel_count,
            status=comic.status,
            character=character,
            panels=panels,
            error=comic.error,
            created_at=comic.created_at,
            updated_at=comic.updated_at,
        )