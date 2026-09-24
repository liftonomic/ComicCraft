"""Comic-related schemas for the public API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.panel import Panel

from .enums import GenerationStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ComicCreateRequest(BaseModel):
    """Payload accepted by ``POST /api/comics``."""

    prompt: str = Field(min_length=1, max_length=4000)
    character_name: str = Field(default="", max_length=120)
    character_description: str = Field(default="", max_length=2000)
    setting: str = Field(default="", max_length=200)
    tone: str = Field(default="", max_length=120)
    art_style: str = Field(default="", max_length=120)
    panel_count: int = Field(default=5, ge=1, le=20)

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "A brave fox exploring an enchanted forest",
                "character_name": "Leo",
                "character_description": "A young orange fox wearing a blue scarf",
                "setting": "Enchanted forest",
                "tone": "adventure",
                "art_style": "anime",
                "panel_count": 5,
            }
        }
    }


class ComicGenerationResponse(BaseModel):
    """Immediate response to creating a comic.

    The endpooint returns this right away; the client polls the status
    endpoint while generation runs in the background.
    """

    comic_id: str
    status: GenerationStatus = GenerationStatus.GENERATING_STORY


class ComicStatusResponse(BaseModel):
    """Polling payload for ``GET /api/comics/{comic_id}/status``."""

    comic_id: str
    status: GenerationStatus
    progress: int = Field(ge=0, le=100)
    current_step: str = ""


class Comic(BaseModel):
    """Full comic record returned to clients."""

    id: str
    title: str = Field(default="", max_length=300)
    prompt: str = Field(default="", max_length=4000)
    tone: str = Field(default="", max_length=120)
    style: str = Field(default="", max_length=120)
    panel_count: int = Field(default=0, ge=0)
    status: GenerationStatus = GenerationStatus.CREATED
    character: dict = Field(default_factory=dict)
    panels: list[Panel] = Field(default_factory=list)
    error: Optional[dict] = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ComicExportResponse(BaseModel):
    """Response to ``POST /api/comics/{comic_id}/export``."""

    comic_id: str = ""
    download_url: str