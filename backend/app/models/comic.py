"""Comic model (in-memory record shape).

Designed to mirror the future PostgreSQL schema:

    User
     └── Comics
           ├── Character
           └── Panels
                 └── Images

For the MVP, panels live directly on the comic record and ``user_id`` is a
free-form placeholder (no auth). A single ``Images`` table would hold the
generated image paths.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.enums import GenerationStatus

from .character import CharacterRecord
from .panel import PanelRecord


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ComicRecord(BaseModel):
    id: str = Field(default="", max_length=40)
    user_id: str = Field(default="anonymous", max_length=80)
    title: str = ""
    prompt: str = ""
    tone: str = ""
    art_style: str = ""
    panel_count: int = 0
    setting: str = ""
    status: GenerationStatus = GenerationStatus.CREATED
    current_step: str = ""
    progress: int = 0
    character: Optional[CharacterRecord] = None
    panels: list[PanelRecord] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)  # file paths
    pdf_path: str = ""
    error: Optional[dict] = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)