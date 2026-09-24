"""Panel model (in-memory record shape)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DialogueLineRecord(BaseModel):
    character: str = ""
    text: str = ""


class PanelRecord(BaseModel):
    id: str = Field(default="", max_length=40)
    comic_id: str = Field(default="", max_length=40)
    panel_number: int = Field(ge=1)
    title: str = ""
    scene: str = ""
    narration: str = ""
    dialogue: list[DialogueLineRecord] = Field(default_factory=list)
    image_prompt: str = ""
    image_url: str = ""
    status: Literal["pending", "generating", "ready", "failed"] = "pending"
    # Optional file system path used exclusively by the PDF export service.
    image_path: str = ""