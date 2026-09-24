"""Panel-related schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DialogueLine(BaseModel):
    """A single line of dialogue attributed to a character."""

    character: str = Field(default="", max_length=120)
    text: str = Field(default="", max_length=1000)


class Panel(BaseModel):
    """A single comic panel (schema view returned to clients)."""

    id: str = Field(default="", max_length=40)
    number: int = Field(ge=1)
    title: str = Field(default="", max_length=200)
    scene: str = Field(default="", max_length=2000)
    narration: str = Field(default="", max_length=1000)
    dialogue: list[DialogueLine] = Field(default_factory=list)
    image_prompt: str = Field(default="", max_length=3000)
    image_url: str = Field(default="", max_length=500)
    status: Literal["pending", "generating", "ready", "failed"] = "pending"


class PanelRegenerateRequest(BaseModel):
    """Instruction for regenerating a single panel.

    Only the targeted panel is regenerated; the rest of the comic is kept.
    """

    instruction: str = Field(min_length=1, max_length=1000)


class ContinueRequest(BaseModel):
    """Instruction for continuing an existing comic with a new chapter."""

    instruction: str = Field(min_length=1, max_length=1000)
    panel_count: int = Field(default=5, ge=1, le=20)

    model_config = {"json_schema_extra": {"example": {"instruction": "Continue the adventure", "panel_count": 5}}}