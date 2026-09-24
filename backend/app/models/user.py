"""User model (placeholder; no auth in the MVP)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UserRecord(BaseModel):
    id: str = Field(default="", max_length=40)
    display_name: str = ""