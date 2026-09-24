"""User schemas.

Authentication is intentionally out of scope for the first MVP. These
objects exist so the API surface is stable and can be wired up without a
breaking change when authentication is introduced later.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class User(BaseModel):
    """Placeholder user record (no auth implemented yet)."""

    id: str = ""
    display_name: str = Field(default="", max_length=200)


class UserPublic(BaseModel):
    id: str
    display_name: str