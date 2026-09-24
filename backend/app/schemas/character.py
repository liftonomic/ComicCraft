"""Character-related schemas.

The character profile is the source of truth used to keep the character
visually consistent across every generated panel.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CharacterCreate(BaseModel):
    """User-supplied starting points for a character."""

    name: str = Field(default="", min_length=0, max_length=120)
    description: str = Field(default="", max_length=2000)


class CharacterProfile(BaseModel):
    """Structured character profile produced by Gemini.

    These fields are folded into every panel's image prompt so the character
    keeps a consistent appearance, clothing, palette, species and personality.
    """

    name: str = Field(default="", max_length=120)
    species: str = Field(default="", max_length=120)
    appearance: str = Field(default="", max_length=1000)
    personality: str = Field(default="", max_length=1000)
    clothing: str = Field(default="", max_length=1000)
    colors: str = Field(default="", max_length=500)
    physical_traits: str = Field(default="", max_length=1000)

    def consistency_block(self) -> str:
        """Return a natural-language description of the character's identity.

        This is the exact text injected into every image prompt to enforce
        character consistency across panels.
        """
        parts = [f"{self.name}"]
        if self.species:
            parts.append(f"a {self.species}")
        traits = [t for t in (self.appearance, self.physical_traits) if t]
        if traits:
            parts.append(f"with {', '.join(traits)}")
        if self.colors:
            parts.append(f"color scheme: {self.colors}")
        if self.clothing:
            parts.append(f"wearing {self.clothing}")
        if self.personality:
            parts.append(f"personality: {self.personality}")
        return ". ".join(parts).strip() + "."


class CharacterInComic(BaseModel):
    """Character profile as stored on a comic record (schema view)."""

    name: str = ""
    appearance: str = ""
    personality: str = ""