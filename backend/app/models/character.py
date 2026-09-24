"""Character model (in-memory record shape mirroring a future DB row).

Mirrors the schema but is kept separate so the storage layer can evolve into
SQLAlchemy/PostgreSQL later without leaking ORM details into schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CharacterRecord(BaseModel):
    id: str = Field(default="", max_length=40)
    comic_id: str = Field(default="", max_length=40)
    name: str = ""
    description: str = ""
    appearance: str = ""
    personality: str = ""
    clothing: str = ""
    colors: str = ""
    species: str = ""
    physical_traits: str = ""

    def consistency_block(self) -> str:
        """Natural-language character identity for image prompts."""
        parts = [self.name or "the character"]
        if self.species:
            parts.append(f"a {self.species}")
        traits = [t for t in (self.appearance, self.physical_traits) if t]
        if traits:
            parts.append("with " + ", ".join(traits))
        if self.colors:
            parts.append(f"color scheme: {self.colors}")
        if self.clothing:
            parts.append(f"wearing {self.clothing}")
        if self.personality:
            parts.append(f"personality: {self.personality}")
        return ". ".join(parts).strip() + "."