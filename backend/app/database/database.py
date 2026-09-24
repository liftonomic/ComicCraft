"""In-memory store for the MVP.

A thread-safe dictionary indexed by comic id. The interface mirrors the
relationships in the future PostgreSQL schema (User -> Comics -> Character /
Panels -> Images) so that swapping this for a real database later is an
isolated change confined to this module.

No generation logic lives here; it is purely a persistence layer.
"""

from __future__ import annotations

import threading
from typing import Optional

from app.models.comic import ComicRecord
from app.models.character import CharacterRecord
from app.models.panel import PanelRecord


class ComicStore:
    """Thread-safe in-memory comic repository."""

    def __init__(self) -> None:
        self._comics: dict[str, ComicRecord] = {}
        self._lock = threading.RLock()

    # --- Comics ----------------------------------------------------------
    def create(self, comic: ComicRecord) -> ComicRecord:
        with self._lock:
            self._comics[comic.id] = comic
        return comic

    def get(self, comic_id: str) -> Optional[ComicRecord]:
        with self._lock:
            return self._comics.get(comic_id)

    def update(self, comic: ComicRecord) -> ComicRecord:
        with self._lock:
            self._comics[comic.id] = comic
        return comic

    def delete(self, comic_id: str) -> bool:
        with self._lock:
            return self._comics.pop(comic_id, None) is not None

    def list_all(self) -> list[ComicRecord]:
        with self._lock:
            return list(self._comics.values())

    def exists(self, comic_id: str) -> bool:
        with self._lock:
            return comic_id in self._comics

    # --- Panels ----------------------------------------------------------
    def get_panel(self, comic_id: str, panel_id: str) -> Optional[PanelRecord]:
        with self._lock:
            comic = self._comics.get(comic_id)
            if comic is None:
                return None
            return next((p for p in comic.panels if p.id == panel_id), None)

    def replace_panel(self, comic: ComicRecord, panel: PanelRecord) -> None:
        """Replace a panel in-place on the comic record."""
        with self._lock:
            comic.panels = [
                p if p.id != panel.id else panel for p in comic.panels
            ]

    # --- Character -------------------------------------------------------
    def get_character(self, comic_id: str) -> Optional[CharacterRecord]:
        with self._lock:
            comic = self._comics.get(comic_id)
            return comic.character if comic else None


# Application-wide singleton store.
store = ComicStore()