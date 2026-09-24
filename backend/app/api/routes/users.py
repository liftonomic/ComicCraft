"""User routes (placeholder).

Authentication is out of scope for the MVP. These endpoints exist so the API
surface is stable and clients can be built against it; they return the
current anonymous user and that user's comics.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_comic_service, get_store
from app.database.database import ComicStore
from app.schemas.comic import Comic
from app.schemas.user import UserPublic
from app.services.comic_service import ComicService

router = APIRouter(prefix="/users", tags=["users"])

#: The single user every MVP request is attributed to.
ANONYMOUS_USER = UserPublic(id="anonymous", display_name="Anonymous")


@router.get("/me", response_model=UserPublic)
async def get_current_user() -> UserPublic:
    """Return the current (anonymous) user."""
    return ANONYMOUS_USER


@router.get("/me/comics", response_model=list[Comic])
async def list_my_comics(
    store: ComicStore = Depends(get_store),
    service: ComicService = Depends(get_comic_service),
) -> list[Comic]:
    """List every comic created by the current user."""
    return [
        service._to_schema(comic)
        for comic in store.list_all()
        if comic.user_id == ANONYMOUS_USER.id
    ]