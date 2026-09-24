"""Comic routes.

Thin HTTP layer: each handler validates input via a Pydantic schema and
delegates all logic to the ComicService. No business logic lives here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_comic_service
from app.schemas.comic import (
    Comic,
    ComicCreateRequest,
    ComicGenerationResponse,
    ComicStatusResponse,
)
from app.schemas.panel import ContinueRequest, PanelRegenerateRequest
from app.services.comic_service import ComicService

router = APIRouter(prefix="/comics", tags=["comics"])


@router.post("", response_model=ComicGenerationResponse, status_code=202)
async def create_comic(
    request: ComicCreateRequest,
    service: ComicService = Depends(get_comic_service),
) -> ComicGenerationResponse:
    """Create a comic and start background generation."""
    return service.create_comic(request)


@router.get("/{comic_id}", response_model=Comic)
async def get_comic(
    comic_id: str,
    service: ComicService = Depends(get_comic_service),
) -> Comic:
    """Return the full comic (title, character, panels)."""
    return service.get_comic(comic_id)


@router.get("/{comic_id}/status", response_model=ComicStatusResponse)
async def get_comic_status(
    comic_id: str,
    service: ComicService = Depends(get_comic_service),
) -> ComicStatusResponse:
    """Return generation status/progress for polling."""
    return service.get_status(comic_id)


@router.post(
    "/{comic_id}/panels/{panel_id}/regenerate", response_model=Comic
)
async def regenerate_panel(
    comic_id: str,
    panel_id: str,
    request: PanelRegenerateRequest,
    service: ComicService = Depends(get_comic_service),
) -> Comic:
    """Regenerate only the targeted panel."""
    return await service.regenerate_panel(
        comic_id, panel_id, request.instruction
    )


@router.post("/{comic_id}/continue", response_model=Comic)
async def continue_comic(
    comic_id: str,
    request: ContinueRequest,
    service: ComicService = Depends(get_comic_service),
) -> Comic:
    """Append a new chapter to an existing comic."""
    return await service.continue_comic(
        comic_id, request.instruction, request.panel_count
    )


@router.delete("/{comic_id}", status_code=204)
async def delete_comic(
    comic_id: str,
    service: ComicService = Depends(get_comic_service),
) -> None:
    """Delete a comic from the store."""
    service.delete_comic(comic_id)