"""Image routes.

Exposes the configured image provider's health and a manual panel-image
regeneration hook. Generation itself runs inside ComicService; this module
only reports provider state and delegates work.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.dependencies import get_comic_service, get_image_provider
from app.ai.image_generator import ImageGenerator
from app.services import image_service
from app.services.comic_service import ComicService
from app.utils.errors import NotFoundError

router = APIRouter(prefix="/images", tags=["images"])


@router.get("/provider")
async def get_provider_info(
    provider: ImageGenerator = Depends(get_image_provider),
) -> dict:
    """Report which image provider is active and whether it is reachable."""
    return await image_service.check_provider(provider)


@router.get("/{comic_id}/panels/{panel_id}")
async def get_panel_image(
    comic_id: str,
    panel_id: str,
    service: ComicService = Depends(get_comic_service),
) -> FileResponse:
    """Serve a single generated panel image by comic + panel id."""
    comic = service._require(comic_id)
    panel = next((p for p in comic.panels if p.id == panel_id), None)
    if panel is None:
        raise NotFoundError(
            f"Panel '{panel_id}' was not found in comic '{comic_id}'"
        )
    if not panel.image_path:
        raise NotFoundError(
            f"Panel '{panel_id}' has no generated image yet"
        )
    path = Path(panel.image_path)
    if not path.exists():
        raise NotFoundError(f"Panel image missing on disk: {panel.image_path}")
    return FileResponse(str(path))