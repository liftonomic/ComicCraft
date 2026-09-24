"""Export routes: build and download comic PDFs.

The heavy lifting lives in ComicService/ExportService; these handlers only
translate HTTP requests into service calls and service results into
download responses.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.dependencies import get_comic_service
from app.schemas.comic import ComicExportResponse
from app.services.comic_service import ComicService
from app.utils.errors import InvalidStateError, NotFoundError

router = APIRouter(prefix="/comics", tags=["exports"])


@router.post("/{comic_id}/export", response_model=ComicExportResponse)
async def export_comic(
    comic_id: str,
    service: ComicService = Depends(get_comic_service),
) -> ComicExportResponse:
    """Render the comic to PDF and return a download URL."""
    await service.export_comic(comic_id)
    return ComicExportResponse(
        comic_id=comic_id,
        download_url=f"/api/comics/{comic_id}/download",
    )


@router.get("/{comic_id}/download")
async def download_comic(
    comic_id: str,
    service: ComicService = Depends(get_comic_service),
) -> FileResponse:
    """Stream the most recently generated PDF for the comic."""
    pdf_path = service.pdf_path_for(comic_id)
    if not pdf_path:
        raise InvalidStateError(
            f"Comic '{comic_id}' has not been exported yet; POST to export first."
        )
    path = Path(pdf_path)
    if not path.exists():
        raise NotFoundError(f"Exported PDF missing on disk: {pdf_path}")
    return FileResponse(
        str(path),
        media_type="application/pdf",
        filename=os.path.basename(pdf_path),
    )