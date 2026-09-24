"""Global application state and FastAPI setup."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import comics, exports, health, images, users
from app.config import FRONTEND_DIR, get_settings
from app.utils.errors import AppError
from app.utils import files


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    _setup_logging(settings.log_level)
    # Ensure storage directories exist at startup.
    files.ensure_dir(settings.storage_base)
    files.images_dir()
    files.pdfs_dir()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "ComicCraft backend API — generate AI comic stories using Gemini "
            "for the narrative and a Stable Diffusion image pipeline for art."
        ),
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_prefix = "/api"

    application.include_router(health.router, prefix=api_prefix)
    application.include_router(comics.router, prefix=api_prefix)
    application.include_router(images.router, prefix=api_prefix)
    application.include_router(exports.router, prefix=api_prefix)
    application.include_router(users.router, prefix=api_prefix)

    # Serve generated media (images / pdfs) for previews and downloads.
    application.mount(
        "/generated",
        StaticFiles(directory=settings.storage_base),
        name="generated",
    )

    # Serve the static frontend from the same origin so the page, hero video,
    # fonts and generated artwork all load without CORS or a separate server.
    # Mounted last so /api and /generated keep priority.
    if FRONTEND_DIR.is_dir():
        application.mount(
            "/",
            StaticFiles(directory=FRONTEND_DIR, html=True),
            name="frontend",
        )
    else:
        logging.getLogger("app").warning(
            "Frontend directory not found at %s; serving API only.", FRONTEND_DIR
        )

    @application.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_payload())

    @application.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        # Safety net: never leak stack traces or internal state to clients.
        logging.getLogger("app").exception("Unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": {"code": "INTERNAL_ERROR", "message": "Unexpected internal error"},
            },
        )

    return application


app = create_app()