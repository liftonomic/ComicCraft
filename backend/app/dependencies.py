"""FastAPI dependencies.

A small seam where shared resources (settings, store, services) can be
injected into route handlers. Kept deliberately thin; providers are resolved
from the centralized settings.
"""

from __future__ import annotations

from fastapi import Depends

from app.ai.gemini import BaseGemini, GeminiClient, MockGeminiClient
from app.ai.image_generator import ImageGenerator, get_image_generator
from app.config import Settings, get_settings
from app.database.database import ComicStore, store
from app.services.comic_service import ComicService


def get_store() -> ComicStore:
    """Dependency: return the shared in-memory store."""
    return store


def get_app_settings() -> Settings:
    """Dependency: return the cached settings object."""
    return get_settings()


def get_gemini(settings: Settings = Depends(get_app_settings)) -> BaseGemini:
    """Dependency: return a Gemini client.

    Returns a real client when a GEMINI_API_KEY is configured, otherwise a
    deterministic mock so the pipeline still works offline (dev/tests).
    """
    if settings.gemini_api_key:
        return GeminiClient(settings)
    return MockGeminiClient()


def get_image_provider(
    settings: Settings = Depends(get_app_settings),
) -> ImageGenerator:
    """Dependency: return the configured image provider."""
    return get_image_generator(settings)


def get_comic_service(
    store: ComicStore = Depends(get_store),
    gemini: BaseGemini = Depends(get_gemini),
    image_provider: ImageGenerator = Depends(get_image_provider),
    settings: Settings = Depends(get_app_settings),
) -> ComicService:
    """Dependency: build the comic orchestration service."""
    return ComicService(
        store=store,
        gemini=gemini,
        image_generator=image_provider,
        settings=settings,
    )