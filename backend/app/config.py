"""Application configuration loaded from environment variables.

All configuration is centralized here so paths and credentials are never
hardcoded in the rest of the application. Values come from a ``.env`` file
or the process environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Root folder of the backend package (…/backend).
BACKEND_DIR = Path(__file__).resolve().parent.parent

#: Repository root (…/ComicCraft) — parent of ``backend`` and ``frontend``.
REPO_ROOT = BACKEND_DIR.parent

#: Folder holding the static frontend (index.html + assets).
FRONTEND_DIR = REPO_ROOT / "frontend"

#: Accepted values for the ``IMAGE_PROVIDER`` setting.
IMAGE_PROVIDERS = ("zimage", "mock")


class Settings(BaseSettings):
    """Typed settings object backed by environment variables."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application -----------------------------------------------------
    app_name: str = "ComicCraft"
    app_version: str = "0.1.0"
    app_env: str = "development"

    # Comma separated list of allowed CORS origins for the Canva frontend.
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- AI providers -----------------------------------------------------
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"

    # Image generation provider selection: zimage | mock.
    image_provider: str = "zimage"

    # Z-Image-Turbo (CPU). Only the transformer comes from the GGUF file; the
    # text encoder / VAE / scheduler are pulled from the base repo.
    zimage_gguf_path: str = str(BACKEND_DIR / "model" / "z-image-turbo-Q4_0.gguf")
    zimage_gguf_repo: str = "unsloth/Z-Image-Turbo-GGUF"
    zimage_base_repo: str = "Tongyi-MAI/Z-Image-Turbo"
    zimage_width: int = 512
    zimage_height: int = 512
    zimage_steps: int = 9
    zimage_threads: int = 0  # 0 = torch default

    # --- Comic generation -------------------------------------------------
    min_panel_count: int = 1
    max_panel_count: int = 20

    # --- Storage ----------------------------------------------------------
    # Base directory holding the generated/ images + pdfs.
    storage_base: Path = BACKEND_DIR / "generated"
    images_dir_name: str = "images"
    pdfs_dir_name: str = "pdfs"

    # --- Logging ----------------------------------------------------------
    log_level: str = "INFO"

    @property
    def images_dir(self) -> Path:
        return self.storage_base / self.images_dir_name

    @property
    def pdfs_dir(self) -> Path:
        return self.storage_base / self.pdfs_dir_name

    def validate_image_provider(self) -> None:
        """Ensure IMAGE_PROVIDER maps to a known implementation."""
        if self.image_provider not in IMAGE_PROVIDERS:
            raise ValueError(
                f"Unknown IMAGE_PROVIDER '{self.image_provider}'. "
                f"Choose from {', '.join(IMAGE_PROVIDERS)}."
            )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (reads .env once per process)."""
    settings = Settings()
    settings.validate_image_provider()
    return settings