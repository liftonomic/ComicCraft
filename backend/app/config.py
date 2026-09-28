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
IMAGE_PROVIDERS = ("gemini", "huggingface", "stable_diffusion", "mock")


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
    gemini_model: str = "gemini-2.0-flash"

    # Image generation provider selection.
    # One of: gemini, huggingface, stable_diffusion, mock.
    image_provider: str = "gemini"

    # Gemini image model ("Nano Banana"). Reuses gemini_api_key.
    gemini_image_model: str = "gemini-2.5-flash-image"
    # e.g. 1:1, 3:2, 2:3, 4:3, 16:9. Empty = model default.
    gemini_image_aspect_ratio: str = "4:3"

    # Locally downloaded Hugging Face diffusers model (folder or single file).
    hf_model_path: str = ""
    hf_device: str = "auto"  # auto | cuda | mps | cpu
    hf_dtype: str = "auto"  # auto | float16 | bfloat16 | float32
    hf_single_file_arch: str = "sdxl"  # sd15 | sdxl (single-file checkpoints only)
    hf_width: int = 768
    hf_height: int = 512
    hf_steps: int = 28
    hf_guidance_scale: float = 7.0

    # AUTOMATIC1111 / compatible Stable Diffusion WebUI endpoint.
    sd_api_url: str = "http://127.0.0.1:7860"

    # Optional basic-auth for the Stable Diffusion WebUI (user:pass). Empty if none.
    sd_api_auth: str = ""

    # txt2img tuning defaults.
    sd_width: int = 768
    sd_height: int = 512
    sd_steps: int = 28
    sd_cfg_scale: float = 7.0
    sd_sampler_name: str = "DPM++ 2M Karras"

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