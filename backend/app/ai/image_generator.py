"""Image generation abstraction.

The rest of the application depends only on :class:`ImageGenerator`, never on
a specific provider. This keeps the provider swappable: today it talks to a
local Stable Diffusion WebUI (AUTOMATIC1111-compatible) via HTTP; later a
Flux / ComfyUI / cloud provider can be added by implementing the same
interface without touching the rest of the codebase.

Providers:

* ``StableDiffusionImageGenerator`` — POST to ``SD_API_URL/sdapi/v1/txt2img``.
* ``MockImageGenerator`` — deterministic offline PNGs for dev and tests.
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

from app.config import Settings
from app.utils.errors import ImageGenerationError
from app.utils.files import images_dir

logger = logging.getLogger(__name__)


class ImageGenerator(ABC):
    """Interface implemented by every image provider.

    ``generate`` returns the absolute path of the saved image file.
    """

    @abstractmethod
    async def generate(self, *, prompt: str, style: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    async def is_available(self) -> bool:
        """Return True if the backing provider looks reachable."""
        raise NotImplementedError


def _save_png(image_bytes: bytes) -> Path:
    """Persist raw PNG bytes under generated/images/ and return the path."""
    directory = images_dir()
    filename = f"{uuid.uuid4().hex}-{int(time.time())}.png"
    path = directory / filename
    path.write_bytes(image_bytes)
    return path


class StableDiffusionImageGenerator(ImageGenerator):
    """Provider for a local/remote AUTOMATIC1111 (or compatible) WebUI.

    Calls ``POST {SD_API_URL}/sdapi/v1/txt2img`` and saves the first returned
    base64 image to ``generated/images/``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._timeout = httpx.Timeout(timeout=300.0, connect=10.0)

    def _endpoint(self) -> str:
        base = self._settings.sd_api_url.rstrip("/")
        return f"{base}/sdapi/v1/txt2img"

    def _auth(self):
        if self._settings.sd_api_auth:
            user, _, pwd = self._settings.sd_api_auth.partition(":")
            return (user, pwd)
        return None

    def _payload(self, *, prompt: str) -> dict:
        s = self._settings
        return {
            "prompt": prompt,
            "steps": s.sd_steps,
            "width": s.sd_width,
            "height": s.sd_height,
            "cfg_scale": s.sd_cfg_scale,
            "sampler_name": s.sd_sampler_name,
            "seed": -1,
            "batch_size": 1,
            "negative_prompt": (
                "text, words, speech bubble, watermark, logo, low quality, "
                "blurry, deformed"
            ),
        }

    async def generate(self, *, prompt: str, style: str) -> Path:
        payload = self._payload(prompt=prompt)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._endpoint(), json=payload, auth=self._auth()
                )
                response.raise_for_status()
                body = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.error("Stable Diffusion request failed: %s", exc)
            raise ImageGenerationError(str(exc)) from exc

        images = body.get("images") or []
        if not images:
            raise ImageGenerationError(
                "Stable Diffusion returned no images", code="EMPTY_RESPONSE"
            )

        # AUTOMATIC1111 returns base64 PNG data (may or may not be prefixed).
        raw = images[0]
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(raw)
        except Exception as exc:  # noqa: BLE001
            raise ImageGenerationError("Could not decode image response") from exc

        return _save_png(image_bytes)

    async def is_available(self) -> bool:
        base = self._settings.sd_api_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{base}/", auth=self._auth())
                return response.status_code < 500
        except Exception:  # noqa: BLE001
            return False


class MockImageGenerator(ImageGenerator):
    """Deterministic offline provider that produces a real saved PNG.

    The image is a generated rectangle sized by hashing the prompt, with a
    small caption drawn in. This lets the full pipeline, PDF export and tests
    run without a live Stable Diffusion server.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._calls = 0

    async def generate(self, *, prompt: str, style: str) -> Path:
        self._calls += 1
        seed = sum(ord(ch) for ch in (prompt + style)) % 360
        width = min(384 + (seed % 128), 768)
        height = min(256 + (seed % 128), 512)

        img = Image.new(
            "RGB",
            (width, height),
            (seed % 256, (seed * 2) % 256, (seed * 3) % 256),
        )
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [8, 8, width - 9, height - 9], outline=(255, 255, 255), width=3
        )
        draw.rectangle(
            [16, 16, width - 17, height - 17], outline=(200, 200, 200), width=2
        )
        draw.text((24, 24), f"Panel {self._calls}", fill=(255, 255, 255))

        directory = images_dir()
        filename = f"mock-{uuid.uuid4().hex}-{int(time.time())}.png"
        path = directory / filename
        img.save(path)
        return path

    async def is_available(self) -> bool:
        return True


def get_image_generator(settings: Settings) -> ImageGenerator:
    """Return the configured image provider instance."""
    if settings.image_provider == "mock":
        return MockImageGenerator(settings)
    if settings.image_provider == "stable_diffusion":
        return StableDiffusionImageGenerator(settings)
    raise ImageGenerationError(
        f"Unknown IMAGE_PROVIDER '{settings.image_provider}'",
        code="UNKNOWN_PROVIDER",
    )