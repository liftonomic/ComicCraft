"""Image generation abstraction.

The rest of the application depends only on :class:`ImageGenerator`, never on
a specific provider, so the backend stays swappable.

Providers:

* ``ZImageGenerator`` - Z-Image-Turbo (GGUF-quantised transformer) run
  in-process on the CPU with diffusers.
* ``MockImageGenerator`` - deterministic offline PNGs for dev and tests.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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
        """Return True if the backing provider looks usable."""
        raise NotImplementedError


def _save_png(image_bytes: bytes) -> Path:
    """Persist raw PNG bytes under generated/images/ and return the path."""
    directory = images_dir()
    filename = f"{uuid.uuid4().hex}-{int(time.time())}.png"
    path = directory / filename
    path.write_bytes(image_bytes)
    return path


# The pipeline takes minutes to load and ~12 GB RAM, so keep one per process.
_ZIMAGE_PIPELINE: Any = None
_ZIMAGE_KEY: tuple | None = None
_ZIMAGE_LOCK = threading.Lock()


class ZImageGenerator(ImageGenerator):
    """Z-Image-Turbo on the CPU (see poc/zimage_t2i.py for the original POC).

    Only the transformer comes from the GGUF file (``ZIMAGE_GGUF_PATH``); the
    Qwen3 text encoder, VAE and scheduler come from ``ZIMAGE_BASE_REPO``
    (downloaded once into the Hugging Face cache). Requires
    ``pip install -r requirements-zimage.txt``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _gguf_path(self) -> Path:
        s = self._settings
        path = Path(s.zimage_gguf_path).expanduser()
        if path.exists():
            return path
        from huggingface_hub import hf_hub_download

        logger.info("%s not found - downloading from %s", path, s.zimage_gguf_repo)
        return Path(
            hf_hub_download(s.zimage_gguf_repo, path.name, local_dir=path.parent)
        )

    def _load_pipeline(self):
        global _ZIMAGE_PIPELINE, _ZIMAGE_KEY
        s = self._settings
        key = (s.zimage_gguf_path, s.zimage_base_repo)
        if _ZIMAGE_PIPELINE is not None and _ZIMAGE_KEY == key:
            return _ZIMAGE_PIPELINE
        try:
            import torch
            from diffusers import (
                GGUFQuantizationConfig,
                ZImagePipeline,
                ZImageTransformer2DModel,
            )
        except ImportError as exc:
            raise ImageGenerationError(
                "IMAGE_PROVIDER=zimage needs torch + diffusers + gguf: "
                "pip install -r requirements-zimage.txt",
                code="MISSING_DEPENDENCY",
            ) from exc

        if s.zimage_threads:
            torch.set_num_threads(s.zimage_threads)
        gguf_path = self._gguf_path()
        logger.info(
            "Loading Z-Image-Turbo from %s on cpu (%d threads)",
            gguf_path,
            torch.get_num_threads(),
        )
        # bf16 keeps RAM at ~12 GB (fp32 would roughly double it).
        dtype = torch.bfloat16
        transformer = ZImageTransformer2DModel.from_single_file(
            str(gguf_path),
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            config=s.zimage_base_repo,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        pipe = ZImagePipeline.from_pretrained(
            s.zimage_base_repo, transformer=transformer, torch_dtype=dtype
        ).to("cpu")
        pipe.set_progress_bar_config(disable=True)
        _ZIMAGE_PIPELINE, _ZIMAGE_KEY = pipe, key
        return pipe

    def _generate_sync(self, prompt: str) -> bytes:
        s = self._settings
        # One load/generation at a time: the CPU is already saturated.
        with _ZIMAGE_LOCK:
            pipe = self._load_pipeline()
            # Turbo is distilled: no CFG, so no negative prompt either.
            image = pipe(
                prompt=prompt,
                width=s.zimage_width,
                height=s.zimage_height,
                num_inference_steps=s.zimage_steps,
                guidance_scale=0.0,
            ).images[0]
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    async def generate(self, *, prompt: str, style: str) -> Path:
        try:
            image_bytes = await asyncio.to_thread(self._generate_sync, prompt)
        except ImageGenerationError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Z-Image generation failed: %s", exc)
            raise ImageGenerationError(str(exc)) from exc
        return _save_png(image_bytes)

    async def is_available(self) -> bool:
        try:
            import diffusers  # noqa: F401
            import gguf  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False
        return True


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
    if settings.image_provider == "zimage":
        return ZImageGenerator(settings)
    raise ImageGenerationError(
        f"Unknown IMAGE_PROVIDER '{settings.image_provider}'",
        code="UNKNOWN_PROVIDER",
    )
