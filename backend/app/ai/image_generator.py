"""Image generation abstraction.

The rest of the application depends only on :class:`ImageGenerator`, never on
a specific provider. This keeps the provider swappable: today it talks to a
local Stable Diffusion WebUI (AUTOMATIC1111-compatible) via HTTP; later a
Flux / ComfyUI / cloud provider can be added by implementing the same
interface without touching the rest of the codebase.

Providers:

* ``GeminiImageGenerator`` — Google "Nano Banana" (``gemini-2.5-flash-image``).
* ``HuggingFaceImageGenerator`` — a locally downloaded diffusers model
  (``HF_MODEL_PATH``), run in-process with PyTorch.
* ``StableDiffusionImageGenerator`` — POST to ``SD_API_URL/sdapi/v1/txt2img``.
* ``MockImageGenerator`` — deterministic offline PNGs for dev and tests.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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


NEGATIVE_PROMPT = (
    "text, words, speech bubble, watermark, logo, low quality, blurry, deformed"
)


def _save_image_bytes(image_bytes: bytes) -> Path:
    """Normalise arbitrary image bytes (PNG/JPEG/WebP) to PNG and save."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
    except Exception as exc:  # noqa: BLE001
        raise ImageGenerationError("Could not decode image response") from exc
    return _save_png(buf.getvalue())


class GeminiImageGenerator(ImageGenerator):
    """Provider for Google's Gemini image model ("Nano Banana").

    Uses ``google-genai`` with ``GEMINI_API_KEY`` and ``GEMINI_IMAGE_MODEL``
    (default ``gemini-2.5-flash-image``).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            if not self._settings.gemini_api_key:
                raise ImageGenerationError(
                    "GEMINI_API_KEY is required for IMAGE_PROVIDER=gemini",
                    code="MISSING_API_KEY",
                )
            from google import genai

            self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    def _generate_sync(self, prompt: str) -> bytes:
        from google.genai import types

        s = self._settings
        config: dict[str, Any] = {"response_modalities": ["IMAGE"]}
        if s.gemini_image_aspect_ratio:
            config["image_config"] = types.ImageConfig(
                aspect_ratio=s.gemini_image_aspect_ratio
            )
        response = self._lazy_client().models.generate_content(
            model=s.gemini_image_model,
            contents=f"{prompt}\n\nAvoid: {NEGATIVE_PROMPT}.",
            config=types.GenerateContentConfig(**config),
        )
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                if inline is not None and inline.data:
                    data = inline.data
                    return base64.b64decode(data) if isinstance(data, str) else data
        raise ImageGenerationError(
            "Gemini returned no image (possibly blocked by safety filters)",
            code="EMPTY_RESPONSE",
        )

    async def generate(self, *, prompt: str, style: str) -> Path:
        try:
            image_bytes = await asyncio.to_thread(self._generate_sync, prompt)
        except ImageGenerationError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini image request failed: %s", exc)
            raise ImageGenerationError(str(exc)) from exc
        return _save_image_bytes(image_bytes)

    async def is_available(self) -> bool:
        return bool(self._settings.gemini_api_key)


# The diffusers pipeline is expensive to load, so keep one per process.
_HF_PIPELINE: Any = None
_HF_PIPELINE_KEY: tuple | None = None
_HF_LOCK = threading.Lock()


class HuggingFaceImageGenerator(ImageGenerator):
    """Provider for a manually downloaded Hugging Face diffusers model.

    ``HF_MODEL_PATH`` may be either:

    * a diffusers model folder (contains ``model_index.json``), e.g. from
      ``huggingface-cli download stabilityai/sdxl-turbo --local-dir ...``, or
    * a single ``.safetensors`` / ``.ckpt`` checkpoint file (set
      ``HF_SINGLE_FILE_ARCH`` to ``sd15`` or ``sdxl``).

    Requires ``torch`` and ``diffusers`` (see requirements-hf.txt).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _model_path(self) -> Path:
        raw = self._settings.hf_model_path
        if not raw:
            raise ImageGenerationError(
                "HF_MODEL_PATH is required for IMAGE_PROVIDER=huggingface",
                code="MISSING_MODEL_PATH",
            )
        path = Path(raw).expanduser()
        if not path.exists():
            raise ImageGenerationError(
                f"HF_MODEL_PATH does not exist: {path}", code="MISSING_MODEL_PATH"
            )
        return path

    def _device(self, torch) -> str:
        device = self._settings.hf_device
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _load_pipeline(self):
        global _HF_PIPELINE, _HF_PIPELINE_KEY
        s = self._settings
        path = self._model_path()
        key = (str(path), s.hf_device, s.hf_dtype, s.hf_single_file_arch)
        if _HF_PIPELINE is not None and _HF_PIPELINE_KEY == key:
            return _HF_PIPELINE
        try:
            import torch
            from diffusers import (
                AutoPipelineForText2Image,
                StableDiffusionPipeline,
                StableDiffusionXLPipeline,
            )
        except ImportError as exc:
            raise ImageGenerationError(
                "IMAGE_PROVIDER=huggingface needs torch + diffusers: "
                "pip install -r requirements-hf.txt",
                code="MISSING_DEPENDENCY",
            ) from exc

        device = self._device(torch)
        dtype_name = s.hf_dtype
        if dtype_name == "auto":
            dtype_name = "float32" if device == "cpu" else "float16"
        dtype = getattr(torch, dtype_name)

        logger.info(
            "Loading Hugging Face model from %s on %s (%s)", path, device, dtype_name
        )
        if path.is_file():
            cls = (
                StableDiffusionXLPipeline
                if s.hf_single_file_arch == "sdxl"
                else StableDiffusionPipeline
            )
            pipe = cls.from_single_file(str(path), torch_dtype=dtype)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(
                str(path), torch_dtype=dtype, local_files_only=True
            )
        pipe = pipe.to(device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        _HF_PIPELINE, _HF_PIPELINE_KEY = pipe, key
        return pipe

    def _generate_sync(self, prompt: str) -> bytes:
        s = self._settings
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "width": s.hf_width,
            "height": s.hf_height,
            "num_inference_steps": s.hf_steps,
            "guidance_scale": s.hf_guidance_scale,
        }
        if s.hf_guidance_scale > 1.0:
            kwargs["negative_prompt"] = NEGATIVE_PROMPT
        # One load/generation at a time on the shared GPU pipeline.
        with _HF_LOCK:
            pipe = self._load_pipeline()
            image = pipe(**kwargs).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    async def generate(self, *, prompt: str, style: str) -> Path:
        try:
            image_bytes = await asyncio.to_thread(self._generate_sync, prompt)
        except ImageGenerationError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Hugging Face generation failed: %s", exc)
            raise ImageGenerationError(str(exc)) from exc
        return _save_png(image_bytes)

    async def is_available(self) -> bool:
        try:
            self._model_path()
        except ImageGenerationError:
            return False
        return True


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
            "negative_prompt": NEGATIVE_PROMPT,
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
    if settings.image_provider == "gemini":
        return GeminiImageGenerator(settings)
    if settings.image_provider == "huggingface":
        return HuggingFaceImageGenerator(settings)
    if settings.image_provider == "stable_diffusion":
        return StableDiffusionImageGenerator(settings)
    raise ImageGenerationError(
        f"Unknown IMAGE_PROVIDER '{settings.image_provider}'",
        code="UNKNOWN_PROVIDER",
    )