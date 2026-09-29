"""POC: text-to-image with Z-Image-Turbo (GGUF Q4_0) via diffusers, CPU only.

Standalone on purpose - no dependency on the app package - so it can be
run outside the API. The app uses app.ai.image_generator.ZImageGenerator.

Only the transformer is GGUF-quantised; the Qwen3 text encoder, VAE,
tokenizer and scheduler come from the base repo ``Tongyi-MAI/Z-Image-Turbo``
(downloaded once into the HF cache).

Usage (from backend/):
    python poc/zimage_t2i.py "a comic panel of a cat detective in the rain"
    python poc/zimage_t2i.py "..." --width 768 --height 1024 --seed 42 --out out.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from diffusers import GGUFQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel
from huggingface_hub import hf_hub_download

BASE_REPO = "Tongyi-MAI/Z-Image-Turbo"
GGUF_REPO = "unsloth/Z-Image-Turbo-GGUF"
GGUF_FILE = "z-image-turbo-Q4_0.gguf"
DEFAULT_GGUF = Path(__file__).resolve().parents[1] / "model" / GGUF_FILE


def resolve_gguf(path: Path) -> Path:
    """Use the local file if present, otherwise download it next to it."""
    if path.exists():
        return path
    print(f"{path} not found - downloading {GGUF_FILE} from {GGUF_REPO}...")
    return Path(hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir=path.parent))


def load_pipeline(gguf_path: Path) -> ZImagePipeline:
    dtype = torch.bfloat16
    transformer = ZImageTransformer2DModel.from_single_file(
        str(gguf_path),
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        config=BASE_REPO,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    pipe = ZImagePipeline.from_pretrained(
        BASE_REPO, transformer=transformer, torch_dtype=dtype
    )
    # CPU-only: bf16 keeps RAM at ~12 GB (fp32 would roughly double it).
    pipe.to("cpu")
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompt")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    # 512px default: CPU cost scales with pixel count (1024px is ~4x slower).
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    # Turbo is distilled: ~8 DiT forwards (9 steps) and no CFG.
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("zimage_out.png"))
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"device=cpu threads={torch.get_num_threads()}")

    t0 = time.perf_counter()
    pipe = load_pipeline(resolve_gguf(args.gguf))
    t1 = time.perf_counter()

    generator = None
    if args.seed is not None:
        generator = torch.Generator("cpu").manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    t2 = time.perf_counter()

    image.save(args.out)
    print(f"load {t1 - t0:.1f}s | generate {t2 - t1:.1f}s | saved {args.out.resolve()}")


if __name__ == "__main__":
    main()
