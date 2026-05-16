"""Inference wrapper for Gemma 4 E4B (MLX-VLM, Apple Silicon native).

Gemma 4 is multimodal (text + vision + audio) — we only use text. mlx-vlm loads the
multimodal stack but text-only `generate()` works (no image/audio passed).

Two modes:
- baseline 0: target alone, autoregressive
- baseline 1: target + MTP assistant drafter (speculative decoding via mlx_vlm.speculative)

4-bit target ~5GB, drafter bf16 ~160MB. Much friendlier than transformers + MPS on M1 Pro.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from mlx_vlm import generate, load
from mlx_vlm.speculative.drafters import load_drafter

from eval.prompts import build_prompt, extract_regex

# MLX-VLM-converted Gemma 4 E4B-it (4-bit quantized) — ~5GB on disk
TARGET_MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
# Official MTP drafter, MLX bf16 port — ~160MB
ASSISTANT_MODEL_ID = "mlx-community/gemma-4-E4B-it-assistant-bf16"


@dataclass
class GenerationResult:
    raw_text: str
    regex: str
    elapsed_s: float
    n_new_tokens: int
    tokens_per_s: float


class RegexInferencer:
    """Loads target (and optional drafter) once; reuses across many prompts."""

    def __init__(
        self,
        target_id: str = TARGET_MODEL_ID,
        assistant_id: Optional[str] = None,
    ) -> None:
        self.target_id = target_id
        self.assistant_id = assistant_id

        self.model, self.processor = load(target_id)

        self.drafter = None
        if assistant_id is not None:
            # `load_drafter` returns (drafter, kind) — kind is e.g. "mtp"
            self.drafter, _ = load_drafter(assistant_id, kind="mtp")

    @property
    def device(self) -> str:
        return "mlx-metal"

    @property
    def dtype(self) -> str:
        return "int4" if "4bit" in self.target_id else "bf16"

    def generate(
        self,
        nl: str,
        *,
        max_new_tokens: int = 64,
        few_shot: bool = True,
    ) -> GenerationResult:
        msgs = build_prompt(nl, few_shot=few_shot)
        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

        kwargs = dict(max_tokens=max_new_tokens, verbose=False)
        if self.drafter is not None:
            kwargs["drafter"] = self.drafter

        t0 = time.perf_counter()
        result = generate(self.model, self.processor, prompt=prompt, **kwargs)
        elapsed = time.perf_counter() - t0

        # mlx_vlm.generate returns a GenerationResult with .text and .generation_tokens
        if hasattr(result, "text"):
            text = result.text
            n_new = getattr(result, "generation_tokens", None)
            if n_new is None:
                n_new = len(self.processor.tokenizer.encode(text))
        else:
            text = str(result)
            n_new = len(self.processor.tokenizer.encode(text))

        n_new = max(int(n_new), 1)

        return GenerationResult(
            raw_text=text,
            regex=extract_regex(text),
            elapsed_s=elapsed,
            n_new_tokens=n_new,
            tokens_per_s=n_new / elapsed if elapsed > 0 else 0.0,
        )
