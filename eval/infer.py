"""Inference wrapper for Gemma 4 E4B with optional MTP drafter (speculative decoding).

Two modes:
- baseline 0: target model alone, autoregressive generation
- baseline 1: target + assistant drafter (speculative decoding via HF Transformers)

Designed for M1 Pro (Apple Silicon). Uses MPS where available, falls back to CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.prompts import build_prompt, extract_regex

TARGET_MODEL_ID = "google/gemma-4-E4B-it"
ASSISTANT_MODEL_ID = "google/gemma-4-E4B-it-assistant"


@dataclass
class GenerationResult:
    raw_text: str
    regex: str
    elapsed_s: float
    n_new_tokens: int
    tokens_per_s: float


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class RegexInferencer:
    """Loads target (and optional drafter) once; reuses across many prompts."""

    def __init__(
        self,
        target_id: str = TARGET_MODEL_ID,
        assistant_id: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or _pick_device()
        # bfloat16 on MPS is supported but float16 is more memory-efficient
        self.dtype = dtype or (torch.float16 if self.device == "mps" else torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(target_id)
        self.target = AutoModelForCausalLM.from_pretrained(
            target_id, dtype=self.dtype, device_map=self.device
        )
        self.target.eval()

        self.assistant = None
        if assistant_id is not None:
            self.assistant = AutoModelForCausalLM.from_pretrained(
                assistant_id, dtype=self.dtype, device_map=self.device
            )
            self.assistant.eval()

    @torch.inference_mode()
    def generate(
        self,
        nl: str,
        *,
        max_new_tokens: int = 64,
        few_shot: bool = True,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> GenerationResult:
        msgs = build_prompt(nl, few_shot=few_shot)
        prompt_ids = self.tokenizer.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
        if self.assistant is not None:
            gen_kwargs["assistant_model"] = self.assistant

        t0 = time.perf_counter()
        out = self.target.generate(prompt_ids, **gen_kwargs)
        elapsed = time.perf_counter() - t0

        new_tokens = out[0, prompt_ids.shape[1]:]
        n_new = int(new_tokens.shape[0])
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return GenerationResult(
            raw_text=raw,
            regex=extract_regex(raw),
            elapsed_s=elapsed,
            n_new_tokens=n_new,
            tokens_per_s=n_new / elapsed if elapsed > 0 else 0.0,
        )
