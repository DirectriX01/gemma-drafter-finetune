"""Exploration: trace one example through target + drafter to verify the training pipeline.

This is a smoke test for the training step:
1. Tokenize prompt+regex into one sequence
2. Forward target with return_hidden=True, return_shared_kv=True
3. Extract per-position hidden states + shared KV
4. Build drafter inputs (token embeds concatenated with prev hidden states)
5. Forward drafter for the regex positions
6. Compute CE loss vs. gold tokens
7. Print everything we need to verify before writing the real trainer

Usage:
    python -m train.probe
"""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn

from eval.infer import ASSISTANT_MODEL_ID, TARGET_MODEL_ID
from eval.prompts import build_prompt
from mlx_vlm import load
from mlx_vlm.speculative.drafters import load_drafter


def main() -> None:
    print("Loading target...")
    target, processor = load(TARGET_MODEL_ID)
    target.eval()
    tokenizer = processor.tokenizer

    print("Loading drafter...")
    drafter, kind = load_drafter(ASSISTANT_MODEL_ID, kind="mtp")
    print(f"Drafter kind={kind}, type={type(drafter).__name__}")
    drafter.bind(target)

    # The centroid-routed sparse LM head uses scatter_axis which has no VJP. For training,
    # swap to the dense tied-embedding head (differentiable). At inference, restore the
    # sparse head via `drafter.bind(target)` which resets `_lm_head_fn`.
    drafter._lm_head_fn = drafter.model.embed_tokens.as_linear

    # Pull one example
    with open("data/processed/KB13_train.jsonl") as f:
        ex = json.loads(f.readline())
    print(f"\nExample: nl={ex['nl']!r}  gold={ex['gold_regex']!r}")

    # Build prompt + gold completion
    msgs = build_prompt(ex["nl"], few_shot=True)
    prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    gold_text = ex["gold_regex"]

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    # Gold regex tokens — what we want the drafter to predict.
    gold_ids = tokenizer.encode(gold_text, add_special_tokens=False)
    print(f"prompt tokens: {len(prompt_ids)}  gold tokens: {len(gold_ids)}")

    # Full sequence = prompt + gold (target gets to see everything to populate shared KV)
    full_ids = prompt_ids + gold_ids
    full = mx.array(full_ids)[None, :]  # [1, T]
    T = full.shape[1]
    P = len(prompt_ids)

    # 1) Target forward — capture last-layer hidden + shared KV
    print("\nTarget forward (return_hidden=True, return_shared_kv=True) ...")
    # The Gemma 4 model wrapper expects "input_ids" (and uses internal embed).
    # We bypass higher-level helpers and call the language model directly.
    lm = target.language_model if hasattr(target, "language_model") else target
    out = lm(inputs=full, return_hidden=True, return_shared_kv=True)
    print(f"  logits: {out.logits.shape}")
    print(f"  hidden_states: list of {len(out.hidden_states)}  last shape: {out.hidden_states[-1].shape}")
    print(f"  shared_kv_states: dict keys = {list(out.shared_kv_states.keys())}")
    for k, v in out.shared_kv_states.items():
        kv_k, kv_v = v
        print(f"    [{k}] k.shape={kv_k.shape}  v.shape={kv_v.shape}")
        break

    # 2) Build drafter inputs at the regex positions.
    # The drafter predicts token t given (token[t-1], target_hidden[t-1]).
    # So for predicting gold position [P, P+1, ..., T-1]:
    #   input tokens = full_ids[P-1 : T-1]
    #   input hidden = target_hidden[P-1 : T-1]
    target_hidden = out.hidden_states[-1]   # [1, T, H_hidden]
    H = target_hidden.shape[-1]
    backbone_h = drafter.config.backbone_hidden_size
    assert H == backbone_h, f"Hidden mismatch: target {H} vs drafter backbone {backbone_h}"

    prev_token_ids = mx.array(full_ids[P - 1 : T - 1])[None, :]   # [1, R]
    prev_hidden = target_hidden[:, P - 1 : T - 1, :]               # [1, R, H]
    R = prev_token_ids.shape[1]

    embed = drafter._input_embed
    embed_scale = drafter._input_embed_scale
    prev_embeds = embed(prev_token_ids) * embed_scale              # [1, R, H_embed]
    inputs_embeds = mx.concatenate([prev_embeds, prev_hidden], axis=-1)
    print(f"\nDrafter inputs: tokens [P-1:T-1]={R}  embeds.shape={inputs_embeds.shape}")

    # position_ids = [P, P+1, ..., T-1]
    position_ids = mx.arange(P, T)[None, :]                        # [1, R]
    print(f"  position_ids: {position_ids.shape}  range=[{P},{T - 1}]")

    # 3) Drafter forward — produces logits at each regex position.
    print("\nDrafter forward ...")
    # The drafter's __call__ uses make_drafter_masks with query_offset = position_ids[0, 0].
    # For multi-position teacher-forced training, we need causal masking. The current
    # make_drafter_masks may not produce a causal triangle — verify and patch if needed.
    last_hidden, logits = drafter(inputs_embeds, out.shared_kv_states, position_ids)
    print(f"  logits.shape={logits.shape}  last_hidden.shape={last_hidden.shape}")

    # 4) Compute CE on regex tokens
    gold = mx.array(gold_ids)[None, :]   # [1, R]
    # cross_entropy expects logits [N, C] and targets [N]
    logits_flat = logits.reshape(-1, logits.shape[-1])
    gold_flat = gold.reshape(-1)
    loss = nn.losses.cross_entropy(logits_flat, gold_flat, reduction="mean")
    print(f"\nLoss (CE on regex tokens, no training yet): {loss.item():.4f}")

    # 5) Verify gradients flow into drafter params
    print("\nVerifying gradient flow ...")

    def loss_fn(drafter):
        last_hidden, logits = drafter(inputs_embeds, out.shared_kv_states, position_ids)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        return nn.losses.cross_entropy(logits_flat, gold_flat, reduction="mean")

    grad_fn = nn.value_and_grad(drafter, loss_fn)
    loss2, grads = grad_fn(drafter)
    # Count nonzero gradient leaves
    leaf_count = 0
    nonzero_count = 0

    def walk(g):
        nonlocal leaf_count, nonzero_count
        if isinstance(g, dict):
            for v in g.values():
                walk(v)
        elif isinstance(g, list):
            for v in g:
                walk(v)
        elif isinstance(g, mx.array):
            leaf_count += 1
            if mx.any(g != 0).item():
                nonzero_count += 1

    walk(grads)
    print(f"  loss (recomputed): {loss2.item():.4f}")
    print(f"  gradient leaves: {leaf_count}, nonzero: {nonzero_count}")


if __name__ == "__main__":
    main()
