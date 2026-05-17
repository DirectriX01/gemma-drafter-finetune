"""Sample-based RL for the Gemma 4 MTP drafter (Path 4 — the project's headline).

Per prompt:
  1) Sample K=4 regex completions via target+drafter spec decoding with temperature.
  2) Score each completion with the verifier (cell-F1 shaped reward).
  3) Recompute drafter's per-token logprobs on the sampled sequences (sparse-aware).
  4) REINFORCE loss with mean-baseline subtraction:
         L = -mean_k[(r_k - mean(r)) * sum_t log p_drafter(token_k_t)]

The drafter's masked_embedding is frozen (training preserves cluster routing).
Loss is only counted on tokens that are in the sparse head's materialized set —
others contribute zero gradient (matches the constraint we discovered in W3).

Usage:
    python -m train.rl --train data/processed/KB13_train.jsonl \\
                       --eval  data/processed/KB13_eval.jsonl  \\
                       --epochs 3 --K 4 --temperature 0.7 \\
                       --out adapters/rl_kb13
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_vlm import load
from mlx_vlm.speculative.drafters import load_drafter
from tqdm import tqdm

from eval.infer import ASSISTANT_MODEL_ID, TARGET_MODEL_ID
from eval.prompts import build_prompt, extract_regex
from train.sparse_loss import materialize_sparse
from verifier import verify


def _load_examples(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _temperature_sampler(temperature: float):
    """Return a sampler closure for mlx_vlm.generate."""
    def sampler(logits: mx.array) -> mx.array:
        return mx.random.categorical(logits * (1.0 / temperature), axis=-1)
    return sampler


def _sample_one_rollout(model, drafter, processor, prompt_text: str, temperature: float,
                        max_new_tokens: int) -> tuple[str, list[int]]:
    """Sample one completion. Returns (text, token_ids)."""
    from mlx_vlm.generate import stream_generate

    sampler = _temperature_sampler(temperature)
    out_tokens: list[int] = []
    out_text_parts: list[str] = []
    for chunk in stream_generate(
        model, processor, prompt_text,
        max_tokens=max_new_tokens,
        sampler=sampler,
        draft_model=drafter,
        draft_kind="mtp",
    ):
        # stream_generate yields GenerationResponse objects with .text, .token, etc.
        if hasattr(chunk, "token"):
            tok = int(chunk.token)
            out_tokens.append(tok)
        if hasattr(chunk, "text"):
            out_text_parts.append(chunk.text)
    return "".join(out_text_parts), out_tokens


def _target_forward(target, full_ids: mx.array):
    lm = target.language_model if hasattr(target, "language_model") else target
    out = lm(inputs=full_ids, return_hidden=True, return_shared_kv=True)
    hidden = mx.stop_gradient(out.hidden_states[-1])
    shared_kv = {k: (mx.stop_gradient(kv[0]), mx.stop_gradient(kv[1]))
                 for k, kv in out.shared_kv_states.items()}
    return hidden, shared_kv


def _drafter_forward_hidden(drafter, full_ids, target_hidden, shared_kv, prompt_len):
    T = full_ids.shape[1]
    P = prompt_len
    prev_token_ids = full_ids[:, P - 1 : T - 1]
    prev_hidden = target_hidden[:, P - 1 : T - 1, :]
    prev_embeds = drafter._input_embed(prev_token_ids) * drafter._input_embed_scale
    inputs_embeds = mx.concatenate([prev_embeds, prev_hidden], axis=-1)
    position_ids = mx.arange(P, T)[None, :]
    prev_lm_head = drafter._lm_head_fn
    drafter._lm_head_fn = lambda h: h
    try:
        _, h = drafter(inputs_embeds, shared_kv, position_ids)
    finally:
        drafter._lm_head_fn = prev_lm_head
    return h


def _drafter_seq_logprob(drafter, prompt_ids: mx.array, rollout_ids: mx.array,
                         target_hidden: mx.array, shared_kv: dict) -> mx.array:
    """Sum log-probs the drafter assigns to the rollout tokens (sparse-aware).

    Tokens not in the materialized top-K set contribute zero logprob (they're
    structurally unreachable, so they can't be steered by the drafter anyway).
    """
    full_ids = mx.concatenate([prompt_ids, rollout_ids], axis=-1)
    P = prompt_ids.shape[1]
    h = _drafter_forward_hidden(drafter, full_ids, target_hidden, shared_kv, P)
    embed_w = drafter.model.embed_tokens.weight
    selected_logits, selected_ids = materialize_sparse(drafter, h, embed_w)

    rollout_expanded = rollout_ids[:, :, None]                              # [1, R, 1]
    match = (selected_ids == rollout_expanded)                              # [1, R, M] bool
    log_probs = nn.log_softmax(selected_logits, axis=-1)                    # [1, R, M]
    token_lp = (match.astype(log_probs.dtype) * log_probs).sum(axis=-1)     # [1, R]
    # has_token=False positions contribute 0 (token wasn't materialized)
    return token_lp.sum()


def _shuffled(items, seed):
    rng = random.Random(seed)
    perm = list(range(len(items)))
    rng.shuffle(perm)
    for i in perm:
        yield items[i]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--K", type=int, default=4, help="rollouts per prompt")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--lr", type=float, default=5e-5)  # lower than SFT — RL is noisy
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-adapter", type=Path, default=None,
                   help="warm-start from a sparse-SFT or sparse-RS adapter (recommended)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))

    print(f"Loading target ({TARGET_MODEL_ID}) ...")
    target, processor = load(TARGET_MODEL_ID)
    target.eval()

    print(f"Loading drafter ({ASSISTANT_MODEL_ID}) ...")
    drafter, _ = load_drafter(ASSISTANT_MODEL_ID, kind="mtp")
    drafter.bind(target)
    drafter.masked_embedding.freeze()

    if args.init_adapter is not None:
        from mlx.utils import tree_unflatten
        flat = mx.load(str(args.init_adapter))
        drafter.update(tree_unflatten(list(flat.items())))
        drafter.bind(target)  # restore sparse head
        drafter.masked_embedding.freeze()
        print(f"Warm-started from {args.init_adapter}")

    drafter.train()
    n_params = sum(int(p.size) for _, p in tree_flatten(drafter.trainable_parameters()))
    print(f"Drafter trainable params: {n_params:,}")
    print(f"K={args.K} rollouts per prompt, T={args.temperature}, max_new={args.max_new_tokens}")

    train_examples = _load_examples(args.train)
    eval_examples = _load_examples(args.eval)
    print(f"train={len(train_examples)} eval={len(eval_examples)}")

    # SGD: no internal state means optimizer.update is structurally simple. AdamW
    # tripped MLX's tree_map after several updates (gradient tree structure shifts
    # subtly between rollouts).
    optimizer = optim.SGD(learning_rate=args.lr)

    def single_rollout_loss(drafter, prompt_ids, rollout_ids, advantage):
        """Per-rollout REINFORCE term. Single scalar loss → single grad call per rollout."""
        full_ids = mx.concatenate([prompt_ids, rollout_ids], axis=-1)
        target_hidden, shared_kv = _target_forward(target, full_ids)
        lp = _drafter_seq_logprob(drafter, prompt_ids, rollout_ids,
                                   target_hidden, shared_kv)
        return -advantage * lp

    grad_fn = nn.value_and_grad(drafter, single_rollout_loss)
    log_path = args.out / "train_log.jsonl"
    log_path.unlink(missing_ok=True)

    print(f"\nStarting RL: {args.epochs} epochs, lr={args.lr}\n")
    step = 0
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_rewards = []
        pbar = tqdm(list(_shuffled(train_examples, args.seed + epoch)),
                    desc=f"epoch {epoch + 1}/{args.epochs}", unit="ex")
        for ex in pbar:
            tok = processor.tokenizer
            prompt_text = tok.apply_chat_template(
                build_prompt(ex["nl"], few_shot=True),
                tokenize=False, add_generation_prompt=True,
            )
            prompt_ids = mx.array(tok.encode(prompt_text, add_special_tokens=False))[None, :]

            # --- Sample K rollouts (no grad) ---
            rollouts_text, rollouts_ids, rewards = [], [], []
            for _ in range(args.K):
                txt, tok_ids = _sample_one_rollout(
                    target, drafter, processor, prompt_text,
                    args.temperature, args.max_new_tokens,
                )
                regex = extract_regex(txt)
                vr = verify(regex, ex["pos_examples"], ex["neg_examples"])
                r = vr.f1  # shaped reward
                if not tok_ids:
                    continue
                rollouts_text.append(regex)
                rollouts_ids.append(mx.array(tok_ids)[None, :])
                rewards.append(r)

            if not rewards:
                continue

            mean_r = sum(rewards) / len(rewards)
            advantages = [r - mean_r for r in rewards]

            # If all rewards identical, no signal — skip backprop
            if all(a == 0.0 for a in advantages):
                epoch_rewards.append(mean_r)
                step += 1
                with log_path.open("a") as f:
                    f.write(json.dumps({"event": "step", "step": step, "epoch": epoch,
                                         "mean_reward": mean_r, "skipped": True}) + "\n")
                continue

            # --- K micro-steps: one optimizer.update per rollout (advantage-scaled) ---
            # Avoids manual grad-tree accumulation which broke MLX's tree_map.
            total_loss = 0.0
            n_updates = 0
            for r_ids, adv in zip(rollouts_ids, advantages):
                if adv == 0.0:
                    continue
                # Scale advantage by 1/K so the effective LR matches a single REINFORCE step.
                scaled_adv = mx.array(adv / args.K)
                loss_k, grads_k = grad_fn(drafter, prompt_ids, r_ids, scaled_adv)
                optimizer.update(drafter, grads_k)
                mx.eval(drafter.parameters(), optimizer.state)
                total_loss += float(loss_k.item())
                n_updates += 1
            mx.clear_cache()
            loss = mx.array(total_loss / max(1, n_updates))

            epoch_rewards.append(mean_r)
            step += 1
            pbar.set_postfix(reward=f"{mean_r:.3f}", loss=f"{float(loss.item()):.3f}")
            with log_path.open("a") as f:
                f.write(json.dumps({"event": "step", "step": step, "epoch": epoch,
                                     "mean_reward": mean_r,
                                     "loss": float(loss.item())}) + "\n")

        mean_epoch_r = sum(epoch_rewards) / max(1, len(epoch_rewards))
        print(f"Epoch {epoch + 1} mean reward: {mean_epoch_r:.4f}  (over {len(epoch_rewards)} prompts)")

        ckpt = args.out / f"drafter_epoch{epoch + 1}.npz"
        flat = dict(tree_flatten(drafter.trainable_parameters()))
        mx.savez(str(ckpt), **flat)
        print(f"  saved {ckpt}")

    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
