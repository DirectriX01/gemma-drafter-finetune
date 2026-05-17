"""Sample-based RL with fixed-shape padded rollouts (W4 attempt 2).

Hypothesis: W3's RL hit `tree_map IndexError` because variable-length rollouts make
the gradient tree's effective shape vary across iterations. Fix: pad all K rollouts
to `max_new_tokens` length, stack into a fixed `[K, max_new_tokens]` tensor, run
drafter forward once in batch mode. One value_and_grad call → one optimizer.update.

Usage:
    python -m train.rl_padded \\
        --train data/processed/KB13_train_100.jsonl \\
        --eval  data/processed/KB13_eval.jsonl \\
        --out   adapters/rl_kb13 \\
        --epochs 1 --K 4 --temperature 0.7 --max-new-tokens 24 \\
        --init-adapter adapters/sft_sparse_kb13/drafter_epoch3.npz
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_vlm import load
from mlx_vlm.generate import stream_generate
from mlx_vlm.speculative.drafters import load_drafter
from tqdm import tqdm

from eval.infer import ASSISTANT_MODEL_ID, TARGET_MODEL_ID
from eval.prompts import build_prompt, extract_regex
from train.sparse_loss import materialize_sparse
from verifier import verify


def _load_examples(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _temperature_sampler(temperature: float):
    def sampler(logits: mx.array) -> mx.array:
        return mx.random.categorical(logits * (1.0 / temperature), axis=-1)
    return sampler


def _sample_one_rollout(model, drafter, processor, prompt_text: str,
                        temperature: float, max_new_tokens: int) -> list[int]:
    sampler = _temperature_sampler(temperature)
    tokens: list[int] = []
    for chunk in stream_generate(
        model, processor, prompt_text,
        max_tokens=max_new_tokens,
        sampler=sampler,
        draft_model=drafter,
        draft_kind="mtp",
    ):
        if hasattr(chunk, "token"):
            tokens.append(int(chunk.token))
    return tokens


def _pad_rollouts(rollout_token_lists: list[list[int]], max_len: int,
                  pad_id: int) -> tuple[mx.array, mx.array]:
    """Pad to fixed max_len; return (rollouts[K, max_len], mask[K, max_len])."""
    K = len(rollout_token_lists)
    padded = [[pad_id] * max_len for _ in range(K)]
    mask = [[False] * max_len for _ in range(K)]
    for k, toks in enumerate(rollout_token_lists):
        L = min(len(toks), max_len)
        for i in range(L):
            padded[k][i] = toks[i]
            mask[k][i] = True
    return mx.array(padded, dtype=mx.int32), mx.array(mask)


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


def _rl_loss_padded(
    drafter, target, prompt_ids: mx.array, rollouts_padded: mx.array,
    rollout_mask: mx.array, advantages: mx.array,
) -> mx.array:
    """REINFORCE loss with fixed-shape padded rollouts.

    All K rollouts processed via a Python loop (unrolled at trace time). The
    per-rollout shape is `[1, max_len]` — constant across iterations.
    """
    K, max_len = rollouts_padded.shape
    P = prompt_ids.shape[1]
    total = mx.array(0.0)
    for k in range(K):
        r_ids = rollouts_padded[k:k + 1]                      # [1, max_len]
        m_k = rollout_mask[k:k + 1].astype(mx.float32)        # [1, max_len]
        full_ids = mx.concatenate([prompt_ids, r_ids], axis=-1)  # [1, P + max_len]
        target_hidden, shared_kv = _target_forward(target, full_ids)
        h = _drafter_forward_hidden(drafter, full_ids, target_hidden, shared_kv, P)
        embed_w = drafter.model.embed_tokens.weight
        selected_logits, selected_ids = materialize_sparse(drafter, h, embed_w)

        # Per-position logprob of the (possibly padded) token
        rollout_expanded = r_ids[:, :, None]                  # [1, max_len, 1]
        match = (selected_ids == rollout_expanded).astype(selected_logits.dtype)
        log_probs = nn.log_softmax(selected_logits, axis=-1)
        token_lp = (match * log_probs).sum(axis=-1)            # [1, max_len]
        # Mask out padded positions (their token_lp would otherwise include the pad token)
        seq_lp = (token_lp * m_k).sum()
        total = total - advantages[k] * seq_lp
    return total / K


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
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-adapter", type=Path, default=None)
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
        drafter.bind(target)
        drafter.masked_embedding.freeze()
        print(f"Warm-started from {args.init_adapter}")

    drafter.train()
    n_params = sum(int(p.size) for _, p in tree_flatten(drafter.trainable_parameters()))
    print(f"Drafter trainable params: {n_params:,}")
    print(f"K={args.K}, T={args.temperature}, max_new={args.max_new_tokens}")

    train_examples = _load_examples(args.train)
    print(f"train={len(train_examples)}")

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    def loss_fn(drafter, prompt_ids, rollouts_padded, rollout_mask, advantages):
        return _rl_loss_padded(drafter, target, prompt_ids, rollouts_padded,
                                rollout_mask, advantages)

    grad_fn = nn.value_and_grad(drafter, loss_fn)

    log_path = args.out / "train_log.jsonl"
    log_path.unlink(missing_ok=True)
    pad_id = processor.tokenizer.pad_token_id or 0

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

            # Sample K rollouts
            rollouts_lists: list[list[int]] = []
            rewards: list[float] = []
            for _ in range(args.K):
                toks = _sample_one_rollout(
                    target, drafter, processor, prompt_text,
                    args.temperature, args.max_new_tokens,
                )
                if not toks:
                    toks = [pad_id]  # avoid empty rollout
                text = tok.decode(toks, skip_special_tokens=True)
                regex = extract_regex(text)
                vr = verify(regex, ex["pos_examples"], ex["neg_examples"])
                rollouts_lists.append(toks)
                rewards.append(vr.f1)

            mean_r = sum(rewards) / max(1, len(rewards))
            advantages_py = [r - mean_r for r in rewards]
            step += 1

            if all(a == 0.0 for a in advantages_py):
                epoch_rewards.append(mean_r)
                with log_path.open("a") as f:
                    f.write(json.dumps({"event": "step", "step": step, "epoch": epoch,
                                         "mean_reward": mean_r, "skipped": True}) + "\n")
                continue

            rollouts_padded, rollout_mask = _pad_rollouts(
                rollouts_lists, args.max_new_tokens, pad_id
            )
            advantages = mx.array(advantages_py)

            # Reset drafter's per-inference state (accept_lens list, _shared_kv tensor).
            # If left as-is, these attributes can confuse MLX's gradient-tree walker
            # — accept_lens grows across samples, _shared_kv changes shape.
            drafter.reset(target)

            loss, grads = grad_fn(drafter, prompt_ids, rollouts_padded, rollout_mask, advantages)
            optimizer.update(drafter, grads)
            mx.eval(drafter.parameters(), optimizer.state)
            mx.clear_cache()

            epoch_rewards.append(mean_r)
            pbar.set_postfix(reward=f"{mean_r:.3f}", loss=f"{float(loss.item()):.3f}")
            with log_path.open("a") as f:
                f.write(json.dumps({"event": "step", "step": step, "epoch": epoch,
                                     "mean_reward": mean_r,
                                     "loss": float(loss.item())}) + "\n")

        mean_epoch_r = sum(epoch_rewards) / max(1, len(epoch_rewards))
        print(f"Epoch {epoch + 1} mean reward: {mean_epoch_r:.4f}  ({len(epoch_rewards)} prompts)")
        ckpt = args.out / f"drafter_epoch{epoch + 1}.npz"
        flat = dict(tree_flatten(drafter.trainable_parameters()))
        mx.savez(str(ckpt), **flat)
        print(f"  saved {ckpt}")

    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
