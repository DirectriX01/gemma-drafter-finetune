"""SFT trainer for the Gemma 4 MTP drafter on regex synthesis.

Baseline 2 of the project: drafter trained on (NL → gold regex) pairs, using the target
model's hidden states + shared K/V at every position (matching the inference-time data
flow). Target is frozen; only drafter weights update.

Note: the centroid-routed sparse LM head (`use_ordered_embeddings=True`) is not
differentiable through MLX's scatter_axis op. We swap `_lm_head_fn` to the dense
tied-embedding head for training; the sparse head is reinstated at inference by
`bind(target)`.

Usage:
    python -m train.sft --train data/processed/KB13_train.jsonl \\
                        --eval  data/processed/KB13_eval.jsonl  \\
                        --epochs 3 --lr 1e-4 --out adapters/sft_kb13
"""

from __future__ import annotations

import argparse
import json
import random
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
from eval.prompts import build_prompt


@dataclass
class Batch:
    full_ids: mx.array       # [1, T] target input tokens (prompt + gold regex)
    prompt_len: int          # P (= len of prompt before regex)
    gold_ids: mx.array       # [1, R] tokens the drafter must predict, R = T - P


def _load_examples(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _make_batch(ex: dict, processor) -> Batch:
    tok = processor.tokenizer
    prompt_text = tok.apply_chat_template(
        build_prompt(ex["nl"], few_shot=True),
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    gold_ids = tok.encode(ex["gold_regex"], add_special_tokens=False)
    full_ids = prompt_ids + gold_ids
    return Batch(
        full_ids=mx.array(full_ids)[None, :],
        prompt_len=len(prompt_ids),
        gold_ids=mx.array(gold_ids)[None, :],
    )


def _drafter_loss(drafter, batch: Batch, target_hidden: mx.array, shared_kv: dict) -> mx.array:
    """Teacher-forced cross-entropy loss for the drafter on the regex tokens."""
    full_ids = batch.full_ids
    P = batch.prompt_len
    T = full_ids.shape[1]

    # Drafter predicts token t given (token[t-1], target_hidden[t-1]).
    prev_token_ids = full_ids[:, P - 1 : T - 1]                # [1, R]
    prev_hidden = target_hidden[:, P - 1 : T - 1, :]            # [1, R, H]

    embed_scale = drafter._input_embed_scale
    prev_embeds = drafter._input_embed(prev_token_ids) * embed_scale
    inputs_embeds = mx.concatenate([prev_embeds, prev_hidden], axis=-1)
    position_ids = mx.arange(P, T)[None, :]

    _, logits = drafter(inputs_embeds, shared_kv, position_ids)

    gold_flat = batch.gold_ids.reshape(-1)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    return nn.losses.cross_entropy(logits_flat, gold_flat, reduction="mean")


def _target_forward(target, full_ids: mx.array) -> tuple[mx.array, dict]:
    """Frozen target forward — extracts last-layer hidden + shared K/V for the drafter.

    Outputs are detached via stop_gradient so backprop does not try to flow back into
    the (frozen) target weights.
    """
    lm = target.language_model if hasattr(target, "language_model") else target
    out = lm(inputs=full_ids, return_hidden=True, return_shared_kv=True)
    hidden = mx.stop_gradient(out.hidden_states[-1])
    shared_kv = {k: (mx.stop_gradient(kv[0]), mx.stop_gradient(kv[1]))
                 for k, kv in out.shared_kv_states.items()}
    return hidden, shared_kv


def _shuffled(items: list, seed: int) -> Iterator:
    rng = random.Random(seed)
    perm = list(range(len(items)))
    rng.shuffle(perm)
    for i in perm:
        yield items[i]


def _count_params(drafter) -> int:
    return sum(int(p.size) for _, p in tree_flatten(drafter.trainable_parameters()))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=200, help="steps")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading target ({TARGET_MODEL_ID}) ...")
    target, processor = load(TARGET_MODEL_ID)
    target.eval()
    # We don't call target.freeze() — some Gemma 4 multimodal submodules don't
    # implement the freeze hooks cleanly. Instead, target outputs are wrapped in
    # mx.stop_gradient inside _target_forward, and nn.value_and_grad(drafter, ...)
    # only computes gradients w.r.t. drafter parameters.

    print(f"Loading drafter ({ASSISTANT_MODEL_ID}) ...")
    drafter, kind = load_drafter(ASSISTANT_MODEL_ID, kind="mtp")
    drafter.bind(target)
    # Swap to dense LM head for differentiable training.
    drafter._lm_head_fn = drafter.model.embed_tokens.as_linear
    drafter.train()
    print(f"Drafter trainable params: {_count_params(drafter):,}")

    train_examples = _load_examples(args.train)
    eval_examples = _load_examples(args.eval)
    print(f"train={len(train_examples)} eval={len(eval_examples)}")

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    def loss_fn(drafter, batch, target_hidden, shared_kv):
        return _drafter_loss(drafter, batch, target_hidden, shared_kv)

    grad_fn = nn.value_and_grad(drafter, loss_fn)

    log_path = args.out / "train_log.jsonl"
    log_path.unlink(missing_ok=True)

    def eval_pass(step: int):
        drafter.eval()
        losses = []
        for ex in eval_examples[:50]:  # cap eval set for speed
            batch = _make_batch(ex, processor)
            target_hidden, shared_kv = _target_forward(target, batch.full_ids)
            loss = _drafter_loss(drafter, batch, target_hidden, shared_kv)
            losses.append(float(loss.item()))
            mx.clear_cache()
        drafter.train()
        m = sum(losses) / max(1, len(losses))
        print(f"  [eval @ step {step}] mean loss = {m:.4f} over {len(losses)} examples")
        with log_path.open("a") as f:
            f.write(json.dumps({"event": "eval", "step": step, "loss": m}) + "\n")
        return m

    print(f"\nStarting training: {args.epochs} epochs, lr={args.lr}\n")
    step = 0
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_losses = []
        pbar = tqdm(list(_shuffled(train_examples, args.seed + epoch)),
                    desc=f"epoch {epoch + 1}/{args.epochs}",
                    unit="ex")
        for ex in pbar:
            batch = _make_batch(ex, processor)
            target_hidden, shared_kv = _target_forward(target, batch.full_ids)
            loss, grads = grad_fn(drafter, batch, target_hidden, shared_kv)
            optimizer.update(drafter, grads)
            mx.eval(drafter.parameters(), optimizer.state)
            mx.clear_cache()

            loss_v = float(loss.item())
            epoch_losses.append(loss_v)
            pbar.set_postfix(loss=f"{loss_v:.3f}")

            step += 1
            if step % args.eval_every == 0:
                eval_pass(step)

            with log_path.open("a") as f:
                f.write(json.dumps({"event": "step", "step": step,
                                     "epoch": epoch, "loss": loss_v}) + "\n")

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} mean train loss: {mean_loss:.4f}")
        eval_pass(step)

        # Save checkpoint after each epoch
        ckpt = args.out / f"drafter_epoch{epoch + 1}.npz"
        flat = dict(tree_flatten(drafter.trainable_parameters()))
        mx.savez(str(ckpt), **flat)
        print(f"  saved {ckpt}")

    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
