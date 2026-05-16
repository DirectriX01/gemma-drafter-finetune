"""Acceptance-only trainer — baseline 3 (the critical control vs. baseline 4).

Trains the Gemma 4 MTP drafter to match the target's output distribution via KL
divergence at every regex position. This is the existing-literature objective: optimize
the drafter to maximize target acceptance, with no task-level verifier signal.

Mirrors the training pipeline of `train/sft.py`. Differences:
- Loss is `KL(drafter_softmax || target_softmax)` instead of CE against gold tokens.
- The target's logits at each regex position provide the supervisory signal.

The drafter never sees the gold regex; it only learns to mimic the target. That makes
this a clean isolating control: baseline 4 = this + verifier signal.

Usage:
    python -m train.accept_only \\
        --train data/processed/KB13_train.jsonl \\
        --eval  data/processed/KB13_eval.jsonl  \\
        --epochs 3 --lr 1e-4 --out adapters/accept_only_kb13
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
    full_ids: mx.array
    prompt_len: int
    gold_ids: mx.array


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


def _target_forward(target, full_ids: mx.array):
    """Runs target; returns (last hidden, shared_kv, logits) all stop_gradient-detached.

    The logits at the regex positions become the soft labels for KL.
    """
    lm = target.language_model if hasattr(target, "language_model") else target
    out = lm(inputs=full_ids, return_hidden=True, return_shared_kv=True)
    hidden = mx.stop_gradient(out.hidden_states[-1])
    shared_kv = {k: (mx.stop_gradient(kv[0]), mx.stop_gradient(kv[1]))
                 for k, kv in out.shared_kv_states.items()}
    target_logits = mx.stop_gradient(out.logits)
    return hidden, shared_kv, target_logits


def _kl_drafter_to_target(drafter_logits: mx.array, target_logits: mx.array) -> mx.array:
    """Forward KL: D_KL(softmax(drafter) || softmax(target)).

    Per-position then mean-reduce. Computes:
        sum_i p_drafter(i) * (log p_drafter(i) - log p_target(i))
    """
    log_p_d = nn.log_softmax(drafter_logits, axis=-1)
    log_p_t = nn.log_softmax(target_logits, axis=-1)
    p_d = mx.exp(log_p_d)
    kl_per_position = (p_d * (log_p_d - log_p_t)).sum(axis=-1)
    return kl_per_position.mean()


def _drafter_loss(drafter, batch: Batch, target_hidden, shared_kv, target_logits) -> mx.array:
    full_ids = batch.full_ids
    P = batch.prompt_len
    T = full_ids.shape[1]

    prev_token_ids = full_ids[:, P - 1 : T - 1]
    prev_hidden = target_hidden[:, P - 1 : T - 1, :]
    embed_scale = drafter._input_embed_scale
    prev_embeds = drafter._input_embed(prev_token_ids) * embed_scale
    inputs_embeds = mx.concatenate([prev_embeds, prev_hidden], axis=-1)
    position_ids = mx.arange(P, T)[None, :]

    _, drafter_logits = drafter(inputs_embeds, shared_kv, position_ids)
    # Target's logits at the SAME positions — i.e., where the gold regex tokens live.
    target_at_regex = target_logits[:, P : T, :]
    return _kl_drafter_to_target(drafter_logits, target_at_regex)


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
    p.add_argument("--eval-every", type=int, default=200)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading target ({TARGET_MODEL_ID}) ...")
    target, processor = load(TARGET_MODEL_ID)
    target.eval()

    print(f"Loading drafter ({ASSISTANT_MODEL_ID}) ...")
    drafter, _ = load_drafter(ASSISTANT_MODEL_ID, kind="mtp")
    drafter.bind(target)
    drafter._lm_head_fn = drafter.model.embed_tokens.as_linear
    drafter.train()
    print(f"Drafter trainable params: {_count_params(drafter):,}")

    train_examples = _load_examples(args.train)
    eval_examples = _load_examples(args.eval)
    print(f"train={len(train_examples)} eval={len(eval_examples)}")

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    def loss_fn(drafter, batch, target_hidden, shared_kv, target_logits):
        return _drafter_loss(drafter, batch, target_hidden, shared_kv, target_logits)

    grad_fn = nn.value_and_grad(drafter, loss_fn)
    log_path = args.out / "train_log.jsonl"
    log_path.unlink(missing_ok=True)

    def eval_pass(step: int) -> float:
        drafter.eval()
        losses = []
        for ex in eval_examples[:50]:
            batch = _make_batch(ex, processor)
            target_hidden, shared_kv, target_logits = _target_forward(target, batch.full_ids)
            loss = _drafter_loss(drafter, batch, target_hidden, shared_kv, target_logits)
            losses.append(float(loss.item()))
            mx.clear_cache()
        drafter.train()
        m = sum(losses) / max(1, len(losses))
        print(f"  [eval @ step {step}] mean KL = {m:.4f} over {len(losses)} examples")
        with log_path.open("a") as f:
            f.write(json.dumps({"event": "eval", "step": step, "kl": m}) + "\n")
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
            target_hidden, shared_kv, target_logits = _target_forward(target, batch.full_ids)
            loss, grads = grad_fn(drafter, batch, target_hidden, shared_kv, target_logits)
            optimizer.update(drafter, grads)
            mx.eval(drafter.parameters(), optimizer.state)
            mx.clear_cache()

            loss_v = float(loss.item())
            epoch_losses.append(loss_v)
            pbar.set_postfix(kl=f"{loss_v:.3f}")

            step += 1
            if step % args.eval_every == 0:
                eval_pass(step)
            with log_path.open("a") as f:
                f.write(json.dumps({"event": "step", "step": step,
                                     "epoch": epoch, "kl": loss_v}) + "\n")

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} mean train KL: {mean_loss:.4f}")
        eval_pass(step)

        ckpt = args.out / f"drafter_epoch{epoch + 1}.npz"
        flat = dict(tree_flatten(drafter.trainable_parameters()))
        mx.savez(str(ckpt), **flat)
        print(f"  saved {ckpt}")

    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
