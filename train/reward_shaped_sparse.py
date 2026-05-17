"""Sparse-aware reward-shaped trainer — Path 1 + B4.

Loss = alpha * sparse_aware_KL(drafter || target) + beta * sparse_aware_CE(drafter, gold)

Same convex-combination idea as `train/reward_shaped.py` but evaluated over the
sparse head's materialized vocab subset. Path 1 fix means the trained drafter is
actually optimized for the inference-time LM head.

Usage:
    python -m train.reward_shaped_sparse \\
        --train data/processed/KB13_train.jsonl \\
        --eval  data/processed/KB13_eval.jsonl \\
        --epochs 3 --alpha 0.5 --beta 0.5 \\
        --out adapters/reward_shaped_sparse_a05b05_kb13
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
from train.sparse_loss import materialize_sparse, sparse_aware_ce, sparse_aware_kl


@dataclass
class Batch:
    full_ids: mx.array
    prompt_len: int
    gold_ids: mx.array


def _load_examples(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _make_batch(ex, processor):
    tok = processor.tokenizer
    prompt_text = tok.apply_chat_template(
        build_prompt(ex["nl"], few_shot=True),
        tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    gold_ids = tok.encode(ex["gold_regex"], add_special_tokens=False)
    full_ids = prompt_ids + gold_ids
    return Batch(
        full_ids=mx.array(full_ids)[None, :],
        prompt_len=len(prompt_ids),
        gold_ids=mx.array(gold_ids)[None, :],
    )


def _target_forward(target, full_ids):
    lm = target.language_model if hasattr(target, "language_model") else target
    out = lm(inputs=full_ids, return_hidden=True, return_shared_kv=True)
    hidden = mx.stop_gradient(out.hidden_states[-1])
    shared_kv = {k: (mx.stop_gradient(kv[0]), mx.stop_gradient(kv[1]))
                 for k, kv in out.shared_kv_states.items()}
    target_logits = mx.stop_gradient(out.logits)
    return hidden, shared_kv, target_logits


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


def _drafter_loss(drafter, batch, target_hidden, shared_kv, target_logits, alpha, beta):
    h = _drafter_forward_hidden(drafter, batch.full_ids, target_hidden, shared_kv, batch.prompt_len)
    embed_w = drafter.model.embed_tokens.weight
    selected_logits, selected_ids = materialize_sparse(drafter, h, embed_w)
    P = batch.prompt_len
    T = batch.full_ids.shape[1]
    target_at_regex = target_logits[:, P:T, :]
    kl = sparse_aware_kl(selected_logits, selected_ids, target_at_regex)
    ce, _has_gold = sparse_aware_ce(selected_logits, selected_ids, batch.gold_ids)
    return alpha * kl + beta * ce, kl, ce


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
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=200)
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
    drafter.train()
    n_params = sum(int(p.size) for _, p in tree_flatten(drafter.trainable_parameters()))
    print(f"Drafter trainable params (masked_embedding frozen): {n_params:,}")
    print(f"Loss = {args.alpha} * sparse_KL + {args.beta} * sparse_CE")

    train_examples = _load_examples(args.train)
    eval_examples = _load_examples(args.eval)
    print(f"train={len(train_examples)} eval={len(eval_examples)}")

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    def loss_fn(drafter, batch, target_hidden, shared_kv, target_logits):
        total, _, _ = _drafter_loss(drafter, batch, target_hidden, shared_kv,
                                     target_logits, args.alpha, args.beta)
        return total

    grad_fn = nn.value_and_grad(drafter, loss_fn)
    log_path = args.out / "train_log.jsonl"
    log_path.unlink(missing_ok=True)

    def eval_pass(step: int):
        drafter.eval()
        kls, ces = [], []
        for ex in eval_examples[:50]:
            batch = _make_batch(ex, processor)
            target_hidden, shared_kv, target_logits = _target_forward(target, batch.full_ids)
            _, kl, ce = _drafter_loss(drafter, batch, target_hidden, shared_kv,
                                       target_logits, args.alpha, args.beta)
            kls.append(float(kl.item()))
            ces.append(float(ce.item()))
            mx.clear_cache()
        drafter.train()
        kl_m = sum(kls) / max(1, len(kls))
        ce_m = sum(ces) / max(1, len(ces))
        print(f"  [eval @ step {step}] sparse_KL = {kl_m:.4f}  sparse_CE = {ce_m:.4f}")
        with log_path.open("a") as f:
            f.write(json.dumps({"event": "eval", "step": step, "kl": kl_m, "ce": ce_m}) + "\n")

    print(f"\nStarting sparse-aware reward-shaped training: {args.epochs} epochs, lr={args.lr}\n")
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

        ckpt = args.out / f"drafter_epoch{epoch + 1}.npz"
        flat = dict(tree_flatten(drafter.trainable_parameters()))
        mx.savez(str(ckpt), **flat)
        print(f"  saved {ckpt}")

    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
