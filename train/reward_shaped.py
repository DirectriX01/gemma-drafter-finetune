"""Reward-shaped trainer — baseline 4, the contribution.

Trains the drafter with a convex combination:
    loss = alpha * KL(drafter || target) + beta * CE(drafter, gold_tokens)

Interpretation:
    - alpha = 1, beta = 0 → pure acceptance-only training (baseline 3)
    - alpha = 0, beta = 1 → pure SFT on gold (baseline 2)
    - in between → trades off target-acceptance vs task-level correctness

The gold tokens are verifier-correct by construction (we sampled the test cases from
them). For a proper sampling-based verifier signal (R3 stretch goal), see comments at
the bottom of this file.

Usage:
    python -m train.reward_shaped \\
        --train data/processed/KB13_train.jsonl \\
        --eval  data/processed/KB13_eval.jsonl  \\
        --epochs 3 --alpha 0.5 --beta 0.5 \\
        --out adapters/reward_shaped_a05b05_kb13
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
    lm = target.language_model if hasattr(target, "language_model") else target
    out = lm(inputs=full_ids, return_hidden=True, return_shared_kv=True)
    hidden = mx.stop_gradient(out.hidden_states[-1])
    shared_kv = {k: (mx.stop_gradient(kv[0]), mx.stop_gradient(kv[1]))
                 for k, kv in out.shared_kv_states.items()}
    target_logits = mx.stop_gradient(out.logits)
    return hidden, shared_kv, target_logits


def _drafter_loss(
    drafter, batch: Batch, target_hidden, shared_kv, target_logits, alpha: float, beta: float
) -> mx.array:
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

    # KL(drafter || target) at regex positions
    target_at_regex = target_logits[:, P : T, :]
    log_p_d = nn.log_softmax(drafter_logits, axis=-1)
    log_p_t = nn.log_softmax(target_at_regex, axis=-1)
    p_d = mx.exp(log_p_d)
    kl = (p_d * (log_p_d - log_p_t)).sum(axis=-1).mean()

    # CE on gold tokens — verifier-correct supervision
    gold_flat = batch.gold_ids.reshape(-1)
    logits_flat = drafter_logits.reshape(-1, drafter_logits.shape[-1])
    ce = nn.losses.cross_entropy(logits_flat, gold_flat, reduction="mean")

    return alpha * kl + beta * ce, kl, ce


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
    p.add_argument("--alpha", type=float, default=0.5, help="weight on KL term (acceptance)")
    p.add_argument("--beta", type=float, default=0.5, help="weight on CE term (verifier)")
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
    drafter._lm_head_fn = drafter.model.embed_tokens.as_linear
    drafter.train()
    print(f"Drafter trainable params: {_count_params(drafter):,}")
    print(f"Loss = {args.alpha} * KL(drafter||target) + {args.beta} * CE(drafter, gold)")

    train_examples = _load_examples(args.train)
    eval_examples = _load_examples(args.eval)
    print(f"train={len(train_examples)} eval={len(eval_examples)}")

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    def loss_fn(drafter, batch, target_hidden, shared_kv, target_logits):
        total, _kl, _ce = _drafter_loss(
            drafter, batch, target_hidden, shared_kv, target_logits, args.alpha, args.beta
        )
        return total

    grad_fn = nn.value_and_grad(drafter, loss_fn)
    log_path = args.out / "train_log.jsonl"
    log_path.unlink(missing_ok=True)

    def eval_pass(step: int) -> float:
        drafter.eval()
        kls, ces = [], []
        for ex in eval_examples[:50]:
            batch = _make_batch(ex, processor)
            target_hidden, shared_kv, target_logits = _target_forward(target, batch.full_ids)
            _, kl, ce = _drafter_loss(
                drafter, batch, target_hidden, shared_kv, target_logits,
                args.alpha, args.beta,
            )
            kls.append(float(kl.item()))
            ces.append(float(ce.item()))
            mx.clear_cache()
        drafter.train()
        kl_m = sum(kls) / max(1, len(kls))
        ce_m = sum(ces) / max(1, len(ces))
        print(f"  [eval @ step {step}] KL={kl_m:.4f} CE={ce_m:.4f}")
        with log_path.open("a") as f:
            f.write(json.dumps({"event": "eval", "step": step, "kl": kl_m, "ce": ce_m}) + "\n")
        return args.alpha * kl_m + args.beta * ce_m

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


# -----------------------------------------------------------------------------
# Future work (W3+): proper sampling-based verifier reward.
#
# The convex combination above uses gold tokens as a proxy for "verifier-correct".
# Stronger formulation: at training time, sample regex completions from the drafter,
# run the verifier on each, and use the verifier reward as a per-sequence weight in
# the policy-gradient (REINFORCE) loss term — adding a third term to the loss:
#
#     loss = α * KL + β * CE_gold + γ * REINFORCE(drafter | verifier)
#
# This requires per-step sampling + verifier execution; ~10-50x more compute per
# step. Leave for W3 once the cheaper proxy is benchmarked.
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(main())
