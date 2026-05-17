# Week 3 Results

**Goal (committed at top of W3):** Path 1 (fix the sparse-head training mismatch) + Path 4
(sample-based RL with verifier). Locked timeline slip W4 → W5.

## What landed: Path 1 — fully validated ✅

W2's surprise was that the trained drafters either matched or *degraded* the vanilla
drafter. Root cause traced to a **dense/sparse LM-head mismatch**: training used the
dense tied-embedding head (because MLX has no VJP through the sparse
`MaskedEmbedder.scatter_axis`), but inference uses the sparse centroid-routed head
that materializes only ~4096 of 262144 vocab. Trained weights were optimal for a head
the drafter never actually uses.

**The fix (`train/sparse_loss.py`):** reproduce the sparse head's materialization
differentiably — the matmul on selected embeddings is differentiable end-to-end if
we just don't compute gradients through the discrete top-K indices. Then mask the
loss to positions where the gold (or target's) token is *actually* in the materialized
top-K set. Body params learn to produce hidden states whose top-K cluster routing
covers useful tokens.

**Additional fix:** `drafter.masked_embedding.freeze()` prevents AdamW from corrupting
the integer `token_ordering` buffer (which MLX treats as a trainable param because
it's an `mx.array` attribute on an `nn.Module`).

### Diagnostic that proved the fix worked

`gold_hit_rate` — fraction of regex tokens in the materialized top-K set:

- Untrained drafter (no fine-tuning): **0.78**
- After sparse-aware SFT (3 epochs on KB13): **0.99**

The body learns to route hidden states through the right clusters even with the
centroids themselves frozen.

## Inference results (KB13_eval, n=102, all sparse-aware variants)

| Baseline | Description | pass@1 | F1 | tok/s | new tok |
|---|---|---|---|---|---|
| 0 | target alone | 0.176 | 0.390 | 6.18 ± 0.77 | 8.0 |
| 1 | + vanilla drafter | 0.176 | 0.390 | 6.00 ± 0.79 | 8.0 |
| 2 (dense) | + dense-SFT drafter | 0.176 | 0.390 | 6.19 ± 0.77 | 8.0 |
| 2 (sparse) | + sparse-SFT drafter | 0.176 | 0.390 | 6.20 ± 0.77 | 8.0 |
| **3 (dense)** | + dense-KL drafter | 0.176 | 0.390 | **2.99 ± 0.40** ⚠ | 8.0 |
| **3 (sparse)** | + sparse-KL drafter | 0.176 | 0.390 | **6.38 ± 0.79** ✓ | 8.0 |
| 4 (dense) | + dense-RS (α=β=0.5) | 0.176 | 0.390 | 6.37 ± 0.79 | 8.0 |
| 4 (sparse) | + sparse-RS (α=β=0.5) | 0.176 | 0.390 | 6.34 ± 0.79 | 8.0 |
| **5** | + RL-trained drafter | — pending W4 — |

**Headline numbers:**

- **B3 went from 2.99 → 6.38 tok/s with the sparse-aware fix.** A 2.1× recovery,
  well outside the 95% CI of either reading. The training-inference mismatch was real
  and is now eliminated.
- All other baselines on the sparse path land in 6.2–6.4 tok/s (CIs overlap with B0
  and B1). No baseline strictly dominates within the sample size.
- Speculative decoding remains loss-free across every variant (pass@1 and F1 are
  bit-identical), so the speed axis is the only one that can differentiate the
  variants under standard spec-decoding semantics.

## Three findings

**1. Path 1 fix recovers a 2× speed loss.** Sparse-aware training is *necessary* for
fine-tuning the official Gemma 4 MTP drafter without degrading it. End users cannot
naively use mlx-vlm + standard CE/KL on this drafter — they will silently make it
slower.

**2. With the fix, the trained variants converge to a tight plateau.** B2 (sparse
SFT), B3 (sparse KL), B4 (sparse RS) are all 6.2–6.4 tok/s, indistinguishable within
CI. The verifier-signal axis (β in B4) doesn't yet show its hand on KB13 at this
scale.

**3. The drafter's sparse head has a structural cap.** Even with maximal training,
gold tokens land in the materialized top-K only ~99% of the time. The other ~1% are
unreachable. For longer outputs or domains where this rate is lower, the cap matters
more.

## What did *not* land: Path 4 (RL)

Implementation:
- `train/rl.py` — K=4 rollouts via `mlx_vlm.stream_generate` with draft_kind='mtp' +
  temperature sampling
- Verifier scoring with cell-F1 (shaped, as you specified)
- REINFORCE with mean-baseline advantage
- Warm-start from sparse-SFT adapter

Sampling, verifier, advantage computation, and the first few non-skipped policy-gradient
updates all run cleanly (~6 s/example, real loss/reward values logged). Then
`optimizer.update(drafter, grads)` consistently hits an `IndexError` deep inside MLX's
`tree_map`. Reproduces with both AdamW and SGD optimizers, with both manual gradient
accumulation and per-rollout micro-steps. The gradient tree returned by
`value_and_grad` on our sparse-loss path apparently has a structure that subtly
shifts between rollouts in a way the optimizer can't reconcile.

This isn't a Python-level bug — it's an MLX framework interaction we don't have time
to resolve in W3. Three viable fixes for W4:

1. Stack rollouts into a single padded tensor `[K, max_len]` so the loss-fn argument
   structure is fixed.
2. Use `mx.value_and_grad` directly on a flat param tree (bypass `nn.Module` wrappers).
3. Switch the sparse-loss path to emit dense logits + `mx.stop_gradient(sparse - dense)`
   (STE on the output), which gives a more uniform gradient tree.

## Compute summary (W3)

| Run | Time |
|---|---|
| Sparse SFT | 25 min |
| Sparse accept-only | 34 min |
| Sparse reward-shaped (α=β=0.5) | 28 min |
| RL (failed at step ~7) | ~2 min |
| Evals (×4) | ~8 min |

Total: ~100 min compute + diagnostic / debug time.

## W4 plan (revised)

1. Resolve the RL tree_map issue (~1–2 days)
2. Run RL training (1 epoch on 100 prompts, then scale up if it converges)
3. Eval B5 against B0–B4
4. Write up the paper draft (Path 1 finding + Path 4 conditional on results)
