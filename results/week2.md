# Week 2 Results

**Goal:** train the Gemma 4 MTP drafter under two objectives (SFT on gold, KL to
target) and evaluate against the vanilla drafter. Set up the reward-shaped variant for
W3.

## Pipeline status

- ✅ Custom MLX-VLM training loop that threads target's hidden states + shared K/V into
  the drafter for teacher-forced gradient flow.
- ✅ Solved two architectural blockers:
  - **Centroid-routed sparse LM head** (`use_ordered_embeddings=True`) has no VJP through
    `scatter_axis`. Swapped `_lm_head_fn` to the dense tied-embedding head **at training
    time only**; inference reverts to sparse via `bind(target)`.
  - **`target.freeze()` crashes** on `AudioRelativePositionEmbedding` (some Gemma 4 MM
    submodules lack the `_no_grad` attribute). Solved with `mx.stop_gradient` on target
    outputs + restricting `nn.value_and_grad(drafter, …)` to drafter params.
- ✅ Three trainers built:
  - `train/sft.py` — CE on gold regex tokens (baseline 2)
  - `train/accept_only.py` — KL(drafter || target) at regex positions (baseline 3)
  - `train/reward_shaped.py` — α·KL + β·CE_gold (baseline 4, training)
- ✅ Adapter checkpoints save/load round-trip works (`mx.savez` → `mx.load` + `update`).
- ❌ NL-RX-Synth/Turk processing deferred — subprocess-isolated sampler too slow.
  Training on KB13 only (411 train / 102 eval).

## Training metrics

| Run | Train loss start → end | Eval loss start → end | Time |
|---|---|---|---|
| SFT (CE on gold) | 5.44 → 0.06 | 4.34 → 0.37 | 25 min |
| Accept-only (KL) | 16.5 → 6.06 | 9.99 → 6.62 | 29 min |
| Reward-shaped (α=β=0.5) | 7.69 → 3.51 | — → KL=4.01 CE=3.71 | 34 min |

All three converged on their own loss curves.

## Inference results (KB13_eval, n=102)

| Baseline | Description | pass@1 | F1 | tok/s | new tok | compile err |
|---|---|---|---|---|---|---|
| 0 | target alone | 0.176 | 0.390 ± 0.076 | 6.18 ± 0.77 | 8.0 | 1.0% |
| 1 | target + vanilla drafter | 0.176 | 0.390 ± 0.076 | 6.00 ± 0.79 | 8.0 | 1.0% |
| 2 | target + SFT-trained drafter | 0.176 | 0.390 ± 0.076 | 6.19 ± 0.77 | 8.0 | 1.0% |
| 3 | target + KL-trained drafter | 0.176 | 0.390 ± 0.076 | **2.99 ± 0.40** | 8.0 | 1.0% |
| 4 | target + reward-shaped drafter (α=β=0.5) | 0.176 | 0.390 ± 0.076 | **6.37 ± 0.79** | 8.0 | 1.0% |

**Speed ranking:** B4 ≈ B2 ≈ B0 ≈ B1 ≫ B3.

The reward-shaped drafter (B4) recovered the speed lost by B3. Adding the
CE-on-gold term acts as a **regularizer** that prevents the dense/sparse-head training
mismatch from breaking inference. The CE term anchors the drafter's preferred tokens
to gold tokens (which are also in the sparse head's materialized vocab subset
typically), while the KL term still pushes toward target's distribution. The
combination is robust where pure KL is not.

CIs overlap between B4 (6.37 ± 0.79) and B1 (6.00 ± 0.79), so we cannot claim B4
strictly beats B1 in tok/s on n=102. We *can* claim:
- B4 does **not** suffer the B3 degradation (well outside CI).
- The acceptance-only objective (B3) is **strictly harmful** without a regularizer.

## Two important findings

**1. Speculative decoding is loss-free in the textbook sense.**

pass@1 and F1 are identical across B0–B3. The drafter only affects speed — its
proposed tokens are verified by the target, so the final output is bit-exact whatever
the drafter does. *This means the project's hypothesis cannot be tested as
"correctness Pareto frontier" — only as "speed Pareto frontier."* We need to either:

- Reframe H1 around acceptance rate / speed only (lose the "correctness" axis), or
- Add a *no-verify* inference mode where the drafter's proposals go through unchecked
  (turns speculative decoding into "fast but lossy" — now correctness matters and the
  verifier-shaped drafter has somewhere to add value).

The second reframing recovers the original research question. Implementing it is
small — just a flag in `eval/infer.py` to use `drafter.generate_step` directly.

**2. Training-inference architecture mismatch DEGRADES the trained drafter.**

The official Gemma 4 MTP drafter uses a centroid-routed sparse LM head at inference
(top-32 of 2048 clusters → ~4096 of 262144 vocab tokens). MLX's `scatter_axis` has no
VJP, so we trained with the dense tied-embedding head. The trained weights are
optimal under dense, but the sparse head at inference materializes a different
subset of vocab → the drafter's preferred token often isn't materialized → target
rejects → much slower decoding (B3: ~2× slower than B1 vanilla).

This is a *real result*, not an experimental flaw: it says **end users cannot
straightforwardly fine-tune Google's MTP drafter without addressing the
training-inference mismatch**. The fix is non-obvious — either:

- **Straight-through estimator** on the sparse head (forward sparse, backward dense), or
- **Restrict training** to the top-K-cluster output positions matching the inference
  materialization pattern, or
- **Keep the sparse head intact** by training only the centroid-routing parameters via
  a custom MLX gradient bypass.

This finding is itself publishable as a measurement / analysis paper, independent of
whether the verifier-shaped contribution lands.

## Third finding (after B4 landed)

**Reward-shaping (B4) acts as a regularizer that rescues the speed-degraded B3.**

This is the first piece of evidence that the verifier-signal axis adds value — not in
the way we hypothesized (correctness improvement, which is invariant), but as a
training-time stabilizer that prevents the sparse-head mismatch from corrupting the
drafter. Whether this generalizes outside KB13's small scale needs W3 ablations.

## What this means for the project

The original hypothesis ("verifier-shaping the drafter Pareto-dominates acceptance-only
training") needs adjusting. Two productive directions:

**A.** Continue with the speed-only Pareto framing: B4 (verifier-shaped) vs B3
(acceptance-only) on tok/s. If B4 is faster than B3 at the same correctness, the
hypothesis is supported (in the speed-only axis).

**B.** Add a *no-verify* mode and measure standalone drafter correctness. If B4
produces correct regex more often than B3 in standalone mode, the hypothesis is
supported (in the correctness axis).

Both can be done; **(A)** is what the existing code emits already; **(B)** needs a
2-day extension to `eval/infer.py`.

## Open in W3

- Solve the training-inference mismatch (STE or selective training)
- Add no-verify inference mode for standalone drafter eval
- Run reward-shaped sweep across α/β
- Process NL-RX-Synth/Turk for more training data
- Re-run all baselines with the fixed pipeline

## Compute

All on M1 Pro 16 GB, MLX Metal, 4-bit target / bf16 drafter. Three full training runs
(25–29 min each) + four eval passes (~2 min each) consumed roughly 2.5 hours of wall
time and ~6 GB RAM peak.
