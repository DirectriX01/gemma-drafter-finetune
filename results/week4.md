# Week 4 Results

**Goal (committed at start of W4):** unblock Path 4 RL, run training, evaluate B5 against
B0–B4, draft the writeup.

## Path 4 unblocked ✅

W3's RL pipeline ran sampling + verifier + REINFORCE cleanly but tripped MLX's
`tree_map` after a few non-skipped policy-gradient steps. Root cause turned out to be
**stateful attributes on the drafter** that mutate between sampling calls:

- `accept_lens: List[int]` grows each spec-decoding round → tree structure shifts
- `_shared_kv: dict | None` set by `set_shared_kv()` during inference → shape changes
- `_kv_offset: int`, `_position`, etc.

MLX's gradient-tree walker apparently picks these module attributes up alongside the
trainable parameters; when their structure drifts between iterations, the walker
indexes past the end and crashes with `IndexError`.

**Fix:** call `drafter.reset(target)` before each `grad_fn` call. This clears
`accept_lens = []`, sets `_shared_kv = None`, and rebinds the LM head. One line in
`train/rl_padded.py`.

After the fix, training ran to completion: 100 prompts × K=4 rollouts × 1 epoch in
11.5 min, mean reward 0.386.

## Full inference results (KB13_eval, n=102)

| Baseline | Description | pass@1 | F1 | tok/s |
|---|---|---|---|---|
| 0 | target alone | 0.176 | 0.390 ± 0.076 | 6.18 ± 0.77 |
| 1 | + vanilla drafter | 0.176 | 0.390 ± 0.076 | 6.00 ± 0.79 |
| 2 | + sparse SFT drafter | 0.176 | 0.390 ± 0.076 | 6.20 ± 0.77 |
| 3 (dense) | + dense KL drafter | 0.176 | 0.390 ± 0.076 | **2.99 ± 0.40 ⚠** |
| 3 (sparse) | + sparse KL drafter | 0.176 | 0.390 ± 0.076 | **6.38 ± 0.79 ✓** |
| 4 | + sparse RS (α=β=0.5) | 0.176 | 0.390 ± 0.076 | 6.34 ± 0.79 |
| **5** | **+ RL drafter (REINFORCE on cell-F1)** | 0.176 | 0.390 ± 0.076 | **6.29 ± 0.79** |

## What the numbers say

**Two clean findings:**

1. **The training-inference mismatch is real and fixable.** Naive dense-head training
   (W2 B3) drops the drafter to 2.99 tok/s — a 2× regression vs. vanilla. The
   sparse-aware fix recovers to 6.38, well outside CI. **End users cannot naively
   fine-tune Google's MTP drafter without addressing the centroid-routed LM head.**

2. **With the architectural fix applied, all training objectives plateau.** B2 (SFT),
   B3 (KL), B4 (reward-shaped), B5 (RL) are all in 6.2–6.4 tok/s. Pairwise CIs
   overlap; no objective strictly dominates on n=102.

**One quiet finding:**

3. **The verifier reward signal does not visibly differentiate the variants at this
   scale.** REINFORCE on cell-F1 with K=4 rollouts × 100 prompts produces a drafter
   essentially indistinguishable in speed from KL- or SFT-trained ones. Possible
   reasons:
   - The drafter's *output* under spec decoding is constrained by the target's
     verification — there's no correctness Pareto frontier to exploit, only an
     acceptance-rate ceiling that's already close to its structural limit.
   - 100 prompts × 1 epoch is too small a training budget to differentiate.
   - The reward signal is mostly degenerate (all K rollouts get identical reward on
     "easy" or "hard" prompts → advantage = 0 → no gradient).

## Project status

The original locked hypothesis was: *"verifier-shaped drafter Pareto-dominates
acceptance-only on (speed × correctness)."* What the data shows:

- **Correctness axis:** invariant — speculative decoding is loss-free in every
  variant. The Pareto formulation doesn't apply.
- **Speed axis:** flat plateau across trained variants once the mismatch is fixed.
  Verifier signal contributes no measurable lift.

The result that *does* land is a clean architectural finding: end users cannot
naively fine-tune Google's published MTP drafter. The path that works is documented
in `train/sparse_loss.py` + the freeze-masked-embedding + drafter.reset incantation.
This is a publishable measurement / methods paper independent of whether the verifier-
shaping experiment succeeded.

## Compute

| Run | Time |
|---|---|
| Sparse SFT (B2) | 25 min |
| Sparse accept-only (B3) | 34 min |
| Sparse reward-shaped (B4) | 28 min |
| RL (B5) — 100 prompts × K=4 × 1 ep | 11.5 min |
| Evals (×5) | ~10 min |

Total W3+W4 training compute: ~110 min on M1 Pro 16 GB, MLX Metal, $0 cloud.

## Possible W5 directions (in priority order if you want to extend)

1. **Scale up RL training** — full KB13 (411 prompts), more epochs, see if the
   plateau breaks. Cheap to run.
2. **Process NL-RX-Synth / NL-RX-Turk** properly so we have ~5–7k training examples.
   The W3 plateau may be a small-data ceiling.
3. **No-verify inference mode** — let the drafter output be the final answer, no
   target verification. Now correctness varies across baselines and the verifier
   signal has somewhere to add value.
4. **Standalone-drafter eval** — run the drafter without target K/V (simulated
   inference) and compare per-baseline correctness. Probably needs a different
   inference pipeline.

## Writeup status

W5 should be the paper draft. Materials in `results/week{1..4}.md` cover all the
findings; the headline is the **architectural finding (mismatch + fix)** with the
verifier-shaping result as a (negative) ablation.
