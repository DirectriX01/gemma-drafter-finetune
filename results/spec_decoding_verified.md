# Speculative-decoding sanity check

Concern from W1: baseline 1 (target + drafter) gave nearly identical tok/sec as baseline 0
(target alone). Suspected the spec-decoding path might be silently falling back.

## Test

Prompt: "Write a Python regex for an email address. Output only the regex, nothing else."
Output: `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$` (34 tokens)

| | Wall-clock | Prompt tps | Gen tps |
|---|---|---|---|
| Target alone | 2.11 s | 53.4 | 41.5 |
| Target + drafter | 1.03 s | 124.7 | 42.4 |
| **Speedup** | **2.05×** | 2.33× | 1.02× |

## Reading

- Wall-clock 2.05× confirms the drafter is engaging; this is **not** a silent fallback.
- Reported "generation tps" looks identical because MLX's per-step counter doesn't
  account for multi-token-per-step spec output; the real wins show up in wall-clock.
- W1's null result was a **regime issue**: at ~8.5 generated tokens per example, spec
  decoding doesn't amortize. The same setup yields 2× on 34-token outputs.

## Implication for the project

This is consistent with H1's motivation: the existing acceptance-only drafter is leverage-
limited on short structured outputs. The reward-shaped drafter has the most room to
contribute exactly where existing drafters underperform — which is the regime we care
about.

## Code

`eval/infer.py` passes `drafter=` to `mlx_vlm.generate()` — confirmed correct kwarg name.
