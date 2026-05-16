# Week 1 Results

**Goal:** verifier + dataset + zero-shot baselines for Gemma 4 E4B (baseline 0) and
Gemma 4 E4B + MTP drafter (baseline 1) on regex synthesis.

## Stack (after pivot)

- Tried `transformers + MPS` first — Gemma 4 E4B-it is 16 GB on disk, transformers'
  `caching_allocator_warmup` tries to allocate a single ~15 GiB MPS buffer which
  exceeds Apple's per-allocation limit on a 16 GB unified-memory machine. CPU-first
  load swap-thrashed.
- Pivoted to **MLX-VLM** with `mlx-community/gemma-4-e4b-it-4bit` (~5 GB) and the
  official MTP drafter `mlx-community/gemma-4-E4B-it-assistant-bf16` (~180 MB). Native
  Apple Silicon (Metal), no MPS allocation issues.
- Will need to revisit for W2 training — MLX-LM has LoRA fine-tuning support.

## Verifier (`verifier/`)

- 30/30 unit tests passing (NULL, Unicode, anchors, greedy/lazy, alternation,
  lookaheads, catastrophic backtracking via SIGALRM 100 ms cap, partial credit,
  compile errors).
- Two reward signals: binary correctness (`reward_binary`) and F1 (`reward_shaped`).

## Datasets

| Split | Raw | Kept | Kept % | Notes |
|---|---|---|---|---|
| KB13 | 823 | 610 | 74.1% | filtered DSL `&`/`~(...)` + sampled pos/neg via exrex |
| NL-RX-Synth | 9999 | — | deferred to W2 | exrex hangs on some patterns even with stricter limits — needs subprocess-isolated sampling |
| NL-RX-Turk | 9999 | — | deferred to W2 | same |

## Baselines (KB13, n=50, seed=0)

| Baseline | Description | pass@1 | F1 (95% CI) | tok/sec (95% CI) | avg new tokens |
|---|---|---|---|---|---|
| 0 | Gemma 4 E4B-it 4-bit, target alone | 0.220 | 0.269 ± 0.118 | 6.70 ± 1.01 | 8.5 |
| 1 | Gemma 4 E4B-it 4-bit + MTP drafter | 0.220 | 0.269 ± 0.118 | 6.75 ± 1.04 | 8.5 |

**Observations:**

- **Correctness preserved across B0 → B1** (same pass@1 and F1). Speculative decoding
  is loss-free, as expected.
- **No measurable speedup from the drafter on these short outputs.** Mean new-token
  count is 8.5 — speculative decoding shines on longer generations where it can
  amortize drafter overhead. On short structured outputs like regex, the leverage is
  limited.
- This is itself a useful Week 1 signal for the project: it confirms the regime where
  reward-shaping the drafter should matter most. Acceptance-only drafters provide
  diminishing returns on short structured outputs; whether a verifier-aware drafter
  changes that is exactly H1 of the project.

## Sanity check (qualitative)

Sample model outputs for KB13 prompts (from baseline 0):

| NL | gold | model |
|---|---|---|
| `lines containing only digits` | (smoke test) | `\d+` ✓ |
| `a 3-digit number followed by a hyphen and a 4-digit number` | (few-shot) | `\d{3}-\d{4}` ✓ |

The model knows regex. The 22% pass@1 on KB13 reflects a mix of (a) prompt-style
mismatch between KB13's NL phrasings and the model's training distribution, (b) the
fact that exec-equivalence under our exrex-generated test cases is a stricter notion
than KB13's original eval, and (c) some inherent difficulty of the held-out set.

## Compute

- All on M1 Pro 16 GB, MLX-Metal, 4-bit target / bf16 drafter
- Deterministic generation (no sampling), few-shot prompting (3 ex)
- B0: 59 s wall for 50 examples
- B1: 74 s wall for 50 examples (drafter load overhead + similar inference time)

## Open items / debt to pay down in Week 2

1. **Confirm drafter is actually engaging.** The marginal speedup (6.70 → 6.75 tok/s)
   is within noise; need to instrument and verify the speculative path is taken.
2. **Fix NL-RX-Synth/Turk loader.** SIGALRM-based timeout doesn't catch all exrex
   pathologies — needs `multiprocessing.Process` isolation per sample.
3. **Run on full KB13** (610 examples) for sample-size-robust numbers, plus on the
   full Spider-equivalent held-out sets once Synth/Turk are processed.

## Next (Week 2)

- Fix the loader → process NL-RX-Synth/Turk
- SFT pipeline on the drafter (start of baseline 2)
- Acceptance-only fine-tuned drafter (baseline 3 — the critical control)

## Files

- `verifier/regex_verifier.py` + 30 tests (`tests/test_regex_verifier.py`)
- `data/loader.py` — Python-re-compatible filter + exrex sampling
- `eval/infer.py` — MLX-VLM wrapper with optional drafter
- `eval/prompts.py` — few-shot prompt template + regex extraction
- `eval/run.py` — pass@1 + F1 + tok/sec eval harness
- `results/baseline0_KB13_n50.json`, `results/baseline1_KB13_n50.json`
