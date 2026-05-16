# Week 1 Results

**Goal:** verifier + dataset + zero-shot baselines for Gemma 4 E4B (baseline 0) and
Gemma 4 E4B + assistant drafter (baseline 1) on regex synthesis.

## Verifier

- 30/30 unit tests passing (NULL, Unicode, anchors, greedy/lazy, alternation, lookaheads,
  catastrophic backtracking, partial credit, compile errors).
- 100ms per-example timeout via `SIGALRM`.
- Two reward signals: binary correctness (`reward_binary`) and F1 (`reward_shaped`).

## Datasets

Processed via `data/loader.py`. Filtered to Python-`re`-compatible examples
(no DSL `&`/`~(...)`). Positive examples sampled via `exrex` (cap 100 chars, 2s timeout per
call); negative examples drawn from other items' positive sets.

| Split | Raw | Kept | Kept % |
|---|---|---|---|
| KB13 | 823 | 610 | 74.1% |
| NL-RX-Synth | 9999 | TBD | TBD |
| NL-RX-Turk | 9999 | TBD | TBD |

## Baselines (KB13 eval, n=TBD)

| Baseline | Description | pass@1 | F1 | tokens/sec | mean new tokens |
|---|---|---|---|---|---|
| 0 | Gemma 4 E4B-it (target alone) | TBD | TBD | TBD | TBD |
| 1 | Gemma 4 E4B-it + assistant drafter | TBD | TBD | TBD | TBD |

## Notes

- All runs on M1 Pro 16GB, MPS, fp16
- Seeded deterministic generation (`do_sample=False`)
- Few-shot prompting with 3 examples
- Held-out evaluation only (no training data leakage into prompt)

## Next (Week 2)

- SFT pipeline (drafter + target) on NL-RX-Synth
- Acceptance-only fine-tuned drafter (baseline 3 — the critical control)
- Reproducible eval numbers with seeded splits
