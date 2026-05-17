# Drafter RLVR

Fine-tuning Google's Gemma 4 E4B Multi-Token-Prediction (MTP) drafter on regex
synthesis. The repo documents an **architectural failure mode**: naive
fine-tuning of the published drafter silently halves its
speculative-decoding speed because of a training-inference mismatch in its
centroid-routed sparse LM head. It then ships the recipe that fixes it and
evaluates five drafter-training objectives against vanilla and target-only
baselines.

## Setup

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"
.venv/bin/hf auth login
python -m data.loader
python -m data.split --src data/processed/KB13.jsonl --eval-frac 0.2
```

Models pulled automatically on first use:
- Target: [`mlx-community/gemma-4-e4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit) (~5 GB)
- Drafter: [`mlx-community/gemma-4-E4B-it-assistant-bf16`](https://huggingface.co/mlx-community/gemma-4-E4B-it-assistant-bf16) (~180 MB)

Tested on M1 Pro, 16 GB unified memory, MLX Metal.

## Results

KB13 held-out eval (n=102), deterministic generation. Speculative decoding is
loss-free, so `pass@1` and `F1` are bit-identical across every baseline; the
discriminating axis is `tok/s`.

| # | Baseline | tok/s |
|---|---|---|
| B0 | target alone | 6.18 ± 0.77 |
| B1 | + vanilla MTP drafter | 6.00 ± 0.79 |
| B2 | + sparse SFT (CE on gold) | 6.20 ± 0.77 |
| B3 (dense, naive) | + KL-distill drafter | **2.99 ± 0.40 ⚠** |
| B3 (sparse, fixed) | + KL-distill drafter | **6.38 ± 0.79 ✓** |
| B4 | + reward-shaped (α=β=0.5) | 6.34 ± 0.79 |
| B5 | + REINFORCE on cell-F1 (K=4) | 6.29 ± 0.79 |

All variants invariant at `pass@1 = 0.176`, `F1 = 0.390 ± 0.076`.

### The three architectural traps

| Trap | Symptom | Fix |
|---|---|---|
| `MaskedEmbedder.scatter_axis` has no VJP | training crashes or grads bypass the sparse head | `train/sparse_loss.py` reproduces the sparse forward differentiably (only the discrete cluster selection is non-differentiable; the matmul on selected embeddings is) |
| `token_ordering` is an integer buffer treated as a trainable parameter | AdamW converts it to float → next gather throws `indices must be integral` | `drafter.masked_embedding.freeze()` |
| Stateful attributes (`accept_lens`, `_shared_kv`, `_kv_offset`) drift between sampling calls | optimizer `tree_map` `IndexError` after a handful of updates | `drafter.reset(target)` before each `grad_fn` call |

Without these fixes, the standard KL-distillation objective drops the drafter
from 6.00 → 2.99 tok/s. With the fixes applied, every training objective
plateaus in the 6.2–6.4 band on KB13.

## Reproducing each baseline

```bash
# B0 — target alone
python -m eval.run --dataset data/processed/KB13_eval.jsonl --baseline 0

# B1 — vanilla drafter
python -m eval.run --dataset data/processed/KB13_eval.jsonl --baseline 1

# B2 — sparse SFT
python -m train.sft_sparse \
    --train data/processed/KB13_train.jsonl --eval data/processed/KB13_eval.jsonl \
    --out adapters/sft_sparse_kb13 --epochs 3 --lr 1e-4
python -m eval.run --dataset data/processed/KB13_eval.jsonl --baseline 2 \
    --adapter adapters/sft_sparse_kb13/drafter_epoch3.npz --tag baseline2_sft_sparse

# B3 — sparse KL
python -m train.accept_only_sparse \
    --train data/processed/KB13_train.jsonl --eval data/processed/KB13_eval.jsonl \
    --out adapters/accept_only_sparse_kb13 --epochs 3 --lr 1e-4
python -m eval.run --dataset data/processed/KB13_eval.jsonl --baseline 3 \
    --adapter adapters/accept_only_sparse_kb13/drafter_epoch3.npz --tag baseline3_accept_only_sparse

# B4 — sparse reward-shaped (α=β=0.5)
python -m train.reward_shaped_sparse \
    --train data/processed/KB13_train.jsonl --eval data/processed/KB13_eval.jsonl \
    --out adapters/reward_shaped_sparse_a05b05_kb13 --epochs 3 --alpha 0.5 --beta 0.5 --lr 1e-4
python -m eval.run --dataset data/processed/KB13_eval.jsonl --baseline 4 \
    --adapter adapters/reward_shaped_sparse_a05b05_kb13/drafter_epoch3.npz \
    --tag baseline4_reward_shaped_sparse_a05b05

# B5 — REINFORCE on cell-F1 (warm-start from B2)
python -m train.rl_padded \
    --train data/processed/KB13_train_100.jsonl --eval data/processed/KB13_eval.jsonl \
    --out adapters/rl_kb13 --epochs 1 --K 4 --temperature 0.7 --max-new-tokens 24 \
    --init-adapter adapters/sft_sparse_kb13/drafter_epoch3.npz --lr 5e-5
python -m eval.run --dataset data/processed/KB13_eval.jsonl --baseline 4 \
    --adapter adapters/rl_kb13/drafter_epoch1.npz --tag baseline5_rl
```

Per-run wall-clock on M1 Pro:

| Run | Time |
|---|---|
| Sparse SFT (3 ep) | 25 min |
| Sparse KL (3 ep) | 34 min |
| Sparse reward-shaped (3 ep) | 28 min |
| REINFORCE (100 prompts × K=4 × 1 ep) | 12 min |
| One eval pass | ~2 min |

## Repo layout

```
verifier/regex_verifier.py    SIGALRM-bounded regex eval; binary + cell-F1 reward
data/loader.py, split.py      KB13 download, filter, pos/neg sampling; train/eval split
eval/infer.py, run.py         MLX-VLM inferencer; pass@1 / F1 / tok/s harness
train/sparse_loss.py          differentiable sparse-head materialization
train/sft_sparse.py           B2 — CE on gold over sparse-head logits
train/accept_only_sparse.py   B3 — KL(drafter || target) over sparse-head logits
train/reward_shaped_sparse.py B4 — α·KL + β·CE_gold
train/rl_padded.py            B5 — REINFORCE on cell-F1, K rollouts, padded
tests/test_regex_verifier.py  30 unit tests on the regex verifier
```

The W2-era trainers (`train/sft.py`, `accept_only.py`, `reward_shaped.py`) use
the dense LM head and produce the broken B3-dense result; kept for
reproducibility.

## License

Apache 2.0 (matches Gemma 4). Built on
[`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) and Google's
[Gemma 4 MTP drafter release](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/).
