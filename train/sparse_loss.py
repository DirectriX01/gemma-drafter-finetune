"""Sparse-aware losses for training the Gemma 4 MTP drafter.

The drafter's inference-time LM head materializes only the top-K cluster's tokens
(~4096 of 262144 vocab). Training with the dense head produces weights optimized for
positions the sparse head never reads — that's the root cause of W2's B3 speed
degradation.

These helpers reproduce the sparse head's forward pass differentiably (the matmul
is differentiable; the topk index selection has no VJP but that's fine since we
don't backprop into centroids here), then mask the loss to positions where the
gold/target token is *actually* in the materialized set. Body params learn to
produce hidden states whose top-K cluster materializes useful tokens.
"""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


def materialize_sparse(
    drafter, hidden: mx.array, embed_weight: mx.array
) -> Tuple[mx.array, mx.array]:
    """Sparse forward pass — returns (selected_logits, selected_token_ids).

    Differentiable through `selected_logits` w.r.t. `hidden` and `embed_weight`.
    The token_ids are integer indices (no VJP needed).

    Args:
        drafter: Gemma4AssistantDraftModel
        hidden: [B, L, H] last-layer hidden states from drafter's decoder body
        embed_weight: [V, H] tied embedding table from target (or drafter)

    Returns:
        selected_logits: [B, L, M] where M = top_k * vocab_size_per_centroid
        selected_token_ids: [B, L, M] canonical vocab token IDs at each position
    """
    me = drafter.masked_embedding
    B, L = hidden.shape[:2]

    # Centroid scores → top-K cluster indices (discrete, no VJP — OK, we don't
    # backprop into centroids here)
    centroid_logits = me.centroids(hidden)  # [B, L, C]
    topk_idx = mx.argpartition(centroid_logits, kth=-me.top_k, axis=-1)[..., -me.top_k:]  # [B, L, K]

    ordering = me.token_ordering.reshape(me.num_centroids, me.vocab_size_per_centroid)
    selected_canonical = ordering[topk_idx]  # [B, L, K, vsc]
    selected_canonical = selected_canonical.reshape(B, L, -1)  # [B, L, M]

    # Gather embeddings for materialized tokens; differentiable w.r.t. embed_weight
    flat_idx = selected_canonical.reshape(-1)
    selected_emb = embed_weight[flat_idx].reshape(B, L, -1, hidden.shape[-1])

    # Logits = hidden @ selected_emb.T  — differentiable w.r.t. hidden and embed_weight
    selected_logits = mx.matmul(
        hidden[..., None, :], selected_emb.swapaxes(-1, -2)
    ).squeeze(-2)  # [B, L, M]

    return selected_logits, selected_canonical


def sparse_aware_ce(
    selected_logits: mx.array,
    selected_token_ids: mx.array,
    gold_tokens: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Cross-entropy on positions where the gold token is in the materialized set.

    Args:
        selected_logits: [B, L, M] from `materialize_sparse`
        selected_token_ids: [B, L, M] from `materialize_sparse`
        gold_tokens: [B, L] target token IDs

    Returns:
        loss: scalar mean CE over positions where gold was materialized
        has_gold: [B, L] boolean — fraction is the "gold hit rate"
    """
    gold_expanded = gold_tokens[:, :, None]                       # [B, L, 1]
    gold_match = (selected_token_ids == gold_expanded)            # [B, L, M] bool
    has_gold = gold_match.any(axis=-1)                            # [B, L]

    log_probs = nn.log_softmax(selected_logits, axis=-1)          # [B, L, M]
    # Gold's log-prob = sum_i 1{token_id[i] == gold} * log_prob[i]
    gold_log_prob = (gold_match.astype(log_probs.dtype) * log_probs).sum(axis=-1)  # [B, L]

    nll = -gold_log_prob * has_gold.astype(gold_log_prob.dtype)
    denom = has_gold.sum().astype(nll.dtype) + 1e-6
    loss = nll.sum() / denom
    return loss, has_gold


def sparse_aware_kl(
    selected_logits: mx.array,
    selected_token_ids: mx.array,
    target_full_logits: mx.array,
) -> mx.array:
    """KL(drafter || target) restricted to the materialized vocab subset at each position.

    Args:
        selected_logits: [B, L, M] drafter logits over materialized tokens
        selected_token_ids: [B, L, M] vocab IDs
        target_full_logits: [B, L, V] target's logits over the full vocab

    Returns:
        scalar mean KL.
    """
    # Gather target logits at the same materialized positions
    B, L, M = selected_logits.shape
    flat_ids = selected_token_ids.reshape(B * L, M)
    flat_target = target_full_logits.reshape(B * L, -1)
    # Per-(B*L)-row gather: t_at_M[i, j] = flat_target[i, flat_ids[i, j]]
    target_at_M = mx.take_along_axis(flat_target, flat_ids, axis=-1).reshape(B, L, M)

    log_p_d = nn.log_softmax(selected_logits, axis=-1)
    log_p_t = nn.log_softmax(target_at_M, axis=-1)
    p_d = mx.exp(log_p_d)
    kl_per_pos = (p_d * (log_p_d - log_p_t)).sum(axis=-1)
    return kl_per_pos.mean()
