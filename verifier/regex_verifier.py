"""Regex verifier — execute a generated regex on (positive, negative) examples and score it.

The binary reward is `all positives match AND no negatives match`. A continuous F1 reward is
also returned for partial-credit shaping during training.

Catastrophic backtracking is bounded by a SIGALRM-based timeout (UNIX only — fine for macOS /
Linux but not portable to Windows). The verifier is single-threaded; do not call from multiple
threads simultaneously without serializing.
"""

from __future__ import annotations

import re
import signal
from dataclasses import dataclass
from typing import Literal, Optional

MatchMode = Literal["fullmatch", "search"]


class _RegexTimeout(Exception):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _RegexTimeout()


@dataclass
class VerificationResult:
    """Result of verifying a regex against examples.

    - `correct`: binary — all positives match AND no negatives match
    - `accuracy`: fraction of correct decisions across all examples
    - `f1`: F1 with positives as the relevant class
    - `compile_error` / `timeout`: failure modes
    """

    correct: bool
    accuracy: float
    f1: float
    n_pos_matched: int
    n_pos_total: int
    n_neg_matched: int
    n_neg_total: int
    compile_error: Optional[str] = None
    timeout: bool = False

    @property
    def reward_binary(self) -> float:
        return 1.0 if self.correct else 0.0

    @property
    def reward_shaped(self) -> float:
        if self.compile_error is not None or self.timeout:
            return 0.0
        return self.f1


def _run_matcher(
    compiled: re.Pattern[str], text: str, mode: MatchMode, timeout_ms: int
) -> Optional[bool]:
    """Run `compiled.fullmatch(text)` or `compiled.search(text)` with a wall-clock timeout.

    Returns None on timeout, True/False otherwise.
    """
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_ms / 1000.0)
    try:
        if mode == "fullmatch":
            return compiled.fullmatch(text) is not None
        return compiled.search(text) is not None
    except _RegexTimeout:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def verify(
    pattern: str,
    pos_examples: list[str],
    neg_examples: list[str],
    *,
    mode: MatchMode = "fullmatch",
    timeout_ms: int = 100,
) -> VerificationResult:
    """Verify a generated regex against positive and negative examples.

    Args:
        pattern: the regex pattern string (Python `re` flavor)
        pos_examples: strings that *should* match
        neg_examples: strings that *should not* match
        mode: "fullmatch" (default — regex must match the whole string) or "search"
        timeout_ms: per-example wall-clock cap, default 100ms

    Returns:
        VerificationResult with binary + shaped reward signals.
    """
    n_pos = len(pos_examples)
    n_neg = len(neg_examples)

    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return VerificationResult(
            correct=False,
            accuracy=0.0,
            f1=0.0,
            n_pos_matched=0,
            n_pos_total=n_pos,
            n_neg_matched=n_neg,  # treat as "all negatives incorrectly matched"
            n_neg_total=n_neg,
            compile_error=str(e),
        )

    n_pos_matched = 0
    n_neg_matched = 0

    for ex in pos_examples:
        r = _run_matcher(compiled, ex, mode, timeout_ms)
        if r is None:
            return _timeout_result(n_pos, n_neg)
        if r:
            n_pos_matched += 1

    for ex in neg_examples:
        r = _run_matcher(compiled, ex, mode, timeout_ms)
        if r is None:
            return _timeout_result(n_pos, n_neg)
        if r:
            n_neg_matched += 1

    correct = (n_pos_matched == n_pos) and (n_neg_matched == 0)
    correct_decisions = n_pos_matched + (n_neg - n_neg_matched)
    accuracy = correct_decisions / max(1, n_pos + n_neg)

    tp = n_pos_matched
    fp = n_neg_matched
    fn = n_pos - n_pos_matched
    f1_denom = 2 * tp + fp + fn
    f1 = (2 * tp) / f1_denom if f1_denom > 0 else 0.0

    return VerificationResult(
        correct=correct,
        accuracy=accuracy,
        f1=f1,
        n_pos_matched=n_pos_matched,
        n_pos_total=n_pos,
        n_neg_matched=n_neg_matched,
        n_neg_total=n_neg,
    )


def _timeout_result(n_pos: int, n_neg: int) -> VerificationResult:
    return VerificationResult(
        correct=False,
        accuracy=0.0,
        f1=0.0,
        n_pos_matched=0,
        n_pos_total=n_pos,
        n_neg_matched=0,
        n_neg_total=n_neg,
        timeout=True,
    )
