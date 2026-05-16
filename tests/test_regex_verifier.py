"""Unit tests for the regex verifier — covers 25+ edge cases."""

from __future__ import annotations

import pytest

from verifier import VerificationResult, verify


def test_trivial_correct():
    r = verify(r"\d+", pos_examples=["123", "9"], neg_examples=["abc", ""])
    assert r.correct is True
    assert r.accuracy == 1.0
    assert r.f1 == 1.0
    assert r.reward_binary == 1.0
    assert r.reward_shaped == 1.0


def test_trivial_wrong():
    r = verify(r"[a-z]+", pos_examples=["123"], neg_examples=["abc"])
    assert r.correct is False
    assert r.n_pos_matched == 0
    assert r.n_neg_matched == 1


def test_fullmatch_default_is_strict():
    # `\d+` only matches strings that are entirely digits under fullmatch
    r = verify(r"\d+", pos_examples=["12a3"], neg_examples=[])
    assert r.correct is False
    assert r.n_pos_matched == 0


def test_search_mode_is_lenient():
    r = verify(r"\d+", pos_examples=["abc12def"], neg_examples=["abc"], mode="search")
    assert r.correct is True


def test_anchored_pattern():
    r = verify(r"^abc$", pos_examples=["abc"], neg_examples=["xyzabc", "abcxyz"])
    assert r.correct is True


def test_unanchored_alternation():
    r = verify(r"cat|dog", pos_examples=["cat", "dog"], neg_examples=["bird"])
    assert r.correct is True


def test_invalid_regex_returns_compile_error():
    r = verify("[a-", pos_examples=["a"], neg_examples=[])
    assert r.correct is False
    assert r.compile_error is not None
    assert r.reward_binary == 0.0
    assert r.reward_shaped == 0.0


def test_invalid_regex_unbalanced_paren():
    r = verify("(abc", pos_examples=["abc"], neg_examples=[])
    assert r.compile_error is not None


def test_unicode_word_class():
    r = verify(r"\w+", pos_examples=["café", "naïve"], neg_examples=["  "])
    assert r.correct is True


def test_character_class():
    r = verify(r"[A-Z][a-z]+", pos_examples=["Hello", "World"], neg_examples=["hello", "WORLD"])
    assert r.correct is True


def test_alternation():
    r = verify(r"(cat|dog|bird)", pos_examples=["cat", "dog", "bird"], neg_examples=["fish"])
    assert r.correct is True


def test_capturing_groups():
    r = verify(r"(abc)+", pos_examples=["abc", "abcabc"], neg_examples=["ab", ""])
    assert r.correct is True


def test_negative_lookahead():
    # Match any 3 chars not followed by "abc"
    r = verify(
        r"(?!abc).{3}",
        pos_examples=["xyz", "123"],
        neg_examples=["abc"],
    )
    assert r.correct is True


def test_escaping_dot():
    r = verify(r"\d+\.com", pos_examples=["123.com"], neg_examples=["123xcom"])
    assert r.correct is True


def test_repetition_quantifier():
    r = verify(r"a{3,5}", pos_examples=["aaa", "aaaa", "aaaaa"], neg_examples=["aa", "aaaaaa"])
    assert r.correct is True


def test_case_sensitive_default():
    r = verify(r"abc", pos_examples=["abc"], neg_examples=["ABC"])
    assert r.correct is True


def test_whitespace_class():
    r = verify(r"\s+", pos_examples=[" ", "\t", "  \n"], neg_examples=["abc"])
    assert r.correct is True


def test_empty_pos_examples():
    r = verify(r"\d+", pos_examples=[], neg_examples=["abc"])
    assert r.correct is True  # vacuously: all 0 positives "matched", no negatives matched
    assert r.n_pos_total == 0


def test_empty_neg_examples():
    r = verify(r"\d+", pos_examples=["123"], neg_examples=[])
    assert r.correct is True


def test_both_empty():
    r = verify(r"\d+", pos_examples=[], neg_examples=[])
    assert r.correct is True


def test_partial_correctness_f1():
    # 1/2 positives match, 0/2 negatives match → tp=1, fp=0, fn=1
    r = verify(r"abc", pos_examples=["abc", "xyz"], neg_examples=["123", "456"])
    assert r.correct is False
    assert r.n_pos_matched == 1
    assert r.n_neg_matched == 0
    # F1 = 2*1 / (2*1 + 0 + 1) = 2/3
    assert abs(r.f1 - 2 / 3) < 1e-9
    assert r.reward_binary == 0.0
    assert r.reward_shaped == pytest.approx(2 / 3)


def test_all_wrong_f1_zero():
    r = verify(r"abc", pos_examples=["xyz", "qrs"], neg_examples=["abc"])
    assert r.correct is False
    assert r.f1 == 0.0


def test_accuracy_metric():
    # 1 of 2 pos matched, 1 of 2 neg incorrectly matched → 2 correct / 4 total = 0.5
    r = verify(r"a.*", pos_examples=["abc", "xyz"], neg_examples=["abc", "xyz"])
    # Both "abc" instances match (pos: tp, neg: fp); both "xyz" don't match (pos: fn, neg: tn)
    assert r.n_pos_matched == 1
    assert r.n_neg_matched == 1
    assert r.accuracy == 0.5


def test_catastrophic_backtracking_times_out():
    # Classic backtracking bomb: (a+)+b on long "a" string with no terminating b
    r = verify(
        r"(a+)+b",
        pos_examples=["a" * 30],
        neg_examples=[],
        timeout_ms=50,
    )
    assert r.timeout is True
    assert r.correct is False
    assert r.reward_binary == 0.0
    assert r.reward_shaped == 0.0


def test_empty_string_examples():
    r = verify(r".*", pos_examples=[""], neg_examples=[])
    assert r.correct is True  # .* matches empty under fullmatch


def test_digit_followed_by_letters():
    r = verify(
        r"\d+[a-z]+",
        pos_examples=["1a", "42abc"],
        neg_examples=["a1", "abc", "123"],
    )
    assert r.correct is True


def test_zero_or_more():
    r = verify(r"a*", pos_examples=["", "a", "aaa"], neg_examples=["b"])
    assert r.correct is True


def test_one_or_more():
    r = verify(r"a+", pos_examples=["a", "aaa"], neg_examples=["", "b"])
    assert r.correct is True


def test_reward_properties_consistent():
    r = VerificationResult(
        correct=True,
        accuracy=1.0,
        f1=1.0,
        n_pos_matched=1,
        n_pos_total=1,
        n_neg_matched=0,
        n_neg_total=1,
    )
    assert r.reward_binary == 1.0
    assert r.reward_shaped == 1.0


def test_compile_error_zero_reward():
    r = verify("[unclosed", pos_examples=["a"], neg_examples=["b"])
    assert r.reward_binary == 0.0
    assert r.reward_shaped == 0.0
    assert r.compile_error is not None
