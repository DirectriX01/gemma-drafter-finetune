"""Dataset loader for regex-synthesis benchmarks.

Downloads KB13, NL-RX-Synth, and NL-RX-Turk from the deep-regex repo, filters out
DSL-specific syntax (intersection `&`, negation `~(...)`) that Python `re` does not
support, generates positive/negative examples by sampling from the gold regex via
`exrex`, and writes JSONL files.

Output schema (one line per example):
    {
        "id": "<split>_<idx>",
        "nl": "lines starting with a capital letter",
        "gold_regex": "[A-Z].*",
        "pos_examples": ["Hello", "World", ...],
        "neg_examples": ["abc", "123", ...],
    }
"""

from __future__ import annotations

import argparse
import json
import random
import re
import signal
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import exrex
from tqdm import tqdm


class _ExrexTimeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise _ExrexTimeout()

REPO_BASE = "https://raw.githubusercontent.com/nicholaslocascio/deep-regex/master/datasets"
SPLITS = ("KB13", "NL-RX-Synth", "NL-RX-Turk")

# DSL operators not supported by Python `re`.
DSL_INTERSECTION = re.compile(r"(?<!\\)&")
DSL_NEGATION = re.compile(r"(?<!\\)~\(")
# Patterns with too many .* are exrex-pathological (long random strings, low signal).
MULTI_WILDCARD = re.compile(r"\.\*.*\.\*.*\.\*")


@dataclass
class RawExample:
    idx: int
    split: str
    nl: str
    gold_regex: str


def _download(split: str, dest_dir: Path) -> tuple[Path, Path]:
    """Download src.txt and targ.txt for a split. Caches to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_path = dest_dir / "src.txt"
    targ_path = dest_dir / "targ.txt"
    if not src_path.exists():
        urllib.request.urlretrieve(f"{REPO_BASE}/{split}/src.txt", src_path)
    if not targ_path.exists():
        urllib.request.urlretrieve(f"{REPO_BASE}/{split}/targ.txt", targ_path)
    return src_path, targ_path


def _load_raw(split: str, raw_dir: Path) -> list[RawExample]:
    src_path, targ_path = _download(split, raw_dir / split)
    nls = src_path.read_text().splitlines()
    regexes = targ_path.read_text().splitlines()
    assert len(nls) == len(regexes), f"{split}: nl/regex count mismatch"
    return [RawExample(i, split, nl.strip(), r.strip()) for i, (nl, r) in enumerate(zip(nls, regexes))]


def _is_dsl_only(pattern: str) -> bool:
    """Return True if pattern uses DSL operators Python `re` does not support."""
    return bool(DSL_INTERSECTION.search(pattern) or DSL_NEGATION.search(pattern))


def _compiles_in_python_re(pattern: str) -> bool:
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


def _sample_positives(
    pattern: str, n: int, max_tries: int = 20, total_timeout_s: float = 1.0
) -> list[str]:
    """Sample up to `n` unique positive matches for `pattern` via exrex.

    Caps generated string length to 100 chars to avoid pathological cases like `.*`.
    Aborts after `total_timeout_s` wall-clock seconds — exrex can hang on certain patterns.
    """
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, total_timeout_s)
    seen: set[str] = set()
    tries = 0
    try:
        while len(seen) < n and tries < max_tries:
            try:
                s = exrex.getone(pattern, limit=3)
            except _ExrexTimeout:
                return []
            except Exception:  # noqa: BLE001 — exrex can blow up on weird patterns
                return []
            tries += 1
            if s is None or len(s) > 100:
                continue
            seen.add(s)
    except _ExrexTimeout:
        return []
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return list(seen)


def _build_examples(
    raw: list[RawExample], n_pos: int = 5, n_neg: int = 5, seed: int = 0
) -> list[dict]:
    """Filter + materialize examples with pos/neg test strings."""
    rng = random.Random(seed)

    # First pass: filter to Python-re-compatible patterns and sample positives.
    candidates: list[tuple[RawExample, list[str]]] = []
    for ex in tqdm(raw, desc=f"sampling {raw[0].split}", unit="ex"):
        if _is_dsl_only(ex.gold_regex):
            continue
        if MULTI_WILDCARD.search(ex.gold_regex):
            continue  # pathological for exrex
        if not _compiles_in_python_re(ex.gold_regex):
            continue
        pos = _sample_positives(ex.gold_regex, n=n_pos)
        if len(pos) < n_pos:
            continue
        candidates.append((ex, pos))

    if not candidates:
        return []

    # Build negative-example pool by collecting positives from OTHER items.
    # A positive for regex X is very likely a negative for unrelated regex Y.
    all_positives = [p for _, pos in candidates for p in pos]

    results = []
    for ex, pos in candidates:
        compiled = re.compile(ex.gold_regex)
        rng.shuffle(all_positives)
        negs: list[str] = []
        for cand in all_positives:
            if cand in pos:
                continue
            if compiled.fullmatch(cand):
                continue  # actually a positive, skip
            negs.append(cand)
            if len(negs) >= n_neg:
                break
        if len(negs) < n_neg:
            continue
        results.append(
            {
                "id": f"{ex.split}_{ex.idx}",
                "split": ex.split,
                "nl": ex.nl,
                "gold_regex": ex.gold_regex,
                "pos_examples": pos,
                "neg_examples": negs,
            }
        )
    return results


def build_split(split: str, raw_dir: Path, out_path: Path, seed: int = 0) -> dict:
    """Build a single split and write a JSONL file. Returns stats dict."""
    raw = _load_raw(split, raw_dir)
    examples = _build_examples(raw, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return {
        "split": split,
        "raw_count": len(raw),
        "kept_count": len(examples),
        "kept_fraction": len(examples) / max(1, len(raw)),
        "out_path": str(out_path),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw", type=Path)
    p.add_argument("--out-dir", default="data/processed", type=Path)
    p.add_argument("--seed", default=0, type=int)
    args = p.parse_args()

    stats = []
    for split in SPLITS:
        out_path = args.out_dir / f"{split}.jsonl"
        stat = build_split(split, args.raw_dir, out_path, seed=args.seed)
        stats.append(stat)
        print(f"{split:14s}: kept {stat['kept_count']:>5d} / {stat['raw_count']:>5d} "
              f"({stat['kept_fraction']:.1%}) -> {stat['out_path']}")

    summary_path = args.out_dir / "stats.json"
    summary_path.write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
