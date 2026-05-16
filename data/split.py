"""Deterministic train/eval split of a JSONL dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def split_jsonl(src: Path, eval_frac: float, out_dir: Path, seed: int = 0) -> tuple[int, int]:
    items = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(items)
    n_eval = max(1, int(eval_frac * len(items)))
    eval_items, train_items = items[:n_eval], items[n_eval:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{src.stem}_train.jsonl"
    eval_path = out_dir / f"{src.stem}_eval.jsonl"
    train_path.write_text("\n".join(json.dumps(x) for x in train_items) + "\n")
    eval_path.write_text("\n".join(json.dumps(x) for x in eval_items) + "\n")
    return len(train_items), len(eval_items)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--eval-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    n_tr, n_ev = split_jsonl(args.src, args.eval_frac, args.out_dir, args.seed)
    print(f"train={n_tr}, eval={n_ev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
