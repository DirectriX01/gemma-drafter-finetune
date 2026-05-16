"""Evaluation harness — score a model on a regex-synthesis split.

Usage:
    python -m eval.run --dataset data/processed/KB13.jsonl --baseline 0 --limit 100
    python -m eval.run --dataset data/processed/KB13.jsonl --baseline 1 --limit 100

Outputs `results/baseline{0|1}_<dataset>_<n>.json` with pass@1, tokens/sec, and per-example
records.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from eval.infer import RegexInferencer, ASSISTANT_MODEL_ID, TARGET_MODEL_ID
from verifier import verify


@dataclass
class ExampleResult:
    id: str
    nl: str
    gold_regex: str
    pred_regex: str
    raw_text: str
    correct: bool
    f1: float
    accuracy: float
    compile_error: str | None
    timeout: bool
    elapsed_s: float
    n_new_tokens: int
    tokens_per_s: float


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


def _mean_ci(values: list[float]) -> tuple[float, float]:
    """Return (mean, half-width 95% CI). Uses normal approx, fine for n>=30."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = 1.96 * sd / math.sqrt(len(values))
    return mean, half


def evaluate(
    dataset_path: Path,
    baseline: int,
    *,
    limit: int | None = None,
    out_dir: Path = Path("results"),
    seed: int = 0,
    adapter_path: str | None = None,
    tag: str | None = None,
) -> Path:
    """Run a single baseline over a dataset; write a result JSON; return its path.

    baseline=0: target alone
    baseline=1: target + drafter (vanilla, no adapter)
    baseline>=2: target + drafter + adapter (specify --adapter)
    """
    assistant_id = ASSISTANT_MODEL_ID if baseline > 0 else None
    print(f"Loading models (target={TARGET_MODEL_ID}, assistant={assistant_id}, "
          f"adapter={adapter_path}) ...")
    infer = RegexInferencer(assistant_id=assistant_id, adapter_path=adapter_path)
    print(f"Device: {infer.device}, dtype: {infer.dtype}")

    examples = _load_jsonl(dataset_path, limit=limit)
    print(f"Evaluating {len(examples)} examples from {dataset_path}")

    results: list[ExampleResult] = []
    t_total_start = time.perf_counter()
    for ex in tqdm(examples, desc=f"baseline{baseline}", unit="ex"):
        gen = infer.generate(ex["nl"])
        vr = verify(
            gen.regex,
            pos_examples=ex["pos_examples"],
            neg_examples=ex["neg_examples"],
        )
        results.append(
            ExampleResult(
                id=ex["id"],
                nl=ex["nl"],
                gold_regex=ex["gold_regex"],
                pred_regex=gen.regex,
                raw_text=gen.raw_text,
                correct=vr.correct,
                f1=vr.f1,
                accuracy=vr.accuracy,
                compile_error=vr.compile_error,
                timeout=vr.timeout,
                elapsed_s=gen.elapsed_s,
                n_new_tokens=gen.n_new_tokens,
                tokens_per_s=gen.tokens_per_s,
            )
        )
    wall_elapsed = time.perf_counter() - t_total_start

    pass_at_1 = sum(r.correct for r in results) / len(results)
    f1_mean, f1_ci = _mean_ci([r.f1 for r in results])
    tps_mean, tps_ci = _mean_ci([r.tokens_per_s for r in results])
    compile_err_rate = sum(r.compile_error is not None for r in results) / len(results)
    timeout_rate = sum(r.timeout for r in results) / len(results)
    new_tokens_mean = statistics.mean(r.n_new_tokens for r in results)

    summary = {
        "dataset": str(dataset_path),
        "baseline": baseline,
        "n": len(results),
        "seed": seed,
        "device": infer.device,
        "dtype": str(infer.dtype),
        "wall_elapsed_s": wall_elapsed,
        "pass_at_1": pass_at_1,
        "f1_mean": f1_mean,
        "f1_ci95": f1_ci,
        "tokens_per_s_mean": tps_mean,
        "tokens_per_s_ci95": tps_ci,
        "new_tokens_mean": new_tokens_mean,
        "compile_error_rate": compile_err_rate,
        "timeout_rate": timeout_rate,
        "examples": [asdict(r) for r in results],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    name_tag = tag or f"baseline{baseline}"
    out_path = out_dir / f"{name_tag}_{dataset_path.stem}_n{len(results)}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== Baseline {baseline} on {dataset_path.name} (n={len(results)}) ===")
    print(f"pass@1         : {pass_at_1:.3f}")
    print(f"F1 mean        : {f1_mean:.3f} ± {f1_ci:.3f}")
    print(f"tokens/sec     : {tps_mean:.2f} ± {tps_ci:.2f}")
    print(f"new tokens avg : {new_tokens_mean:.1f}")
    print(f"compile errors : {compile_err_rate:.1%}")
    print(f"timeouts       : {timeout_rate:.1%}")
    print(f"-> {out_path}")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--baseline", required=True, type=int, choices=[0, 1, 2, 3, 4])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adapter", type=str, default=None,
                   help="path to .npz adapter checkpoint (required for baselines >= 2)")
    p.add_argument("--tag", type=str, default=None,
                   help="optional name tag for output filename (defaults to 'baseline{N}')")
    args = p.parse_args()
    evaluate(
        args.dataset, args.baseline,
        limit=args.limit, out_dir=args.out_dir, seed=args.seed,
        adapter_path=args.adapter, tag=args.tag,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
