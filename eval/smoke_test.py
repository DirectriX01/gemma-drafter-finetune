"""Quick smoke test: load target (+ optional drafter), run one inference, print result.

Use this to verify the inference stack works before running a full eval.

    python -m eval.smoke_test          # target only
    python -m eval.smoke_test --spec   # target + assistant drafter (speculative)
"""

from __future__ import annotations

import argparse

from eval.infer import RegexInferencer, ASSISTANT_MODEL_ID


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", action="store_true", help="enable speculative decoding")
    p.add_argument("--nl", default="lines containing only digits", help="NL prompt to test")
    p.add_argument("--max-new-tokens", type=int, default=32)
    args = p.parse_args()

    infer = RegexInferencer(assistant_id=ASSISTANT_MODEL_ID if args.spec else None)
    print(f"device={infer.device} dtype={infer.dtype} spec={args.spec}")

    print(f"\nPrompt: {args.nl!r}")
    res = infer.generate(args.nl, max_new_tokens=args.max_new_tokens)
    print(f"Raw output : {res.raw_text!r}")
    print(f"Regex      : {res.regex!r}")
    print(f"Tokens     : {res.n_new_tokens}")
    print(f"Elapsed    : {res.elapsed_s:.2f}s ({res.tokens_per_s:.1f} tok/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
