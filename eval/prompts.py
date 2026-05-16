"""Prompt templates for regex synthesis from natural language."""

from __future__ import annotations

SYSTEM = (
    "You are a regex expert. Given a natural-language description, output a single Python `re` "
    "regular expression that matches the described strings. Output ONLY the regex pattern — "
    "no prose, no quotes, no code fences. The regex will be evaluated with `re.fullmatch`."
)

FEW_SHOT = [
    (
        "lines that contain only lowercase letters",
        r"[a-z]+",
    ),
    (
        "a 3-digit number followed by a hyphen and a 4-digit number",
        r"\d{3}-\d{4}",
    ),
    (
        "strings starting with a capital letter and ending with a period",
        r"[A-Z].*\.",
    ),
]


def build_prompt(nl: str, *, few_shot: bool = True) -> list[dict]:
    """Build a chat-formatted prompt for regex synthesis."""
    msgs: list[dict] = [{"role": "system", "content": SYSTEM}]
    if few_shot:
        for ex_nl, ex_regex in FEW_SHOT:
            msgs.append({"role": "user", "content": ex_nl})
            msgs.append({"role": "assistant", "content": ex_regex})
    msgs.append({"role": "user", "content": nl})
    return msgs


def extract_regex(text: str) -> str:
    """Extract a regex from the model's raw output.

    Strips code fences, surrounding whitespace, and `re.compile(...)` wrappers if present.
    """
    text = text.strip()
    # Strip code fences ```regex ... ``` or ```python ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 2:
            # drop the first line (```lang) and last line (```)
            inner = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = inner.strip()
    # Strip a single pair of surrounding quotes (single or double)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    # Take only the first line (regex shouldn't span lines)
    text = text.split("\n", 1)[0].strip()
    return text
