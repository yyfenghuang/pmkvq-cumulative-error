"""Prompt source for the run scripts.

Kept out of the canonical modules so ``run_experiment`` stays free of a dataset
dependency. ``load_prompts`` reads one prompt per line from a file, or falls
back to a small in-repo default so the scaffold runs without external data.
Replace the default with the paper's long-CoT eval set before reporting numbers
and record the swap in ``notes/deviations.md``.
"""
from __future__ import annotations

from pathlib import Path

_DEFAULT = [
    "Solve step by step, showing every line of reasoning: "
    "what is the remainder when 7^100 is divided by 13?",
    "Work through this methodically, one step at a time: "
    "find all real solutions of x^4 - 5x^2 + 6 = 0.",
    "Think carefully and derive the answer in full: "
    "how many trailing zeros does 100! have?",
    "Reason it out completely before answering: "
    "prove that the sum of the first n odd numbers is n^2.",
]


def load_prompts(path: str | None, n: int) -> list[str]:
    if path and Path(path).exists():
        lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    else:
        lines = _DEFAULT
    if not lines:
        raise SystemExit(f"no prompts found in {path!r}")
    out = []
    i = 0
    while len(out) < n:
        out.append(lines[i % len(lines)])
        i += 1
    return out
