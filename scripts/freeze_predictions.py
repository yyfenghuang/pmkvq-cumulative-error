#!/usr/bin/env python
"""Hash and lock predictions/h1_predictions.json.

Writes a sidecar ``h1_predictions.lock.json`` holding the sha256 of the frozen
file. Re-running verifies the hash still matches; a mismatch means the frozen
predictions were edited after the lock, which the gate treats as a hard error.
Predictions must be frozen *before* the first run (todo Section 6).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "predictions" / "h1_predictions.json"
LOCK = REPO / "predictions" / "h1_predictions.lock.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if not PRED.exists():
        print(f"missing {PRED}", file=sys.stderr)
        return 1
    digest = sha256(PRED)
    if LOCK.exists():
        prev = json.loads(LOCK.read_text())["sha256"]
        if prev != digest:
            print("FROZEN PREDICTIONS CHANGED AFTER LOCK", file=sys.stderr)
            print(f"  locked : {prev}", file=sys.stderr)
            print(f"  current: {digest}", file=sys.stderr)
            return 1
        return 0
    LOCK.write_text(json.dumps({"sha256": digest, "file": PRED.name}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
