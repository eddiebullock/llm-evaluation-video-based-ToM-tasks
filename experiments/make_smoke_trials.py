from __future__ import annotations

"""
Create tiny "smoke test" trial JSONs by slicing existing trial-definition files.

Writes JSON files that preserve the input schema (list or {"trials":[...]}), but
with only the first N trials.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_trials(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("trials"), list):
        return list(data["trials"]), "dict"
    if isinstance(data, list):
        # list of trials
        return list(data), "list"
    raise ValueError(f"Unrecognized trial JSON format: {path}")


def _save_trials(path: Path, trials: List[Dict[str, Any]], fmt: str, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "dict":
        out: Dict[str, Any] = {"trials": trials, **meta}
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    if fmt == "list":
        path.write_text(json.dumps(trials, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    raise ValueError(f"Unknown fmt={fmt!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Slice trial-definition JSONs for smoke testing.")
    p.add_argument("--input", required=True, help="Input trial-definition JSON path")
    p.add_argument("--output", required=True, help="Output trial-definition JSON path")
    p.add_argument("--n", type=int, default=20, help="Number of trials to keep")
    args = p.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    trials, fmt = _load_trials(in_path)
    n = max(1, min(int(args.n), len(trials)))
    sliced = trials[:n]

    meta = {
        "smoke_test": True,
        "source_file": str(in_path),
        "num_trials": n,
    }
    _save_trials(out_path, sliced, fmt, meta)

    print(f"Wrote {n} trials: {out_path}")


if __name__ == "__main__":
    main()

