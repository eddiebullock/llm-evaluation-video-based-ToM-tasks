from __future__ import annotations

"""
Load evaluation summaries and CSVs for analysis.

Merges publication_repo reruns (results/full_run/) with imported legacy video-only
runs (results/legacy_import/) for GPT/Claude and legacy Flash MR video-only.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from .study_config import EXCLUDED_MODELS, VIDEO_ONLY_LEGACY_KEYS, VIDEO_ONLY_MODELS
from .mr_fair_subset import (
    fair_mr_video_trial_ids,
    filter_mr_video_only_fair,
    mr_fair_subset_note,
    mr_video_only_counts_on_trials,
)

Counts = Tuple[int, int]  # (n_correct, n_total)
ResultKey = Tuple[str, str, str]


def _read_summaries_from_dir(
    root: Path,
    *,
    excluded: Set[str],
) -> Dict[ResultKey, Counts]:
    out: Dict[ResultKey, Counts] = {}
    if not root.exists():
        return out
    for path in sorted(root.glob("*_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        model = str(data["model"])
        if model in excluded:
            continue
        dataset = str(data["dataset"])
        condition = str(data["condition"])
        n_total = int(data["n_valid"])
        accuracy = float(data["accuracy"])
        n_correct = int(round(accuracy * n_total))
        out[(model, dataset, condition)] = (n_correct, n_total)
    return out


def _nested_from_flat(flat: Dict[ResultKey, Counts]) -> Dict[str, Dict[str, Dict[str, Counts]]]:
    nested: Dict[str, Dict[str, Dict[str, Counts]]] = {}
    for (model, dataset, condition), counts in flat.items():
        nested.setdefault(model, {}).setdefault(dataset, {})[condition] = counts
    return nested


def load_results_from_summaries(
    results_dir: str | Path | None = None,
    *,
    legacy_dir: str | Path | None = None,
    exclude_models: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Counts]]]:
    """
    Build nested counts from summary JSON files.

    Merges legacy_import (if present) with full_run. For Mindreading video-only
    cross-model tables, legacy counts are preferred for keys in VIDEO_ONLY_LEGACY_KEYS.
  """
    excluded: Set[str] = set(exclude_models if exclude_models is not None else EXCLUDED_MODELS)
    repo_root = Path(__file__).resolve().parent.parent
    full_run = Path(results_dir) if results_dir is not None else repo_root / "results" / "full_run"
    legacy = Path(legacy_dir) if legacy_dir is not None else repo_root / "results" / "legacy_import"

    legacy_flat = _read_summaries_from_dir(legacy, excluded=excluded)
    full_flat = _read_summaries_from_dir(full_run, excluded=excluded)

    merged: Dict[ResultKey, Counts] = {}
    merged.update(legacy_flat)
    for key, counts in full_flat.items():
        if key in VIDEO_ONLY_LEGACY_KEYS and key in legacy_flat:
            continue
        merged[key] = counts

    if not merged:
        raise FileNotFoundError(
            f"No summaries found in {full_run} or {legacy}. "
            "Run evaluations and/or: python analysis/import_legacy_results.py"
        )

    return _nested_from_flat(merged)


def apply_mr_fair_video_subset(
    results: Dict[str, Dict[str, Dict[str, Counts]]],
    results_df: pd.DataFrame,
    *,
    models: Iterable[str] | None = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, Counts]]], Set[str], str]:
    """
    Replace MR video_only counts with the intersection of video-evaluated trials.

    Returns (patched_results, common_trial_ids, note).
    """
    model_list = [m for m in (models if models is not None else VIDEO_ONLY_MODELS) if m in results]
    common_ids = fair_mr_video_trial_ids(results_df, model_list)
    note = mr_fair_subset_note(common_ids)

    patched = {m: {ds: dict(cond) for ds, cond in per_ds.items()} for m, per_ds in results.items()}
    for model in model_list:
        if model not in patched:
            continue
        n_correct, n_total = mr_video_only_counts_on_trials(results_df, model, common_ids)
        patched[model].setdefault("mindreading", {})["video_only"] = (n_correct, n_total)

    return patched, common_ids, note


def load_results_for_analysis(
    results_dir: str | Path | None = None,
    *,
    legacy_dir: str | Path | None = None,
    exclude_models: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, Counts]]], pd.DataFrame, Dict[str, Any]]:
    """
    Load summaries, apply MR video_only fair subset, return (counts, filtered_df, metadata).
    """
    results = load_results_from_summaries(results_dir, legacy_dir=legacy_dir, exclude_models=exclude_models)
    results_df = load_results_csvs(results_dir, exclude_models=exclude_models)
    results_df = filter_mr_video_only_fair(results_df)
    results, common_ids, note = apply_mr_fair_video_subset(results, results_df)
    meta = {
        "mr_video_only_fair_subset": {
            "n_trials": len(common_ids),
            "models": list(VIDEO_ONLY_MODELS),
            "note": note,
        }
    }
    return results, results_df, meta


def load_results_csvs(
    results_dirs: str | Path | Iterable[str | Path] | None = None,
    *,
    exclude_models: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load and concatenate ``*_results.csv`` from one or more result directories."""
    excluded: Set[str] = set(exclude_models if exclude_models is not None else EXCLUDED_MODELS)
    repo_root = Path(__file__).resolve().parent.parent

    if results_dirs is None:
        dirs = [repo_root / "results" / "legacy_import", repo_root / "results" / "full_run"]
    elif isinstance(results_dirs, (str, Path)):
        dirs = [Path(results_dirs)]
    else:
        dirs = [Path(d) for d in results_dirs]

    frames: List[pd.DataFrame] = []
    seen_keys: Set[ResultKey] = set()

    for root in dirs:
        if not root.exists():
            continue
        for path in sorted(root.glob("*_results.csv")):
            if any(path.name.startswith(f"{m}_") for m in excluded):
                continue
            parts = path.stem.replace("_results", "").split("_")
            if len(parts) < 3:
                continue
            condition = parts[-1]
            dataset = parts[-2]
            model = "_".join(parts[:-2])
            key = (model, dataset, condition)
            if key in seen_keys:
                continue
            if key in VIDEO_ONLY_LEGACY_KEYS and root.name == "full_run" and (repo_root / "results" / "legacy_import" / path.name).exists():
                continue

            df = pd.read_csv(path)
            if "model" in df.columns and df["model"].isin(excluded).any():
                df = df[~df["model"].isin(excluded)]
            if df.empty:
                continue
            df["source_file"] = path.name
            frames.append(df)
            seen_keys.add(key)

    if not frames:
        raise FileNotFoundError(f"No '*_results.csv' files left after exclusions in: {dirs}")

    return pd.concat(frames, ignore_index=True)


def summarize_loaded_results(results: Dict[str, Any]) -> str:
    """Human-readable inventory of loaded summary counts."""
    lines: List[str] = []
    for model in sorted(results):
        for dataset in sorted(results[model]):
            for condition in sorted(results[model][dataset]):
                n_correct, n_total = results[model][dataset][condition]
                acc = 100.0 * n_correct / n_total if n_total else 0.0
                lines.append(f"  {model} | {dataset} | {condition}: {acc:.2f}% ({n_correct}/{n_total})")
    return "\n".join(lines)
