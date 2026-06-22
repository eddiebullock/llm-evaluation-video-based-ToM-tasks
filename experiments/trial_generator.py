from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg"}


def _infer_group(label: str) -> str:
    """
    Infer a coarse "mental state group" from a label string.

    Datasets often encode group/category information in the label itself (e.g., prefix tokens).
    This function implements a heuristic grouping so we can stratify foil sampling.
    """
    s = label.strip()
    if not s:
        return "unknown"
    # Split on common separators; use the first non-empty token as group.
    tokens = [t for t in re.split(r"[\s:_/\-]+", s) if t]
    return tokens[0].casefold() if tokens else "unknown"


@dataclass(frozen=True)
class StimulusItem:
    """
    Input stimulus item for forced-choice trial generation.

    Attributes:
        correct_label: Ground-truth label for this stimulus.
        video_path: Path to the video stimulus.
        audio_path: Optional path to the audio stimulus.
    """

    correct_label: str
    video_path: str
    audio_path: Optional[str] = None


class TrialGenerator:
    """
    Generate four-alternative forced-choice trials for mental state recognition evaluation.

    The generator:
    - Builds an inventory of available labels from a `data_dir`
    - Samples 3 foil labels per stimulus using stratified sampling where possible
    - Shuffles candidate labels deterministically given a fixed seed
    - Saves/loads trials as JSON for reproducibility
    """

    def __init__(self, dataset_name: str, data_dir: str, random_seed: int = 42) -> None:
        """
        Args:
            dataset_name: Either 'eu_emotion' or 'mindreading'.
            data_dir: Path to directory containing video/audio files.
            random_seed: Seed for deterministic sampling and shuffling.
        """
        if dataset_name not in {"eu_emotion", "mindreading"}:
            raise ValueError("dataset_name must be either 'eu_emotion' or 'mindreading'")

        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.random_seed = random_seed

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        self._inventory: Optional[Dict[str, List[str]]] = None
        self._label_groups: Optional[Dict[str, str]] = None

    def _load_stimulus_inventory(self) -> Dict[str, List[str]]:
        """
        Load all available mental state labels and associated stimulus file paths from `data_dir`.

        Expected layout (flexible):
        - Labels are inferred from the immediate parent directory name of each video file.
          For example: `data_dir/<label>/<anything>.mp4`

        Returns:
            Dict mapping label -> list of video file paths (as strings).
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir does not exist: {self.data_dir}")

        inventory: Dict[str, List[str]] = {}
        for p in self.data_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in VIDEO_EXTS:
                continue
            label = p.parent.name
            inventory.setdefault(label, []).append(str(p))

        if not inventory:
            raise ValueError(f"No video files found under data_dir={self.data_dir}")

        # Build group map for stratification.
        self._label_groups = {label: _infer_group(label) for label in inventory.keys()}
        return inventory

    def _ensure_inventory(self) -> Dict[str, List[str]]:
        if self._inventory is None:
            self._inventory = self._load_stimulus_inventory()
        return self._inventory

    def _sample_foils(self, correct_label: str, k: int = 3) -> List[str]:
        """
        Stratified random sampling of foil labels.

        Foils are sampled from the complete mental state inventory excluding `correct_label`.
        The method attempts to draw foils from different inferred mental state groups where possible.

        Args:
            correct_label: The ground-truth label to exclude.
            k: Number of foils to sample (default 3).

        Returns:
            List of `k` foil labels.
        """
        inventory = self._ensure_inventory()
        labels = [lbl for lbl in inventory.keys() if lbl != correct_label]
        if len(labels) < k:
            raise ValueError(f"Not enough labels to sample {k} foils (available={len(labels)})")

        groups = self._label_groups or {lbl: _infer_group(lbl) for lbl in inventory.keys()}
        correct_group = groups.get(correct_label, _infer_group(correct_label))

        # Prefer foils from groups different from the correct label's group.
        labels_diff_group = [lbl for lbl in labels if groups.get(lbl, "") != correct_group]

        selected: List[str] = []

        # Step 1: try to maximize group diversity.
        if labels_diff_group:
            # Group -> labels mapping for candidates
            by_group: Dict[str, List[str]] = {}
            for lbl in labels_diff_group:
                by_group.setdefault(groups.get(lbl, "unknown"), []).append(lbl)

            group_keys = list(by_group.keys())
            random.shuffle(group_keys)

            for g in group_keys:
                if len(selected) >= k:
                    break
                options = by_group[g]
                random.shuffle(options)
                pick = options[0]
                if pick not in selected:
                    selected.append(pick)

        # Step 2: fill remaining uniformly at random from all remaining labels.
        if len(selected) < k:
            remaining = [lbl for lbl in labels if lbl not in selected]
            random.shuffle(remaining)
            selected.extend(remaining[: (k - len(selected))])

        return selected[:k]

    def generate_trials(self, stimulus_items: Sequence[StimulusItem]) -> List[Dict[str, Any]]:
        """
        Generate four-alternative forced-choice trials.

        Args:
            stimulus_items: Iterable of stimulus items, each with correct_label, video_path, and optional audio_path.

        Returns:
            List of dicts, each containing:
            - trial_id
            - video_path
            - audio_path
            - correct_label
            - candidate_labels (shuffled list of 4)
            - dataset
        """
        self._ensure_inventory()

        trials: List[Dict[str, Any]] = []
        for idx, item in enumerate(stimulus_items):
            foils = self._sample_foils(item.correct_label, k=3)
            candidate_labels = [item.correct_label, *foils]

            # Deterministic shuffle given the fixed seed and consistent iteration order.
            random.shuffle(candidate_labels)

            trials.append(
                {
                    "trial_id": f"{self.dataset_name}_{idx:06d}",
                    "video_path": item.video_path,
                    "audio_path": item.audio_path,
                    "correct_label": item.correct_label,
                    "candidate_labels": candidate_labels,
                    "dataset": self.dataset_name,
                }
            )

        return trials

    def save_trials(self, trials: Sequence[Dict[str, Any]], output_path: str) -> None:
        """
        Save generated trials to a JSON file for reproducibility.

        Args:
            trials: Trial definitions as returned by `generate_trials`.
            output_path: File path to write JSON to.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(list(trials), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_trials(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Load previously saved trial definitions from JSON.

        Args:
            input_path: JSON path previously written by `save_trials`.

        Returns:
            List of trial dicts.
        """
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"Trials JSON not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Trials JSON must be a list of trial dicts")
        return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demonstration with dummy paths (will require a real data_dir to load inventory).
    gen = TrialGenerator(dataset_name="eu_emotion", data_dir="data", random_seed=42)

    dummy_items = [
        StimulusItem(correct_label="happy", video_path="data/happy/clip_001.mp4", audio_path="data/happy/clip_001.wav"),
        StimulusItem(correct_label="sad", video_path="data/sad/clip_002.mp4", audio_path="data/sad/clip_002.wav"),
    ]

    try:
        trials = gen.generate_trials(dummy_items)
        print("Generated trials:", len(trials))
        print(json.dumps(trials[0], indent=2))
    except Exception as e:
        print("Demo failed (expected if data_dir is not populated):", str(e))
