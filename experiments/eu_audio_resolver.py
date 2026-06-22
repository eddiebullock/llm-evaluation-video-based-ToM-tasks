from __future__ import annotations

"""
EU-Emotions audio resolver.

Face videos live under `emotions N/.../Faces.../<Emotion>/`. UK voice clips live under a
separate tree, typically:

  <data_dir>/EU Emotion - UK Voices/Fixed - amplified volume/<Emotion>/...
  <data_dir>/EU Emotion - UK Voices/Original/<Emotion>/...

For multimodal EU runs we pair each face video trial to a voice clip by normalized emotion
label (`correct_label` / `emotion` field, or the video's parent folder name). This is
emotion-category matching, not actor-matched performance pairing.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".aif", ".aiff"}
UK_VOICES_DIRNAME = "EU Emotion - UK Voices"
UK_VOICES_SUBDIRS = ("Fixed - amplified volume", "Original")

_VOICE_INDEX_CACHE: Dict[str, Dict[str, List[Path]]] = {}


def normalize_emotion_label(label: str) -> str:
    """Normalize labels/folder names for lookup (e.g. 'Angry - Low Intensity' -> 'angry low intensity')."""
    s = label.strip().lower().replace("-", " ")
    return re.sub(r"\s+", " ", s)


def find_eu_voices_root(base_data_dir: Path) -> Optional[Path]:
    """Locate `EU Emotion - UK Voices` under the dataset root (or nearby parents)."""
    base_data_dir = base_data_dir.resolve()
    search_roots = [base_data_dir]
    search_roots.extend(base_data_dir.parents[:4])
    seen: set[Path] = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = root / UK_VOICES_DIRNAME
        if candidate.is_dir():
            return candidate
    return None


def build_uk_voice_index(voices_root: Path) -> Dict[str, List[Path]]:
    """
    Map normalized emotion label -> sorted audio paths.

    `Fixed - amplified volume` is indexed before `Original` so picks prefer fixed clips.
    """
    index: Dict[str, List[Path]] = {}
    for subname in UK_VOICES_SUBDIRS:
        sub = voices_root / subname
        if not sub.is_dir():
            continue
        for folder in sorted(sub.iterdir()):
            if not folder.is_dir():
                continue
            label = normalize_emotion_label(folder.name)
            files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
            if not files:
                continue
            index.setdefault(label, []).extend(files)
    return index


def get_uk_voice_index(base_data_dir: Path) -> Dict[str, List[Path]]:
    voices_root = find_eu_voices_root(base_data_dir)
    if voices_root is None:
        return {}
    cache_key = str(voices_root.resolve())
    if cache_key not in _VOICE_INDEX_CACHE:
        _VOICE_INDEX_CACHE[cache_key] = build_uk_voice_index(voices_root)
    return _VOICE_INDEX_CACHE[cache_key]


def _pick_deterministic(candidates: Sequence[Path], *, trial_id: str, seed: int) -> Path:
    ordered = sorted(candidates, key=lambda p: str(p))
    digest = hashlib.sha256(f"{trial_id}|{seed}".encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(ordered)
    return ordered[idx]


def emotion_label_for_trial(trial: Mapping[str, Any], video_path: Path) -> str:
    for key in ("correct_label", "emotion"):
        val = trial.get(key)
        if val:
            return str(val)
    return video_path.parent.name


def resolve_audio_next_to_video(video_path: Path) -> Tuple[Optional[Path], str]:
    """
    Resolve audio for a EU-Emotions face video by searching the video folder.

    Strategy:
    1) Prefer same-stem audio (any supported audio extension).
    2) Else, if exactly one audio file exists in the folder, use it.
    3) Else, return None.
    """
    if not video_path.exists():
        return None, "video_missing"
    folder = video_path.parent
    if not folder.exists():
        return None, "folder_missing"

    stem = video_path.stem
    for ext in sorted(AUDIO_EXTS):
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate, "same_stem"

    audio_files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if len(audio_files) == 1:
        return audio_files[0], "single_audio_in_folder"

    return None, "not_found"


def resolve_uk_voice_by_label(
    *,
    emotion_label: str,
    base_data_dir: Path,
    trial_id: str,
    seed: int,
) -> Tuple[Optional[Path], str]:
    """Pick a UK voice clip from the emotion-matched folder."""
    norm = normalize_emotion_label(emotion_label)
    index = get_uk_voice_index(base_data_dir)
    if not index:
        return None, "uk_voices_root_missing"

    candidates = index.get(norm)
    if not candidates:
        return None, f"uk_voices_label_missing:{norm}"

    if len(candidates) == 1:
        return candidates[0], "uk_voices_single_match"

    return _pick_deterministic(candidates, trial_id=trial_id, seed=seed), "uk_voices_label_hash_pick"


def resolve_eu_multimodal_audio(
    video_path: Path,
    *,
    emotion_label: str,
    base_data_dir: Path,
    trial_id: str,
    seed: int,
) -> Tuple[Optional[Path], str]:
    """
    Resolve audio for EU multimodal evaluation.

    1) Sidecar audio next to the face video (if present).
    2) UK Voices clip matched by normalized emotion label.
    """
    ap, rule = resolve_audio_next_to_video(video_path)
    if ap is not None:
        return ap, rule

    return resolve_uk_voice_by_label(
        emotion_label=emotion_label,
        base_data_dir=base_data_dir,
        trial_id=trial_id,
        seed=seed,
    )


def build_audio_mapping_audit(
    trials: Sequence[Dict[str, Any]],
    *,
    base_data_dir: Path,
    seed: int = 0,
    max_audit_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, t in enumerate(trials):
        if max_audit_rows is not None and i >= max_audit_rows:
            break
        trial_id = str(t.get("trial_id", f"trial_{i}"))
        stimulus_path = t.get("stimulus_path") or t.get("video_path")
        if not stimulus_path:
            out.append(
                {
                    "trial_id": trial_id,
                    "stimulus_path": None,
                    "emotion_label": None,
                    "resolved_video_path": None,
                    "resolved_audio_path": None,
                    "matching_rule": "missing_stimulus_path",
                }
            )
            continue

        vp = Path(str(stimulus_path))
        if not vp.is_absolute():
            vp = (base_data_dir / vp).resolve()

        if _is_video_file(vp):
            emotion_label = emotion_label_for_trial(t, vp)
            ap, rule = resolve_eu_multimodal_audio(
                vp,
                emotion_label=emotion_label,
                base_data_dir=base_data_dir,
                trial_id=trial_id,
                seed=seed,
            )
        else:
            emotion_label = str(t.get("correct_label") or t.get("emotion") or "")
            ap, rule = None, "stimulus_not_video"

        out.append(
            {
                "trial_id": trial_id,
                "stimulus_path": str(stimulus_path),
                "emotion_label": emotion_label if _is_video_file(vp) else emotion_label or None,
                "resolved_video_path": str(vp) if _is_video_file(vp) else None,
                "resolved_audio_path": str(ap) if ap else None,
                "matching_rule": rule,
            }
        )
    return out


def _is_video_file(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in {".mp4", ".m4v", ".avi", ".webm"}:
        return True
    return ext == ".mov"


def save_audio_mapping_audit(audit_rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(list(audit_rows), indent=2, ensure_ascii=False), encoding="utf-8")
