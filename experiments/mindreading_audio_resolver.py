from __future__ import annotations

"""
Mindreading audio resolver (item-folder audio, not leakage directory).

The Mindreading dataset in this project is expected to have audio files stored alongside
the video files within each item folder, e.g.:
  .../MindReading/Emotions/07/0700803/<video file>
  .../MindReading/Emotions/07/0700803/<audio file>

This module intentionally does NOT support resolving audio from:
  .../MindReading/Emotions/Audio
because that directory contains known leakage files.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".aif", ".aiff"}
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".webm"}


def resolve_mr_v_video_from_t_stimulus(stimulus_path: Path) -> Optional[Path]:
    """
    For Mindreading audio-only .mov files (T marker), locate the paired V video clip
    in the same item folder (same code prefix + tail label).
    """
    try:
        name = stimulus_path.name
        if "T" not in name or "V" in name:
            return None
        prefix = name[:7]
        tail = name.split("T", 1)[1]
        candidates = sorted(stimulus_path.parent.glob(f"{prefix}*V{tail}"))
        return candidates[0] if candidates else None
    except Exception:
        return None


class LeakageAudioPathError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioResolution:
    trial_id: str
    video_path: str
    audio_path: Optional[str]
    matching_rule: str


def _hard_guard_no_leakage(audio_path: Optional[Path]) -> None:
    if audio_path is None:
        return
    s = str(audio_path)
    if "/Emotions/Audio/" in s.replace("\\", "/"):
        raise LeakageAudioPathError(f"Resolved audio path points to leakage directory: {audio_path}")


def resolve_item_folder_audio(video_path: Path) -> Tuple[Optional[Path], str]:
    """
    Resolve audio for a Mindreading video by searching the video item folder.

    Matching strategy (deterministic):
    1) Prefer audio with the same stem as the video (any supported audio extension).
    2) Else, if there is exactly one audio file in the folder, use it.
    3) Else, fall back to None.
    """
    if not video_path.exists():
        return None, "video_missing"

    item_dir = video_path.parent
    if not item_dir.exists():
        return None, "item_dir_missing"

    # 1) Same-stem sidecar match (rare in Mindreading)
    stem = video_path.stem
    for ext in sorted(AUDIO_EXTS):
        candidate = item_dir / f"{stem}{ext}"
        if candidate.exists():
            _hard_guard_no_leakage(candidate)
            return candidate, "same_stem"

    # 2) Mindreading convention: audio-only .mov files are stored in the same folder, using a 'T' marker.
    # Example:
    #   0700803C4Vcalming.mov (video-only)
    #   0700803Z6Tcalming.mov (audio-only)
    #
    # We match by shared code prefix (first 7 digits) + shared trailing label (after the V/T marker).
    try:
        vname = video_path.name
        if len(vname) >= 10 and "V" in vname:
            prefix = vname[:7]
            tail = vname.split("V", 1)[1]  # e.g., "calming.mov"
            candidates = sorted(item_dir.glob(f"{prefix}*T{tail}"))
            if len(candidates) == 1:
                extracted = extract_audio_from_video(candidates[0])
                if extracted is not None and extracted.exists():
                    _hard_guard_no_leakage(extracted)
                    return extracted, "matched_T_file_extracted"
            elif len(candidates) > 1:
                # Fall back: pick the first deterministically
                extracted = extract_audio_from_video(candidates[0])
                if extracted is not None and extracted.exists():
                    _hard_guard_no_leakage(extracted)
                    return extracted, "matched_T_file_extracted_first"
    except Exception as e:
        logger.warning("T-file matching failed for %s: %s", str(video_path), str(e))

    # 2) Single-audio-in-folder match
    audio_files = sorted([p for p in item_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS])
    if len(audio_files) == 1:
        candidate = audio_files[0]
        _hard_guard_no_leakage(candidate)
        if candidate.suffix.lower() == ".mov":
            extracted = extract_audio_from_video(candidate)
            if extracted is not None and extracted.exists():
                return extracted, "single_audio_in_folder_extracted"
        return candidate, "single_audio_in_folder"

    # 4) If no sidecar audio exists, try extracting an audio stream from the container.
    # Mindreading frequently uses:
    # - video-only .mov clips for face stimuli
    # - audio-only .mov clips for spoken labels (no video stream)
    #
    # We attempt extraction only if ffprobe indicates an audio stream exists.
    if video_path.suffix.lower() in VIDEO_EXTS:
        try:
            if has_audio_stream(video_path):
                extracted = extract_audio_from_video(video_path)
                if extracted is not None and extracted.exists():
                    _hard_guard_no_leakage(extracted)
                    return extracted, "extracted_from_video"
        except Exception as e:
            logger.warning("Audio extraction failed for %s: %s", str(video_path), str(e))

    # 5) None
    return None, "not_found"


def has_audio_stream(video_path: Path) -> bool:
    """
    Return True if ffprobe detects at least one audio stream.
    """
    import json
    import subprocess

    if not video_path.exists():
        return False
    cmd = ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return False
    try:
        d = json.loads(proc.stdout.decode("utf-8", errors="ignore") or "{}")
        streams = d.get("streams", []) or []
        return any(s.get("codec_type") == "audio" for s in streams)
    except Exception:
        return False


def extract_audio_from_video(video_path: Path, *, out_root: Optional[Path] = None) -> Optional[Path]:
    """
    Extract the audio track from a video container to a wav file using ffmpeg.

    Output is cached deterministically under:
      <out_root>/extracted_audio/<video_stem>.wav
    Default out_root is `publication_repo/cache/` inferred relative to this file.
    """
    import subprocess

    if not video_path.exists():
        return None

    if out_root is None:
        # publication_repo/experiments/mindreading_audio_resolver.py -> publication_repo
        pub_root = Path(__file__).resolve().parent.parent
        out_root = pub_root / "cache"

    out_dir = out_root / "extracted_audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{video_path.stem}.wav"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    # ffmpeg command: extract audio only, convert to mono 16kHz wav for API friendliness
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(out_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (code={proc.returncode}): {proc.stderr.decode('utf-8', errors='ignore')[:400]}")

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    return None


def build_audio_mapping_audit(
    trials: Sequence[Dict[str, Any]],
    *,
    base_data_dir: Path,
    max_audit_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build an auditable mapping list from trial definitions.

    Each row includes:
      {trial_id, stimulus_path, resolved_video_path, resolved_audio_path, matching_rule}
    """
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
                    "resolved_video_path": None,
                    "resolved_audio_path": None,
                    "matching_rule": "missing_stimulus_path",
                }
            )
            continue

        vp = Path(stimulus_path)
        if not vp.is_absolute():
            vp = (base_data_dir / vp).resolve()

        audio_path, rule = resolve_item_folder_audio(vp)
        out.append(
            {
                "trial_id": trial_id,
                "stimulus_path": str(stimulus_path),
                "resolved_video_path": str(vp),
                "resolved_audio_path": str(audio_path) if audio_path else None,
                "matching_rule": rule,
            }
        )
    return out


def save_audio_mapping_audit(audit_rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(list(audit_rows), indent=2, ensure_ascii=False), encoding="utf-8")

