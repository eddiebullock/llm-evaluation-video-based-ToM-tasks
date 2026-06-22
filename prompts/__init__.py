from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def _load_text(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


MULTIMODAL_PROMPT: str = _load_text("multimodal_prompt.txt")
VIDEO_ONLY_PROMPT: str = _load_text("video_only_prompt.txt")


def get_prompt(model_name: str, *, condition: str = "auto") -> str:
    """
    Return the appropriate prompt template for a given model name.

    Conditions:
    - "auto": keep legacy behavior (Gemini -> multimodal prompt; others -> video-only prompt)
    - "video_only": force video-only prompt
    - "audio_only": force multimodal prompt (audio is referenced in instructions)
    - "multimodal": force multimodal prompt (video + audio)
    """
    name = (model_name or "").strip().lower()
    c = (condition or "auto").strip().lower()

    if c not in {"auto", "video_only", "audio_only", "multimodal"}:
        raise ValueError(f"Unknown condition={condition!r}. Expected one of: auto, video_only, audio_only, multimodal")

    if c in {"audio_only", "multimodal"}:
        return MULTIMODAL_PROMPT
    if c == "video_only":
        return VIDEO_ONLY_PROMPT

    # auto (legacy default)
    if "gemini" in name:
        return MULTIMODAL_PROMPT
    return VIDEO_ONLY_PROMPT
