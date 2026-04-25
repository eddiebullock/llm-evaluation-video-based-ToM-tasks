from __future__ import annotations

from pathlib import Path


def get_prompt(model_name: str) -> str:
    """
    Return the prompt template string for a given model.

    Gemini models use a multimodal prompt (video frames + audio).
    GPT and Claude models use a video-only prompt (video frames only).
    """

    prompts_dir = Path(__file__).resolve().parent

    if model_name.startswith("gemini"):
        prompt_path = prompts_dir / "multimodal_prompt.txt"
    else:
        prompt_path = prompts_dir / "video_only_prompt.txt"

    return prompt_path.read_text(encoding="utf-8")
