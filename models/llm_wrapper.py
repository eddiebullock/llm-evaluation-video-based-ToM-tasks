from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .preprocessing import encode_audio, extract_frames

logger = logging.getLogger(__name__)

# Matched to Gemini 3 Flash full_run settings (see _call_google_gemini).
STUDY_TEMPERATURE = 0.1
STUDY_MAX_OUTPUT_TOKENS = 1000


SUPPORTED_MODELS = {
    "gemini-3-pro",
    "gemini-3-flash",
    "gpt-5",
    "gpt-5-mini",
    "claude-opus-4-5",
}

GEMINI_API_MODEL_IDS: Dict[str, str] = {
    # Map study short names to Google API model ids (see `list_models()` for this key).
    "gemini-3-flash": "models/gemini-3-flash-preview",
    # gemini-3-pro-preview returns 404 as of 2026-06-04; 3.1 pro preview is the successor.
    "gemini-3-pro": "models/gemini-3.1-pro-preview",
}

def _find_dotenv_path() -> Optional[Path]:
    """
    Find a dotenv file containing API keys.

    Search order:
    1) `.env` in current working directory (load_dotenv default behavior)
    2) `.env` in parent directories of this file (up to repo root)
    3) legacy location used elsewhere in this monorepo:
       `experiments/cam_human_like/training/.env`
    """
    # (1) Let load_dotenv() handle cwd by default; here we only return explicit fallbacks.
    here = Path(__file__).resolve()
    # publication_repo/models/llm_wrapper.py -> publication_repo -> mr_ts_play
    candidates: List[Path] = []
    for p in [here.parent.parent.parent, here.parent.parent, here.parent]:
        candidates.append(p / ".env")

    # Walk upwards a bit more defensively
    cur = here.parent
    for _ in range(6):
        candidates.append(cur / ".env")
        if cur.parent == cur:
            break
        cur = cur.parent

    # (3) Known legacy path within this workspace
    repo_root_guess = here.parent.parent.parent  # .../mr_ts_play
    candidates.append(repo_root_guess / "experiments" / "cam_human_like" / "training" / ".env")

    for c in candidates:
        try:
            if c.exists() and c.is_file():
                return c
        except Exception:
            continue
    return None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _guess_audio_mime(audio_path: str) -> str:
    ext = Path(audio_path).suffix.lower()
    if ext == ".wav":
        return "audio/wav"
    if ext in {".mp3", ".mpeg"}:
        return "audio/mpeg"
    if ext in {".m4a", ".mov"}:
        return "audio/mp4"
    if ext == ".ogg":
        return "audio/ogg"
    return "audio/wav"


class LLMWrapper:
    """
    Core API wrapper for four-alternative forced-choice mental state recognition.

    This wrapper supports:
    - Gemini 3 Pro / Gemini 3 Flash (video + audio; temperature=STUDY_TEMPERATURE)
    - GPT-5 / GPT-5 Mini (video only; temperature omitted — OpenAI API only allows default)
    - Claude Opus 4.5 (video only; temperature=STUDY_TEMPERATURE)

    All models use max output tokens = STUDY_MAX_OUTPUT_TOKENS (1000) where the API permits.

    Caching:
    - Successful calls are cached on disk as JSON, keyed by model/video/audio/labels/prompt hash.
    - Failed calls (predicted_label is None) are NOT cached.
    """

    def __init__(self, model_name: str, cache_dir: str, prompt_template: str) -> None:
        """
        Args:
            model_name: One of SUPPORTED_MODELS.
            cache_dir: Directory for JSON cache files.
            prompt_template: Prompt template string containing `{labels}` placeholder.
        """
        # Load env keys from cwd if present, then from fallback locations in this repo.
        load_dotenv()
        dotenv_path = _find_dotenv_path()
        if dotenv_path is not None:
            load_dotenv(dotenv_path=dotenv_path, override=False)

        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model_name={model_name!r}. Supported: {sorted(SUPPORTED_MODELS)}")

        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_template = prompt_template

        self._google_api_key = os.getenv("GOOGLE_API_KEY")
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        self._anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        if self.model_name.startswith("gemini") and not self._google_api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment/.env")
        if self.model_name.startswith("gpt") and not self._openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment/.env")
        if self.model_name.startswith("claude") and not self._anthropic_api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY in environment/.env")

    def _get_cache_key(
        self,
        *,
        video_path: Optional[str],
        audio_path: Optional[str],
        candidate_labels: List[str],
    ) -> str:
        """
        Generate a deterministic cache key.

        Key is computed from:
        - model_name
        - video_path
        - audio_path (or None)
        - sorted candidate labels
        - hash of prompt template

        The key MUST NOT include temperature because temperature is fixed per model.
        """
        prompt_hash = _sha256_text(self.prompt_template)
        labels_sorted = sorted(candidate_labels, key=lambda s: s.casefold())
        payload = {
            "model_name": self.model_name,
            "video_path": video_path,
            "audio_path": audio_path,
            "candidate_labels": labels_sorted,
            "prompt_hash": prompt_hash,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Load a cached result if present.

        Returns:
            Cached dict if found, else None.
        """
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to read cache file: %s", str(path))
            return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """
        Save a successful result to the cache as JSON.

        Failed API calls (where predicted_label is None) are NOT cached.
        """
        if result.get("predicted_label") is None:
            return

        path = self._cache_path(cache_key)
        try:
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            logger.exception("Failed to write cache file: %s", str(path))

    def _parse_response(self, raw_text: str, candidate_labels: List[str]) -> Optional[str]:
        """
        Extract the predicted label from model output.

        Strategy:
        1) First exact case-insensitive match against candidate labels.
        2) Then case-insensitive substring search as fallback.

        Returns:
            The matched label (original casing from candidate_labels) or None if no match is found.
        """
        if not raw_text:
            return None

        text_cf = raw_text.casefold()
        labels_by_cf = {lbl.casefold(): lbl for lbl in candidate_labels}

        # Exact match against tokenized lines / common "EMOTION: X" pattern
        for line in raw_text.splitlines():
            candidate = line.strip()
            if ":" in candidate:
                candidate = candidate.split(":", 1)[1].strip()
            cand_cf = candidate.casefold()
            if cand_cf in labels_by_cf:
                return labels_by_cf[cand_cf]

        # Exact match anywhere
        for lbl in candidate_labels:
            if text_cf.strip() == lbl.casefold():
                return lbl

        # Substring fallback
        for lbl in candidate_labels:
            if lbl.casefold() in text_cf:
                return lbl

        return None

    def _call_google_gemini(self, *, prompt: str, video_path: Optional[str], audio_path: Optional[str]) -> str:
        """
        Call the Google Generative AI API (Gemini) with video frames and optional audio.

        Settings (study standard — same as all Gemini Flash full_run evaluations):
        - temperature=STUDY_TEMPERATURE (0.1)
        - max_output_tokens=STUDY_MAX_OUTPUT_TOKENS (1000)
        - retry: exponential backoff (max 10 attempts, base delay 3s, max delay 90s)
        - 2.5s delay between requests
        - Safety settings disabled (BLOCK_NONE) as per the study
        """
        import google.generativeai as genai

        genai.configure(api_key=self._google_api_key)

        # Disable safety settings (BLOCK_NONE) as per study configuration.
        try:
            from google.generativeai.types import HarmBlockThreshold, HarmCategory

            safety_settings = [
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            ]
        except Exception:
            safety_settings = []

        api_model_id = GEMINI_API_MODEL_IDS.get(self.model_name, self.model_name)
        model = genai.GenerativeModel(
            model_name=api_model_id,
            safety_settings=safety_settings if safety_settings else None,
        )

        parts: List[Any] = [prompt]

        if video_path is not None:
            frames_b64 = extract_frames(video_path)
            for b64 in frames_b64:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64decode(b64),
                        }
                    }
                )

        if audio_path is not None:
            audio_b64 = encode_audio(audio_path)
            parts.append(
                {
                    "inline_data": {
                        "mime_type": _guess_audio_mime(audio_path),
                        "data": base64.b64decode(audio_b64),
                    }
                }
            )

        generation_config = {
            "temperature": STUDY_TEMPERATURE,
            "max_output_tokens": STUDY_MAX_OUTPUT_TOKENS,
        }

        attempts = 10
        base_delay = 3.0
        max_delay = 90.0

        last_err: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    delay = min(max_delay, base_delay * (2 ** (attempt - 2)))
                    delay = delay * (0.9 + 0.2 * random.random())
                    logger.warning("Gemini retry %s/%s after %.1fs", attempt, attempts, delay)
                    time.sleep(delay)

                time.sleep(2.5)
                resp = model.generate_content(parts, generation_config=generation_config)
                return getattr(resp, "text", "") or ""
            except Exception as e:
                last_err = e
                logger.exception("Gemini call failed (attempt %s/%s)", attempt, attempts)
                # Daily per-model quota (e.g. 250 RPD on free tier): retries won't help for hours.
                err_text = str(e)
                if "ResourceExhausted" in type(e).__name__ or "quota" in err_text.casefold():
                    if "per_day" in err_text.casefold() or "GenerateRequestsPerDay" in err_text:
                        raise RuntimeError(
                            "Gemini daily request quota exceeded for this model; "
                            "stop the run and retry after the quota resets (see ai.dev/rate-limit). "
                            f"Last error: {err_text[:300]}"
                        ) from e

        raise RuntimeError(f"Gemini call failed after {attempts} attempts: {last_err}")

    def _call_openai(self, *, prompt: str, video_path: str) -> str:
        """
        Call the OpenAI API with base64 video frames as images.

        Settings (aligned with Gemini Flash):
        - vision detail='low'
        - max_completion_tokens=STUDY_MAX_OUTPUT_TOKENS
        - temperature=STUDY_TEMPERATURE for non-GPT-5 models
        - gpt-5 / gpt-5-mini: temperature omitted (OpenAI only allows API default)
        """
        from openai import OpenAI

        client = OpenAI(api_key=self._openai_api_key)

        frames_b64 = extract_frames(video_path)
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        for b64 in frames_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                }
            )

        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
        }
        if self.model_name.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = STUDY_MAX_OUTPUT_TOKENS
            # OpenAI GPT-5 family rejects temperature != 1; cannot match Gemini's 0.1.
        else:
            kwargs["max_tokens"] = STUDY_MAX_OUTPUT_TOKENS
            kwargs["temperature"] = STUDY_TEMPERATURE

        resp = client.chat.completions.create(**kwargs)
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return str(resp)

    def _call_anthropic(self, *, prompt: str, video_path: str) -> str:
        """
        Call the Anthropic API with base64 video frames as images.

        Settings (aligned with Gemini Flash):
        - max_tokens=STUDY_MAX_OUTPUT_TOKENS
        - temperature=STUDY_TEMPERATURE
        """
        import anthropic

        client = anthropic.Anthropic(api_key=self._anthropic_api_key)

        frames_b64 = extract_frames(video_path)
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        for b64 in frames_b64:
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                }
            )

        resp = client.messages.create(
            model=self.model_name,
            max_tokens=STUDY_MAX_OUTPUT_TOKENS,
            temperature=STUDY_TEMPERATURE,
            messages=[{"role": "user", "content": content}],
        )
        # anthropic returns a list of content blocks
        try:
            blocks = getattr(resp, "content", []) or []
            texts = [b.text for b in blocks if getattr(b, "type", None) == "text" and getattr(b, "text", None)]
            if texts:
                return "\n".join(texts)
        except Exception:
            pass
        return str(resp)

    def classify(
        self,
        *,
        video_path: Optional[str],
        audio_path: Optional[str],
        candidate_labels: List[str],
    ) -> Dict[str, Any]:
        """
        Classify a stimulus using a four-alternative forced-choice label set.

        Args:
            video_path: Path to video stimulus.
            audio_path: Optional path to audio stimulus (Gemini only).
            candidate_labels: List of candidate emotion labels (expected length=4).

        Returns:
            Dict with keys:
            - predicted_label: str | None
            - raw_response: str
            - model_name: str
            - video_path: str
            - cached: bool
        """
        if not candidate_labels:
            raise ValueError("candidate_labels must be non-empty")

        cache_key = self._get_cache_key(video_path=video_path, audio_path=audio_path, candidate_labels=candidate_labels)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            cached["cached"] = True
            return cached

        labels_block = "\n".join(f"- {lbl}" for lbl in candidate_labels)
        prompt = self.prompt_template.format(labels=labels_block)

        raw_response = ""
        predicted_label: Optional[str] = None

        try:
            if self.model_name.startswith("gemini"):
                raw_response = self._call_google_gemini(prompt=prompt, video_path=video_path, audio_path=audio_path)
            elif self.model_name.startswith("gpt"):
                if video_path is None:
                    raise ValueError("OpenAI models in this repo are video-only; video_path cannot be None")
                raw_response = self._call_openai(prompt=prompt, video_path=video_path)
            elif self.model_name.startswith("claude"):
                if video_path is None:
                    raise ValueError("Anthropic models in this repo are video-only; video_path cannot be None")
                raw_response = self._call_anthropic(prompt=prompt, video_path=video_path)
            else:
                raise ValueError(f"Unsupported model_name: {self.model_name}")

            predicted_label = self._parse_response(raw_response, candidate_labels)
        except Exception as e:
            if "daily request quota exceeded" in str(e).casefold():
                raise
            logger.exception("LLM classification failed for model=%s video=%s", self.model_name, video_path)

        result: Dict[str, Any] = {
            "predicted_label": predicted_label,
            "raw_response": raw_response,
            "model_name": self.model_name,
            "video_path": video_path,
            "audio_path": audio_path,
            "cached": False,
        }

        self._save_to_cache(cache_key, result)
        return result

