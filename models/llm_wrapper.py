
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


SUPPORTED_MODELS = {
    "gemini-3-pro",
    "gemini-3-flash",
    "gpt-5",
    "gpt-5-mini",
    "claude-opus-4-5",
}

# return the hash of a string as a hex string
def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# return audio file extension as MIME type, required by gemini API
def _guess_audio_mime(audio_path: str) -> str:
    ext = Path(audio_path).suffix.lower()
    if ext == ".wav":
        return "audio/wav"
    if ext in {".mp3", ".mpeg"}:
        return "audio/mpeg"
    if ext == ".m4a":
        return "audio/mp4"
    if ext == ".ogg":
        return "audio/ogg"
    return "application/octet-stream"

# core wrapper for ToM task 
class LLMWrapper:
    def __init__(self, model_name: str, cache_dir: str, prompt_template: str) -> None:
        load_dotenv()

        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model_name={model_name!r}. Supported: {sorted(SUPPORTED_MODELS)}")

        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_template = prompt_template

        self._google_api_key = os.getenv("GOOGLE_API_KEY")
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        self._anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        # raise an error if the API key is not set
        if self.model_name.startswith("gemini") and not self._google_api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment/.env")
        if self.model_name.startswith("gpt") and not self._openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment/.env")
        if self.model_name.startswith("claude") and not self._anthropic_api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY in environment/.env")

    # generate a deterministic cache key, cache results to prevent redundant API calls
    def _get_cache_key(
        self,
        *,
        video_path: str,
        audio_path: Optional[str],
        candidate_labels: List[str],
    ) -> str:
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
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to read cache file: %s", str(path))
            return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        if result.get("predicted_label") is None:
            return

        path = self._cache_path(cache_key)
        try:
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            logger.exception("Failed to write cache file: %s", str(path))

    # parse the response from the LLM, return the predicted label
    def _parse_response(self, raw_text: str, candidate_labels: List[str]) -> Optional[str]:
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

    def _call_google_gemini(self, *, prompt: str, video_path: str, audio_path: Optional[str]) -> str:
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

        model = genai.GenerativeModel(
            model_name=self.model_name,
            safety_settings=safety_settings if safety_settings else None,
        )

        frames_b64 = extract_frames(video_path)
        parts: List[Any] = [prompt]

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

        generation_config = {"temperature": 0.1, "max_output_tokens": 1000}

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

        raise RuntimeError(f"Gemini call failed after {attempts} attempts: {last_err}")

    def _call_openai(self, *, prompt: str, video_path: str) -> str:
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
            "max_tokens": 1000,
        }
        if self.model_name == "gpt-5-mini":
            kwargs["temperature"] = 0.1

        resp = client.chat.completions.create(**kwargs)
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return str(resp)

    def _call_anthropic(self, *, prompt: str, video_path: str) -> str:
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
            max_tokens=1000,
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
        video_path: str,
        audio_path: Optional[str],
        candidate_labels: List[str],
    ) -> Dict[str, Any]:
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
                raw_response = self._call_openai(prompt=prompt, video_path=video_path)
            elif self.model_name.startswith("claude"):
                raw_response = self._call_anthropic(prompt=prompt, video_path=video_path)
            else:
                raise ValueError(f"Unsupported model_name: {self.model_name}")

            predicted_label = self._parse_response(raw_response, candidate_labels)
        except Exception:
            logger.exception("LLM classification failed for model=%s video=%s", self.model_name, video_path)

        result: Dict[str, Any] = {
            "predicted_label": predicted_label,
            "raw_response": raw_response,
            "model_name": self.model_name,
            "video_path": video_path,
            "cached": False,
        }

        self._save_to_cache(cache_key, result)
        return result