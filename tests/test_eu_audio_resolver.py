from __future__ import annotations

from pathlib import Path

from experiments.eu_audio_resolver import (
    build_uk_voice_index,
    normalize_emotion_label,
    resolve_eu_multimodal_audio,
    resolve_uk_voice_by_label,
)


def test_normalize_emotion_label_collapses_dashes_and_spaces():
    assert normalize_emotion_label("Angry - Low Intensity") == "angry low intensity"
    assert normalize_emotion_label("Afraid-Low Intensity") == "afraid low intensity"


def test_build_uk_voice_index_prefers_fixed_before_original(tmp_path: Path):
    voices = tmp_path / "EU Emotion - UK Voices"
    fixed = voices / "Fixed - amplified volume" / "Afraid"
    orig = voices / "Original" / "Afraid"
    fixed.mkdir(parents=True)
    orig.mkdir(parents=True)
    (fixed / "fixed_a.mp3").write_bytes(b"a")
    (orig / "orig_b.mp3").write_bytes(b"b")

    index = build_uk_voice_index(voices)
    assert [p.name for p in index["afraid"]] == ["fixed_a.mp3", "orig_b.mp3"]


def test_resolve_uk_voice_by_label_is_deterministic(tmp_path: Path):
    voices = tmp_path / "EU Emotion - UK Voices" / "Original" / "Sneaky"
    voices.mkdir(parents=True)
    for name in ("a.mp3", "b.mp3", "c.mp3"):
        (voices / name).write_bytes(b"x")

    ap1, _ = resolve_uk_voice_by_label(
        emotion_label="sneaky", base_data_dir=tmp_path, trial_id="eu_trial_1", seed=42
    )
    ap2, _ = resolve_uk_voice_by_label(
        emotion_label="sneaky", base_data_dir=tmp_path, trial_id="eu_trial_1", seed=42
    )
    ap3, _ = resolve_uk_voice_by_label(
        emotion_label="sneaky", base_data_dir=tmp_path, trial_id="eu_trial_2", seed=42
    )
    assert ap1 == ap2
    assert ap1 is not None
    assert ap3 is not None


def test_resolve_eu_multimodal_audio_uses_sidecar_when_present(tmp_path: Path):
    video = tmp_path / "faces" / "Happy" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"vid")
    sidecar = video.parent / "clip.wav"
    sidecar.write_bytes(b"wav")

    ap, rule = resolve_eu_multimodal_audio(
        video,
        emotion_label="happy",
        base_data_dir=tmp_path,
        trial_id="t1",
        seed=0,
    )
    assert ap == sidecar
    assert rule == "same_stem"
