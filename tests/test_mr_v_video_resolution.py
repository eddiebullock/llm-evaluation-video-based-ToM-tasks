from pathlib import Path

from experiments.mindreading_audio_resolver import resolve_mr_v_video_from_t_stimulus


def test_resolve_mr_v_video_from_t_stimulus_finds_paired_v(tmp_path: Path) -> None:
    folder = tmp_path / "05" / "0507601"
    folder.mkdir(parents=True)
    t_file = folder / "0507601M4Tcommitted.mov"
    v_file = folder / "0507601M4Vcommitted.mov"
    t_file.write_bytes(b"t")
    v_file.write_bytes(b"v")

    assert resolve_mr_v_video_from_t_stimulus(t_file) == v_file


def test_resolve_mr_v_video_from_t_stimulus_returns_none_for_v_file(tmp_path: Path) -> None:
    folder = tmp_path / "05" / "0507601"
    folder.mkdir(parents=True)
    v_file = folder / "0507601M4Vcommitted.mov"
    v_file.write_bytes(b"v")

    assert resolve_mr_v_video_from_t_stimulus(v_file) is None


def test_resolve_mr_v_video_from_t_stimulus_returns_none_when_missing_pair(tmp_path: Path) -> None:
    folder = tmp_path / "05" / "0507601"
    folder.mkdir(parents=True)
    t_file = folder / "0507601M4Tcommitted.mov"
    t_file.write_bytes(b"t")

    assert resolve_mr_v_video_from_t_stimulus(t_file) is None
