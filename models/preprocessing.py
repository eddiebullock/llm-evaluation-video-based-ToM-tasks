from __future__ import annotations

import base64
import os
import subprocess
from typing import Any, Dict, List

import cv2


def extract_frames(video_path: str, num_frames: int = 4) -> List[str]:
    """
    Extract frames from a video and return them as base64-encoded JPEG strings.

    Frames are sampled at 0%, 25%, 50%, and 75% of the total duration using:
        index = i * (total_frames / 4)  for i in {0, 1, 2, 3}

    Notes:
    - Exactly 4 frames are sampled regardless of `num_frames` (kept for API stability).
    - Native video resolution is preserved (e.g., EU-Emotion: 1920x1080, Mindreading: 320x240).

    Args:
        video_path: Path to the input video file.
        num_frames: Kept for compatibility; frames are sampled at four fixed positions.

    Returns:
        List of base64-encoded JPEG strings (one per sampled frame).

    Raises:
        ValueError: If the video cannot be opened, has no frames, or frames cannot be read/encoded.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Fallback: try ffmpeg decode/transcode for legacy codecs (e.g., SVQ1 Mindreading .mov)
        frames = _extract_frames_via_ffmpeg(video_path, num_frames=4)
        if frames:
            return frames
        raise ValueError(f"Could not open video: {video_path}")

    try:
        total_frames_float = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(total_frames_float) if total_frames_float is not None else 0
        if total_frames <= 0:
            raise ValueError(f"Video has no frames (frame_count={total_frames_float}): {video_path}")

        encoded_frames: List[str] = []
        for i in range(4):
            frame_index = int(i * (total_frames / 4))
            frame_index = max(0, min(frame_index, total_frames - 1))

            ok = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            if not ok:
                raise ValueError(f"Failed to seek to frame {frame_index} in video: {video_path}")

            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                raise ValueError(f"Failed to read frame {frame_index} from video: {video_path}")

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            success, jpg_buf = cv2.imencode(".jpg", frame_rgb)
            if not success:
                raise ValueError(f"Failed to JPEG-encode frame {frame_index} from video: {video_path}")

            b64 = base64.b64encode(jpg_buf.tobytes()).decode("utf-8")
            encoded_frames.append(b64)

        return encoded_frames
    finally:
        cap.release()


def _extract_frames_via_ffmpeg(video_path: str, num_frames: int = 4) -> List[str]:
    """
    Fallback frame extraction via ffmpeg for videos OpenCV can't decode.

    Strategy:
    - sample 4 frames at ~evenly spaced timestamps using select='not(mod(n,step))' is tricky without duration;
      instead extract 4 fps=1 frames then downsample if needed.
    - output frames as jpeg bytes to stdout is complex; instead write to a temp dir under workspace.

    This is intended for smoke runs and legacy codecs; prefer OpenCV path when available.
    """
    import tempfile
    from pathlib import Path

    tmpdir = Path(tempfile.mkdtemp(prefix="frames_"))
    out_pattern = str(tmpdir / "frame_%03d.jpg")

    # Extract a small set of frames (up to 12) and then take 4 evenly spaced.
    # -vsync vfr avoids dup frames; -frames:v limits work.
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        "fps=2",
        "-frames:v",
        "12",
        out_pattern,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return []

    jpgs = sorted(tmpdir.glob("frame_*.jpg"))
    if not jpgs:
        return []

    # Pick 4 roughly evenly spaced frames from what we have
    if len(jpgs) <= num_frames:
        selected = jpgs
    else:
        idxs = [int(i * (len(jpgs) - 1) / (num_frames - 1)) for i in range(num_frames)]
        selected = [jpgs[i] for i in idxs]

    encoded: List[str] = []
    for p in selected[:num_frames]:
        data = p.read_bytes()
        encoded.append(base64.b64encode(data).decode("utf-8"))
    return encoded


def encode_audio(audio_path: str) -> str:
    """
    Read an audio file from disk and return it as a base64-encoded string.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Base64 string of the raw audio file bytes.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file exists but cannot be read.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with open(audio_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Return basic metadata for a video file.

    Args:
        video_path: Path to a video file readable by OpenCV.

    Returns:
        Dict with keys:
        - width (int)
        - height (int)
        - fps (float)
        - total_frames (int)

    Raises:
        ValueError: If the video cannot be opened or contains no readable metadata.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if total_frames <= 0 or width <= 0 or height <= 0:
            raise ValueError(
                "Invalid video metadata "
                f"(width={width}, height={height}, fps={fps}, total_frames={total_frames}): {video_path}"
            )

        return {"width": width, "height": height, "fps": fps, "total_frames": total_frames}
    finally:
        cap.release()


if __name__ == "__main__":
    dummy_video_path = "path/to/video.mp4"

    try:
        meta = get_video_metadata(dummy_video_path)
        print("Video metadata:", meta)

        frames = extract_frames(dummy_video_path)
        print("Extracted frames:", len(frames))
        print("First frame base64 prefix:", frames[0][:48] + "..." if frames else "(none)")
    except Exception as e:
        print("Demo failed (expected for dummy path):", str(e))
