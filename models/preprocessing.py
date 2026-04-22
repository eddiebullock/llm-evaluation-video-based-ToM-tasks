
from __future__ import annotations

import base64
import os
from typing import Any, Dict, List

import cv2

# extract frames from a video and return them as base64-encoded JPEG strings
def extract_frames(video_path: str, num_frames: int = 4) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
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


def encode_audio(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with open(audio_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

# return basic metadata for a video file, not essential for the ToM task but useful for debugging
def get_video_metadata(video_path: str) -> Dict[str, Any]:
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
