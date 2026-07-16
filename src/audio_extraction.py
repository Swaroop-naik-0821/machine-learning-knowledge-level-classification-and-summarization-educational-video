"""
Audio Extraction & Whisper Transcription

This module:
1. Loads an educational video.
2. Uses OpenAI Whisper to generate a transcript.
3. Preserves timestamps for each speech segment.
4. Saves the transcript as JSON.
"""

import os
import whisper
from config import (
    VIDEO_PATH,
    RAW_TRANSCRIPT_JSON,
    WHISPER_MODEL
)
from utils import save_json


def transcribe_video(video_path=VIDEO_PATH):
    """
    Transcribe an educational video using Whisper.

    Parameters
    ----------
    video_path : str
        Path to input video.

    Returns
    -------
    dict
        Whisper transcription output.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"Loading Whisper model ({WHISPER_MODEL})...")

    model = whisper.load_model(WHISPER_MODEL)

    print("Transcribing video...")

    result = model.transcribe(
        video_path,
        verbose=True
    )

    save_json(result, RAW_TRANSCRIPT_JSON)

    print(f"Transcript saved to: {RAW_TRANSCRIPT_JSON}")

    return result


if __name__ == "__main__":
    transcribe_video()
