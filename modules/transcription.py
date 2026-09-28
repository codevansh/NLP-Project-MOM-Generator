import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import streamlit as st


SUPPORTED_AUDIO_TYPES = {".wav", ".mp3", ".m4a"}


@st.cache_resource
def _load_whisper_model():
    """Load Whisper once and share it across Streamlit reruns."""
    from faster_whisper import WhisperModel

    return WhisperModel("base", device="cpu", compute_type="int8")


@contextmanager
def temporary_audio_file(audio_bytes: bytes, suffix: str):
    """Write uploaded bytes outside the project and remove them on exit."""
    suffix = suffix.lower()
    if suffix not in SUPPORTED_AUDIO_TYPES:
        raise ValueError("Upload a WAV, MP3, or M4A audio file.")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as audio_file:
            audio_file.write(audio_bytes)
            temporary_path = audio_file.name
        yield temporary_path
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def transcribe_audio(audio_path: str) -> list[dict[str, float | str]]:
    """Return Whisper segments with their timestamps for speaker alignment."""
    path = Path(audio_path)
    if path.suffix.lower() not in SUPPORTED_AUDIO_TYPES:
        raise ValueError("Unsupported audio type. Use WAV, MP3, or M4A.")
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        model = _load_whisper_model()
        segments, _ = model.transcribe(str(path))
        transcript_segments = [
            {"start": float(segment.start), "end": float(segment.end), "text": segment.text.strip()}
            for segment in segments
            if segment.text.strip()
        ]
        if not transcript_segments:
            raise ValueError("No speech was detected in this audio file.")
        return transcript_segments
    except (FileNotFoundError, ValueError):
        raise
    except Exception as error:
        raise RuntimeError(f"Audio transcription failed: {error}") from error
