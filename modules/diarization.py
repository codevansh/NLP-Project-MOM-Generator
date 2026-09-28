"""Speaker diarization and simple timestamp based transcript alignment."""

import os

import streamlit as st
from dotenv import load_dotenv


MODEL_NAME = "pyannote/speaker-diarization-community-1"


@st.cache_resource
def _load_diarization_pipeline(token: str):
    """Load the pyannote pipeline only when audio diarization is requested."""
    from pyannote.audio import Pipeline

    return Pipeline.from_pretrained(MODEL_NAME, token=token)


def diarize_audio(audio_path: str) -> list[dict[str, float | str]]:
    """Return speaker turns from Community-1 using an in-memory audio waveform."""
    load_dotenv()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add your Hugging Face access token to the environment and restart Streamlit."
        )

    try:
        import soundfile as sf
        import torch

        pipeline = _load_diarization_pipeline(token)
        try:
            audio, sample_rate = sf.read(audio_path, always_2d=True, dtype="float32")
        except Exception as error:
            raise RuntimeError(
                "Could not read the audio file. Make sure it is a supported WAV/audio format."
            ) from error

        # soundfile returns (time, channels); pyannote expects (channels, time).
        waveform = torch.from_numpy(audio.T).float()
        audio_file = {"waveform": waveform, "sample_rate": int(sample_rate)}
        result = pipeline(audio_file)
        # pyannote.audio 4 returns a DiarizeOutput; earlier community pipeline
        # releases returned the Annotation directly.
        annotation = getattr(result, "speaker_diarization", result)
        return [
            {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
    except ImportError as error:
        raise RuntimeError(
            "A diarization dependency could not be imported. Install soundfile and project dependencies, then restart Streamlit."
        ) from error
    except Exception as error:
        message = str(error).casefold()
        if "401" in message or "403" in message or "gated" in message or "token" in message:
            raise RuntimeError(
                "Hugging Face access was denied. Accept the pyannote model agreement and check HF_TOKEN."
            ) from error
        if isinstance(error, RuntimeError) and str(error).startswith("Could not read the audio file."):
            raise
        raise RuntimeError("Speaker diarization failed. Check the audio file and model access.") from error


def align_segments(
    whisper_segments: list[dict[str, float | str]],
    speaker_segments: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Assign each Whisper segment to the speaker with the greatest overlap.

    If a Whisper segment spans multiple speakers, only its greatest-overlap
    speaker is kept; this simple approach can hide brief speaker changes.
    """
    aligned = []
    for segment in whisper_segments:
        start, end = float(segment["start"]), float(segment["end"])
        overlaps = [
            (min(end, float(turn["end"])) - max(start, float(turn["start"])), str(turn["speaker"]))
            for turn in speaker_segments
        ]
        duration, speaker = max(overlaps, default=(0.0, "Unknown"), key=lambda match: match[0])
        if duration <= 0:
            speaker = "Unknown"
        aligned.append({**segment, "speaker": speaker})
    return aligned


def format_labeled_transcript(
    segments: list[dict[str, float | str]], speaker_names: dict[str, str] | None = None
) -> str:
    """Format aligned segments in time order, replacing labels with user names."""
    speaker_names = speaker_names or {}
    lines = []
    for segment in sorted(segments, key=lambda item: float(item["start"])):
        label = str(segment["speaker"])
        # Keep the detected ID visible until a real name is explicitly mapped.
        name = speaker_names.get(label, "").strip() or label
        text = str(segment["text"]).strip()
        if text:
            lines.append(f"{name}: {text}")
    return "\n".join(lines)
