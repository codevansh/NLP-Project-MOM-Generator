from pathlib import Path
from types import SimpleNamespace

import pytest

from modules import transcription


def test_transcribe_audio_joins_segments_in_order(tmp_path, monkeypatch):
    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"fake audio bytes")

    class FakeModel:
        def transcribe(self, path):
            assert path == str(audio_path)
            segments = [
                SimpleNamespace(start=0.0, end=1.2, text=" First sentence."),
                SimpleNamespace(start=1.2, end=1.3, text=""),
                SimpleNamespace(start=1.3, end=2.5, text="Second sentence."),
            ]
            return iter(segments), SimpleNamespace(language="en")

    monkeypatch.setattr(transcription, "_load_whisper_model", lambda: FakeModel())

    assert transcription.transcribe_audio(str(audio_path)) == [
        {"start": 0.0, "end": 1.2, "text": "First sentence."},
        {"start": 1.3, "end": 2.5, "text": "Second sentence."},
    ]


def test_transcribe_audio_rejects_unsupported_extensions(tmp_path):
    transcript_path = tmp_path / "meeting.txt"
    transcript_path.write_text("not audio", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported audio type"):
        transcription.transcribe_audio(str(transcript_path))


def test_transcribe_audio_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        transcription.transcribe_audio(str(tmp_path / "missing.wav"))


def test_temporary_audio_file_is_removed_after_use():
    temporary_path = None
    with transcription.temporary_audio_file(b"audio bytes", ".wav") as path:
        temporary_path = Path(path)
        assert temporary_path.is_file()
        assert temporary_path.read_bytes() == b"audio bytes"

    assert temporary_path is not None
    assert not temporary_path.exists()


def test_temporary_audio_file_rejects_unsupported_suffix():
    with pytest.raises(ValueError, match="Upload a WAV, MP3, or M4A"):
        with transcription.temporary_audio_file(b"audio bytes", ".txt"):
            pass
