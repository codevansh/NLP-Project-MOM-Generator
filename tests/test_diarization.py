from modules.diarization import align_segments, format_labeled_transcript


def test_alignment_uses_speaker_with_greatest_overlap():
    whisper = [{"start": 0.0, "end": 4.0, "text": "Hello there."}]
    speakers = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 4.0, "speaker": "SPEAKER_01"},
    ]

    assert align_segments(whisper, speakers)[0]["speaker"] == "SPEAKER_01"


def test_alignment_marks_uncovered_segments_unknown():
    aligned = align_segments([{"start": 3.0, "end": 4.0, "text": "Hi."}], [])

    assert aligned[0]["speaker"] == "Unknown"


def test_transcript_uses_mapping_and_keeps_chronological_order():
    segments = [
        {"start": 2.0, "end": 3.0, "text": "Second.", "speaker": "SPEAKER_01"},
        {"start": 0.0, "end": 1.0, "text": "First.", "speaker": "SPEAKER_00"},
    ]

    assert format_labeled_transcript(segments, {"SPEAKER_00": "Rahul"}) == (
        "Rahul: First.\nSPEAKER_01: Second."
    )
