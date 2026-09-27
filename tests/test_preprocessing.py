from modules.preprocessing import clean_transcript, split_sentences


def test_clean_transcript_removes_fillers_and_normalizes_spaces():
    assert clean_transcript("  Um, we   agreed uh to review it.  ") == "we agreed to review it."


def test_split_sentences_preserves_sentence_content():
    assert split_sentences("We met today. Testing starts next week!") == [
        "We met today.",
        "Testing starts next week!",
    ]


def test_clean_transcript_preserves_speaker_line_breaks():
    assert clean_transcript("Rahul: I will test.\n\nPriya: I can review.") == (
        "Rahul: I will test.\nPriya: I can review."
    )