from modules.action_items import extract_action_items
from modules.information_extraction import extract_speaker_labels
from modules.ner import extract_entities
from modules.preprocessing import clean_transcript, split_sentences
from modules.report import build_meeting_report
from modules.summarizer import summarize_discussion_points, summarize_text


def generate_meeting_minutes(
    transcript: str,
    audio_transcript: bool = False,
) -> tuple[str, dict]:
    """Run the existing Phase 1 steps for any reviewed transcript text."""
    clean_text = clean_transcript(transcript)
    sentences = split_sentences(clean_text)
    entities = extract_entities(clean_text)
    action_items = extract_action_items(
        sentences,
        entities,
        allow_unlabelled_first_person_owner=not audio_transcript,
    )
    discussion_points = summarize_discussion_points(clean_text, action_items=action_items)
    executive_summary = summarize_text(clean_text, action_items=action_items)
    report = build_meeting_report(
        sentences=sentences,
        entities=entities,
        action_items=action_items,
        executive_summary=executive_summary,
        discussion_points=discussion_points,
        source_text=clean_text,
        audio_transcript=audio_transcript,
    )
    if audio_transcript and not extract_speaker_labels(clean_text):
        report["meeting_metadata"]["attendees"] = []
    return clean_text, report