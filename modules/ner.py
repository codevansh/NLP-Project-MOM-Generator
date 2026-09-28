import re

import spacy
import streamlit as st

from modules.information_extraction import extract_speaker_labels


_TECHNICAL_TERMS = {"API", "UI", "UX", "URL", "HTTP", "HTTPS", "SQL", "CPU"}
_MEETING_ENTITY_LABELS = {"PERSON", "DATE", "TIME", "ORG", "GPE", "LOC"}
_DATE = re.compile(
    r"\b(?:"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
    r"|today|tomorrow|tonight|(?:this|next|following)\s+(?:week|month|year)"
    r")\b",
    re.IGNORECASE,
)
_TIME = re.compile(
    r"\b(?:1[0-2]|0?[1-9]):[0-5]\d\s*(?:a\.m|p\.m|am|pm)\b"
    r"|\b(?:1[0-2]|0?[1-9])\s*(?:a\.m|p\.m|am|pm)\b",
    re.IGNORECASE,
)
_HEADER_LINE = re.compile(
    r"(?im)^\s*(?:meeting\s+title|date|meeting\s+date|time|meeting\s+time|attendees)\s*:[^\n]*"
)

# Streamlit reruns this script after interactions, so share one loaded NLP model across reruns.
@st.cache_resource
def _load_english_model():
    """Load the installed English model once per application process."""
    return spacy.load("en_core_web_sm")


def extract_entities(text: str, nlp=None) -> list[dict[str, str]]:
    """Return filtered spaCy entities plus exact dates, times, and speaker names."""
    if nlp is None:
        nlp = _load_english_model()

    doc = nlp(text)
    speakers = extract_speaker_labels(text)
    speaker_keys = {name.casefold() for name in speakers}
    header_spans = [match.span() for match in _HEADER_LINE.finditer(text)]
    if re.match(r"^\s*meeting\s+title\b", text, re.IGNORECASE):
        first_date = _DATE.search(text)
        if first_date:
            header_spans.append((0, first_date.start()))
    found = []

    for entity in doc.ents:
        if any(entity.start_char < end and entity.end_char > start for start, end in header_spans):
            continue
        if entity.label_ in {"DATE", "TIME"}:
            continue
        label = "PERSON" if entity.text.casefold() in speaker_keys else entity.label_
        if label not in _MEETING_ENTITY_LABELS:
            continue
        if label == "ORG" and entity.text.upper() in _TECHNICAL_TERMS:
            continue
        found.append((entity.start_char, entity.text, label))

    for match in _DATE.finditer(text):
        found.append((match.start(), match.group(0), "DATE"))
    for match in _TIME.finditer(text):
        found.append((match.start(), match.group(0), "TIME"))
    for name in speakers:
        start = text.casefold().find(name.casefold())
        if start >= 0:
            found.append((start, name, "PERSON"))

    label_priority = {"PERSON": 0, "DATE": 1, "TIME": 2, "ORG": 3, "GPE": 4, "LOC": 5}
    entities = []
    seen_text = set()
    for _, entity_text, label in sorted(
        found,
        key=lambda item: (item[0], label_priority.get(item[2], 10)),
    ):
        key = entity_text.casefold()
        if key not in seen_text:
            entities.append({"text": entity_text, "label": label})
            seen_text.add(key)
    return entities