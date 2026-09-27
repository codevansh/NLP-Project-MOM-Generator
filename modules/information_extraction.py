import re


_SPEAKER_LABEL = re.compile(
    r"(?m)^\s*([A-Z][a-z]+(?:['-][A-Z]?[a-z]+)*(?:\s+[A-Z][a-z]+(?:['-][A-Z]?[a-z]+)*)*)\s*:"
)
_HEADER_FIELDS = {"meeting title", "date", "meeting date", "time", "meeting time", "attendees"}


def extract_speaker_labels(text: str) -> list[str]:
    """Return unique names used as transcript speaker labels, in order."""
    speakers = []
    for match in _SPEAKER_LABEL.finditer(text):
        name = match.group(1).strip()
        if name.casefold() not in _HEADER_FIELDS and name.casefold() not in {
            speaker.casefold() for speaker in speakers
        }:
            speakers.append(name)
    return speakers


def extract_meeting_metadata(text: str, entities: list[dict[str, str]]) -> dict:
    """Use labeled headers for meeting date/time and combine NER with speakers."""
    date = _header_value(text, r"(?:meeting\s+)?date")
    time = _header_value(text, r"(?:meeting\s+)?time")
    date = _matching_entity_value(date, "DATE", entities)
    time = _matching_entity_value(time, "TIME", entities)

    attendees = []
    for name in [
        *(entity["text"] for entity in entities if entity.get("label") == "PERSON"),
        *extract_speaker_labels(text),
    ]:
        if name.casefold() not in {attendee.casefold() for attendee in attendees}:
            attendees.append(name)

    return {"date": date, "time": time, "attendees": attendees}


def _header_value(text: str, field_pattern: str) -> str | None:
    match = re.search(rf"(?im)^\s*{field_pattern}\s*:\s*(.+?)\s*$", text)
    return match.group(1).strip() if match else None


def _matching_entity_value(value: str | None, label: str, entities: list[dict[str, str]]) -> str | None:
    if value is None:
        return None
    for entity in entities:
        if entity.get("label") == label and entity["text"].casefold() in value.casefold():
            return entity["text"]
    return value