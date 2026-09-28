import re


_SPEAKER_LABEL = re.compile(
    r"(?m)(?:^\s*|[.!?]\s+)([A-Z][a-z]+(?:['-][A-Z]?[a-z]+)*(?:\s+[A-Z][a-z]+(?:['-][A-Z]?[a-z]+)*)*)\s*:"
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


def extract_meeting_metadata(
    text: str,
    entities: list[dict[str, str]],
    audio_transcript: bool = False,
) -> dict:
    """Use labeled headers for meeting date/time and combine NER with speakers."""
    date = _header_value(text, r"(?:meeting\s+)?date")
    time = _header_value(text, r"(?:meeting\s+)?time")

    if audio_transcript and re.match(r"^\s*meeting\s+title\b", text, re.IGNORECASE):
        date = date or _first_entity_value("DATE", entities)
        time = time or _first_entity_value("TIME", entities)

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
    if match is None:
        match = re.search(
            rf"\b{field_pattern}\s*:\s*(.+?)"
            r"(?=\b(?:meeting\s+)?(?:date|time|attendees)\s*:|$)",
            text,
            re.IGNORECASE,
        )
    return match.group(1).strip() if match else None


def _first_entity_value(label: str, entities: list[dict[str, str]]) -> str | None:
    return next(
        (entity["text"] for entity in entities if entity.get("label") == label),
        None,
    )


def _matching_entity_value(value: str | None, label: str, entities: list[dict[str, str]]) -> str | None:
    if value is None:
        return None
    for entity in entities:
        if entity.get("label") == label and entity["text"].casefold() in value.casefold():
            return entity["text"]
    return value