from modules.information_extraction import extract_meeting_metadata


def build_meeting_report(
    sentences: list[str],
    entities: list[dict[str, str]],
    action_items: list[dict[str, str]],
    executive_summary: str,
    discussion_points: list[str],
    source_text: str = "",
) -> dict:
    """Combine pipeline results into the agreed meeting report structure."""
    unique_points = []
    seen_points = set()
    for point in discussion_points:
        key = " ".join(point.casefold().split())
        if key and key not in seen_points:
            unique_points.append(point)
            seen_points.add(key)

    return {
        "meeting_metadata": extract_meeting_metadata(source_text, entities),
        "executive_summary": executive_summary,
        "key_discussion_points": unique_points[:7],
        "entities": entities,
        "action_items": action_items,
    }