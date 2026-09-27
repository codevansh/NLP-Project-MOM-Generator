from modules.report import build_meeting_report


def test_report_has_required_sections_and_metadata():
    summary = (
        "The team reviewed the student portal. An authentication issue affects login. "
        "Testing and documentation were assigned for the coming week."
    )
    report = build_meeting_report(
        sentences=["Rahul will finish testing by Friday.", "The team approved the new design."],
        entities=[
            {"text": "Rahul", "label": "PERSON"},
            {"text": "Friday", "label": "DATE"},
        ],
        action_items=[
            {"task": "finish testing", "assigned_to": "Rahul", "deadline": "Friday"}
        ],
        executive_summary=summary,
        discussion_points=[
            "The student portal was reviewed.",
            "An authentication issue affects login.",
            "Testing and documentation were assigned for the coming week.",
        ],
        source_text=(
            "Date: September 26, 2026\nTime: 10:00 AM\n"
            "Rahul: The team approved the new design."
        ),
    )

    assert set(report) == {
        "meeting_metadata",
        "executive_summary",
        "key_discussion_points",
        "entities",
        "action_items",
    }
    assert report["meeting_metadata"] == {
        "date": "September 26, 2026",
        "time": "10:00 AM",
        "attendees": ["Rahul"],
    }
    assert report["key_discussion_points"] == [
        "The student portal was reviewed.",
        "An authentication issue affects login.",
        "Testing and documentation were assigned for the coming week.",
    ]
    assert report["executive_summary"].count(".") == 3


def test_metadata_without_labeled_date_or_time_remains_null():
    report = build_meeting_report(
        sentences=["Testing will happen next week."],
        entities=[{"text": "next week", "label": "DATE"}],
        action_items=[],
        executive_summary="Testing is planned for next week.",
        discussion_points=["Testing is planned for next week."],
        source_text="Testing will happen next week.",
    )

    assert report["meeting_metadata"]["date"] is None
    assert report["meeting_metadata"]["time"] is None