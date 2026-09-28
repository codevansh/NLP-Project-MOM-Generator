from modules import pipeline


def _stub_summarizers(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "summarize_discussion_points",
        lambda text, action_items: ["Responsive layout needs attention."],
    )
    monkeypatch.setattr(
        pipeline,
        "summarize_text",
        lambda text, action_items: "The responsive layout requires work.",
    )


def test_edited_transcript_uses_the_existing_phase_one_pipeline(monkeypatch):
    _stub_summarizers(monkeypatch)
    monkeypatch.setattr(pipeline, "extract_entities", lambda text: [])
    reviewed_text = "Nitya: I will fix the responsive layout tomorrow."

    clean_text, report = pipeline.generate_meeting_minutes(reviewed_text)

    assert clean_text == reviewed_text
    assert report["meeting_metadata"] == {"date": None, "time": None, "attendees": ["Nitya"]}
    assert report["action_items"] == [
        {"task": "fix the responsive layout", "assigned_to": "Nitya", "deadline": "tomorrow"}
    ]
    assert set(report) == {
        "meeting_metadata",
        "executive_summary",
        "key_discussion_points",
        "entities",
        "action_items",
    }


def test_unlabelled_audio_transcript_does_not_invent_speaker_or_metadata(monkeypatch):
    _stub_summarizers(monkeypatch)
    monkeypatch.setattr(pipeline, "extract_entities", lambda text: [])

    _, report = pipeline.generate_meeting_minutes(
        "I will fix the responsive layout tomorrow.", audio_transcript=True
    )

    assert report["meeting_metadata"] == {"date": None, "time": None, "attendees": []}
    assert report["action_items"] == [
        {"task": "fix the responsive layout", "assigned_to": "Unknown", "deadline": "tomorrow"}
    ]


def test_audio_transcript_does_not_count_mentioned_person_as_attendee(monkeypatch):
    _stub_summarizers(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "extract_entities",
        lambda text: [{"text": "Nitya", "label": "PERSON"}],
    )

    _, report = pipeline.generate_meeting_minutes(
        "We discussed Nitya's suggestion. I will fix the responsive layout.",
        audio_transcript=True,
    )

    assert report["meeting_metadata"]["attendees"] == []


def test_audio_reviewed_speaker_label_can_assign_first_person_commitment(monkeypatch):
    _stub_summarizers(monkeypatch)
    monkeypatch.setattr(pipeline, "extract_entities", lambda text: [])

    _, report = pipeline.generate_meeting_minutes(
        "Nitya: I will fix the responsive layout tomorrow.",
        audio_transcript=True,
    )

    assert report["meeting_metadata"]["attendees"] == ["Nitya"]
    assert report["action_items"][0]["assigned_to"] == "Nitya"


def test_audio_explicit_named_assignment_remains_assigned(monkeypatch):
    _stub_summarizers(monkeypatch)
    monkeypatch.setattr(pipeline, "extract_entities", lambda text: [])

    _, report = pipeline.generate_meeting_minutes(
        "Mit will fix the notification API tomorrow.",
        audio_transcript=True,
    )

    assert report["meeting_metadata"]["attendees"] == []
    assert report["action_items"][0]["assigned_to"] == "Mit"


def test_audio_flat_meeting_title_uses_detected_date_and_time(monkeypatch):
    _stub_summarizers(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "extract_entities",
        lambda text: [
            {"text": "September 26, 2026", "label": "DATE"},
            {"text": "10 AM", "label": "TIME"},
        ],
    )
    transcript = (
        "Meeting Title, Student Portal Development Meeting, September 26, 2026, 10 AM. "
        "The frontend dashboard is nearly complete."
    )

    _, report = pipeline.generate_meeting_minutes(transcript, audio_transcript=True)

    assert report["meeting_metadata"] == {
        "date": "September 26, 2026",
        "time": "10 AM",
        "attendees": [],
    }


def test_audio_does_not_guess_metadata_from_deadline_entities(monkeypatch):
    _stub_summarizers(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "extract_entities",
        lambda text: [{"text": "Friday", "label": "DATE"}],
    )

    _, report = pipeline.generate_meeting_minutes(
        "I will finish the report by Friday.", audio_transcript=True
    )

    assert report["meeting_metadata"]["date"] is None
    assert report["meeting_metadata"]["time"] is None


def test_corrected_flat_audio_transcript_recovers_metadata_and_speaker_owner(monkeypatch):
    _stub_summarizers(monkeypatch)
    transcript = (
        "Meeting Title, Student Portal Development Meeting, September 26, 2026, 10 AM.\n"
        "Rahul: Good morning everyone.\n"
        "Priya: I can test the login API and report the exact issue by tomorrow."
    )

    _, report = pipeline.generate_meeting_minutes(transcript, audio_transcript=True)

    assert report["meeting_metadata"] == {
        "date": "September 26, 2026",
        "time": "10 AM",
        "attendees": ["Rahul", "Priya"],
    }
    assert {("Rahul", "PERSON"), ("Priya", "PERSON")} <= {
        (entity["text"], entity["label"]) for entity in report["entities"]
    }
    assert report["action_items"] == [
        {
            "task": "test the login API and report the exact issue",
            "assigned_to": "Priya",
            "deadline": "tomorrow",
        }
    ]
    assert not any(entity["text"] == "Student Portal Development Meeting" for entity in report["entities"])