from pathlib import Path

from modules.action_items import extract_action_items
from modules.ner import extract_entities
from modules.preprocessing import clean_transcript, split_sentences


def test_extract_action_items_uses_person_and_date_entities():
    sentences = ["Rahul will complete the login API by Friday.", "We discussed the portal."]
    entities = [
        {"text": "Rahul", "label": "PERSON"},
        {"text": "Friday", "label": "DATE"},
    ]

    assert extract_action_items(sentences, entities) == [
        {"task": "complete the login API", "assigned_to": "Rahul", "deadline": "Friday"}
    ]


def test_extract_action_items_handles_first_person_and_deadline_pattern():
    assert extract_action_items(["I will finish testing tomorrow."]) == [
        {"task": "finish testing", "assigned_to": "Unknown", "deadline": "tomorrow"}
    ]


def test_speaker_context_and_explicit_person_assignments():
    sentences = [
        "Rahul: I will complete the backend API for the attendance module by Friday.",
        "Priya: I can test the login API and report the exact issue by tomorrow.",
        "Neha: I will prepare the presentation for the project review and share it with everyone by Monday.",
        "Rahul: Priya will prepare the presentation by Monday.",
    ]
    entities = [
        {"text": name, "label": "PERSON"} for name in ["Rahul", "Priya", "Neha"]
    ] + [
        {"text": day, "label": "DATE"} for day in ["Friday", "tomorrow", "Monday"]
    ]

    assert extract_action_items(sentences, entities) == [
        {
            "task": "complete the backend API for the attendance module",
            "assigned_to": "Rahul",
            "deadline": "Friday",
        },
        {
            "task": "test the login API and report the exact issue",
            "assigned_to": "Priya",
            "deadline": "tomorrow",
        },
        {
            "task": "prepare the presentation for the project review and share it with everyone",
            "assigned_to": "Neha",
            "deadline": "Monday",
        },
        {"task": "prepare the presentation", "assigned_to": "Priya", "deadline": "Monday"},
    ]


def test_other_person_name_is_not_hardcoded_and_time_deadline_is_clean():
    assert extract_action_items(["Arjun will submit the database report before Wednesday."])[0] == {
        "task": "submit the database report",
        "assigned_to": "Arjun",
        "deadline": "Wednesday",
    }
    assert extract_action_items(["Mina: I should present the results by Tuesday at 11 AM."])[0] == {
        "task": "present the results",
        "assigned_to": "Mina",
        "deadline": "Tuesday at 11 AM",
    }


def test_existing_sample_attendees_and_contextual_action_owners():
    text = clean_transcript(Path("samples/transcript.txt").read_text(encoding="utf-8"))
    sentences = split_sentences(text)
    entities = extract_entities(text)

    action_items = extract_action_items(sentences, entities)

    assert {item["assigned_to"] for item in action_items} >= {"Rahul", "Priya", "Neha", "Team"}


def test_agenda_sentences_are_not_action_items_but_real_commitments_remain():
    sentences = [
        "Rahul: Today we need to discuss the portal progress and finalize tasks.",
        "Priya: Let's discuss the login issue.",
        "Today we will discuss testing.",
        "The meeting will cover the dashboard.",
        "Neha: I will update the project documentation after testing.",
        "We should conduct system testing next week.",
    ]

    assert extract_action_items(sentences) == [
        {
            "task": "update the project documentation after testing",
            "assigned_to": "Neha",
            "deadline": "",
        },
        {"task": "conduct system testing", "assigned_to": "Team", "deadline": "next week"},
    ]


def test_duplicate_action_items_ignore_punctuation_and_deadline_changes():
    sentences = [
        "Rahul will update the project documentation by Friday.",
        "Rahul will update the project documentation, by Monday.",
    ]

    assert extract_action_items(sentences) == [
        {
            "task": "update the project documentation",
            "assigned_to": "Rahul",
            "deadline": "Friday",
        }
    ]


def test_question_is_not_a_task_and_confirming_speaker_owns_the_commitment():
    sentences = [
        "Vansh: Can you fix the API issue by tomorrow?",
        "Mit: Yes, I will fix the notification API and push the changes by tomorrow evening.",
    ]
    entities = [{"text": "Yes", "label": "PERSON"}]

    assert extract_action_items(sentences, entities) == [
        {
            "task": "fix the notification API and push the changes",
            "assigned_to": "Mit",
            "deadline": "tomorrow evening",
        }
    ]


def test_relative_weekday_and_preposition_are_removed_with_deadline():
    assert extract_action_items(["Shravani: We should schedule integration testing for next Monday."]) == [
        {
            "task": "schedule integration testing",
            "assigned_to": "Team",
            "deadline": "next Monday",
        }
    ]
    assert extract_action_items(["Nitya: I will prepare the Instagram posts and schedule them for Friday."])[0][
        "task"
    ] == "prepare the Instagram posts and schedule them"
    assert extract_action_items(["Mit: I can investigate the login performance issue this afternoon."])[0] == {
        "task": "investigate the login performance issue",
        "assigned_to": "Mit",
        "deadline": "this afternoon",
    }


def test_explicit_person_replaces_overlapping_team_requirement():
    sentences = [
        "We still need to complete the shopping cart interface.",
        "Nitya: I can work on the shopping cart interface and complete it by Thursday.",
    ]

    assert extract_action_items(sentences) == [
        {
            "task": "work on the shopping cart interface and complete it",
            "assigned_to": "Nitya",
            "deadline": "Thursday",
        }
    ]


def test_video_coordination_context_completes_task_description():
    sentences = [
        "Shravani: Who is responsible for the video?",
        "Atharva: I am coordinating with the design team.",
        "The final video should be ready by Thursday.",
    ]

    assert extract_action_items(sentences) == [
        {
            "task": "coordinate with the design team and finalize final video",
            "assigned_to": "Atharva",
            "deadline": "Thursday",
        }
    ]