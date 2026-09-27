from modules.summarizer import _discussion_text, summarize_discussion_points, summarize_text


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(token_ids)


class FakeSummarizer:
    tokenizer = FakeTokenizer()

    def __init__(self, output="A concise, grounded meeting summary."):
        self.received = []
        self.output = output

    def __call__(self, text, **kwargs):
        self.received.append(text)
        return [{"summary_text": self.output}]


def test_summarizer_handles_empty_input_without_loading_a_model():
    assert summarize_text("  ") == ""


def test_summarizer_sends_all_long_input_chunks_to_the_model():
    model = FakeSummarizer()
    words = [f"word{number}" for number in range(12)]

    summarize_text(" ".join(words), summarizer=model, max_input_tokens=5)

    first_pass_text = " ".join(model.received[:3])
    assert all(word in first_pass_text for word in words)


def test_summarizer_requests_paraphrase_without_inventing_facts():
    model = FakeSummarizer()

    summary = summarize_text("The team reviewed the project and assigned testing.", summarizer=model)

    assert summary == "A concise, grounded meeting summary."
    assert "The team reviewed the project" in model.received[0]


def test_summary_removes_repeated_sentences_and_uses_transcript_text():
    model = FakeSummarizer(
        "The portal is progressing well. Login testing is needed. "
        "The portal is progressing well. Testing is planned for next week."
    )
    transcript = "The portal is progressing well. Login testing is needed."

    summary = summarize_text(transcript, summarizer=model)

    assert summary == (
        "The portal is progressing well. Login testing is needed. Testing is planned for next week."
    )
    assert transcript in model.received[0]
    assert "The portal is progressing well" in model.received[0]


def test_summary_uses_source_discussion_without_repeating_action_sentences():
    model = FakeSummarizer()
    transcript = "The portal discussion is underway. Rahul will complete the backend API by Friday."
    actions = [{"task": "complete the backend API", "assigned_to": "Rahul", "deadline": "Friday"}]

    summarize_text(transcript, summarizer=model, action_items=actions)

    assert "The portal discussion is underway" in model.received[0]
    assert "complete the backend API" not in model.received[0]


def test_executive_summary_uses_cleaned_source_not_an_action_item_list():
    model = FakeSummarizer("The portal is progressing. A login issue remains. Testing is planned.")
    transcript = (
        "The portal is progressing. A login issue remains. "
        "Rahul will complete the backend API by Friday."
    )

    summary = summarize_text(
        transcript,
        summarizer=model,
        action_items=[
            {"task": "complete the backend API", "assigned_to": "Rahul", "deadline": "Friday"}
        ],
    )

    assert summary == "The portal is progressing. A login issue remains. Testing is planned."
    assert "The portal is progressing" in model.received[0]
    assert "A login issue remains" in model.received[0]
    assert "complete the backend API" not in model.received[0]


def test_discussion_points_are_paraphrased_and_deduplicated():
    model = FakeSummarizer("The dashboard is nearly complete. The dashboard is almost complete.")

    points = summarize_discussion_points(
        "The dashboard is almost complete. Login errors were reported.", summarizer=model
    )

    assert points == ["The dashboard is nearly complete."]


def test_discussion_source_skips_headers_and_agenda_but_keeps_major_decisions():
    transcript = (
        "Meeting Title: Student Portal Development Meeting\n"
        "Date: September 26, 2026\n"
        "Time: 10:00 AM\n"
        "Rahul: Today we need to discuss the portal progress.\n"
        "Rahul: I will complete the backend API by Friday.\n"
        "Rahul: Once the login issue is fixed, we should integrate the frontend with the backend.\n"
        "Amit: We should conduct system testing next week."
    )
    actions = [
        {"task": "complete the backend API", "assigned_to": "Rahul", "deadline": "Friday"},
        {"task": "integrate the frontend with the backend", "assigned_to": "Team", "deadline": ""},
        {"task": "conduct system testing", "assigned_to": "Team", "deadline": "next week"},
    ]

    discussion = _discussion_text(transcript, actions, keep_major_decisions=True)

    assert "Student Portal Development Meeting" not in discussion
    assert "September 26, 2026" not in discussion
    assert "10:00 AM" not in discussion
    assert "complete the backend API" not in discussion
    assert "integrate the frontend with the backend" in discussion
    assert "conduct system testing next week" in discussion


def test_standalone_questions_are_excluded_from_summary_and_discussion_source():
    transcript = (
        "Can you fix the API issue by tomorrow?\n"
        "Mit: Yes, I will fix the notification API and push the changes by tomorrow evening."
    )

    discussion = _discussion_text(transcript, [], keep_major_decisions=True)

    assert "Can you fix the API issue" not in discussion
    assert "I will fix the notification API" in discussion


def test_discussion_points_do_not_repeat_extracted_action_sentences():
    transcript = (
        "The login screen is loading slowly.\n"
        "Mit: I can investigate the login performance issue this afternoon.\n"
        "We should complete regression testing by Friday."
    )
    actions = [
        {"task": "investigate the login performance issue", "assigned_to": "Mit", "deadline": "this afternoon"},
        {"task": "complete regression testing", "assigned_to": "Team", "deadline": "Friday"},
    ]

    discussion = _discussion_text(transcript, actions, keep_major_decisions=False)

    assert "The login screen is loading slowly" in discussion
    assert "investigate the login performance issue" not in discussion
    assert "complete regression testing" not in discussion


def test_important_issue_is_preserved_when_model_omits_it():
    transcript = "The homepage is complete. I noticed that the mobile layout has a problem."
    summary_model = FakeSummarizer("The homepage is complete.")
    point_model = FakeSummarizer("The homepage is complete.")

    summary = summarize_text(transcript, summarizer=summary_model)
    points = summarize_discussion_points(transcript, summarizer=point_model)

    assert "mobile layout has a problem" in summary.casefold()
    assert any("mobile layout has a problem" in point.casefold() for point in points)