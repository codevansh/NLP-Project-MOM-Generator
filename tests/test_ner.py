import spacy

from modules.ner import extract_entities


def test_extract_entities_returns_text_and_labels():
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(
        [
            {"label": "PERSON", "pattern": "Rahul"},
            {"label": "ORG", "pattern": "Acme Labs"},
        ]
    )

    assert extract_entities("Rahul works at Acme Labs.", nlp) == [
        {"text": "Rahul", "label": "PERSON"},
        {"text": "Acme Labs", "label": "ORG"},
    ]


def test_extract_entities_cleans_date_time_and_technical_labels():
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(
        [
            {"label": "ORG", "pattern": "API"},
            {"label": "ORG", "pattern": "UI"},
            {"label": "ORG", "pattern": "Priya"},
        ]
    )
    text = (
        "Date: September 26, 2026\nTime: 10:00 AM\n"
        "Rahul: The API is ready.\nPriya: I will finish the UI by Friday."
    )

    entities = extract_entities(text, nlp)

    assert {("Rahul", "PERSON"), ("Priya", "PERSON")} <= {
        (entity["text"], entity["label"]) for entity in entities
    }
    assert {("September 26, 2026", "DATE"), ("10:00 AM", "TIME")} <= {
        (entity["text"], entity["label"]) for entity in entities
    }
    assert not any(entity["text"] in {"API", "UI"} and entity["label"] == "ORG" for entity in entities)
    assert not any("Rahul:" in entity["text"] or "Time" in entity["text"] for entity in entities)


def test_meeting_title_header_is_not_reported_as_an_entity():
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "ORG", "pattern": "E-Commerce Project Planning"}])
    text = "Meeting Title: E-Commerce Project Planning\nDate: September 27, 2026\nVansh: Hello."

    entities = extract_entities(text, nlp)

    assert not any(entity["label"] == "ORG" for entity in entities)
    assert {("September 27, 2026", "DATE"), ("Vansh", "PERSON")} <= {
        (entity["text"], entity["label"]) for entity in entities
    }


def test_irrelevant_labels_are_filtered_and_speaker_labels_are_people():
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(
        [
            {"label": "LANGUAGE", "pattern": "Rahul"},
            {"label": "NORP", "pattern": "Priya"},
            {"label": "ORG", "pattern": "Acme Labs"},
        ]
    )
    text = "Rahul: Hello. Priya: See you at Acme Labs on Friday."

    entities = extract_entities(text, nlp)

    assert {("Rahul", "PERSON"), ("Priya", "PERSON"), ("Acme Labs", "ORG"), ("Friday", "DATE")} <= {
        (entity["text"], entity["label"]) for entity in entities
    }
    assert len([entity for entity in entities if entity["text"].casefold() == "rahul"]) == 1
    assert not any(entity["label"] in {"LANGUAGE", "NORP"} for entity in entities)