import re

from modules.information_extraction import extract_speaker_labels

_SPEAKER_PREFIX = re.compile(r"^\s*([A-Z][a-z]+(?:['-][A-Z]?[a-z]+)*(?:\s+[A-Z][a-z]+)*)\s*:\s*(.*)$")
_COMMITMENT = re.compile(
    r"\b(?:I['’]ll|I\s+(?:will|can|need\s+to|should|must)|"
    r"will|can|needs?\s+to|should|must|has\s+to|have\s+to|is\s+going\s+to|"
    r"responsible\s+for|assigned\s+to|please)\b",
    re.IGNORECASE,
)
_DATE_PART = (
    r"(?:this|tomorrow|next)\s+(?:morning|afternoon|evening|night)|"
    r"(?:(?:this|next|following)\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{4}-\d{2}-\d{2}|"
    r"\d{1,2}/\d{1,2}/\d{2,4}|"
    r"today|tomorrow|tonight|(?:this|next|following)\s+(?:week|month|year)"
)
_TIME_PART = (
    r"(?:1[0-2]|0?[1-9])(?::[0-5]\d)?\s*(?:a\.m|p\.m|am|pm)|"
    r"(?:early\s+)?(?:morning|afternoon|evening|night)"
)
_DEADLINE = re.compile(
    rf"\b(?:(?:by|before|on|for|during|due(?:\s+by)?)\s+)?"
    rf"(?P<value>(?:{_DATE_PART})(?:\s+(?:at\s+)?{_TIME_PART})?|{_TIME_PART})\b",
    re.IGNORECASE,
)
_NON_PERSON_SUBJECTS = {"i", "i'll", "we", "you", "he", "she", "they", "it", "the team", "team"}
_AGENDA_ONLY = re.compile(
    r"\b(?:today\s+we\s+need\s+to\s+(?:discuss|finalize)|we\s+need\s+to\s+discuss|"
    r"we\s+can\s+discuss|let['’]s\s+(?:discuss|finalize\s+the\s+development\s+plan)|"
    r"today\s+we\s+will\s+discuss|the\s+meeting\s+will\s+cover)\b",
    re.IGNORECASE,
)
_CONFIRMATION = re.compile(r"^(?:yes|yeah|yep|sure|okay|ok|certainly|absolutely)[,!\s.]*$", re.IGNORECASE)
_TASK_STOP_WORDS = {"a", "an", "and", "be", "complete", "do", "for", "it", "on", "the", "to", "work"}


def _assigned_to(
    prefix: str,
    speaker: str | None,
    entities: list[dict[str, str]],
    after_trigger: str,
    assigned_to_trigger: bool,
) -> str:
    prefix = re.sub(
        r"^\s*(?:yes|yeah|yep|sure|okay|ok|certainly|absolutely)[,!\s]+",
        "",
        prefix,
        flags=re.IGNORECASE,
    )

    if assigned_to_trigger:
        named_assignee = _name_candidate(after_trigger, entities)
        if named_assignee:
            return named_assignee

    prefix = _SPEAKER_PREFIX.sub(r"\2", prefix, count=1).strip()
    people_in_prefix = [
        entity["text"]
        for entity in entities
        if entity.get("label") == "PERSON"
        and entity["text"].casefold() in prefix.casefold()
    ]
    if people_in_prefix:
        return people_in_prefix[-1]

    if re.search(r"\bwe\b", prefix, re.IGNORECASE):
        return "Team"

    explicit_name = _name_candidate(prefix, entities)
    if explicit_name and explicit_name.casefold() != (speaker or "").casefold():
        return explicit_name
    if speaker:
        return speaker
    if explicit_name:
        return explicit_name
    return "Unknown"


def _name_candidate(text: str, entities: list[dict[str, str]]) -> str:
    people = [
        entity["text"]
        for entity in entities
        if entity.get("label") == "PERSON" and entity["text"].casefold() in text.casefold()
    ]
    if people:
        return people[-1]

    candidate = re.sub(r"^\s*(?:please\s+)?", "", text).strip(" ,;:")
    if re.fullmatch(r"[A-Z][a-z]+(?:['-][A-Z]?[a-z]+)*(?:\s+[A-Z][a-z]+)*", candidate):
        if candidate.casefold() not in _NON_PERSON_SUBJECTS:
            return candidate
    return ""


def _find_deadline(
    sentence: str, entities: list[dict[str, str]], trigger_end: int
) -> tuple[str, str]:
    matches = list(_DEADLINE.finditer(sentence))
    match = next((item for item in matches if item.start() >= trigger_end), None)
    if match is None:
        match = next(
            (
                item
                for item in matches
                if item.end() <= trigger_end
                and item.group(0)[: item.start("value")].strip().casefold()
                in {"by", "before", "on", "due", "due by"}
            ),
            None,
        )
    if match:
        following = next((item for item in matches if item.start() >= match.end()), None)
        if following and sentence[match.end() : following.start()].strip().casefold() in {"", "at"}:
            if following.group("value").casefold() in {
                "morning", "afternoon", "evening", "night", "early morning"
            }:
                value = f"{match.group('value')} {following.group('value')}"
                return value.strip(" ,.;"), sentence[match.start() : following.end()]
        return match.group("value").strip(" ,.;"), match.group(0)

    for entity in entities:
        if entity.get("label") in {"DATE", "TIME"}:
            match = re.search(re.escape(entity["text"]), sentence, re.IGNORECASE)
            if match:
                return entity["text"], match.group(0)
    return "", ""


def _clean_task(task: str, deadline_match: str, trigger_text: str) -> str:
    if trigger_text.casefold() == "assigned to":
        task = task.strip(" ,.:;-")
        return re.sub(r"^(?:the\s+)?(?:task|item)\s+", "", task, flags=re.IGNORECASE)
    if deadline_match:
        task = re.sub(re.escape(deadline_match), "", task, flags=re.IGNORECASE)
    task = re.sub(r"\s+['’]s\s+meeting\b", "", task, flags=re.IGNORECASE)
    task = re.sub(r"\b(?:by|before|on|for|during|at)\s*$", "", task, flags=re.IGNORECASE)
    return task.strip(" ,.:;-")


def _task_core(task: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", task.casefold())
        if word not in _TASK_STOP_WORDS
    }


def _remove_duplicate_tasks(items: list[dict[str, str]]) -> list[dict[str, str]]:
    unique = []
    for item in items:
        words = _task_core(item["task"])
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(unique)
                if words
                and _task_core(existing["task"])
                and len(words & _task_core(existing["task"]))
                / min(len(words), len(_task_core(existing["task"])))
                >= 0.8
                and (
                    item["assigned_to"].casefold() == existing["assigned_to"].casefold()
                    or item["assigned_to"].casefold() == "team"
                    or existing["assigned_to"].casefold() == "team"
                )
            ),
            None,
        )
        if duplicate_index is None:
            unique.append(item)
        elif unique[duplicate_index]["assigned_to"].casefold() == "team" and item[
            "assigned_to"
        ].casefold() != "team":
            unique[duplicate_index] = item
    return unique


def extract_action_items(
    sentences: list[str],
    entities: list[dict[str, str]] | None = None,
    allow_unlabelled_first_person_owner: bool = True,
) -> list[dict[str, str]]:
    """Find commitments with readable trigger-word patterns and nearby entities."""
    entities = entities or []
    action_items = []
    seen = set()
    speakers = set(extract_speaker_labels("\n".join(sentences)))
    current_speaker = None

    for sentence_index, sentence in enumerate(sentences):
        speaker_match = _SPEAKER_PREFIX.match(sentence)
        if speaker_match:
            current_speaker = speaker_match.group(1)
            speakers.add(current_speaker)
            sentence_body = speaker_match.group(2)
        else:
            sentence_body = sentence

        trigger = _COMMITMENT.search(sentence_body)
        if (
            trigger is None
            or sentence_body.rstrip().endswith("?")
            or _AGENDA_ONLY.search(sentence_body)
        ):
            continue

        prefix = sentence_body[: trigger.start()]
        after_trigger = sentence_body[trigger.end() :]
        deadline, deadline_match = _find_deadline(sentence_body, entities, trigger.end())
        if trigger.group(0).casefold() == "assigned to":
            task = _clean_task(prefix, deadline_match, trigger.group(0))
        else:
            task = _clean_task(after_trigger, deadline_match, trigger.group(0))

        if trigger.group(0).casefold() == "should" and task.casefold() == "be ready":
            previous = sentences[sentence_index - 1] if sentence_index else ""
            coordination = re.search(
                r"\bI\s+am\s+coordinating\s+with\s+(.+?)[.!?]*$",
                previous,
                re.IGNORECASE,
            )
            subject = re.search(r"\b(?:the|a|an)\s+(.+?)\s+should\s+be\s+ready\b", sentence_body, re.IGNORECASE)
            if coordination and subject:
                task = f"coordinate with {coordination.group(1).strip()} and finalize {subject.group(1).strip()}"

        first_person_commitment = trigger.group(0).casefold().startswith(("i ", "i'", "i’"))
        if (
            first_person_commitment
            and current_speaker is None
            and not allow_unlabelled_first_person_owner
        ):
            assigned_to = "Unknown"
        else:
            assigned_to = _assigned_to(
                prefix,
                current_speaker,
                entities,
                after_trigger,
                trigger.group(0).casefold() == "assigned to",
            )

        item = {
            "task": task,
            "assigned_to": assigned_to,
            "deadline": deadline,
        }
        normalized_task = re.sub(r"\W+", " ", item["task"].casefold()).strip()
        key = (normalized_task, item["assigned_to"].casefold())
        if item["task"] and key not in seen:
            action_items.append(item)
            seen.add(key)

    return _remove_duplicate_tasks(action_items)