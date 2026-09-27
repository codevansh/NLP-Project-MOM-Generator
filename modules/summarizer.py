import re

import streamlit as st


MODEL_NAME = "sshleifer/distilbart-cnn-6-6"
MAX_INPUT_TOKENS = 400


# Share the expensive pipeline across Streamlit reruns; Transformers reuses its user-level disk cache.
@st.cache_resource
def _load_summarizer():
    """Load a local CPU summarization pipeline once, on first use."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to("cpu")
    return tokenizer, model


def _split_into_chunks(text: str, tokenizer, max_input_tokens: int) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(token_ids), max_input_tokens):
        chunk_ids = token_ids[start : start + max_input_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks


def _summarize_chunk(text: str, summarizer, final: bool = False) -> str:
    if summarizer is None:
        tokenizer, model = _load_summarizer()
    else:
        tokenizer = summarizer.tokenizer
        model = summarizer

    input_length = len(tokenizer.encode(text, add_special_tokens=True))
    max_length = min(150 if final else 110, max(24, int(input_length * 1.1)))
    min_length = min(max_length - 2, max(10, int(input_length * 0.3)))
    if callable(model) and not hasattr(model, "generate"):
        result = model(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
            truncation=False,
        )
        return result[0].get("summary_text", result[0].get("generated_text", "")).strip()

    model_inputs = tokenizer(text, return_tensors="pt", truncation=False)
    generated = model.generate(
        **model_inputs,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()


_SUMMARY_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "but", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "the", "to", "was", "were", "with",
}


def _comparison_words(text: str) -> set[str]:
    text = re.sub(r"\bfront\s+end\b", "frontend", text.casefold())
    text = re.sub(r"\bnearly\b", "almost", text)
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text)
        if word not in _SUMMARY_STOP_WORDS
    }


def _is_duplicate(text: str, previous: list[str]) -> bool:
    words = _comparison_words(text)
    if not words:
        return True
    for existing in previous:
        old_words = _comparison_words(existing)
        if words == old_words or len(words & old_words) / min(len(words), len(old_words)) >= 0.8:
            return True
    return False


def _unique_sentences(text: str, limit: int = 5) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    unique = []
    for sentence in sentences:
        sentence = sentence.strip(" -*\t\r\n")
        sentence = re.sub(r"\s+([,.!?])", r"\1", sentence)
        if sentence and not _is_duplicate(sentence, unique):
            unique.append(sentence)
        if len(unique) == limit:
            break
    return " ".join(unique)


def _unique_points(text: str, limit: int = 7) -> list[str]:
    candidates = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    unique = []
    for point in candidates:
        point = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", point).strip(" -*\t\r\n")
        point = re.sub(r"\s+([,.!?])", r"\1", point)
        if point and not _is_duplicate(point, unique):
            unique.append(point)
        if len(unique) == limit:
            break
    return unique


_ISSUE_WORDS = re.compile(
    r"\b(?:problem|issue|error|crash(?:es|ed)?|slow(?:ly)?|unable|fail(?:ed|ure)?|"
    r"broken|needs? improvement|needs? work)\b",
    re.IGNORECASE,
)


def _include_important_issues(text: str, generated: list[str], limit: int) -> list[str]:
    result = list(generated)
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not _ISSUE_WORDS.search(sentence):
            continue
        sentence = re.sub(
            r"^\s*(?:I noticed(?: that)?|I found(?: that)?|we found(?: that)?)\s+",
            "",
            sentence,
            flags=re.IGNORECASE,
        ).strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
            if sentence[-1] not in ".!?":
                sentence += "."
            if not _is_duplicate(sentence, result):
                result.append(sentence)
        if len(result) >= limit:
            break
    return result[:limit]


def _discussion_text(
    text: str,
    action_items: list[dict[str, str]],
    keep_major_decisions: bool = False,
) -> str:
    task_keys = [
        (
            re.sub(r"[^a-z0-9]+", " ", item["task"].casefold()).strip(),
            item.get("assigned_to", "").casefold(),
        )
        for item in action_items
        if item.get("task")
    ]
    discussion_blocks = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if re.match(r"^(?:meeting title|date|time)\s*:", line, re.IGNORECASE):
            continue

        line = re.sub(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*:\s*", "", line)
        kept_sentences = []
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            normalized = re.sub(r"[^a-z0-9]+", " ", sentence.casefold()).strip()
            if not normalized or "main action items are" in normalized:
                continue
            if sentence.rstrip().endswith("?"):
                continue
            if re.match(
                r"^(?:good\s+(?:morning|afternoon|evening)|hello|hi|okay|ok|good|great|"
                r"agreed|perfect|thank you everyone)\b",
                normalized,
            ):
                continue
            if re.search(
                r"\b(?:we need to discuss|let['’]s (?:discuss|review|continue)|today we (?:need to finalize|will discuss|are reviewing)|the meeting will cover|let['’]s meet again)\b",
                normalized,
            ):
                continue
            matching_tasks = [(task, owner) for task, owner in task_keys if task and task in normalized]
            is_team_plan = keep_major_decisions and any(
                owner == "team" for _, owner in matching_tasks
            ) and re.search(r"\b(?:should|can|once|after|schedule|planned|let['’]s)\b", normalized)
            if matching_tasks and not is_team_plan:
                continue
            if normalized.startswith("that covers everything for today"):
                continue
            kept_sentences.append(sentence.strip())
        if kept_sentences:
            discussion_blocks.append(" ".join(kept_sentences))

    return "\n".join(discussion_blocks)


def summarize_discussion_points(
    text: str,
    summarizer=None,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    action_items: list[dict[str, str]] | None = None,
) -> list[str]:
    """Summarize pairs of related source sentences into topic-level bullets."""
    if not text.strip():
        return []

    text = _discussion_text(text, action_items or [], keep_major_decisions=False)
    tokenizer = summarizer.tokenizer if summarizer is not None else _load_summarizer()[0]
    summaries = [
        _summarize_chunk(chunk, summarizer, final=True)
        for chunk in _split_into_chunks(text, tokenizer, max_input_tokens)
    ]
    combined = " ".join(summaries)
    if len(tokenizer.encode(combined, add_special_tokens=False)) > max_input_tokens:
        combined = _summarize_chunk(combined, summarizer, final=True)
    points = _unique_points(combined)
    return _include_important_issues(text, points, limit=7)


def summarize_text(
    text: str,
    summarizer=None,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    action_items: list[dict[str, str]] | None = None,
) -> str:
    """Summarize the cleaned source discussion, including confirmed commitments."""
    if not text.strip():
        return ""

    text = _discussion_text(text, action_items or [], keep_major_decisions=True)
    tokenizer = summarizer.tokenizer if summarizer is not None else _load_summarizer()[0]
    chunks = _split_into_chunks(text, tokenizer, max_input_tokens)
    summaries = [_summarize_chunk(chunk, summarizer) for chunk in chunks]
    combined = " ".join(summaries)
    if len(chunks) == 1 or len(tokenizer.encode(combined, add_special_tokens=False)) <= max_input_tokens:
        combined = _summarize_chunk(combined, summarizer, final=True)
    summary = _unique_sentences(combined)
    summary_sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
    sentences = _include_important_issues(text, summary_sentences, limit=5)
    return " ".join(sentences)