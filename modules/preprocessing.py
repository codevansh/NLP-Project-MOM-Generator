import re

import spacy


_FILLER_WORDS = re.compile(r"\b(?:um+|uh+|erm+|ah+)\b[, ]*", re.IGNORECASE)
_SENTENCE_PIPELINE = spacy.blank("en")
_SENTENCE_PIPELINE.add_pipe("sentencizer")


def clean_transcript(text: str) -> str:
    """Normalize line whitespace while keeping speaker labels on their lines."""
    text = _FILLER_WORDS.sub(" ", text)
    lines = [re.sub(r"[\t ]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def split_sentences(text: str) -> list[str]:
    """Split clean text into sentences without loading a language model."""
    return [sentence.text.strip() for sentence in _SENTENCE_PIPELINE(text).sents if sentence.text.strip()]