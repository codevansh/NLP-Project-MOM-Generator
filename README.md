# AI Meeting Minutes Generator

A beginner-friendly NLP lab project that turns a written meeting transcript into structured minutes. Phase 1 processes pasted or uploaded `.txt` transcripts. Audio transcription is deliberately not implemented yet.

## Pipeline

1. Normalize whitespace and remove a few unambiguous speech fillers.
2. Split the cleaned transcript into sentences.
3. Extract named entities with spaCy's pretrained English model.
4. Find likely commitments with readable trigger-word patterns and use nearby PERSON, DATE, and TIME entities when available.
5. Generate an abstractive summary with the Hugging Face `sshleifer/distilbart-cnn-6-6` model.
6. Assemble metadata, discussion points, entities, and action items into a Python dictionary.

Action-item extraction is a small explainable heuristic, not a trained classifier. It can miss commitments or assign an unclear speaker imperfectly. spaCy model quality and the transcript itself also affect extracted metadata.

## Setup

Use Python 3.10 or newer. From this project folder, create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The summarizer downloads its Hugging Face model the first time a report is generated and caches it locally. The model runs on CPU by default and needs an internet connection for its first download. No database is used.

## Run

```powershell
streamlit run app.py
```

Paste a transcript or upload a UTF-8 `.txt` file, then select **Generate meeting minutes**. The report can be downloaded as JSON or plain text. A sample transcript is in `samples/transcript.txt`.

## Tests

```powershell
python -m pytest
```

The tests use a small fake summarizer and an in-memory spaCy entity ruler, so they do not need to download model weights.

## Project structure

```text
app.py
modules/
  action_items.py
  ner.py
  preprocessing.py
  report.py
  summarizer.py
samples/
  transcript.txt
  audio/       # Reserved for a later phase
outputs/       # Reserved for optional saved reports
tests/
```

Phase 2 can add audio upload and speech-to-text in a separate module after the transcript workflow is working.