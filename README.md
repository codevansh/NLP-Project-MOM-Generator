# AI Meeting Minutes Generator

A beginner-friendly Python and Streamlit NLP project. Text transcripts continue through the Phase 1 pipeline. Audio uses local faster-whisper transcription, pyannote speaker diarization, timestamp alignment, and user-provided speaker names before entering the same Phase 1 pipeline.

## Workflow

**Text:** Transcript → preprocessing → NER and information extraction → heuristic action items → abstractive summarization → meeting minutes.

**Audio:** Audio → faster-whisper → pyannote speaker diarization → timestamp alignment → user speaker mapping → Phase 1 NLP pipeline → meeting minutes.

Whisper supplies words and timestamps; pyannote identifies distinct voices. Diarization does not know the speakers' real names. In Audio File mode, enter a name for each detected `SPEAKER_XX`, or leave it blank to use **Unknown / Other**. If one Whisper segment overlaps multiple speakers, the app assigns the speaker with the largest time overlap, so a brief change within a segment can be missed. WAV audio is read through SoundFile; MP3/M4A support depends on the installed libsndfile build.

Action-item extraction uses readable commitment patterns rather than a trained classifier. It can miss commitments or assign an unclear speaker imperfectly. Transcript quality and spaCy's model also affect extracted metadata.

## Setup

Use Python 3.10 or newer, then create and activate a virtual environment. SoundFile reads audio into memory before pyannote processes it, avoiding TorchCodec for audio decoding:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Create a Hugging Face access token at <https://huggingface.co/settings/tokens> and accept the user conditions on the [pyannote Community-1 model page](https://huggingface.co/pyannote/speaker-diarization-community-1). Create a `.env` file in the project folder:

```text
HF_TOKEN=hf_your_token_here
```

The `.env` file is ignored by Git. Restart Streamlit after changing the token. The pipeline downloads model files on first use and caches them in your local user cache. Whisper uses CPU with `int8`; diarization also runs locally. No database is used.

## Run

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Choose **Text Transcript** to paste a transcript or upload a UTF-8 `.txt` file, then generate minutes.

Choose **Audio File** to upload a WAV, MP3, or M4A file. Select **Transcribe audio** to transcribe and detect speakers, map detected speaker IDs to names (optional), review or edit the labeled transcript, then generate minutes. Reports can be downloaded as JSON or plain text. Sample transcripts and `sample_meeting_audio.wav` are in `samples/`.

## Tests

```powershell
python -m pytest
```

Tests use fake model outputs and do not download model weights. Full audio inference additionally requires audio decoding support, model downloads, and accepted Hugging Face model conditions.

## Project structure

```text
app.py
modules/
  action_items.py
  diarization.py
  ner.py
  preprocessing.py
  pipeline.py
  report.py
  summarizer.py
  transcription.py
samples/
  transcripts/
    sample_meeting_audio.wav
tests/
```
