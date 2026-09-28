# AI Meeting Minutes Generator

A beginner-friendly Python and Streamlit NLP project for automatically generating structured meeting minutes and action items from text transcripts and recorded meeting audio.

Text transcripts continue through the Phase 1 NLP pipeline. Recorded audio is processed using local Faster-Whisper transcription, Pyannote speaker diarization, timestamp alignment, and optional user-provided speaker names before entering the same Phase 1 NLP pipeline.

## Workflow

**Text:**

Transcript → preprocessing → NER and information extraction → heuristic action items → abstractive summarization → meeting minutes.

**Audio:**

Audio → Faster-Whisper → timestamped transcript → Pyannote speaker diarization → timestamp alignment → user speaker mapping → Phase 1 NLP pipeline → meeting minutes.

Whisper supplies transcription text and timestamps, while Pyannote identifies distinct speaker segments. Diarization does not know the speakers' real names. In Audio File mode, detected `SPEAKER_XX` IDs can be mapped to participant names by the user, or left as **Unknown / Other**.

The current audio pipeline has been tested with a controlled multi-speaker meeting recording and successfully detected four distinct speaker clusters:

```text
SPEAKER_00
SPEAKER_01
SPEAKER_02
SPEAKER_03

## Setup

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm

Create a Hugging Face access token at: https://huggingface.co/settings/tokens

Accept the required user conditions on the Pyannote Community-1 model page: https://huggingface.co/pyannote/speaker-diarization-community-1

Create a .env file in the project folder: HF_TOKEN=hf_your_token_here


The project was build in 2 phases:

Phase-1:
  Transcript
      ↓
  Preprocessing
      ↓
  NER & Information Extraction
      ↓
  Action Item Extraction
      ↓
  Abstractive Summarization
      ↓
  Meeting Minutes

Phase 1 includes:
-- Text preprocessing
-- Named Entity Recognition
-- Information extraction
-- Action-item extraction
-- Abstractive summarization
-- Meeting report generation

Phase-2:
  Recorded Audio
        ↓
  Faster-Whisper
        ↓
  Timestamped Transcript
        +
  Pyannote Audio
        ↓
  Speaker Diarization
        ↓
  Timestamp Alignment
        ↓
  Speaker-Labeled Transcript
        ↓
  Phase 1 NLP Pipeline
        ↓
  Meeting Minutes

The Phase 2 components have been independently tested:

Faster-Whisper successfully transcribes the recorded meeting audio.
Whisper provides timestamped transcript segments.
Pyannote Community-1 successfully detects four speaker clusters in the controlled multi-speaker test audio.
Timestamp alignment produces a speaker-labelled transcript.
Speaker IDs can be mapped to user-provided participant names.

** Project Structure **
meeting-minutes-generator/
│
├── app.py
│
├── modules/
│   ├── action_items.py
│   ├── diarization.py
│   ├── ner.py
│   ├── preprocessing.py
│   ├── pipeline.py
│   ├── report.py
│   ├── summarizer.py
│   └── transcription.py
│
├── scripts/
│   └── test_speaker_alignment.py
│
├── samples/
│   ├── transcripts/
│   └── audio/
│       └── multi_speaker_meeting.wav
│
├── tests/
│
├── outputs/
│
├── requirements.txt
├── README.md
├── AGENTS.md
├── .gitignore
└── .env

Technology Stack:
  Programming:
    Python

  Interface:
    Streamlit

  Speech-to-Text:
    Faster-Whisper

  Speaker Diarization
    Pyannote Audio
    pyannote/speaker-diarization-community-1
    NLP
    spaCy
    NLTK
    Hugging Face Transformers
    Audio / ML
    PyTorch
    SoundFile
    Configuration
    python-dotenv