# Project guidance

- Keep this an approachable Python and Streamlit project. Prefer small functions and readable code.
- Phase 1 supports transcripts and Phase 2 supports uploaded audio through local faster-whisper. Do not add diarization, databases, authentication, or Phase 3 features.
- Keep preprocessing, NER, action-item extraction, summarization, and report assembly in their own modules.
- Use real NLP libraries where specified; do not replace the pipeline with hardcoded sample output or an LLM-only prompt.
- Explain heuristic action-item extraction honestly. Preserve transcript meaning during cleaning.
- Keep model loading lazy so tests and app startup do not download weights unnecessarily.
- Keep Whisper on CPU with int8 quantization; store uploaded audio in the OS temporary directory and always clean it up.
- Run `python -m pytest` after changing processing logic.
- Keep generated reports and model files out of version control; model weights belong in the local user cache.