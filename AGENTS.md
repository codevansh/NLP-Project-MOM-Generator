# Project guidance

- Keep this an approachable Python and Streamlit project. Prefer small functions and readable code.
- Phase 1 is transcript-only. Do not add audio transcription, a database, authentication, or unrelated frameworks.
- Keep preprocessing, NER, action-item extraction, summarization, and report assembly in their own modules.
- Use real NLP libraries where specified; do not replace the pipeline with hardcoded sample output or an LLM-only prompt.
- Explain heuristic action-item extraction honestly. Preserve transcript meaning during cleaning.
- Keep model loading lazy so tests and app startup do not download weights unnecessarily.
- Run `python -m pytest` after changing processing logic.
- Keep generated reports out of version control; the `outputs/` folder is only a placeholder in Phase 1.