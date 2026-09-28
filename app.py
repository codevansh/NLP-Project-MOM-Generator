import json
from pathlib import Path

import streamlit as st

from modules.pipeline import generate_meeting_minutes
from modules.diarization import align_segments, diarize_audio, format_labeled_transcript
from modules.transcription import transcribe_audio, temporary_audio_file


st.set_page_config(page_title="Meeting Minutes Generator", page_icon="M", layout="wide")
st.title("AI Meeting Minutes Generator")
st.write("Generate structured minutes from a transcript or audio recording.")

input_mode = st.radio("Input type", ["Text Transcript", "Audio File"], horizontal=True)


def process_transcript(transcript: str, audio_transcript: bool = False) -> None:
    if not transcript.strip():
        st.warning("Paste or upload a transcript first.")
        return

    try:
        if audio_transcript:
            st.session_state["phase1_audio_input"] = transcript
        with st.spinner("Processing transcript and generating meeting minutes..."):
            clean_text, report = generate_meeting_minutes(
                transcript,
                audio_transcript=audio_transcript,
            )
        st.session_state["meeting_report"] = report
        st.session_state["clean_transcript"] = clean_text
    except Exception as error:
        st.error(f"Could not process this transcript: {error}")
        st.info(
            "Check that en_core_web_sm is installed and that Hugging Face model downloads are available. "
            "See README.md for setup commands."
        )


if input_mode == "Text Transcript":
    uploaded_file = st.file_uploader("Upload a transcript (.txt)", type=["txt"])
    uploaded_text = ""
    if uploaded_file is not None:
        uploaded_text = uploaded_file.getvalue().decode("utf-8", errors="replace")

    transcript = st.text_area(
        "Meeting transcript",
        value=uploaded_text,
        height=240,
        placeholder="Paste a transcript or upload a .txt file...",
    )
    if st.button("Generate meeting minutes", type="primary"):
        process_transcript(transcript)
else:
    audio_file = st.file_uploader(
        "Upload audio (.wav, .mp3, or .m4a)",
        type=["wav", "mp3", "m4a"],
        key="audio_upload",
    )
    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
        audio_suffix = Path(audio_file.name).suffix.lower()
        audio_identity = (audio_file.name, len(audio_bytes))
        if st.session_state.get("audio_identity") != audio_identity:
            st.session_state["audio_identity"] = audio_identity
            st.session_state["raw_audio_transcript"] = ""
            st.session_state["whisper_segments"] = []
            st.session_state["diarization_segments"] = []
            st.session_state["audio_transcript_editor"] = ""
            st.session_state["aligned_audio_segments"] = []
            st.session_state["audio_speaker_names"] = {}
            st.session_state["audio_mapping_key"] = None

        st.write(f"Uploaded file: {audio_file.name}")
        st.audio(audio_file)

        if st.button("Transcribe audio"):
            try:
                with temporary_audio_file(audio_bytes, audio_suffix) as temporary_path:
                    with st.spinner("Transcribing audio..."):
                        whisper_segments = transcribe_audio(temporary_path)
                    st.session_state["whisper_segments"] = whisper_segments
                    st.session_state["raw_audio_transcript"] = " ".join(
                        str(segment["text"]) for segment in whisper_segments
                    )
                    with st.spinner("Detecting speakers..."):
                        speaker_segments = diarize_audio(temporary_path)
                    st.session_state["diarization_segments"] = speaker_segments
                with st.spinner("Aligning transcript with speakers..."):
                    st.session_state["aligned_audio_segments"] = align_segments(
                        whisper_segments, speaker_segments
                    )
                st.session_state["audio_transcript_editor"] = format_labeled_transcript(
                    st.session_state["aligned_audio_segments"]
                )
                st.session_state["audio_speaker_names"] = {}
                st.success("Transcription and speaker detection complete.")
            except ValueError as error:
                st.error(str(error))
            except FileNotFoundError:
                st.error("The temporary audio file could not be read. Try uploading the audio again.")
            except Exception as error:
                st.error(str(error) or "Audio processing failed. Check the audio format and model setup.")

        aligned_segments = st.session_state.get("aligned_audio_segments", [])
        if aligned_segments:
            st.subheader("Detected speakers")
            diarization_segments = st.session_state.get("diarization_segments", [])
            detected_speakers = sorted(
                {str(segment["speaker"]) for segment in diarization_segments}
            )
            aligned_speaker_count = sum(
                str(segment["speaker"]).startswith("SPEAKER_")
                for segment in aligned_segments
            )
            if not diarization_segments:
                st.warning("Pyannote returned zero speaker turns for this audio.")
            elif aligned_speaker_count == 0:
                st.warning(
                    "No Whisper segments overlapped a detected speaker turn. "
                    "Review the segment timestamps in Developer diagnostics."
                )
            st.caption(
                f"Whisper segments: {len(st.session_state.get('whisper_segments', []))} · "
                f"Diarization turns: {len(diarization_segments)} · "
                f"Aligned segments: {len(aligned_segments)} · "
                f"Detected speakers: {', '.join(detected_speakers) or 'none'}"
            )
            with st.expander("Developer diagnostics: segment data and Phase 1 input"):
                st.write("Whisper segments")
                st.json(st.session_state.get("whisper_segments", []))
                st.write("Diarization segments")
                st.json(diarization_segments)
                st.write("Aligned segments")
                st.json(aligned_segments)
                st.write("Transcript passed to Phase 1 (after generation)")
                st.text(st.session_state.get("phase1_audio_input", "Not generated yet"))
            with st.expander("View original Whisper transcript"):
                st.text(st.session_state.get("raw_audio_transcript", ""))
            speakers = detected_speakers
            speaker_names = {}
            for speaker in speakers:
                speaker_names[speaker] = st.text_input(
                    f"{speaker} — enter a name (leave blank to keep the speaker ID)",
                    key=f"speaker_name_{speaker}_{audio_identity[1]}",
                )
            speaker_names = {speaker: name.strip() for speaker, name in speaker_names.items()}
            st.session_state["audio_speaker_names"] = speaker_names
            mapping_key = tuple(sorted(speaker_names.items()))
            if st.session_state.get("audio_mapping_key") != mapping_key:
                st.session_state["audio_mapping_key"] = mapping_key
                st.session_state["audio_transcript_editor"] = format_labeled_transcript(
                    aligned_segments, speaker_names
                )
            mapped_segment_count = sum(
                bool(speaker_names.get(str(segment["speaker"]), ""))
                for segment in aligned_segments
            )
            st.caption(f"Segments mapped to a real name: {mapped_segment_count} / {len(aligned_segments)}")

            st.subheader("Editable speaker-labeled transcript")
            reviewed_transcript = st.text_area(
                "Review or edit the transcript before generating minutes",
                height=240,
                key="audio_transcript_editor",
            )
            if st.button("Generate meeting minutes from transcript", type="primary"):
                process_transcript(reviewed_transcript, audio_transcript=True)

if "meeting_report" in st.session_state:
    report = st.session_state["meeting_report"]
    st.subheader("Clean transcript")
    st.text(st.session_state["clean_transcript"])

    st.subheader("Meeting metadata")
    metadata = report["meeting_metadata"]
    st.write(f"Date: {metadata['date'] or 'Not detected'}")
    st.write(f"Time: {metadata['time'] or 'Not detected'}")
    st.write(f"Attendees: {', '.join(metadata['attendees']) or 'Not detected'}")

    st.subheader("Named entities")
    if report["entities"]:
        st.dataframe(report["entities"], use_container_width=True, hide_index=True)
    else:
        st.write("No named entities detected.")

    st.subheader("Executive summary")
    st.write(report["executive_summary"])

    st.subheader("Key discussion points")
    if report["key_discussion_points"]:
        for point in report["key_discussion_points"]:
            st.markdown(f"- {point}")
    else:
        st.write("No separate discussion points detected.")

    st.subheader("Action items")
    if report["action_items"]:
        st.dataframe(report["action_items"], use_container_width=True, hide_index=True)
    else:
        st.write("No action items detected.")

    json_report = json.dumps(report, indent=2, ensure_ascii=False)
    text_report = "\n".join(
        [
            "MEETING MINUTES",
            "",
            "Meeting metadata",
            f"Date: {metadata['date'] or 'Not detected'}",
            f"Time: {metadata['time'] or 'Not detected'}",
            f"Attendees: {', '.join(metadata['attendees']) or 'Not detected'}",
            "",
            "Executive summary",
            report["executive_summary"],
            "",
            "Key discussion points",
            *[f"- {point}" for point in report["key_discussion_points"]],
            "",
            "Action items",
            *[
                f"- {item['task']} | Assigned to: {item['assigned_to']} | Deadline: {item['deadline'] or 'Not detected'}"
                for item in report["action_items"]
            ],
        ]
    )
    st.download_button("Download JSON report", json_report, "meeting_minutes.json", "application/json")
    st.download_button("Download TXT report", text_report, "meeting_minutes.txt", "text/plain")
