import json

import streamlit as st

from modules.action_items import extract_action_items
from modules.ner import extract_entities
from modules.preprocessing import clean_transcript, split_sentences
from modules.report import build_meeting_report
from modules.summarizer import summarize_discussion_points, summarize_text


st.set_page_config(page_title="Meeting Minutes Generator", page_icon="M", layout="wide")
st.title("AI Meeting Minutes Generator")
st.write("Turn a meeting transcript into a structured report.")

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
    if not transcript.strip():
        st.warning("Paste or upload a transcript first.")
    else:
        try:
            clean_text = clean_transcript(transcript)
            sentences = split_sentences(clean_text)
            entities = extract_entities(clean_text)
            action_items = extract_action_items(sentences, entities)

            with st.spinner("Generating an abstractive summary. The model downloads on first use."):
                discussion_points = summarize_discussion_points(clean_text, action_items=action_items)
                summary = summarize_text(clean_text, action_items=action_items)

            report = build_meeting_report(
                sentences=sentences,
                entities=entities,
                action_items=action_items,
                executive_summary=summary,
                discussion_points=discussion_points,
                source_text=clean_text,
            )
            st.session_state["meeting_report"] = report
            st.session_state["clean_transcript"] = clean_text
        except Exception as error:
            st.error(f"Could not process this transcript: {error}")
            st.info(
                "Check that en_core_web_sm is installed and that Hugging Face model downloads are available. "
                "See README.md for setup commands."
            )

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