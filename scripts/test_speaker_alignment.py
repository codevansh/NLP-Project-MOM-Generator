import os
from pathlib import Path

import soundfile as sf
import torch
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_PATH = PROJECT_ROOT / "samples" / "audio" / "multi_speaker_meeting.wav"
MODEL_NAME = "pyannote/speaker-diarization-community-1"


def transcribe_audio():
    print("\n[1/3] Loading Whisper and requesting word timestamps...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segment_generator, info = model.transcribe(
        str(AUDIO_PATH),
        beam_size=5,
        language="en",
        condition_on_previous_text=False,
        word_timestamps=True,
    )
    # faster-whisper returns a generator; materialize it before using its words.
    segments = list(segment_generator)
    print(f"Language: {info.language}")
    print(f"Duration: {info.duration:.2f}s")
    print(f"Whisper segments: {len(segments)}")
    return segments


def diarize_audio():
    print("\n[2/3] Loading pyannote...")
    load_dotenv()
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN not found in .env")

    audio, sample_rate = sf.read(str(AUDIO_PATH), always_2d=True, dtype="float32")
    waveform = torch.from_numpy(audio.T).float()
    pipeline = Pipeline.from_pretrained(MODEL_NAME, token=token)
    output = pipeline({"waveform": waveform, "sample_rate": int(sample_rate)})

    # Prefer exclusive turns for transcript alignment; retain compatibility with
    # a pipeline result that does not expose them.
    diarization = getattr(output, "exclusive_speaker_diarization", None)
    if diarization is None:
        diarization = output.speaker_diarization

    speaker_segments = [
        {
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        }
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    speaker_ids = sorted({item["speaker"] for item in speaker_segments})
    print(f"Pyannote speaker segments: {len(speaker_segments)}")
    print(f"Detected speaker IDs: {speaker_ids}")
    return speaker_segments


def word_records(whisper_segments):
    """Flatten Whisper segments while retaining each word and its timestamps."""
    words = []
    for segment in whisper_segments:
        for word in segment.words or []:
            start = None if word.start is None else float(word.start)
            end = None if word.end is None else float(word.end)
            words.append({"start": start, "end": end, "text": word.word})
    return words


def overlap_seconds(word_start, word_end, turn_start, turn_end):
    """Return positive interval overlap; touching or invalid intervals do not match."""
    if word_start is None or word_end is None or word_end <= word_start:
        return 0.0
    if turn_end <= turn_start:
        return 0.0
    return max(0.0, min(word_end, turn_end) - max(word_start, turn_start))


def assign_word_speaker(word, speaker_segments):
    """Choose the speaker with the greatest timestamp overlap for one word."""
    start, end = word["start"], word["end"]
    if start is None or end is None or end <= start:
        return "UNKNOWN"

    word_duration = end - start
    best_speaker = "UNKNOWN"
    best_score = (0.0, 0.0)
    for turn in speaker_segments:
        overlap = overlap_seconds(start, end, turn["start"], turn["end"])
        # The ratio is a secondary tie-breaker, as requested. For a given word
        # its duration is constant, so exact ties remain stable in turn order.
        ratio = overlap / word_duration
        score = (overlap, ratio)
        if overlap > 0 and score > best_score:
            best_score = score
            best_speaker = turn["speaker"]
    return best_speaker


def align_words(words, speaker_segments):
    aligned = []
    for word in words:
        aligned.append({**word, "speaker": assign_word_speaker(word, speaker_segments)})
    return aligned


def group_words_by_speaker(aligned_words):
    """Combine adjacent word records with the same speaker into transcript turns."""
    turns = []
    for word in aligned_words:
        speaker = word["speaker"]
        if not turns or turns[-1]["speaker"] != speaker:
            turns.append(
                {
                    "start": word["start"],
                    "end": word["end"],
                    "speaker": speaker,
                    "text": str(word["text"]),
                }
            )
            continue

        turn = turns[-1]
        if turn["start"] is None and word["start"] is not None:
            turn["start"] = word["start"]
        if word["end"] is not None:
            turn["end"] = word["end"]
        turn["text"] += str(word["text"])

    for turn in turns:
        turn["text"] = turn["text"].strip()
    return turns


def print_turns(title, turns, speaker_mapping=None):
    
    print(title)

    for turn in turns:
        start = "   ?" if turn["start"] is None else f"{turn['start']:6.2f}"
        end = "   ?" if turn["end"] is None else f"{turn['end']:6.2f}"
        speaker = turn["speaker"]
        if speaker_mapping is not None:
            speaker = speaker_mapping.get(speaker, speaker)
        print(f"[{start}s -> {end}s] {speaker}:")
        print(turn["text"])


def main():

    print("WORD-LEVEL SPEAKER ALIGNMENT TEST")

    whisper_segments = transcribe_audio()
    words = word_records(whisper_segments)
    speaker_segments = diarize_audio()
    aligned_words = align_words(words, speaker_segments)
    speaker_turns = group_words_by_speaker(aligned_words)

    detected_speakers = sorted({turn["speaker"] for turn in speaker_segments})
    unknown_words = sum(word["speaker"] == "UNKNOWN" for word in aligned_words)
    print("\nALIGNMENT DIAGNOSTICS")
    print(f"Whisper segments: {len(whisper_segments)}")
    print(f"Whisper words: {len(words)}")
    print(f"Pyannote speaker segments: {len(speaker_segments)}")
    print(f"Detected speaker IDs: {detected_speakers}")
    print(f"UNKNOWN words: {unknown_words}")
    print(f"Final speaker turns: {len(speaker_turns)}")

    print_turns("WORD-LEVEL SPEAKER ALIGNMENT", speaker_turns)
    
    print("ALIGNMENT COMPLETE")

    # Optional display-only mapping. The aligned records retain anonymous IDs.
    speaker_mapping = {
        "SPEAKER_00": "Priya",
        "SPEAKER_01": "Neha",
        "SPEAKER_02": "Rahul",
        "SPEAKER_03": "Amit",
    }
    print_turns("MAPPED SPEAKER TRANSCRIPT", speaker_turns, speaker_mapping)


if __name__ == "__main__":
    main()
