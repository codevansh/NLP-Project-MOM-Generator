from pathlib import Path
import soundfile as sf
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "samples" / "audio"

parts = [
    AUDIO_DIR / "p1.mp3",
    AUDIO_DIR / "p2.mp3",
    AUDIO_DIR / "p3.mp3",
    AUDIO_DIR / "p4.mp3",
]

output = AUDIO_DIR / "multi_speaker_meeting.wav"

audio_parts = []
sample_rate = None
channels = None

for path in parts:
    if not path.exists():
        raise FileNotFoundError(f"Missing audio file: {path}")

    audio, sr = sf.read(path, always_2d=True)

    print(f"\n{path.name}")
    print(f"  Sample rate: {sr}")
    print(f"  Channels: {audio.shape[1]}")
    print(f"  Duration: {len(audio) / sr:.2f} seconds")

    if sample_rate is None:
        sample_rate = sr
        channels = audio.shape[1]
    else:
        if sr != sample_rate:
            raise ValueError(
                f"Sample-rate mismatch: {path.name} has {sr}, "
                f"expected {sample_rate}"
            )

        if audio.shape[1] != channels:
            raise ValueError(
                f"Channel mismatch: {path.name} has {audio.shape[1]}, "
                f"expected {channels}"
            )

    audio_parts.append(audio)

combined = np.concatenate(audio_parts, axis=0)

sf.write(output, combined, sample_rate)

duration = len(combined) / sample_rate


print("COMBINATION COMPLETE")

print(f"Output: {output}")
print(f"Sample rate: {sample_rate}")
print(f"Channels: {channels}")
print(f"Duration: {duration:.2f} seconds")
