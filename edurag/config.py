from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
META_DIR = DATA_DIR / "metadata"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"

for d in (AUDIO_DIR, META_DIR, TRANSCRIPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Whisper (runs locally on CPU)
WHISPER_MODEL = "small"
WHISPER_COMPUTE_TYPE = "int8"
# "hi" handles Hindi/Hinglish lectures; use "en" for English-only, or None to auto-detect
WHISPER_LANGUAGE = "hi"
# Romanised Hinglish prompt nudges Whisper to write Roman script instead of Devanagari
WHISPER_PROMPT = "Aaj hum sliding window ka concept padhenge. Array aur string ke questions solve karenge."
