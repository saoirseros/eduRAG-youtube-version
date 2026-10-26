from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
META_DIR = DATA_DIR / "metadata"

for d in (AUDIO_DIR, META_DIR):
    d.mkdir(parents=True, exist_ok=True)
