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
WHISPER_LANGUAGE = "en"  # None = auto-detect

# Chunking
CHUNK_SECONDS = 75    # target length of each chunk
OVERLAP_SECONDS = 15  # how much of the previous chunk is repeated, so topics aren't cut at a boundary
CHUNK_DIR = DATA_DIR / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# Embeddings (runs locally on CPU)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Qdrant (the Docker container from docker-compose.yml)
QDRANT_URL = "http://localhost:6333"
COLLECTION = "lectures"
TOP_K = 5
