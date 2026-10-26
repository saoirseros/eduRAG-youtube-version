import os
import sys
from pathlib import Path

# video titles often contain emojis, which the default Windows console encoding can't print
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

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
CHUNK_DIR = Path(os.getenv("EDURAG_CHUNK_DIR", DATA_DIR / "chunks"))
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# Embeddings (runs locally on CPU)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Qdrant (the Docker container from docker-compose.yml)
QDRANT_URL = "http://localhost:6333"
COLLECTION = os.getenv("EDURAG_COLLECTION", "lectures")  # set to "lectures_demo" to use the dummy data
TOP_K = 5
# chunks scoring below this are treated as unrelated (MiniLM: relevant ~0.4-0.6, unrelated <0.2); tune if needed
MIN_SCORE = 0.25
# also drop chunks scoring below this fraction of the best match; fewer chunks = much faster LLM on CPU
RELATIVE_SCORE = 0.7
MAX_ANSWER_TOKENS = 300

# LLM (Ollama runs on the host, not in Docker)
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2"
