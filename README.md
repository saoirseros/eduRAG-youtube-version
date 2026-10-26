# EduRAG

Turn any educational YouTube playlist into a searchable knowledge base. Paste a playlist link, let it build, then ask questions. Each answer cites the **video title and timestamp**, and a player jumps straight to that moment.

Everything runs locally: no API keys, no cloud services.

## How it works

```
playlist URL
   -> yt-dlp            list videos (+ English captions when available)
   -> faster-whisper    transcribe audio only when a video has no captions
   -> chunker           ~75 s timestamped chunks (15 s overlap), tagged with video id, title, time
   -> MiniLM            384-d embeddings (sentence-transformers all-MiniLM-L6-v2)
   -> Qdrant            one collection per playlist
question -> embed -> top 5 chunks (weak matches dropped) -> Ollama llama3.2 -> answer + sources
```

- **Captions first:** videos with YouTube captions are ready in seconds. Others are downloaded (audio only) and transcribed locally, which takes roughly as long as the video.
- **Background builds:** one playlist builds at a time. You can search finished videos while the rest build, and cancel or resume at any time.
- **English only.**

## Requirements

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (runs Qdrant)
- [Ollama](https://ollama.com/) with the `llama3.2` model

## Setup

```powershell
# 1. Python environment
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# 2. Vector database (Qdrant) in Docker
docker compose up -d

# 3. Language model
ollama pull llama3.2
```

The first run also downloads the embedding model (about 90 MB) and, if a video needs it, the Whisper `small` model (about 500 MB).

## Run

```powershell
.\.venv\Scripts\python -m uvicorn edurag.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000, paste a playlist link, choose how many videos to include, press **Add**, then **Build**.

The first start takes a little while because the embedding model loads. Answers are generated on the CPU by default, so expect roughly one to two minutes per answer; the sources and player appear immediately.

## Command line

Each stage can also be run on its own:

```powershell
.\.venv\Scripts\python -m edurag.pipeline "<playlist url>" --max-videos 2   # build a playlist
.\.venv\Scripts\python -m edurag.rag "your question"                        # ask from the terminal
```

Stage-by-stage tools: `edurag.downloader`, `edurag.transcriber`, `edurag.chunker`, `edurag.indexer`.

## Project layout

| Path | Purpose |
|---|---|
| `edurag/pipeline.py` | Runs every stage for a video or a whole playlist |
| `edurag/jobs.py` | Background worker with cancel and restart recovery |
| `edurag/playlists.py` | URL validation, playlist listing, registry (`data/playlists.json`) |
| `edurag/captions.py` | YouTube captions (fast path) |
| `edurag/transcriber.py` | Whisper fallback |
| `edurag/chunker.py` | Timestamped chunks |
| `edurag/embedder.py`, `vectorstore.py` | Embeddings and Qdrant |
| `edurag/rag.py` | Retrieval and streamed answers |
| `edurag/api.py`, `static/index.html` | Web server and UI |
| `edurag/config.py` | Settings (models, chunk size, thresholds) |

## Configuration

Settings live in `edurag/config.py`. Optional environment variables:

| Variable | Purpose |
|---|---|
| `EDURAG_COOKIES_FROM_BROWSER` | e.g. `edge` or `chrome`; only needed if YouTube shows "Sign in to confirm you're not a bot" |
| `EDURAG_COOKIES_FILE` | Path to a `cookies.txt`; alternative to the above |
| `EDURAG_COLLECTION` | Qdrant collection used by `edurag.rag` and `edurag.indexer` when no playlist is selected |

## Notes

- Downloaded audio, transcripts and the Qdrant data are stored in `data/` and `qdrant_storage/`. Both are git-ignored.
- Never commit cookie files. `cookies*.txt` and `.env` are git-ignored.
- The server binds to `127.0.0.1` only; it is not meant to be exposed to a network.
- Answers come only from the retrieved transcript excerpts. If nothing relevant is found, the app says so instead of guessing.
- Only use it on videos you are allowed to download and transcribe; respect YouTube's terms and creators' rights.
