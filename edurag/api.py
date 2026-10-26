"""Web server: serves the page, manages playlists (add / build / cancel / delete) and streams answers."""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config, embedder, jobs, pipeline, playlists, rag, vectorstore

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger("edurag")
runner = jobs.JobRunner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder.embed_query("warm up")  # load the embedding model now so the first question isn't slow
    try:
        playlists.seed_existing_collection()
    except Exception:
        log.warning("could not register the existing collection (is Qdrant running?)", exc_info=True)
    jobs.recover_interrupted()
    runner.start()
    yield


app = FastAPI(title="EduRAG", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    playlist_id: str | None = Field(default=None, max_length=64)


class AddPlaylistRequest(BaseModel):
    url: str = Field(min_length=1, max_length=300)
    max_videos: int = Field(default=config.DEFAULT_MAX_VIDEOS, ge=1, le=50)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def _public(record: dict) -> dict:
    """The registry record plus the numbers the UI needs for its progress display."""
    videos = record["videos"]
    return {
        **record,
        "videos_done": sum(v["status"] == "done" for v in videos),
        "videos_total": len(videos),
        "total_duration": sum(v.get("duration") or 0 for v in videos),
        "active": runner.is_active(record["id"]),
    }


def _get_or_404(playlist_id: str) -> dict:
    record = playlists.get_playlist(playlist_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Playlist not found.")
    return record


# these are plain `def`s: FastAPI runs them in a worker thread, so slow network calls don't block the server
@app.get("/api/playlists")
def list_playlists():
    return [_public(p) for p in playlists.list_playlists()]


@app.get("/api/playlists/{playlist_id}")
def get_playlist(playlist_id: str):
    return _public(_get_or_404(playlist_id))


@app.post("/api/playlists")
def add_playlist(req: AddPlaylistRequest):
    """Validate the URL and list the videos. Nothing is downloaded until /start is called."""
    try:
        playlist_id = playlists.parse_playlist_url(req.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    existing = playlists.get_playlist(playlist_id)
    if existing and runner.is_active(playlist_id):
        return _public(existing)  # already building; don't reset its progress
    try:
        record = pipeline.prepare_playlist(req.url, req.max_videos, config.cookie_opts_from_env())
    except Exception as exc:
        log.exception("could not read playlist %s", playlist_id)
        raise HTTPException(status_code=502, detail=f"Could not read that playlist: {str(exc)[:150]}")
    return _public(record)


@app.post("/api/playlists/{playlist_id}/start")
def start_playlist(playlist_id: str):
    record = _get_or_404(playlist_id)
    if not runner.submit(playlist_id):
        raise HTTPException(status_code=409, detail="This playlist is already being built.")
    return _public(playlists.get_playlist(record["id"]))


@app.post("/api/playlists/{playlist_id}/cancel")
def cancel_playlist(playlist_id: str):
    _get_or_404(playlist_id)
    if not runner.cancel(playlist_id):
        raise HTTPException(status_code=409, detail="This playlist isn't being built.")
    return {"ok": True}


@app.delete("/api/playlists/{playlist_id}")
def delete_playlist(playlist_id: str):
    record = _get_or_404(playlist_id)
    if runner.is_active(playlist_id):
        raise HTTPException(status_code=409, detail="Cancel the build before deleting this playlist.")
    if not record["collection"].startswith("pl_"):
        raise HTTPException(status_code=400, detail="The built-in sample can't be deleted.")
    try:
        vectorstore.delete_collection(record["collection"])
    except Exception:
        log.exception("could not drop collection %s", record["collection"])
        raise HTTPException(status_code=503, detail="Could not reach Qdrant. Is the Docker container running?")
    playlists.delete_playlist(playlist_id)
    return {"ok": True}


def _event(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False) + "\n"


def _answer_events(query: str, collection: str):
    """Newline-delimited JSON events: one 'sources', many 'token', then 'done' (or 'error')."""
    try:
        chunks = rag.retrieve(query, collection)
        yield _event({
            "type": "sources",
            "sources": [
                {
                    "n": i,
                    "title": s["title"],
                    "video_id": s["video_id"],
                    "start": s["start"],
                    "start_label": s["start_label"],
                    "link": s["link"],
                    "score": round(s["score"], 3),
                }
                for i, s in enumerate(chunks, 1)
            ],
        })
        if not chunks:
            yield _event({"type": "token", "text": rag.NOT_FOUND})
        else:
            for text in rag.stream_answer(query, chunks):
                yield _event({"type": "token", "text": text})
        yield _event({"type": "done"})
    except Exception:
        log.exception("ask failed")
        yield _event({
            "type": "error",
            "detail": "Could not reach Qdrant or Ollama. Check that Docker (edurag-qdrant) and Ollama are running.",
        })


# a plain generator: Starlette iterates it in a worker thread, so slow LLM calls don't block the server
@app.post("/api/ask")
def ask(req: AskRequest):
    collection = config.COLLECTION
    if req.playlist_id:
        record = _get_or_404(req.playlist_id)
        if not any(v["status"] == "done" for v in record["videos"]):
            raise HTTPException(status_code=409, detail="Nothing in this playlist is searchable yet.")
        collection = record["collection"]
    return StreamingResponse(_answer_events(req.query.strip(), collection), media_type="application/x-ndjson")
