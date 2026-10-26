"""Step 6: tiny web server. GET / serves the page, POST /api/ask streams sources first, then the answer."""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import embedder, rag

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger("edurag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder.embed_query("warm up")  # load the embedding model now so the first question isn't slow
    yield


app = FastAPI(title="EduRAG", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def _event(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False) + "\n"


def _answer_events(query: str):
    """Newline-delimited JSON events: one 'sources', many 'token', then 'done' (or 'error')."""
    try:
        chunks = rag.retrieve(query)
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
    return StreamingResponse(_answer_events(req.query.strip()), media_type="application/x-ndjson")
