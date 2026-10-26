"""Step 6: tiny web server. GET / serves the page, POST /api/ask returns the answer + sources."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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


# plain `def` (not async): FastAPI runs it in a worker thread, so slow LLM calls don't block the server
@app.post("/api/ask")
def ask(req: AskRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Please type a question.")
    try:
        result = rag.answer(query)
    except Exception:
        log.exception("ask failed")
        raise HTTPException(
            status_code=503,
            detail="Could not reach Qdrant or Ollama. Check that Docker (edurag-qdrant) and Ollama are running.",
        )
    return {
        "answer": result["answer"],
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
            for i, s in enumerate(result["sources"], 1)
        ],
    }
