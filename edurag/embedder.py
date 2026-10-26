"""Turn text into vectors with a small local embedding model (all-MiniLM-L6-v2, 384 dimensions)."""
from sentence_transformers import SentenceTransformer

from . import config

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model '{config.EMBEDDING_MODEL}' ...")
        _model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    # normalized vectors make cosine similarity a plain dot product
    vectors = _get_model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
