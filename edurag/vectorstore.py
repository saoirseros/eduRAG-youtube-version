"""Talk to the Qdrant container: create the collection, upload chunk vectors, search."""
import uuid

from qdrant_client import QdrantClient, models

from . import config

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        # collection/index creation can take several seconds on Docker for Windows
        _client = QdrantClient(url=config.QDRANT_URL, timeout=60)
    return _client


def ensure_collection(collection: str | None = None) -> None:
    """Create the collection (Qdrant's 'table') if it doesn't exist yet."""
    collection = collection or config.COLLECTION
    client = get_client()
    if client.collection_exists(collection):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=config.EMBEDDING_DIM, distance=models.Distance.COSINE),
    )
    # lets us filter / delete by video quickly
    client.create_payload_index(collection, "video_id", models.PayloadSchemaType.KEYWORD)


def delete_collection(collection: str) -> None:
    get_client().delete_collection(collection)


def _point_id(chunk_id: str) -> str:
    # Qdrant ids must be ints or UUIDs; a UUID derived from chunk_id makes re-indexing idempotent
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def replace_video_chunks(video_id: str, chunks: list[dict], vectors: list[list[float]],
                         collection: str | None = None) -> None:
    """Remove any old points for this video, then upload the fresh chunks with their vectors."""
    collection = collection or config.COLLECTION
    client = get_client()
    client.delete(
        collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))]
            )
        ),
        wait=True,
    )
    points = [
        models.PointStruct(id=_point_id(c["chunk_id"]), vector=v, payload=c)
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(points), 128):
        client.upsert(collection, points=points[i : i + 128], wait=True)


def search(query_vector: list[float], limit: int = config.TOP_K, collection: str | None = None) -> list[dict]:
    """Return the `limit` most similar chunks (payload + similarity score)."""
    res = get_client().query_points(
        collection or config.COLLECTION, query=query_vector, limit=limit, with_payload=True
    )
    return [{**p.payload, "score": p.score} for p in res.points]
