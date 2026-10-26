"""Step 4: embed every chunk and upload the vectors to Qdrant. Also a quick CLI to test search."""
import argparse
import json

from . import config, embedder, vectorstore


def index_all(video_id: str | None = None) -> None:
    paths = sorted(config.CHUNK_DIR.glob("*.json"))
    if video_id:
        paths = [p for p in paths if p.stem == video_id]
    if not paths:
        print("No chunk files found. Run the chunker first.")
        return

    vectorstore.ensure_collection()
    total = 0
    for p in paths:
        chunks = json.loads(p.read_text(encoding="utf-8"))
        if not chunks:
            continue
        vectors = embedder.embed_texts([c["text"] for c in chunks])
        vectorstore.replace_video_chunks(p.stem, chunks, vectors)
        total += len(chunks)
        print(f"[indexed] {p.stem}  {len(chunks):4d} chunks  {chunks[0]['title']}")

    count = vectorstore.get_client().count(config.COLLECTION, exact=True).count
    print(f"\nDone. Uploaded {total} chunks. Collection '{config.COLLECTION}' now holds {count} points.")


def search_cli(query: str) -> None:
    results = vectorstore.search(embedder.embed_query(query))
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r['score']:.3f}  {r['title']}  @ {r['start_label']}  {r['link']}")
        print(f"    {r['text'][:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunks and upload them to Qdrant")
    parser.add_argument("--video-id", help="Only index this video")
    parser.add_argument("--search", metavar="QUERY", help="Instead of indexing, test a search query")
    args = parser.parse_args()

    if args.search:
        search_cli(args.search)
    else:
        index_all(args.video_id)
