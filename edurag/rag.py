"""Step 5: retrieve the top chunks from Qdrant and have the LLM answer from them, citing video + timestamp."""
import argparse

import ollama

from . import config, embedder, vectorstore

SYSTEM_PROMPT = """You answer questions about lecture videos using ONLY the numbered excerpts provided.
Rules:
- Use only information in the excerpts. If they don't contain the answer, say you couldn't find it in the lectures.
- Give a short, direct answer in your own words. Do not copy the excerpt headers.
- End with where to watch it, in the form: Watch "Video title" at mm:ss [n], where n is the excerpt number.
- Be concise."""


def _build_prompt(query: str, chunks: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"Excerpt [{i}] (video \"{c['title']}\", starts at {c['start_label']}):\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    return f"{excerpts}\n\nQuestion: {query}"


NOT_FOUND = "I couldn't find anything about that in the indexed lectures."


def retrieve(query: str) -> list[dict]:
    """Top chunks from Qdrant, keeping only the relevant ones (best first)."""
    hits = vectorstore.search(embedder.embed_query(query), config.TOP_K)
    if not hits:
        return []
    cutoff = max(config.MIN_SCORE, hits[0]["score"] * config.RELATIVE_SCORE)
    return [h for h in hits if h["score"] >= cutoff]


def stream_answer(query: str, chunks: list[dict]):
    """Yield the LLM's answer piece by piece."""
    client = ollama.Client(host=config.OLLAMA_URL)
    stream = client.chat(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(query, chunks)},
        ],
        options={"temperature": 0.2, "num_ctx": 4096, "num_predict": config.MAX_ANSWER_TOKENS},
        stream=True,
    )
    for part in stream:
        text = part["message"]["content"]
        if text:
            yield text


def answer(query: str) -> dict:
    """Return {"answer": str, "sources": [chunk dicts with title, start_label, link, score, ...]}."""
    chunks = retrieve(query)
    if not chunks:
        return {"answer": NOT_FOUND, "sources": []}
    return {"answer": "".join(stream_answer(query, chunks)).strip(), "sources": chunks}


def format_result(result: dict) -> str:
    lines = [result["answer"]]
    if result["sources"]:
        lines.append("\nSources:")
        for i, s in enumerate(result["sources"], 1):
            lines.append(f"  [{i}] {s['title']} @ {s['start_label']}  (score {s['score']:.2f})  {s['link']}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question about the indexed lectures")
    parser.add_argument("query", help="Your question")
    args = parser.parse_args()
    print(format_result(answer(args.query)))
