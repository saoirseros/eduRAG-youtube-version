"""Step 5: retrieve the top chunks from Qdrant and have the LLM answer from them, citing video + timestamp."""
import argparse

import ollama

from . import config, embedder, vectorstore

SYSTEM_PROMPT = """You answer questions about lecture videos using ONLY the numbered excerpts provided.
Rules:
- Use only information in the excerpts. If they don't contain the answer, say you couldn't find it in the lectures.
- Mention where the topic is taught: the video title and the timestamp (mm:ss), citing the excerpt number like [1].
- Be concise: a short explanation, then the best place(s) to watch."""


def _build_prompt(query: str, chunks: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"[{i}] Video: {c['title']} | Time: {c['start_label']}-{c['end_label']}\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    return f"Excerpts:\n{excerpts}\n\nQuestion: {query}"


def answer(query: str) -> dict:
    """Return {"answer": str, "sources": [chunk dicts with title, start_label, link, score, ...]}."""
    hits = vectorstore.search(embedder.embed_query(query), config.TOP_K)
    chunks = [h for h in hits if h["score"] >= config.MIN_SCORE]
    if not chunks:
        return {"answer": "I couldn't find anything about that in the indexed lectures.", "sources": []}

    client = ollama.Client(host=config.OLLAMA_URL)
    response = client.chat(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(query, chunks)},
        ],
        options={"temperature": 0.2, "num_ctx": 4096},
    )
    return {"answer": response["message"]["content"].strip(), "sources": chunks}


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
