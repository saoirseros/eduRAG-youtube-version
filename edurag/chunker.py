"""Step 3: group transcript segments into ~75 second chunks that carry video id, title and timestamps."""
import argparse
import json

from . import config


def format_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def chunk_segments(segments: list[dict], chunk_seconds: float, overlap_seconds: float) -> list[list[dict]]:
    """Group consecutive segments into chunks of >= chunk_seconds, never splitting a segment.

    Each chunk repeats the last ~overlap_seconds of the previous one so a topic that
    straddles a boundary is still fully contained in at least one chunk.
    """
    chunks = []
    n = len(segments)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and segments[j]["end"] - segments[i]["start"] < chunk_seconds:
            j += 1
        chunks.append(segments[i : j + 1])
        if j == n - 1:
            break
        # step back from j+1 to include segments that start inside the overlap window
        k = j + 1
        while k - 1 > i and segments[k - 1]["start"] >= segments[j]["end"] - overlap_seconds:
            k -= 1
        i = k
    return chunks


def chunk_transcript(transcript: dict, meta: dict | None = None,
                     chunk_seconds: float = config.CHUNK_SECONDS,
                     overlap_seconds: float = config.OVERLAP_SECONDS) -> list[dict]:
    video_id = transcript["video_id"]
    url = (meta or {}).get("url") or f"https://www.youtube.com/watch?v={video_id}"
    result = []
    for idx, group in enumerate(chunk_segments(transcript["segments"], chunk_seconds, overlap_seconds)):
        start, end = group[0]["start"], group[-1]["end"]
        result.append({
            "chunk_id": f"{video_id}_{idx:04d}",
            "video_id": video_id,
            "title": transcript["title"],
            "start": start,
            "end": end,
            "start_label": format_time(start),
            "end_label": format_time(end),
            "link": f"{url}&t={int(start)}s",
            "text": " ".join(s["text"] for s in group),
        })
    return result


def chunk_all(video_id: str | None = None, chunk_seconds: float = config.CHUNK_SECONDS,
              overlap_seconds: float = config.OVERLAP_SECONDS) -> None:
    paths = sorted(config.TRANSCRIPT_DIR.glob("*.json"))
    if video_id:
        paths = [p for p in paths if p.stem == video_id]
    if not paths:
        print("No transcripts found. Run the transcriber first.")
        return

    total = 0
    for p in paths:
        transcript = json.loads(p.read_text(encoding="utf-8"))
        meta_path = config.META_DIR / f"{p.stem}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        chunks = chunk_transcript(transcript, meta, chunk_seconds, overlap_seconds)
        (config.CHUNK_DIR / f"{p.stem}.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        total += len(chunks)
        print(f"[chunked] {p.stem}  {len(chunks):4d} chunks  {transcript['title']}")
    print(f"\nDone. {total} chunks in {config.CHUNK_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk transcripts into timestamped pieces")
    parser.add_argument("--video-id", help="Only chunk this video")
    parser.add_argument("--chunk-seconds", type=float, default=config.CHUNK_SECONDS)
    parser.add_argument("--overlap-seconds", type=float, default=config.OVERLAP_SECONDS)
    args = parser.parse_args()
    chunk_all(args.video_id, args.chunk_seconds, args.overlap_seconds)
