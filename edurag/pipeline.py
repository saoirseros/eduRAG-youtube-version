"""Step 7a: one function that turns a YouTube video (or a whole playlist) into searchable chunks in Qdrant.

Per video: captions (fast) -> else download audio + Whisper -> chunk -> embed -> upload.
"""
import argparse
import json
import logging

from . import captions, chunker, config, downloader, embedder, playlists, transcriber, vectorstore
from .config import cookie_opts_from_env

log = logging.getLogger(__name__)


def _save_json(path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _noop(status, **extra):
    pass


def process_video(video_id: str, collection: str, cookie_opts: dict | None = None, on_status=_noop) -> dict:
    """Run every stage for one video. on_status(status, **extra) is called as it moves along."""
    transcript_path = config.TRANSCRIPT_DIR / f"{video_id}.json"
    meta_path = config.META_DIR / f"{video_id}.json"

    if transcript_path.exists() and meta_path.exists():
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        on_status("fetching info")
        info = downloader.fetch_info(video_id, cookie_opts)

        segments = None
        try:
            segments = captions.fetch_segments(info, cookie_opts)
        except Exception:
            log.warning("captions failed for %s, falling back to Whisper", video_id, exc_info=True)

        if segments:
            meta = downloader.save_metadata(info, audio_file=None)
            transcript = {
                "video_id": video_id, "title": meta["title"], "language": "en", "source": "captions",
                "duration": meta["duration"], "segments": segments,
            }
            _save_json(transcript_path, transcript)
        else:
            if (info.get("duration") or 0) > config.MAX_VIDEO_SECONDS:
                raise RuntimeError("No captions and the video is too long to transcribe locally.")
            on_status("downloading audio")
            audio_file = downloader.download_audio(video_id, cookie_opts)
            meta = downloader.save_metadata(info, audio_file=audio_file)
            on_status("transcribing", progress=0)
            transcript = transcriber.transcribe_video(
                meta, config.WHISPER_LANGUAGE, on_progress=lambda pct: on_status("transcribing", progress=round(pct)),
            )

    chunks = chunker.chunk_transcript(transcript, meta)
    if not chunks:
        raise RuntimeError("No speech was found in this video.")

    on_status("indexing")
    vectors = embedder.embed_texts([c["text"] for c in chunks])
    vectorstore.ensure_collection(collection)
    vectorstore.replace_video_chunks(video_id, chunks, vectors, collection)
    return {"chunks": len(chunks), "source": transcript.get("source", "whisper"), "title": meta["title"]}


def prepare_playlist(url: str, max_videos: int = config.DEFAULT_MAX_VIDEOS, cookie_opts: dict | None = None) -> dict:
    """Validate the URL, list the videos and register the playlist (nothing is downloaded yet)."""
    playlist_id = playlists.parse_playlist_url(url)
    return playlists.register_playlist(playlists.fetch_playlist(playlist_id, cookie_opts), max_videos)


class JobCancelled(Exception):
    """Raised inside a running video when the user cancels the playlist build."""


def run_playlist(playlist_id: str, cookie_opts: dict | None = None, should_stop=lambda: False) -> dict:
    """Process every not-yet-done video of a registered playlist. Failures are recorded per video."""
    record = playlists.get_playlist(playlist_id)
    if record is None:
        raise KeyError(playlist_id)
    playlists.update_playlist(playlist_id, state="building")
    vectorstore.ensure_collection(record["collection"])

    for v in record["videos"]:
        if v["status"] == "done":
            continue
        if should_stop():
            playlists.update_playlist(playlist_id, state="cancelled")
            return playlists.get_playlist(playlist_id)
        vid = v["video_id"]

        def on_status(status, _vid=vid, **extra):
            # called often (every Whisper segment), so this is also where a cancel takes effect
            if should_stop():
                raise JobCancelled()
            playlists.update_video(playlist_id, _vid, status=status, error=None, **extra)

        try:
            result = process_video(vid, record["collection"], cookie_opts, on_status)
            playlists.update_video(playlist_id, vid, status="done", progress=100, error=None,
                                   chunks=result["chunks"], source=result["source"])
        except JobCancelled:
            playlists.update_video(playlist_id, vid, status="queued", progress=0)
            playlists.update_playlist(playlist_id, state="cancelled")
            return playlists.get_playlist(playlist_id)
        except Exception as exc:
            log.exception("video %s failed", vid)
            playlists.update_video(playlist_id, vid, status="failed", error=str(exc)[:200] or type(exc).__name__)

    final = playlists.get_playlist(playlist_id)
    state = "ready" if any(v["status"] == "done" for v in final["videos"]) else "failed"
    playlists.update_playlist(playlist_id, state=state)
    return playlists.get_playlist(playlist_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a searchable knowledge base from a YouTube playlist")
    parser.add_argument("url", help="YouTube playlist URL")
    parser.add_argument("--max-videos", type=int, default=config.DEFAULT_MAX_VIDEOS)
    parser.add_argument("--cookies-from-browser", help="e.g. firefox, chrome, edge (if YouTube asks to sign in)")
    parser.add_argument("--cookies", help="Path to a cookies.txt file")
    args = parser.parse_args()

    cookie_opts = cookie_opts_from_env()
    if args.cookies_from_browser:
        cookie_opts["cookiesfrombrowser"] = (args.cookies_from_browser,)
    if args.cookies:
        cookie_opts["cookiefile"] = args.cookies

    record = prepare_playlist(args.url, args.max_videos, cookie_opts)
    print(f"Playlist: {record['title']}  ({record['total_videos']} videos, building first {len(record['videos'])})")
    result = run_playlist(record["id"], cookie_opts)
    print(f"\nState: {result['state']}   Collection: {result['collection']}")
    for v in result["videos"]:
        extra = f"{v.get('chunks', 0)} chunks via {v.get('source')}" if v["status"] == "done" else v.get("error")
        print(f"  [{v['status']}] {v['title']}  - {extra}")
