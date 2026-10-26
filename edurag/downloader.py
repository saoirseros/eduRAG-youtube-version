"""Step 1: download only the audio of every video in a YouTube playlist."""
import argparse
import json

import yt_dlp

from .config import AUDIO_DIR, META_DIR


def _save_metadata(info: dict, playlist_title: str | None) -> dict | None:
    video_id = info["id"]
    files = [f for f in AUDIO_DIR.glob(f"{video_id}.*") if f.suffix not in (".part", ".ytdl")]
    if not files:
        return None
    meta = {
        "video_id": video_id,
        "title": info.get("title"),
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "channel": info.get("channel") or info.get("uploader"),
        "duration": info.get("duration"),
        "playlist_title": playlist_title,
        "audio_file": files[0].name,
    }
    (META_DIR / f"{video_id}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def _progress(d: dict) -> None:
    if d["status"] == "finished":
        print(f"[downloaded] {d['filename']}")


def download_playlist(url: str, limit: int | None = None) -> list[dict]:
    """Download audio for a playlist (or single video). Already-downloaded files are skipped."""
    opts = {
        # raw audio stream, no ffmpeg conversion needed (faster-whisper decodes it directly)
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
        "ignoreerrors": True,
        "playlistend": limit,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "progress_hooks": [_progress],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    if info is None:
        raise RuntimeError("Could not fetch the URL.")

    playlist_title = info.get("title") if "entries" in info else None
    entries = [e for e in info["entries"] if e] if "entries" in info else [info]

    results = []
    for entry in entries:
        meta = _save_metadata(entry, playlist_title)
        if meta:
            results.append(meta)
        else:
            print(f"[skip] no audio found for {entry.get('id')}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download audio of a YouTube playlist")
    parser.add_argument("url", help="YouTube playlist (or video) URL")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N videos")
    args = parser.parse_args()

    videos = download_playlist(args.url, args.limit)
    print(f"\nDone. {len(videos)} video(s) ready in {AUDIO_DIR}")
    for v in videos:
        print(f"  {v['video_id']}  {v['title']}")
