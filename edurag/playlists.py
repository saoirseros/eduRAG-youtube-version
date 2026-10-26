"""Playlist helpers: validate a playlist URL, list its videos, and keep a small registry of built playlists."""
import json
import re
import threading
import time
import urllib.parse

import yt_dlp

from . import config

_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,64}$")
_UNAVAILABLE = ("[Private video]", "[Deleted video]")


def parse_playlist_url(url: str) -> str:
    """Return the playlist id from a YouTube URL. Only YouTube hosts are accepted (the URL ends up in yt-dlp)."""
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or (parsed.hostname or "").lower() not in _HOSTS:
        raise ValueError("Please paste a youtube.com playlist link.")
    ids = urllib.parse.parse_qs(parsed.query).get("list")
    playlist_id = ids[0] if ids else ""
    if not _ID_RE.match(playlist_id):
        raise ValueError("That link has no playlist in it. Open the playlist page and copy that link.")
    if playlist_id.startswith("RD"):
        raise ValueError("YouTube 'Mix' playlists are endless. Please use a normal playlist.")
    return playlist_id


def collection_name(playlist_id: str) -> str:
    return f"pl_{playlist_id}"


def fetch_playlist(playlist_id: str, cookie_opts: dict | None = None) -> dict:
    """List a playlist's title and videos (fast, nothing is downloaded)."""
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "extract_flat": "in_playlist",
            **(cookie_opts or {})}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/playlist?list={playlist_id}", download=False)
    if not info or not info.get("entries"):
        raise RuntimeError("Could not read that playlist (is it private or empty?).")
    videos = [
        {"video_id": e["id"], "title": e.get("title") or e["id"], "duration": e.get("duration")}
        for e in info["entries"]
        if e and e.get("id") and e.get("title") not in _UNAVAILABLE
    ]
    return {"id": playlist_id, "title": info.get("title") or playlist_id, "videos": videos}


# ---- registry: data/playlists.json  {playlist_id: record} ----
_lock = threading.RLock()


def _load() -> dict:
    if config.PLAYLIST_REGISTRY.exists():
        return json.loads(config.PLAYLIST_REGISTRY.read_text(encoding="utf-8"))
    return {}


def _save(data: dict) -> None:
    # write-then-rename so a crash never leaves a half-written registry
    tmp = config.PLAYLIST_REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(config.PLAYLIST_REGISTRY)


def list_playlists() -> list[dict]:
    with _lock:
        return sorted(_load().values(), key=lambda p: p["created_at"], reverse=True)


def get_playlist(playlist_id: str) -> dict | None:
    with _lock:
        return _load().get(playlist_id)


def register_playlist(playlist: dict, max_videos: int) -> dict:
    """Create (or refresh) the record for a playlist, keeping progress of videos that are already done."""
    with _lock:
        data = _load()
        old = data.get(playlist["id"], {})
        done = {v["video_id"]: v for v in old.get("videos", []) if v["status"] == "done"}
        videos = []
        for v in playlist["videos"][:max_videos]:
            videos.append(done.get(v["video_id"]) or {**v, "status": "queued", "progress": 0, "error": None})
        record = {
            "id": playlist["id"],
            "title": playlist["title"],
            "collection": collection_name(playlist["id"]),
            "state": "ready" if videos and all(v["status"] == "done" for v in videos) else "new",
            "total_videos": len(playlist["videos"]),
            "max_videos": max_videos,
            "created_at": old.get("created_at") or time.time(),
            "videos": videos,
        }
        data[playlist["id"]] = record
        _save(data)
        return record


def update_playlist(playlist_id: str, **fields) -> None:
    with _lock:
        data = _load()
        data[playlist_id].update(fields)
        _save(data)


def update_video(playlist_id: str, video_id: str, **fields) -> None:
    with _lock:
        data = _load()
        for v in data[playlist_id]["videos"]:
            if v["video_id"] == video_id:
                v.update(fields)
        _save(data)


def seed_existing_collection(collection: str = config.COLLECTION) -> None:
    """Show the playlist indexed before this feature existed (the 'lectures' collection) in the registry."""
    from . import vectorstore  # local import: keeps URL parsing usable without a Qdrant client

    with _lock:
        data = _load()
        if any(p["collection"] == collection for p in data.values()):
            return
        client = vectorstore.get_client()
        if not client.collection_exists(collection) or client.count(collection, exact=True).count == 0:
            return
        videos, title = [], None
        for chunk_file in sorted(config.CHUNK_DIR.glob("*.json")):
            meta_file = config.META_DIR / chunk_file.name
            if not meta_file.exists():
                continue
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            title = title or meta.get("playlist_title")
            videos.append({
                "video_id": meta["video_id"], "title": meta["title"], "duration": meta.get("duration"),
                "status": "done", "progress": 100, "error": None, "source": "whisper",
                "chunks": len(json.loads(chunk_file.read_text(encoding="utf-8"))),
            })
        if not videos:
            return
        record_id = f"existing-{collection}"
        data[record_id] = {
            "id": record_id, "title": title or "Existing lectures", "collection": collection,
            "state": "ready", "total_videos": len(videos), "max_videos": len(videos),
            "created_at": 0, "videos": videos,
        }
        _save(data)


def delete_playlist(playlist_id: str) -> bool:
    with _lock:
        data = _load()
        if playlist_id not in data:
            return False
        del data[playlist_id]
        _save(data)
        return True
