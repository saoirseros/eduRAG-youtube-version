"""Fast path: use YouTube's own captions (seconds) instead of running Whisper (about real-time)."""
import json

import yt_dlp


def _pick_track(info: dict) -> list[dict] | None:
    """Prefer human-made English subtitles, then English auto-captions (original track first)."""
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    for source in (manual, auto):
        for lang in sorted(source, key=lambda k: (k != "en-orig", k != "en", k)):
            if lang.split("-")[0] == "en":
                return source[lang]
    return None


def parse_json3(data: dict) -> list[dict]:
    """Turn YouTube's json3 caption events into [{start, end, text}], with no overlapping times."""
    segments = []
    for ev in data.get("events", []):
        text = "".join(s.get("utf8", "") for s in ev.get("segs", [])).replace("\n", " ").strip()
        if not text or "tStartMs" not in ev:
            continue
        start = ev["tStartMs"] / 1000
        segments.append({"start": start, "end": start + ev.get("dDurationMs", 0) / 1000, "text": text})
    # captions on screen overlap the next line; cut each one at the next start so timestamps stay ordered
    for cur, nxt in zip(segments, segments[1:]):
        cur["end"] = max(cur["start"], min(cur["end"], nxt["start"]))
    for s in segments:
        s["start"], s["end"] = round(s["start"], 2), round(s["end"], 2)
    return segments


def fetch_segments(info: dict, cookie_opts: dict | None = None) -> list[dict] | None:
    """Return caption segments for a video (from yt-dlp info), or None if no usable English captions."""
    track = _pick_track(info)
    if not track:
        return None
    fmt = next((f for f in track if f.get("ext") == "json3"), None)
    if not fmt:
        return None
    opts = {"quiet": True, "no_warnings": True, **(cookie_opts or {})}
    with yt_dlp.YoutubeDL(opts) as ydl:
        data = json.loads(ydl.urlopen(fmt["url"]).read())
    segments = parse_json3(data)
    # a handful of caption lines isn't a real transcript
    return segments if len(segments) >= 5 else None
