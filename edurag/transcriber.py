"""Step 2: transcribe downloaded audio with faster-whisper, keeping segment timestamps."""
import argparse
import json
import time

from faster_whisper import WhisperModel

from . import config

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        print(f"Loading Whisper '{config.WHISPER_MODEL}' ({config.WHISPER_COMPUTE_TYPE}) ...")
        _model = WhisperModel(
            config.WHISPER_MODEL, device="cpu", compute_type=config.WHISPER_COMPUTE_TYPE
        )
    return _model


def transcribe_video(meta: dict, language: str | None, force: bool = False) -> dict | None:
    """Transcribe one video and save data/transcripts/<video_id>.json. Skips if already done."""
    video_id = meta["video_id"]
    out_path = config.TRANSCRIPT_DIR / f"{video_id}.json"
    if out_path.exists() and not force:
        print(f"[skip] {video_id} already transcribed")
        return json.loads(out_path.read_text(encoding="utf-8"))

    audio_path = config.AUDIO_DIR / meta["audio_file"]
    print(f"[transcribing] {video_id}  {meta['title']}")

    segments_iter, info = _get_model().transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        beam_size=5,
        # avoids hallucination loops that can cascade across long lectures
        condition_on_previous_text=False,
    )

    total = info.duration or meta.get("duration") or 0
    start_time = time.time()
    segments = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
        if total:
            pct = min(seg.end / total * 100, 100)
            elapsed = time.time() - start_time
            print(f"\r  {pct:5.1f}%  ({elapsed/60:.1f} min elapsed)", end="", flush=True)
    print()

    transcript = {
        "video_id": video_id,
        "title": meta["title"],
        "language": info.language,
        "duration": total,
        "segments": segments,
    }
    # write only once complete, so an interrupted run never leaves a partial transcript
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    print(f"[saved] {out_path.name}  ({len(segments)} segments)")
    return transcript


def transcribe_all(video_id: str | None = None, language: str | None = config.WHISPER_LANGUAGE,
                   force: bool = False) -> None:
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(config.META_DIR.glob("*.json"))]
    if video_id:
        metas = [m for m in metas if m["video_id"] == video_id]
    if not metas:
        print("No videos found. Run the downloader first.")
        return
    for i, meta in enumerate(metas, 1):
        print(f"\n({i}/{len(metas)})")
        transcribe_video(meta, language, force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe downloaded audio with Whisper")
    parser.add_argument("--video-id", help="Only transcribe this video")
    parser.add_argument("--language", default=config.WHISPER_LANGUAGE,
                        help="Language code (en, ...) or 'auto'")
    parser.add_argument("--force", action="store_true", help="Re-transcribe existing transcripts")
    args = parser.parse_args()

    lang = None if args.language == "auto" else args.language
    transcribe_all(args.video_id, lang, args.force)
