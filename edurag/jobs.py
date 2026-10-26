"""Step 7b: background job runner. One worker thread builds one playlist at a time (the CPU is the bottleneck)."""
import logging
import queue
import threading

from . import config, pipeline, playlists

log = logging.getLogger("edurag")

# a video in one of these statuses was cut off by a crash or restart
_IN_PROGRESS = ("fetching info", "downloading audio", "transcribing", "indexing")


def recover_interrupted() -> None:
    """After a restart, playlists left 'queued'/'building' are marked 'interrupted' so the user can resume."""
    for p in playlists.list_playlists():
        if p["state"] not in ("queued", "building"):
            continue
        for v in p["videos"]:
            if v["status"] in _IN_PROGRESS:
                playlists.update_video(p["id"], v["video_id"], status="queued", progress=0)
        playlists.update_playlist(p["id"], state="interrupted")


class JobRunner:
    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._cancel_flags: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._work, daemon=True, name="edurag-jobs")

    def start(self) -> None:
        self._thread.start()

    def is_active(self, playlist_id: str) -> bool:
        with self._lock:
            return playlist_id in self._cancel_flags

    def submit(self, playlist_id: str) -> bool:
        """Queue a playlist build. Returns False if it is already queued or running."""
        with self._lock:
            if playlist_id in self._cancel_flags:
                return False
            self._cancel_flags[playlist_id] = threading.Event()
        playlists.update_playlist(playlist_id, state="queued")
        self._queue.put(playlist_id)
        return True

    def cancel(self, playlist_id: str) -> bool:
        with self._lock:
            flag = self._cancel_flags.get(playlist_id)
        if flag:
            flag.set()
        return flag is not None

    def _work(self) -> None:
        cookie_opts = config.cookie_opts_from_env()
        while True:
            playlist_id = self._queue.get()
            flag = self._cancel_flags[playlist_id]
            try:
                if flag.is_set():
                    playlists.update_playlist(playlist_id, state="cancelled")
                else:
                    pipeline.run_playlist(playlist_id, cookie_opts, flag.is_set)
            except Exception:
                log.exception("playlist %s failed", playlist_id)
                try:
                    playlists.update_playlist(playlist_id, state="failed")
                except KeyError:
                    pass  # the record was deleted while it was building
            finally:
                with self._lock:
                    self._cancel_flags.pop(playlist_id, None)
