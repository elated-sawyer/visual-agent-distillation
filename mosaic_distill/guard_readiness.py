"""Wait without rendering for a fresh lease; retain every canonical safety check."""
from __future__ import annotations
import time
from .render_guard import require_render_guard as _check_guard

MAX_WAIT_SECONDS = 30.0
POLL_SECONDS = 0.25
_events = []

def require_render_guard():
    started = None
    attempts = 0
    while True:
        attempts += 1
        try:
            result = _check_guard()
        except RuntimeError as exc:
            if str(exc) != "Render guard heartbeat is stale":
                raise
            now = time.monotonic()
            if started is None:
                started = now
            if now - started >= MAX_WAIT_SECONDS:
                raise
            time.sleep(min(POLL_SECONDS, MAX_WAIT_SECONDS - (now - started)))
        else:
            if started is not None:
                _events.append({"wait_seconds": time.monotonic() - started,
                                "checks": attempts, "time": time.time()})
            return result

def take_wait_events():
    events = list(_events)
    _events.clear()
    return events
