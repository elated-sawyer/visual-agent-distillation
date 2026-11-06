"""Exception provenance without exception messages, prompts, URLs or credentials."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

_GUARD_ERRORS = {
    "Render guard heartbeat is stale": "guard_heartbeat_stale",
    "Render guard is tripped; automatic restart is forbidden": "guard_tripped",
    "Render guard supervisor is no longer active": "guard_supervisor_inactive",
    "Renderer is outside the supervised process tree": "guard_wrong_ancestry",
    "Render guard host/boot identity mismatch": "guard_identity_mismatch",
}
def exception_context(exc, stage):
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        frames.append({"file": Path(tb.tb_frame.f_code.co_filename).name,
                       "function": tb.tb_frame.f_code.co_name, "line": tb.tb_lineno})
        tb = tb.tb_next
    result = {"stage": stage, "type": type(exc).__name__,
              "code": _GUARD_ERRORS.get(str(exc), "unclassified"),
              "frames": frames}
    lease_path = os.environ.get("MOSAIC_RENDER_GUARD")
    if lease_path:
        try:
            lease = json.loads(Path(lease_path).read_text())
            result["guard"] = {"status": lease.get("status"),
                "heartbeat_age_seconds": time.monotonic() - lease["heartbeat_monotonic"],
                "supervisor_pid": lease.get("pid"),
                "latched": Path(lease["latch"]).exists()}
        except Exception as inspection_error:
            result["guard_inspection_error_type"] = type(inspection_error).__name__
    return result
