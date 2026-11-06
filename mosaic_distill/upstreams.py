"""Verify downloaded sources against the committed byte manifest before importing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify_upstream(name: str, root: Path = ROOT) -> Path:
    lock = json.loads((root / "upstreams.lock.json").read_text())[name]
    path = root / "external" / name
    for relative, expected in lock["files"].items():
        actual = hashlib.sha256((path / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Changed upstream source: {name}/{relative}")
    # Reject unexpected Python modules, which could shadow the pinned implementations.
    extras = {str(p.relative_to(path)) for p in path.rglob("*.py")
              if ".git" not in p.parts} - set(lock["files"])
    if extras:
        raise ValueError(f"Unpinned Python sources in {name}: {sorted(extras)}")
    return path
