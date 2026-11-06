"""Lossless, content-addressed screenshots. Model preprocessing happens downstream."""
from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
from pathlib import Path


class ImageStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def put(self, image) -> dict:
        if image.mode != "RGB":
            raise ValueError("Expected an upstream RGB observation; do not silently convert")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        ref = {"sha256": digest, "mime": "image/png", "width": image.width,
               "height": image.height}
        path = self._path(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Publish only complete bytes. Other process workers may write the same
        # screenshot concurrently; a reader must never observe a partial PNG.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".image-", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError("Existing screenshot content differs from its hash")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return ref

    def _path(self, ref: dict) -> Path:
        digest = ref.get("sha256", "")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid image hash")
        return self.root / digest[:2] / f"{digest}.png"

    def read(self, ref: dict) -> bytes:
        data = self._path(ref).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("Screenshot hash mismatch")
        if ref.get("mime") != "image/png":
            raise ValueError("Unsupported image MIME type")
        from PIL import Image
        with Image.open(io.BytesIO(data)) as image:
            if image.mode != "RGB" or image.size != (ref["width"], ref["height"]):
                raise ValueError("Screenshot metadata mismatch")
        return data
