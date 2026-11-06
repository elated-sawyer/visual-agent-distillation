"""Versioned initial-observation clarification; native steps and scoring unchanged."""
import hashlib
from mosaic_distill.spatial import VagenSpatialGym as NativeSpatialGym


class VagenSpatialGym(NativeSpatialGym):
    def __init__(self, protocol, split, store):
        super().__init__(protocol, split, store)
        entry = self.protocol["initial_image_clarification"]
        assert entry["version"] == "spatial-intro-images-v1"
        assert hashlib.sha256(entry["text"].encode()).hexdigest() == entry["sha256"]

    async def reset(self, seed):
        observation, task = await super().reset(seed)
        for part in observation["content"]:
            if part["type"] == "text":
                part["text"] = self.protocol["initial_image_clarification"]["text"] + part["text"]
                return observation, task
        raise ValueError("No initial observation text to clarify")
