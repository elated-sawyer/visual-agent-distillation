"""SpatialGym adapter: native observations, actions and continuous cogmap metrics."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

from .records import fingerprint, validate_message
from .upstreams import ROOT, verify_upstream


class VagenSpatialGym:
    def __init__(self, protocol, split, store):
        if split not in {"train", "eval"}:
            raise ValueError("Explicit training or evaluation split required")
        if protocol.get("schema") != "spatial-protocol-v1" or protocol["history"] != "full":
            raise ValueError("Unknown SpatialGym protocol")
        cfg = copy.deepcopy(protocol["env_config"])
        if cfg.get("exp_type") != "active" or cfg.get("render_mode") != "vision":
            raise ValueError("This adapter implements native active visual exploration")
        if protocol["max_turns"] != cfg["max_exp_steps"] + 1:
            raise ValueError("Reserve exactly one final cognitive-map turn")
        variant = protocol["dataset"].get("variant", "1-room")
        splits = {"1-room": {"train": list(range(16)), "eval": list(range(16,20))},
                  "2-room": {"train": [], "eval": list(range(20))}}
        if variant not in splits or protocol["room_splits"] != splits[variant]:
            raise ValueError("Preserve the pinned room split and eval-only two-room boundary")
        if variant == "2-room" and split != "eval":
            raise ValueError("Two-room data is final evaluation only")
        path = verify_upstream("VAGEN").resolve()
        revision = json.loads((ROOT / "upstreams.lock.json").read_text())["VAGEN"]["commit"]
        if protocol["upstream"] != {"name": "VAGEN", "commit": revision}:
            raise ValueError("Upstream identity drift")
        for name, module in tuple(sys.modules.items()):
            location = getattr(module, "__file__", None)
            if (name == "vagen" or name.startswith("vagen.")) and location:
                if not Path(location).resolve().is_relative_to(path):
                    raise RuntimeError("Foreign VAGEN already imported")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        from vagen.envs.spatial_gym.spatial_gym_env import SpatialGym
        from vagen.envs.spatial_gym.env_config import SpatialGymConfig
        from vagen.envs._common.observations import convert_obs_to_content
        from vagen.envs import TurnLimit

        # Construct the dataclass first: upstream dict handling silently falls back
        # to defaults on invalid config, which must never change an experiment.
        config = SpatialGymConfig(**cfg)
        self.raw = SpatialGym(config)
        self.env = TurnLimit(self.raw, protocol["max_turns"])
        self._convert = convert_obs_to_content
        self.protocol, self.split, self.store = copy.deepcopy(protocol), split, store
        self.done = True

    def _message(self, obs, role="user"):
        mm = obs.get("multi_modal_input") or {}
        if any(k != "<image>" and v for k, v in mm.items()):
            raise ValueError("Unsupported observation modality")
        images = iter(mm.get("<image>", []))
        content = []
        for part in self._convert(obs):
            if part["type"] == "text":
                content.append(dict(part))
            elif part["type"] == "image":
                content.append({"type": "image_ref", "image": self.store.put(next(images))})
            else:
                raise ValueError("Unsupported observation content")
        if next(images, None) is not None:
            raise ValueError("Unconsumed images")
        message = {"role": role, "content": content}
        validate_message(message, role=role)
        # Invalid-action feedback and final-map instructions are natively text-only.
        return message

    async def reset(self, seed):
        if type(seed) is not int or seed not in self.protocol["room_splits"][self.split]:
            raise ValueError("Room is not in the requested split")
        dataset = self.protocol["dataset"]
        data_root = Path(self.protocol["env_config"]["data_dir"])
        expected = dataset["rooms"][str(seed)]
        for name, digest in expected["files"].items():
            p = data_root / name
            if not p.resolve().is_relative_to(data_root.resolve()):
                raise ValueError("Dataset path escapes root")
            if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                raise ValueError("Dataset content drift: " + name)
        obs, _ = await self.env.reset(seed=seed)
        self.done = False
        # Room index/directory name must not manufacture distinct task identities.
        content_ids = {name.removeprefix(f"run{seed:02d}/"): digest
                       for name, digest in expected["files"].items()}
        if len(content_ids) != len(expected["files"]):
            raise ValueError("Ambiguous dataset file basenames")
        room_sha = fingerprint(content_ids)
        task = {"environment": "VAGEN/SpatialGym", "split": self.split, "seed": seed,
                "room_index": seed, "room_sha256": room_sha,
                "dataset_revision": dataset["revision"],
                "task_id": f"spatial:{dataset.get('variant', '1-room')}:{dataset['revision'][:12]}:{room_sha}"}
        return self._message(obs), task

    async def system_prompt(self):
        return self._message(await self.env.system_prompt(), role="system")

    async def step(self, response):
        if self.done:
            raise RuntimeError("Environment must be reset")
        obs, reward, done, info = await self.env.step(response)
        self.done = bool(done)
        if not math.isfinite(float(reward)):
            raise ValueError("Nonfinite native reward")
        truncated = bool(info.get("truncated", False))
        return self._message(obs), float(reward), bool(done) and not truncated, truncated, info

    async def close(self):
        self.done = True
        await self.env.close()


def episode_metrics(episode):
    """No custom binary threshold: retain native score and its component metrics."""
    if episode.get("status") != "complete":
        raise ValueError("Infrastructure errors are unscored")
    final = episode["turns"][-1].get("info", {}) if episode["turns"] else {}
    names = ("cogmap_score", "cogmap_dir", "cogmap_facing", "cogmap_pos",
             "cogmap_exploration_coverage")
    submitted = "cogmap_score" in final
    return {**{name: float(final.get(name, 0.0)) for name in names},
            "map_scored": submitted, "native_success": bool(episode["success"]),
            "forced_term": bool(final.get("forced_term", False)),
            "context_exhausted": episode.get("stop_reason") == "context_limit",
            "format_correct_turns": sum(bool(t.get("info", {}).get("format_correct"))
                                        for t in episode["turns"]),
            "turns": len(episode["turns"])}
