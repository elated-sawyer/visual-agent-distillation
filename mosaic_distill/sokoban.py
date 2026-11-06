"""Narrow adapter over pinned VAGEN; no changes to its rendering/actions/judge."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

from .render_guard import require_render_guard
from .assets import ImageStore
from .records import fingerprint, validate_message
from .upstreams import ROOT, verify_upstream


class VagenSokoban:
    def __init__(self, protocol: dict, split: str, store: ImageStore):
        require_render_guard()
        if split not in {"train", "eval"}:
            raise ValueError("Explicit train/eval partition required")
        if protocol.get("schema") != "sokoban-protocol-v1":
            raise ValueError("Unknown environment protocol")
        if protocol["history"] != "full":
            raise ValueError("Only explicit full-history protocol is implemented")
        if type(protocol["max_turns"]) is not int or protocol["max_turns"] <= 0:
            raise ValueError("Positive turn budget required")
        cfg = copy.deepcopy(protocol["env_config"])
        if cfg.get("render_mode") != "vision" or cfg.get("image_placeholder") != "<image>":
            raise ValueError("This adapter requires real image observations")
        if "map_partition" in cfg or cfg.get("state_reward"):
            raise ValueError("Partition is derived from split; state judge must remain disabled")
        # Upstream map generation temporarily uses global RNG. Run one episode at a
        # time per process; scale collection through separate worker processes.
        path = verify_upstream("VAGEN").resolve()
        revision = json.loads((ROOT / "upstreams.lock.json").read_text())["VAGEN"]["commit"]
        if protocol.get("upstream") != {"name": "VAGEN", "commit": revision}:
            raise ValueError("Protocol and source lock refer to different VAGEN versions")
        for name, module in tuple(sys.modules.items()):
            if name == "vagen" or name.startswith("vagen."):
                location = getattr(module, "__file__", None)
                if location and not Path(location).resolve().is_relative_to(path):
                    raise RuntimeError("Another VAGEN version was already imported")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        from vagen.envs.sokoban.sokoban_env import Sokoban
        from vagen.envs._common.observations import convert_obs_to_content
        self._convert = convert_obs_to_content
        cfg["map_partition"] = split
        self.raw = Sokoban(cfg)
        # Use upstream TurnLimit directly: no optional judge or reward wrapper.
        from vagen.envs import TurnLimit
        self.env = TurnLimit(self.raw, protocol["max_turns"])
        self.protocol = copy.deepcopy(protocol)
        self.split, self.store = split, store
        self.done = True

    def _message(self, obs: dict, role="user") -> dict:
        mm = obs.get("multi_modal_input") or {}
        if any(k != "<image>" and v for k, v in mm.items()):
            raise ValueError("Unsupported multimodal observation")
        parts = self._convert(obs)
        images = iter(mm.get("<image>", []))
        content = []
        for part in parts:
            if part["type"] == "image":
                content.append({"type": "image_ref", "image": self.store.put(next(images))})
            elif part["type"] == "text":
                content.append(dict(part))
            else:
                raise ValueError("Only images/text supported")
        message = {"role": role, "content": content}
        validate_message(message, role=role)
        if role == "user" and not any(p["type"] == "image_ref" for p in content):
            raise ValueError("Sokoban observation did not contain an image")
        return message

    async def reset(self, seed: int):
        require_render_guard()
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Expected a uint32 reset seed")
        obs, info = await self.env.reset(seed=seed)
        import numpy as np
        fixed = np.asarray(self.raw.env.room_fixed, dtype="<i8", order="C")
        state = np.asarray(self.raw.env.room_state, dtype="<i8", order="C")
        room_sha = hashlib.sha256(fixed.tobytes() + state.tobytes()).hexdigest()
        config = self.protocol["env_config"]
        bucket = int.from_bytes(bytes.fromhex(room_sha)[:8], "big") % config["map_partition_modulus"]
        held_out = bucket == config["map_partition_eval_bucket"]
        if held_out != (self.split == "eval"):
            raise ValueError("Upstream generated the wrong map partition")
        self.task = {"environment": "VAGEN/Sokoban", "split": self.split, "seed": seed,
                     "map_sha256": room_sha, "partition_bucket": bucket,
                     "task_id": f"sokoban:{fingerprint(self.protocol)[:12]}:{room_sha}"}
        self.done = False
        return self._message(obs), copy.deepcopy(self.task)

    async def system_prompt(self):
        return self._message(await self.env.system_prompt(), role="system")

    async def step(self, response: str):
        require_render_guard()
        if self.done:
            raise RuntimeError("Episode ended or reset was not called")
        # Intentionally let infrastructure exceptions propagate. They are unscored,
        # not synthetic task failures. Preserve response exactly, including bad format.
        obs, reward, done, info = await self.env.step(response)
        self.done = bool(done)
        truncated = bool(info.get("truncated", False))
        return self._message(obs), float(reward), bool(done) and not truncated, truncated, info

    async def close(self):
        self.done = True
        await self.env.close()
