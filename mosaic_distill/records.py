"""Causal prefixes are rebuilt from explicit observation/action fields only."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import re

from .assets import ImageStore


def fingerprint(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_message(message: dict, *, role: str) -> None:
    if set(message) != {"role", "content"} or message["role"] != role:
        raise ValueError(f"Expected {role} message with explicit content only")
    if not isinstance(message["content"], list) or not message["content"]:
        raise ValueError("Empty message")
    for item in message["content"]:
        if item.get("type") == "text":
            if set(item) != {"type", "text"} or not isinstance(item["text"], str):
                raise ValueError("Invalid text part")
        elif item.get("type") == "image_ref" and role == "user":
            if set(item) != {"type", "image"} or not isinstance(item["image"], dict):
                raise ValueError("Invalid image part")
            ref = item["image"]
            if set(ref) != {"sha256", "mime", "width", "height"}:
                raise ValueError("Unexpected image metadata")
            if not isinstance(ref["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]):
                raise ValueError("Invalid screenshot identity")
            if ref["mime"] != "image/png" or any(type(ref[k]) is not int or ref[k] <= 0
                                                for k in ("width", "height")):
                raise ValueError("Invalid image metadata")
        else:
            raise ValueError("Unsupported message part")


def decision_messages(episode: dict, turn_index: int) -> list[dict]:
    """Full history, through observation t; exclude action t and every future field."""
    turns = episode["turns"]
    if type(turn_index) is not int or not 0 <= turn_index < len(turns):
        raise ValueError("Invalid decision index")
    validate_message(episode["system"], role="system")
    messages = [copy.deepcopy(episode["system"])]
    for index in range(turn_index + 1):
        turn = turns[index]
        validate_message(turn["observation"], role="user")
        if index and turns[index - 1]["next_observation"] != turn["observation"]:
            raise ValueError("Broken causal observation chain")
        messages.append(copy.deepcopy(turn["observation"]))
        if index < turn_index:
            if not isinstance(turn["response"], str):
                raise ValueError("Missing original response")
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": turn["response"]}]})
    return messages


def export_prefixes(episode: dict, *, image_requirement="current") -> list[dict]:
    if image_requirement not in {"current", "history"}:
        raise ValueError("Unknown image requirement")
    if episode.get("schema") != "mm-episode-v1":
        raise ValueError("Unknown episode schema")
    if episode.get("status") != "complete":
        raise ValueError("Incomplete or infrastructure-error episode cannot enter replay")
    if episode["task"]["split"] != "train":
        raise ValueError("Evaluation episodes cannot enter replay")
    if episode.get("source", {}).get("kind") not in {"student", "teacher"}:
        raise ValueError("Only real model collection enters replay; diagnostics cannot")
    if not episode["turns"]:
        raise ValueError("No decisions to replay")
    source_sha = fingerprint(episode)
    result = []
    for index in range(len(episode["turns"])):
        messages = decision_messages(episode, index)
        visible = messages[-1:] if image_requirement == "current" else messages
        if not any(p["type"] == "image_ref" for m in visible for p in m["content"]):
            raise ValueError("Multimodal replay requires a real image in the allowed context")
        body = {"schema": "mm-prefix-v1", "episode_sha256": source_sha,
                "task": copy.deepcopy(episode["task"]), "turn_index": index,
                "protocol_sha256": episode["protocol_sha256"], "messages": messages}
        if image_requirement == "history":
            body["image_requirement"] = "history"
        # Task identity stays metadata: the model input is exclusively `messages`.
        result.append({**body, "prefix_sha256": fingerprint(body)})
    return result


def to_chat_messages(messages: list[dict], store: ImageStore) -> list[dict]:
    """OpenAI image wire format for future collection; no network calls here."""
    result = []
    for message in messages:
        validate_message(message, role=message["role"])
        parts = []
        for item in message["content"]:
            if item["type"] == "text":
                parts.append(dict(item))
            else:
                data = base64.b64encode(store.read(item["image"])).decode("ascii")
                parts.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{data}"}})
        result.append({"role": message["role"], "content": parts})
    return result
