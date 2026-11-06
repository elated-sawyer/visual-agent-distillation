"""One sequential episode per process; policy callback supplies original text."""
from __future__ import annotations

import copy

from .records import decision_messages, fingerprint
from .limits import ContextBudgetExhausted


async def collect_episode(env, policy, seed: int, source: dict) -> dict:
    episode = {"schema": "mm-episode-v1", "status": "running", "turns": [],
               "source": copy.deepcopy(source), "protocol_sha256": fingerprint(env.protocol)}
    try:
        observation, task = await env.reset(seed)
        episode.update(task=task, system=await env.system_prompt())
        for index in range(env.protocol["max_turns"]):
            episode["turns"].append({"observation": copy.deepcopy(observation)})
            response = await policy(decision_messages(episode, index))
            if not isinstance(response, str):
                raise TypeError("Policy callback must return exact original response text")
            episode["turns"][-1]["response"] = response
            observation, reward, terminated, truncated, info = await env.step(response)
            episode["turns"][-1].update(
                next_observation=copy.deepcopy(observation), reward=reward,
                terminated=terminated, truncated=truncated, info=copy.deepcopy(info))
            if terminated or truncated:
                episode.update(status="complete", success=bool(info.get("success", False)),
                               stop_reason="turn_limit" if truncated else "terminated")
                break
        else:
            raise RuntimeError("Environment did not honor the configured turn limit")
    except ContextBudgetExhausted:
        # No action was generated or executed for this pending decision.
        assert episode["turns"] and "response" not in episode["turns"][-1]
        pending = episode["turns"].pop()
        episode.update(status="complete", success=False, stop_reason="context_limit",
                       final_observation=pending["observation"])
    except Exception as exc:
        episode.update(status="error", success=None, error_type=type(exc).__name__)
        # Do not serialize exception messages: HTTP failures can contain credentials.
    finally:
        try:
            await env.close()
        except Exception as exc:
            episode.update(status="error", success=None, cleanup_error_type=type(exc).__name__)
    return episode
