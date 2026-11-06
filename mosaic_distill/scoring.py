"""Audit sampled response scores before any future learner consumes them."""
import math


def response_logprobs(result: dict, response_token_ids: list[int]) -> list[float]:
    if not response_token_ids or any(type(t) is not int or t < 0 for t in response_token_ids):
        raise ValueError("Nonempty exact sampled token IDs are required")
    records = result["meta_info"]["input_token_logprobs"]
    if len(records) < len(response_token_ids):
        raise ValueError("Teacher returned fewer scores than response tokens")
    values = []
    for item, expected in zip(records[-len(response_token_ids):], response_token_ids):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError("Malformed native teacher score")
        if type(item[1]) is not int or item[1] != expected:
            raise ValueError("Teacher scored different response token IDs")
        lp = item[0]
        if isinstance(lp, bool) or not isinstance(lp, (float, int)) or not math.isfinite(lp):
            raise ValueError("Missing or nonfinite teacher log probability")
        if lp > 1e-6:
            raise ValueError("Positive teacher log probability")
        values.append(float(lp))
    return values
