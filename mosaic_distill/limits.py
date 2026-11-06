"""Interpret only the pinned SGLang context validation, before any sampling."""
import re


class ContextBudgetExhausted(Exception):
    """Full causal history leaves no space for another response."""


def remaining_from_context_error(status, data, context_length, requested):
    if status != 400 or not isinstance(data, dict):
        return None
    error = data.get('error', data)
    if not isinstance(error, dict):
        return None
    message = error.get('message', '')
    if not isinstance(message, str):
        return None
    if 'Requested token count exceeds the model' in message:
        match = re.search(r'(\d+) tokens from the input messages and (\d+) tokens for the completion', message)
        if match is None or int(match[2]) != requested:
            return None
        prompt = int(match[1])
    else:
        match = re.search(r'The input \((\d+) tokens\) is longer than the model', message)
        if match is None:
            return None
        prompt = int(match[1])
    remaining = context_length - prompt
    # Reject a service-capacity issue rather than silently lowering the protocol.
    if remaining >= requested:
        return None
    return max(0, remaining)
