"""One LLM client for every role in the project.

The companion, the fact extractor, the contradiction reconciler, the eval judge,
and the oracle are all just (model, messages) with different prompts, so there's
one client behind a uniform `chat()` / `chat_json()`. It talks to Claude on AWS
Bedrock via the official Anthropic SDK.

Callers pass a list of {"role": "...", "content": "..."} messages that may start
with a system entry. On Bedrock the system prompt is a separate argument, not a
message, so `_split_system` pulls it out.
"""
import json
import re

from . import config

_client = None


def _get_client():
    global _client
    if _client is None:
        config.require_credentials()
        # Lazy import so the memory layer / offline tests don't need the SDK.
        from anthropic import AnthropicBedrock
        if config.BEDROCK_API_KEY:
            # Bearer-token auth (a Bedrock API key). Region still needed for the
            # endpoint. Passing api_key forces bearer mode.
            _client = AnthropicBedrock(
                api_key=config.BEDROCK_API_KEY, aws_region=config.AWS_REGION)
        else:
            # Fall back to the standard AWS credential chain (access key/secret).
            _client = AnthropicBedrock(aws_region=config.AWS_REGION)
    return _client


def _split_system(messages):
    """[{'role':'system'|'user'|'assistant', ...}] ->
    (system_text, conversation_without_system)."""
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    conv = [m for m in messages if m["role"] != "system"]
    return "\n\n".join(system_parts), conv


def chat(messages, model, temperature=0.7, max_tokens=1024):
    """Plain text completion."""
    client = _get_client()
    system, conv = _split_system(messages)
    kwargs = dict(model=model, max_tokens=max_tokens, messages=conv)
    if system:
        kwargs["system"] = system
    # Newer Claude models (Sonnet 4.6+, Opus 5, ...) reject `temperature` with a
    # 400. Older ones (3.5) accept it and it genuinely helps the companion's
    # warmth. So: try with it, and if the model refuses it, retry without -
    # instead of hard-coding an assumption about which model we're pointed at.
    try:
        resp = client.messages.create(temperature=temperature, **kwargs)
    except Exception as e:  # noqa: BLE001 - want a broad net for the 400 shape
        if _is_param_error(e):
            resp = client.messages.create(**kwargs)
        else:
            raise
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _is_param_error(exc):
    s = str(exc).lower()
    return ("temperature" in s or "top_p" in s or "unsupported" in s
            or "not supported" in s) and ("400" in s or "invalid" in s or "unexpected" in s)


def chat_json(messages, model, temperature=0.2, max_tokens=1024):
    """Completion where we expect JSON back.

    I ask for JSON in the prompt and dig it out defensively rather than relying on
    a structured-output flag, so there's one parsing path regardless of model.
    One retry with a louder instruction if the first parse fails - in practice
    that's enough.
    """
    raw = chat(messages, model, temperature=temperature, max_tokens=max_tokens)
    parsed = _try_parse(raw)
    if parsed is not None:
        return parsed

    retry = messages + [{
        "role": "user",
        "content": "That wasn't valid JSON. Reply with ONLY the JSON, no prose, no code fences.",
    }]
    raw = chat(retry, model, temperature=0.0, max_tokens=max_tokens)
    parsed = _try_parse(raw)
    if parsed is None:
        # Give up gracefully rather than crash the chat loop mid-conversation.
        return None
    return parsed


def _try_parse(raw):
    if not raw:
        return None
    # Strip ```json fences if the model added them anyway.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Last resort: grab the outermost {...} or [...] and try that.
        match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
    return None
