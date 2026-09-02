import json
from .. import llm, config


_EXTRACT_SYSTEM = """\
You extract durable memory from a chat between a user and their AI companion.
You are a precise information extractor, not a chatbot. Output JSON only.

Return this shape:
{
  "facts": [
    {"attribute": "<snake_case key, e.g. relationship_status, job, lives_in, has_pet>",
     "value": "<short value>",
     "type": "durable" | "state",
     "confidence": 0.0-1.0,
     "source": "<the user phrase this came from>"}
  ],
  "events": [
    "<one-sentence third-person note about something that happened or was disclosed>"
  ]
}

Rules:
- Only extract things worth remembering next week. Skip greetings, small talk,
  and anything about the conversation itself.
- "durable" = stable identity/relationships/preferences/plans. "state" = true now
  but expected to change (mood, today's task, a passing worry).
- Prefer few high-quality facts over many shaky ones. Empty lists are fine.
- attribute should be a stable key so a later update can match it. Use the SAME
  key for the same kind of fact (always "relationship_status", not sometimes
  "dating_status").
- Do not invent anything not supported by the user's message.
"""

_EXTRACT_FEWSHOT = [
    {"role": "user", "content": "User said: \"hey, how's it going?\""},
    {"role": "assistant", "content": '{"facts": [], "events": []}'},
    {"role": "user", "content": "User said: \"I just started a new job at a fintech startup, it's my third week and honestly I'm a little overwhelmed\""},
    {"role": "assistant", "content": '{"facts": [{"attribute": "job", "value": "works at a fintech startup", "type": "durable", "confidence": 0.9, "source": "started a new job at a fintech startup"}, {"attribute": "mood", "value": "overwhelmed at new job", "type": "state", "confidence": 0.8, "source": "a little overwhelmed"}], "events": ["User started a new job at a fintech startup and is in their third week."]}'},
]


def extract_user_memory(user_message):
    messages = (
        [{"role": "system", "content": _EXTRACT_SYSTEM}]
        + _EXTRACT_FEWSHOT
        + [{"role": "user", "content": f'User said: "{user_message}"'}]
    )
    result = llm.chat_json(messages, model=config.EXTRACT_MODEL, temperature=0.1)
    if not isinstance(result, dict):
        return {"facts": [], "events": []}
    result.setdefault("facts", [])
    result.setdefault("events", [])
    return result


_OPINION_SYSTEM = """\
You read one message written by an AI companion named Juniper and pull out any
OPINION, preference, taste, or self-claim she stated - the kind of thing she'd
need to stay consistent with later. Output JSON only.

Return:
{"opinions": [
  {"topic": "<short topic>",
   "stance": "<one-line canonical statement of her position>",
   "text": "<the actual phrase she used>"}
]}

Only include genuine stances (likes/dislikes, beliefs, self-description,
commitments). Ignore neutral questions, acknowledgements, and things she said
ABOUT the user. Empty list is fine and common.
"""


def extract_opinions(companion_message):
    messages = [
        {"role": "system", "content": _OPINION_SYSTEM},
        {"role": "user", "content": f'Juniper said: "{companion_message}"'},
    ]
    result = llm.chat_json(messages, model=config.EXTRACT_MODEL, temperature=0.1)
    if not isinstance(result, dict):
        return []
    return result.get("opinions", []) or []
