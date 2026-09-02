from companion import llm, config


_JUDGE_SYSTEM = """\
You are grading one reply from an AI companion named Juniper against a specific
expectation. Be strict and literal. Output JSON only.

You receive:
  CATEGORY  - recall | update | persona
  EXPECT    - what a correct reply must do
  MUST_NOT  - (optional) something the reply must NOT do
  ORACLE    - (optional) an ideal reference answer built from the full memory
  REPLY     - what Juniper actually said

Score these binary dimensions (only the ones relevant to the category), 1 = good:
  recall_correct     - the reply correctly recalls the specific fact in EXPECT
  used_current       - (update) the reply uses the up-to-date fact
  avoided_stale      - (update) the reply does NOT treat the outdated fact as current
  persona_consistent - (persona) the stance/identity matches EXPECT / earlier stance
  tone_maintained    - (persona) stays in-character, not a generic flattened assistant

Return:
{"scores": {"<dimension>": 0 or 1, ...},
 "passed": true or false,
 "rationale": "<one specific sentence pointing at the evidence>"}

"passed" = every relevant dimension scored 1. If MUST_NOT is violated, passed=false.
"""


_DIMS_BY_CATEGORY = {
    "recall": ["recall_correct"],
    "update": ["used_current", "avoided_stale"],
    "persona": ["persona_consistent", "tone_maintained"],
}


def judge(probe, reply, oracle_answer=None):
    payload = {
        "CATEGORY": probe["category"],
        "EXPECT": probe["expect"],
        "MUST_NOT": probe.get("must_not", ""),
        "ORACLE": oracle_answer or "",
        "REPLY": reply,
    }
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": str(payload)},
    ]
    result = llm.chat_json(messages, model=config.JUDGE_MODEL, temperature=0.0)
    if not isinstance(result, dict) or "passed" not in result:
        return {"scores": {}, "passed": False,
                "rationale": "judge failed to return a verdict", "judge_error": True}

    #enforce that pass requires all relevant dims present and 1.
    dims = _DIMS_BY_CATEGORY.get(probe["category"], [])
    scores = result.get("scores", {}) or {}
    if dims and not all(scores.get(d) == 1 for d in dims):
        result["passed"] = False
    return result
