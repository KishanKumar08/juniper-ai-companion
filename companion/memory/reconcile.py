from .. import llm, config


def _deterministic_match(new_fact, active_facts):
    for f in active_facts:
        if f["attribute"] == new_fact["attribute"]:
            if f["value"].strip().lower() == str(new_fact["value"]).strip().lower():
                return ("DUPLICATE", f)
            return ("UPDATE", f)
    return None


_RECONCILE_SYSTEM = """\
You maintain a companion's memory of a user. A new fact was just extracted.
Decide how it relates to the facts already on file. Output JSON only.

You get:
  NEW: {attribute, value}
  EXISTING: [{id, attribute, value}]

Return:
{"action": "ADD" | "DUPLICATE" | "UPDATE" | "RETIRE",
 "target_id": <id of the existing fact this affects, or null>,
 "reason": "<one short line>"}

Meaning:
- ADD: unrelated to everything existing; keep both.
- DUPLICATE: the same information as an existing fact; nothing to change.
- UPDATE: it replaces an existing fact whose value is now wrong/outdated
  (target_id = that fact). This includes cases where the attribute keys differ
  but they describe the same slice of the user's life (e.g. NEW
  relationship_status=single UPDATES an existing has_partner fact).
- RETIRE: an existing fact is no longer true and the new fact just negates it
  without giving a fresh value (target_id = that fact).

Be conservative: only UPDATE/RETIRE when the new fact genuinely conflicts with a
specific existing one. When in doubt, ADD.
"""


def _llm_reconcile(new_fact, active_facts):
    if not active_facts:
        return {"action": "ADD", "target_id": None, "reason": "nothing on file"}
    existing = [
        {"id": f["id"], "attribute": f["attribute"], "value": f["value"]}
        for f in active_facts
    ]
    payload = {
        "NEW": {"attribute": new_fact["attribute"], "value": new_fact["value"]},
        "EXISTING": existing,
    }
    messages = [
        {"role": "system", "content": _RECONCILE_SYSTEM},
        {"role": "user", "content": str(payload)},
    ]
    result = llm.chat_json(messages, model=config.EXTRACT_MODEL, temperature=0.0)
    if not isinstance(result, dict) or "action" not in result:
        # Safe default: keep it, don't destroy anything.
        return {"action": "ADD", "target_id": None, "reason": "reconcile failed, defaulting to ADD"}
    return result


def integrate_fact(store, new_fact, turn):
    """Apply one extracted fact to the store with contradiction handling.

    Returns a small dict describing what happened (handy for logging / the demo).
    """
    subject = new_fact.get("subject", "user")
    active = store.active_facts(subject=subject)

    # Stage 1: deterministic same-attribute match. If the user gives a new value
    # for an attribute we already track (job -> job, relationship_status ->
    # relationship_status), that's an update - trust it, no LLM needed.
    det = _deterministic_match(new_fact, active)
    if det and det[0] == "DUPLICATE":
        return {"action": "DUPLICATE", "attribute": new_fact["attribute"]}
    if det and det[0] == "UPDATE":
        old = det[1]
        new_id = store.add_fact(
            subject, new_fact["attribute"], str(new_fact["value"]),
            new_fact.get("type", "durable"), float(new_fact.get("confidence", 0.7)),
            valid_from=turn, source_text=new_fact.get("source", ""),
        )
        store.supersede_fact(old["id"], new_id=new_id, at_turn=turn)
        return {"action": "UPDATE", "superseded_id": old["id"], "new_id": new_id,
                "reason": "same-attribute update (deterministic)"}

    # Stage 2: no same-attribute match, so ask the LLM whether this new fact
    # contradicts a DIFFERENT existing attribute or is genuinely new.
    decision = _llm_reconcile(new_fact, active)
    action = decision.get("action", "ADD")
    target_id = decision.get("target_id")

    if action == "DUPLICATE":
        return {"action": "DUPLICATE", "attribute": new_fact["attribute"]}

    if action == "RETIRE" and target_id is not None:
        store.supersede_fact(target_id, new_id=None, at_turn=turn, retire_only=True)
        # A retire often still carries a new fact worth keeping 
        new_id = store.add_fact(
            subject, new_fact["attribute"], str(new_fact["value"]),
            new_fact.get("type", "durable"), float(new_fact.get("confidence", 0.7)),
            valid_from=turn, source_text=new_fact.get("source", ""),
        )
        return {"action": "RETIRE+ADD", "retired_id": target_id, "new_id": new_id,
                "reason": decision.get("reason", "")}

    if action == "UPDATE" and target_id is not None:
        new_id = store.add_fact(
            subject, new_fact["attribute"], str(new_fact["value"]),
            new_fact.get("type", "durable"), float(new_fact.get("confidence", 0.7)),
            valid_from=turn, source_text=new_fact.get("source", ""),
        )
        store.supersede_fact(target_id, new_id=new_id, at_turn=turn)
        return {"action": "UPDATE", "superseded_id": target_id, "new_id": new_id,
                "reason": decision.get("reason", "")}

    # Default will be add.
    new_id = store.add_fact(
        subject, new_fact["attribute"], str(new_fact["value"]),
        new_fact.get("type", "durable"), float(new_fact.get("confidence", 0.7)),
        valid_from=turn, source_text=new_fact.get("source", ""),
    )
    return {"action": "ADD", "new_id": new_id}


def expire_state_facts(store, now_turn, ttl):
    """Decay: state facts older than ttl turns are retired automatically, so the
    companion stops treating a stale mood/task as current. Durable facts are
    never touched by this."""
    expired = []
    for f in store.active_facts():
        if f["fact_type"] == "state" and (now_turn - f["valid_from"]) > ttl:
            store.supersede_fact(f["id"], new_id=None, at_turn=now_turn, retire_only=True)
            expired.append(f["id"])
    return expired
