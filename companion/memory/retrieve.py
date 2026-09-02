from .. import config


def build_working_memory(store, user_message, now_turn):
    facts = store.active_facts()
    durable = [f for f in facts if f["fact_type"] == "durable"]
    state = [f for f in facts if f["fact_type"] == "state"]

    episodic = store.search_episodic(
        user_message, top_k=config.EPISODIC_TOP_K,
        now_turn=now_turn, half_life=config.RECENCY_HALF_LIFE,
    )
    opinions = store.search_opinions(user_message, top_k=config.OPINION_TOP_K)

    lines = []

    if durable:
        lines.append("What you know about them:")
        for f in durable:
            lines.append(f"  - {_humanize(f['attribute'])}: {f['value']}")

    if state:
        lines.append("\nRecent / current context (may have changed):")
        for f in state:
            lines.append(f"  - {_humanize(f['attribute'])}: {f['value']}")

    if episodic:
        lines.append("\nThings they've told you before that feel relevant now:")
        for text, turn, score, _sim in episodic:
            lines.append(f"  - {text}")

    if opinions:
        lines.append("\nOpinions YOU'VE already expressed (stay consistent with these):")
        for op in opinions:
            lines.append(f"  - on {op['topic']}: {op['stance']}")

    if not lines:
        block = "(You don't have any notes on this person yet - this is early days.)"
    else:
        block = "\n".join(lines)

    debug = {
        "durable_facts": len(durable),
        "state_facts": len(state),
        "episodic_recalled": [(t, round(s, 3)) for t, _turn, s, _sim in episodic],
        "opinions_recalled": [op["topic"] for op in opinions],
    }
    return block, debug


def _humanize(attribute):
    return attribute.replace("_", " ")
