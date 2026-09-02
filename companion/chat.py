"""
Per-turn pipeline:
  1. Read user input.
  2. Assemble working memory (retrieve.build_working_memory).
  3. Build messages = persona system prompt + recent raw turns + this message.
  4. Call the companion model, print the reply.
  5. Learn from the turn: extract user facts -> reconcile into the store;
     extract Juni's opinions -> ledger; store the exchange as episodic + raw log.
  6. Decay: expire stale state facts.
"""
import json

from . import config, llm
from .persona import canon
from .memory import store as store_mod
from .memory import extract, reconcile, retrieve


BANNER = """\
  Juniper (Juni) — a companion who actually remembers.
  Commands:  /memory  show what she remembers
             /facts   show the full fact history (incl. superseded)
             /debug   toggle showing what got recalled each turn
             /quit
"""


def seed_if_new(store):
    if store.all_opinions():
        return
    for topic, stance, text in canon.SEED_OPINIONS:
        store.add_opinion(turn=0, topic=topic, stance=stance, text=text)


def respond(store, user_message):
    now_turn = store.current_turn()

    memory_block, retrieved = retrieve.build_working_memory(store, user_message, now_turn)
    system = canon.system_prompt(memory_block)

    messages = [{"role": "system", "content": system}]
    for role, content in store.recent_turns(config.RECENT_TURNS):
        messages.append({
            "role": "assistant" if role == "companion" else "user",
            "content": content,
        })
    messages.append({"role": "user", "content": user_message})

    reply = llm.chat(messages, model=config.COMPANION_MODEL, temperature=0.75)

    # --- persist + learn ---
    turn = store.bump_turn()
    store.add_turn(turn, "user", user_message)
    store.add_turn(turn, "companion", reply)
    store.add_episodic(turn, f"User: {user_message}")

    learned = _learn_from_turn(store, user_message, reply, turn)
    expired = reconcile.expire_state_facts(store, now_turn=turn, ttl=config.STATE_FACT_TTL)

    trace = {
        "turn": turn,
        "retrieved": retrieved,   # what went into context (facts, episodic w/ scores, opinions)
        "learned": learned,       # what came out (extracted facts, reconcile actions, opinions)
        "expired_state_facts": expired,
    }
    return reply, trace


def _learn_from_turn(store, user_message, reply, turn):
    trace = {"facts": [], "reconcile": [], "events": [], "opinions": []}

    extracted = extract.extract_user_memory(user_message)
    for fact in extracted.get("facts", []):
        if not fact.get("attribute") or fact.get("value") in (None, ""):
            continue
        result = reconcile.integrate_fact(store, fact, turn)
        trace["facts"].append(fact)
        trace["reconcile"].append(result)
    for event in extracted.get("events", []):
        if event:
            store.add_episodic(turn, event)
            trace["events"].append(event)

    for op in extract.extract_opinions(reply):
        if op.get("topic") and op.get("stance"):
            store.add_opinion(turn, op["topic"], op["stance"], op.get("text", ""))
            trace["opinions"].append(op)

    return trace


def _print_memory(store):
    facts = [f for f in store.active_facts()]
    print("\n  What Juni currently believes about you:")
    if not facts:
        print("    (nothing yet)")
    for f in facts:
        tag = "" if f["fact_type"] == "durable" else "  [state]"
        print(f"    - {f['attribute'].replace('_',' ')}: {f['value']}{tag}")
    ops = store.all_opinions()
    print("\n  Opinions she's on record with:")
    for o in ops:
        print(f"    - {o['topic']}: {o['stance']}")
    print()


def _print_facts(store):
    print("\n Full fact history (this is the audit trail contradiction handling leaves):")
    for f in store.all_facts():
        line = f"    #{f['id']} [{f['status']}] {f['attribute']}: {f['value']}"
        if f["status"] != "active":
            line += f"  (valid turns {f['valid_from']}–{f['valid_to']})"
        print(line)
    print()


def main():
    config.require_credentials()
    store = store_mod.MemoryStore(config.DB_PATH)
    seed_if_new(store)

    print(BANNER)
    if store.current_turn() > 0:
        print(f"(picking up where we left off — {store.current_turn()} turns of history)\n")

    debug = False
    try:
        while True:
            try:
                user_message = input("you > ").strip()
            except EOFError:
                break
            if not user_message:
                continue
            if user_message == "/quit":
                break
            if user_message == "/memory":
                _print_memory(store)
                continue
            if user_message == "/facts":
                _print_facts(store)
                continue
            if user_message == "/debug":
                debug = not debug
                print(f"  [debug {'on' if debug else 'off'}]")
                continue

            reply, trace = respond(store, user_message)
            print(f"\njuni > {reply}\n")
            if debug:
                print(f"  [trace: {json.dumps(trace, default=str)}]\n")
    finally:
        store.close()
        print("\n  (memory saved — she'll remember next time.)")


if __name__ == "__main__":
    main()
