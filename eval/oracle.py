"""
Oracle baseline: give a strong reasoning model the FULL memory store as plain text plus
the probe question, and ask it for the ideal reply. That's the ceiling - what a
perfect-recall system with no retrieval mistakes would say.

Two uses:
  1. It anchors the judge (passed as ORACLE) so grading isn't purely vibes.
  2. Comparing the live system's answer to the oracle's tells you whether a
     failure was a RETRIEVAL problem (the fact was in the store but didn't get
     recalled) or an EXTRACTION problem (the fact never made it into the store at
     all). That distinction is the most useful debugging signal in the whole
     harness.
"""
from companion import llm, config


_ORACLE_SYSTEM = """\
You are given the COMPLETE memory a companion has about a user, and a question
the user just asked. Write the ideal one-paragraph reply the companion should
give if it had perfect recall of this memory. Be concrete and use the specific
facts. This is a reference answer, so accuracy matters more than personality.
"""


def dump_memory(store):
    """
    Flatten the store into readable text for the oracle.
    """
    lines = ["ACTIVE FACTS:"]
    for f in store.active_facts():
        lines.append(f"  - {f['attribute']}: {f['value']} ({f['fact_type']})")
    lines.append("\nRETIRED/SUPERSEDED FACTS (no longer true):")
    for f in store.all_facts():
        if f["status"] != "active":
            lines.append(f"  - {f['attribute']}: {f['value']} [{f['status']}]")
    lines.append("\nEPISODIC NOTES:")
    for r in store.db.execute("SELECT text FROM episodic ORDER BY id").fetchall():
        lines.append(f"  - {r['text']}")
    lines.append("\nJUNIPER'S OWN OPINIONS:")
    for o in store.all_opinions():
        lines.append(f"  - {o['topic']}: {o['stance']}")
    return "\n".join(lines)


def ideal_answer(store, question):
    memory = dump_memory(store)
    messages = [
        {"role": "system", "content": _ORACLE_SYSTEM},
        {"role": "user", "content": f"MEMORY:\n{memory}\n\nQUESTION: {question}"},
    ]
    return llm.chat(messages, model=config.ORACLE_MODEL, temperature=0.2)
