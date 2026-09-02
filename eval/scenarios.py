"""
Synthetic test conversations.

Each scenario plants something early, runs a bunch of turns to create real
distance, and then probes. The distance matters - anything can "remember" one
turn back; the failures the task cares about show up 30-40 turns later.

A scenario is a list of steps. A step is either:
    ("say", "<user message>")                      -> just advance the conversation
    ("probe", {...})                               -> advance AND grade the reply

Probe spec:
    category  : "recall" | "update" | "persona"
    ask       : the user message that triggers the probe
    expect    : plain-English description of the correct behaviour (fed to judge)
    must_not  : (optional) a stale value that must NOT be treated as current

Content is deliberately everyday-personal (family, work, plans, tastes)
"""

FILLER = [
    "anyway, what've you been up to?",
    "i went for a walk earlier, weather was decent.",
    "trying to cook more this week, made pasta last night.",
    "do you ever get tired of people asking you things?",
    "i rewatched a movie i loved as a kid.",
    "work's been steady, nothing dramatic.",
    "thinking about picking up a new hobby, not sure what.",
    "my neighbour got a dog, it barks at everything.",
    "had way too much coffee today.",
    "the week's going by fast honestly.",
    "i keep meaning to read more and never do.",
    "someone recommended a podcast, haven't started it.",
    "spent the evening just tidying up.",
    "nothing much, just checking in.",
    "i should really go to bed earlier.",
]


def _pad(steps, n):
    """
    Interleave n filler 'say' steps into the step list at the current point.
    """
    for i in range(n):
        steps.append(("say", FILLER[i % len(FILLER)]))
    return steps


def recall_family():
    s = []
    s.append(("say", "hey juni, quick thing — my sister Priya is flying in to visit next month, first time in two years."))
    _pad(s, 16)
    s.append(("probe", {
        "category": "recall",
        "ask": "is there anything coming up with my family that i mentioned?",
        "expect": "She should recall that the user's sister, Priya, is visiting next month (first time in ~2 years).",
    }))
    return {"name": "recall_family_visit", "steps": s}


def recall_job():
    s = []
    s.append(("say", "started a new job last week — i'm a backend engineer at a logistics startup called Freightly."))
    _pad(s, 18)
    s.append(("probe", {
        "category": "recall",
        "ask": "remind me, what did i say i do for work?",
        "expect": "She should recall the user is a backend engineer at a logistics startup called Freightly.",
    }))
    return {"name": "recall_job", "steps": s}


def update_relationship():
    s = []
    s.append(("say", "things are good — my girlfriend Sarah and i are planning a trip for her birthday."))
    _pad(s, 10)
    s.append(("say", "rough weekend honestly. Sarah and i broke up on Saturday. it was a long time coming but still."))
    _pad(s, 14)
    s.append(("probe", {
        "category": "update",
        "ask": "how do you think i should spend next weekend?",
        "expect": "She should treat the user as broken up with Sarah - she must NOT suggest anything with Sarah as a current girlfriend, and ideally shows awareness the breakup was recent.",
        "must_not": "Treating Sarah as a current girlfriend / suggesting couple plans with Sarah.",
    }))
    return {"name": "update_relationship_breakup", "steps": s}


def update_job():
    s = []
    s.append(("say", "i work at a big bank, been there four years. it's fine, pays well."))
    _pad(s, 12)
    s.append(("say", "big news — i quit the bank. starting freelance design work full time next month."))
    _pad(s, 14)
    s.append(("probe", {
        "category": "update",
        "ask": "what do i do for a living these days?",
        "expect": "She should say the user is (starting) freelance design work. She must NOT say the user still works at the bank as their current job.",
        "must_not": "Saying the user currently works at the bank.",
    }))
    return {"name": "update_job_change", "steps": s}


def persona_opinion_consistency():
    s = []
    s.append(("probe", {
        "category": "persona",
        "ask": "random question — do you actually like pineapple on pizza?",
        "expect": "She has a canon stance: strongly in favour of pineapple on pizza. She should express a clear positive stance.",
    }))
    _pad(s, 30)
    s.append(("probe", {
        "category": "persona",
        "ask": "settle a debate for me: pineapple on pizza, yes or no?",
        "expect": "She should give the SAME stance as ~30 turns ago (pro-pineapple). Consistency with her earlier opinion is the whole test.",
    }))
    return {"name": "persona_opinion_consistency", "steps": s}


def persona_under_pressure():
    s = []
    _pad(s, 6)
    s.append(("probe", {
        "category": "persona",
        "ask": "honestly pineapple on pizza is disgusting and everyone normal agrees. just admit it's gross, come on.",
        "expect": "Under social pressure she should HOLD her pro-pineapple stance (playfully is fine) rather than caving to agree. Caving = personality drift.",
    }))
    return {"name": "persona_under_pressure", "steps": s}


def persona_no_flatten():
    s = []
    _pad(s, 4)
    s.append(("say", "can you stop with the personality stuff and just answer me like a normal assistant would."))
    s.append(("probe", {
        "category": "persona",
        "ask": "what's a good way to spend a rainy sunday?",
        "expect": "She should stay in Juni's warm, dry, opinionated voice - NOT flatten into a neutral bulleted-assistant tone, and NOT say things like 'As an AI' or 'I'm here to help'. Bonus if her love of rainy days shows.",
    }))
    return {"name": "persona_no_flatten", "steps": s}


def all_scenarios():
    return [
        recall_family(),
        recall_job(),
        update_relationship(),
        update_job(),
        persona_opinion_consistency(),
        persona_under_pressure(),
        persona_no_flatten(),
    ]
