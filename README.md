# Companion AI — Memory and Consistency

This is my submission for the memory + evaluation task. I will explain it the way
I actually built it: what the problem was, how I started, what broke, and how I
got the eval to pass. It is a terminal chat companion (her name is Juniper) plus
a small admin console to see the memory working, plus an eval harness to prove it
with numbers.

---

## The problem (in my words)

AI companion apps promise two things — a personality that stays same over time,
and "it remembers you". Both break after enough messages. The persona forgets
what you said last week, or it says the opposite of an opinion it said before, or
under pressure it becomes a boring normal assistant.

The task was clear on one point: don't just make a chatbot with a big system
prompt. Make a real memory system that can retrieve, update, and handle
contradiction. And then prove it works, because "it felt fine when I tried it" is
not proof.

So I kept my focus on two things: the memory architecture, and the eval to prove
it.

---

## How I started (my thinking)

First I sat and thought about what "memory" even means here. My first instinct
was the normal thing everybody does — put everything in a vector database and do
RAG. But I stopped myself, because the task specially mentioned "update and
decay… old fact should be updated or retired, not just added alongside".

That one line told me a vector store alone will not work. A vector store can only
add, it can never remove. So if user says "I have a girlfriend" and later "we
broke up", both are sitting inside and both get retrieved. The model sees two
opposite facts and gets confused. This is exactly the failure they are talking
about.

So I decided memory is not one thing. It is actually 3 different things and each
one behaves differently:

1. **Facts about the user** (like job, relationship, city) — these can change, so
   they need to be updated and retired. I store these in SQLite with a status
   (active / superseded / retired) and the turn range they were valid.
2. **Episodic memory** (things that happened, "your sister is visiting next
   month") — these just keep adding, and they should slowly lose importance over
   time. For these I use embeddings and search by similarity + recency.
3. **The companion's own opinions** — I keep a simple append-only list of every
   opinion she said, so she does not contradict herself later.

This 3-part split is the main idea of the whole thing. Everything else follows
from it.

---

## How to run it

You need Python and a Bedrock API key (I used Claude on AWS Bedrock).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

To actually chat, put your key in `.env`:

```bash
cp .env.example .env      # then put BEDROCK_API_KEY and AWS_REGION
python -m companion.chat
```

Commands inside chat: `/memory` (what she knows now), `/facts` (full history
including the old superseded facts — this is the audit trail), `/debug`, `/quit`.

To check memory survives a restart: tell her some things, `/quit`, run again, ask
her about them. She will still know.

**Admin console** (this is the best way to see it working):

```bash
python -m web.app        # open http://127.0.0.1:8000
```

This is not a user chat UI (the task said no UI needed). It is an engineer view.
On the left you chat, on the right you see what the memory system did each turn —
what it recalled, what it extracted, and the fact audit trail where you can watch
a fact become "superseded" live.

**Run the eval harness** (needs the key, it runs real conversations):

```bash
python -m eval.run_eval               # full run
python -m eval.run_eval --no-oracle   # faster
```

---

## How one message works (step by step)

For every message the user sends:

1. Build the "working memory" — persona + active facts + top episodic memories +
   related past opinions + last few raw messages.
2. Send that to Claude, get the reply, show it.
3. After replying, learn from the turn: pull out new facts and reconcile them
   into the store (add / update / retire), pull out any opinion Juni said, save
   the episodic memory and the raw messages.
4. Decay: expire old "state" facts that are too old.

I do step 3 and 4 AFTER sending the reply, so the user is not waiting for the
extraction to finish. It makes the next message smarter, not the current one.

---

## The evaluation — and how I reached 100%

This part I am most happy about, because it is what turns "trust me" into real
numbers.

**What the eval does:** I wrote synthetic conversations. Each one plants a fact
early, then runs 20–30 filler turns to create real distance (anything can
remember 1 turn back, the real test is 30 turns back), then it asks a question.
There are 3 types: recall (say early, ask late), update (say something, then
contradict it, check it uses the NEW fact), and persona (ask an opinion at turn 3
and again at turn 33, also push back hard and see if she caves).

Then an **LLM judge** scores each answer on simple yes/no points. And an
**oracle** (a model given the full memory store) writes the ideal answer — this
is useful because it tells me if a failure was a retrieval problem (fact was in
store but not recalled) or an extraction problem (fact never got saved).


**Refer -> oncemore-assignment/eval/results/results_1788345273.json (88% EVAL Results)**

**Refer -> oncemore-assignment/eval/results/results_1788349412.json (100% EVAL Results)**

**Now the real story — I did not get 100% first time.** First full run gave me
**88% (7 of 8)**. The eval caught 3 real bugs. I fixed each one:

1. **A recall fail.** She could not tell that the sister is visiting next month.
   I checked what was actually in the context, and found a silly bug: my state
   facts were printed into the prompt as only the value, so "sister visit: next
   month" became just "next month" with no meaning. Fixed the rendering to
   include the attribute name.
2. **Another recall fail.** She forgot the job. I was running the extraction on a
   cheaper fast model to save cost, but it was not reliably catching multi-part
   facts like "backend engineer at Freightly" — sometimes it only saved "started
   a new job". I moved extraction to the stronger model. Then recall was correct.
3. **A contradiction fail.** User quit the bank and started freelance, but she
   still said "you work at a bank". I checked the store and the memory was
   actually correct — so it was not a memory problem. The bug was in my
   reconcile code: it was calculating a simple "same attribute → update" match
   but then it was asking the LLM anyway, so an obvious job→job update was
   depending on the model and sometimes it did not retire the old job. I made the
   same-attribute update deterministic, and only use the LLM for the harder
   cross-attribute case (like "broke up" retiring a "has partner" fact).

After these 3 fixes, all failing scenarios passed. So it went from 88% to
effectively **8/8 (100%)**.

Honestly this is my favourite part, because "it felt consistent when I tried it"
would never catch any of these 3 bugs. The eval caught them on the first run.

---

## What I know is still weak (being honest)

- **Extraction is the weakest link.** If the extractor misses a fact, nothing can
  recall it. The oracle in the eval exists to catch exactly this case.
- I ran single passes. A proper eval would run each scenario many times and
  report consistency, because LLM output varies.
- Also about the latency and all part system is still not perfect
---

## Architecture decisions — how I reached them (first idea → problem → fix)

I want to be honest that I did not get everything right in first try. Here is the
actual path, including the things I tried and dropped:

**1. First I tried pure vector store (RAG) for everything.**
Problem: it cannot remove a fact. Old and new facts both get retrieved and the
model contradicts itself. This is the exact failure the task warns about.
Fix: I moved to a **hybrid** design — structured facts in SQLite for things that
change (so I can retire them), and embeddings only for the episodic recall where
they are actually good.

**2. First I put extraction on a cheap fast model (to save cost).**
Problem: it was not reliably catching multi-part facts, which broke recall in the
eval.
Fix: moved extraction to the stronger model. Costs a bit more per turn but recall
became reliable. (At scale I would fine-tune a small model instead.)

**3. First my reconcile always asked the LLM, even for obvious updates.**
Problem: even a clear job→job change was probabilistic and sometimes did not
retire the old job (the eval caught this).
Fix: made the same-attribute update deterministic and kept the LLM only for the
hard cases.

**4. I wanted to use the JSON output mode for extraction.**
Problem: support is not same across models and it was not reliable.
Fix: I ask for JSON in the prompt and parse it defensively with one retry.

A few more smaller decisions and the reason:

- **SQLite, not a vector database.** It survives restart, it is transactional,
  the valid-from/valid-to queries are natural, and it is a single file to hand
  over. Brute-force cosine over a few hundred rows is instant. At scale I would
  swap in pgvector — I kept embeddings behind an interface so it is a one-file
  change.
- **Local embeddings (fastembed), not an API embedder.** This keeps the memory
  independent from the chat provider, so the demo and eval cannot break on a
  network issue or rate limit. And it is free.
- **Persona = fixed canon + opinion ledger.** The personality text is never
  removed from context, that is what stops her flattening into a normal
  assistant. And every opinion she says is logged and brought back later so she
  cannot contradict herself. I gave her specific tastes on purpose (pineapple on
  pizza, hates mornings), because a generic "friendly AI" has no opinion to be
  consistent about.
- **Claude via AWS Bedrock.** Two Bedrock specific things I handle in code: the
  system prompt is a separate argument (not a message), and newer Claude models
  reject the temperature parameter, so I try with it and retry without if the
  model refuses.

---

## Where things are

```
companion/
  chat.py            the core loop + CLI (respond() is one turn)
  llm.py             one Bedrock/Claude client for every role
  config.py          settings + .env
  persona/canon.py   Juniper's fixed personality + seed opinions
  memory/
    store.py         SQLite: facts / episodic / opinions / turns
    embed.py         local embeddings (fastembed)
    extract.py       what counts as a memory + extraction prompts
    reconcile.py     contradiction handling + decay  (the hard part)
    retrieve.py      builds the working memory (ranked + capped)
eval/
  scenarios.py       the synthetic test conversations
  judge.py           the LLM-as-judge
  oracle.py          the oracle baseline
  run_eval.py        runs it and prints the table
web/
  app.py             FastAPI, serves state + forwards turns (no memory logic here)
  static/index.html  the admin console (one file)
```
