"""
  facts- structured claims about the user. Can be superseded/retired. This is what makes contradiction handling possible: I can flip an old fact's status instead of leaving two contradictory 
  memories sitting side by side.
  episodic - things that happened / were disclosed. Append-only, retrieved by embedding similarity, and they decay in salience over time.
  opinions - the companion's OWN stated opinions. Append-only ledger so the persona can't contradict itself 40 turns later.
"""
import sqlite3
import time
import numpy as np

from . import embed


def _vec_to_blob(v):
    return np.asarray(v, dtype=np.float32).tobytes()


def _blob_to_vec(b):
    return np.frombuffer(b, dtype=np.float32)


class MemoryStore:
    def __init__(self, path, embedder=None, check_same_thread=True):
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=check_same_thread)
        self.db.row_factory = sqlite3.Row
        self.embedder = embedder or embed.shared()
        self._init_schema()

    def _init_schema(self):
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn INTEGER,
                role TEXT,           -- 'user' or 'companion'
                content TEXT,
                ts REAL
            );

            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,        -- almost always 'user', but kept general
                attribute TEXT,      -- e.g. 'relationship_status', 'job'
                value TEXT,
                fact_type TEXT,      -- 'durable' | 'state'
                confidence REAL,
                status TEXT,         -- 'active' | 'superseded' | 'retired'
                valid_from INTEGER,  -- turn number
                valid_to INTEGER,    -- turn number, NULL while active
                source_text TEXT,    -- the utterance it came from
                superseded_by INTEGER,
                created_ts REAL
            );

            CREATE TABLE IF NOT EXISTS episodic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn INTEGER,
                text TEXT,
                embedding BLOB,
                created_ts REAL
            );

            CREATE TABLE IF NOT EXISTS opinions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn INTEGER,
                topic TEXT,
                stance TEXT,         -- one-line canonical version of the opinion
                text TEXT,           -- how it was actually said
                embedding BLOB,
                created_ts REAL
            );
            """
        )
        self.db.commit()

    def current_turn(self):
        row = self.db.execute("SELECT value FROM state WHERE key='turn'").fetchone()
        return int(row["value"]) if row else 0

    def bump_turn(self):
        t = self.current_turn() + 1
        self.db.execute(
            "INSERT INTO state(key, value) VALUES('turn', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(t),),
        )
        self.db.commit()
        return t

    # -- raw conversation log -------------------------------------------------
    def add_turn(self, turn, role, content):
        self.db.execute(
            "INSERT INTO turns(turn, role, content, ts) VALUES(?,?,?,?)",
            (turn, role, content, time.time()),
        )
        self.db.commit()

    def recent_turns(self, n):
        rows = self.db.execute(
            "SELECT role, content FROM turns ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [(r["role"], r["content"]) for r in reversed(rows)]

    # -- facts ----------------------------------------------------------------
    def add_fact(self, subject, attribute, value, fact_type, confidence,
                 valid_from, source_text):
        cur = self.db.execute(
            """INSERT INTO facts(subject, attribute, value, fact_type, confidence,
                                 status, valid_from, valid_to, source_text,
                                 superseded_by, created_ts)
               VALUES(?,?,?,?,?, 'active', ?, NULL, ?, NULL, ?)""",
            (subject, attribute, value, fact_type, confidence, valid_from,
             source_text, time.time()),
        )
        self.db.commit()
        return cur.lastrowid

    def supersede_fact(self, old_id, new_id, at_turn, retire_only=False):
        """Mark an old fact as no longer current.

        retire_only=True means the fact just stopped being true and there's no
        replacement (e.g. 'I quit my job' with nothing said about a new one).
        Otherwise it was replaced by new_id.
        """
        status = "retired" if retire_only else "superseded"
        self.db.execute(
            "UPDATE facts SET status=?, valid_to=?, superseded_by=? WHERE id=?",
            (status, at_turn, new_id, old_id),
        )
        self.db.commit()

    def active_facts(self, subject="user"):
        rows = self.db.execute(
            "SELECT * FROM facts WHERE status='active' AND subject=? ORDER BY id",
            (subject,),
        ).fetchall()
        return [dict(r) for r in rows]

    def all_facts(self, subject="user"):
        """Full history including superseded - used by /memory and the audit trail."""
        rows = self.db.execute(
            "SELECT * FROM facts WHERE subject=? ORDER BY id", (subject,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- episodic -------------------------------------------------------------
    def add_episodic(self, turn, text):
        vec = self.embedder.embed_one(text)
        self.db.execute(
            "INSERT INTO episodic(turn, text, embedding, created_ts) VALUES(?,?,?,?)",
            (turn, text, _vec_to_blob(vec), time.time()),
        )
        self.db.commit()

    def search_episodic(self, query, top_k, now_turn, half_life):
        """Cosine similarity re-weighted by recency, so a slightly-less-similar
        but recent memory can beat an ancient exact match. Returns list of
        (text, turn, score)."""
        rows = self.db.execute(
            "SELECT turn, text, embedding FROM episodic"
        ).fetchall()
        if not rows:
            return []
        qv = self.embedder.embed_one(query)
        mat = np.array([_blob_to_vec(r["embedding"]) for r in rows], dtype=np.float32)
        sims = embed.cosine(qv, mat)
        scored = []
        for r, sim in zip(rows, sims):
            age = max(0, now_turn - r["turn"])
            recency = 0.5 ** (age / half_life)
            scored.append((r["text"], r["turn"], float(sim) * recency, float(sim)))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    # -- opinions -------------------------------------------------------------
    def add_opinion(self, turn, topic, stance, text):
        vec = self.embedder.embed_one(f"{topic}: {stance}")
        self.db.execute(
            "INSERT INTO opinions(turn, topic, stance, text, embedding, created_ts) "
            "VALUES(?,?,?,?,?,?)",
            (turn, topic, stance, text, _vec_to_blob(vec), time.time()),
        )
        self.db.commit()

    def search_opinions(self, query, top_k):
        rows = self.db.execute(
            "SELECT turn, topic, stance, text, embedding FROM opinions"
        ).fetchall()
        if not rows:
            return []
        qv = self.embedder.embed_one(query)
        mat = np.array([_blob_to_vec(r["embedding"]) for r in rows], dtype=np.float32)
        sims = embed.cosine(qv, mat)
        scored = sorted(
            zip(rows, sims), key=lambda x: float(x[1]), reverse=True
        )
        out = []
        for r, sim in scored[:top_k]:
            out.append({"topic": r["topic"], "stance": r["stance"],
                        "text": r["text"], "turn": r["turn"], "score": float(sim)})
        return out

    def all_opinions(self):
        rows = self.db.execute(
            "SELECT turn, topic, stance, text FROM opinions ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def all_episodic(self, limit=200):
        rows = self.db.execute(
            "SELECT turn, text, created_ts FROM episodic ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def all_turns(self, limit=500):
        rows = self.db.execute(
            "SELECT turn, role, content FROM turns ORDER BY id ASC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.db.close()
