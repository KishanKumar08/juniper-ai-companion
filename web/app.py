import os
import threading

from fastapi import FastAPI, Body
from fastapi.responses import FileResponse, JSONResponse

from companion import config
from companion.memory import store as store_mod
from companion.chat import respond, seed_if_new

_HERE = os.path.dirname(__file__)
_lock = threading.Lock()    
_store = None


def get_store():
    global _store
    if _store is None:
        _store = store_mod.MemoryStore(config.DB_PATH, check_same_thread=False)
        seed_if_new(_store)
    return _store


app = FastAPI(title="Juniper memory console")


@app.on_event("startup")
def _gate():
    config.require_credentials()


@app.get("/")
def index():
    return FileResponse(os.path.join(_HERE, "static", "index.html"))


@app.get("/api/state")
def api_state():
    with _lock:
        return _snapshot(get_store())


@app.post("/api/chat")
def api_chat(payload: dict = Body(...)):
    message = (payload or {}).get("message", "").strip()
    if not message:
        return JSONResponse({"error": "empty message"}, status_code=400)
    with _lock:
        store = get_store()
        try:
            reply, trace = respond(store, message)
        except Exception as e:  # surface Bedrock/errors in the UI instead of a 500 blob
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)
        return {"reply": reply, "trace": trace, "state": _snapshot(store)}


@app.post("/api/reset")
def api_reset():
    global _store
    with _lock:
        if _store is not None:
            _store.close()
            _store = None
        try:
            os.remove(config.DB_PATH)
        except OSError:
            pass
        return _snapshot(get_store())


def _snapshot(store):
    """Everything the console renders, in one payload."""
    active = store.active_facts()
    all_facts = store.all_facts()
    opinions = store.all_opinions()
    return {
        "turn": store.current_turn(),
        "active_facts": active,
        "all_facts": all_facts,
        "opinions": opinions,
        "episodic": store.all_episodic(),
        "turns": store.all_turns(),
        "stats": {
            "active": len(active),
            "durable": sum(1 for f in active if f["fact_type"] == "durable"),
            "state": sum(1 for f in active if f["fact_type"] == "state"),
            "superseded": sum(1 for f in all_facts if f["status"] != "active"),
            "opinions": len(opinions),
            "episodic": len(store.all_episodic()),
        },
        "config": {
            "companion_model": config.COMPANION_MODEL,
            "extract_model": config.EXTRACT_MODEL,
            "state_ttl": config.STATE_FACT_TTL,
        },
    }


def main():
    config.require_credentials()
    import uvicorn
    print("Juniper memory console -> http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
